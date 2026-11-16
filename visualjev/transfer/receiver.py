"""Frozen released receiver with the experiment's image slots and candidate readout.

The multimodal wrapper shares the loaded language object. External scoring
constructs slots from image dimensions and never calls native visual inference.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path

import torch
from .upstream import upstream_module

SIZE = {"shortest_edge": 65536, "longest_edge": 262144}
MAX_SEQUENCE = 2048


def stamp(module):
    return [(name, parameter._version, tuple(parameter.shape), str(parameter.dtype),
             str(parameter.device), parameter.requires_grad)
            for name, parameter in module.named_parameters()]


class FrozenReceiver:
    def __init__(self, base_path, checkpoint_path, device="cuda:0"):
        from accelerate import init_empty_weights
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        for name in ("JEV_LOAD_8BIT", "JEV_LOAD_4BIT", "JEV_DEVICE_MAP",
                     "JEV_MAX_MEMORY", "JEV_ALLOW_OFFLOAD", "JEV_TORCH_DTYPE"):
            if os.environ.get(name):
                raise ValueError(f"Unset {name} to use the recorded unquantized receiver")
        if torch.device(device).type != "cuda":
            raise ValueError("The released runtime requires a BF16-capable CUDA device")
        self.device = device
        base = Path(base_path)
        checkpoint = Path(checkpoint_path)
        config = json.loads((checkpoint / "model.json").read_text())
        self.temperature = float(json.loads((checkpoint / "temperature.json").read_text())["temperature"])
        self.candidate_prompts = upstream_module("api").candidate_prompts
        DecisionModel = upstream_module("model").DecisionModel
        self.decision = DecisionModel(str(base), config["revision"], device=device,
                                      lora_rank=0, max_length=MAX_SEQUENCE)
        self.decision.backbone = PeftModel.from_pretrained(
            self.decision.backbone, checkpoint / "adapter", is_trainable=False)
        saved_head = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
        self.decision.head.load_state_dict(saved_head, strict=True)
        expected = load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
        actual = get_peft_model_state_dict(self.decision.backbone)
        assert set(expected) == set(actual), "LoRA tensor names differ from the released checkpoint"
        for key, tensor in expected.items():
            assert torch.equal(tensor, actual[key].detach().cpu().to(tensor.dtype)), key
        for key, tensor in saved_head.items():
            assert torch.equal(tensor, self.decision.head.state_dict()[key].detach().cpu()), key
        self.decision.eval().requires_grad_(False)
        self.decision._validate_loaded_parameters()
        del actual, expected, saved_head

        cfg = AutoConfig.from_pretrained(base, local_files_only=True)
        cfg._attn_implementation = "sdpa"
        cfg.vision_config._attn_implementation = "sdpa"
        with init_empty_weights():
            self.mm = Qwen3_5Model(cfg)
        self.mm.language_model = self.decision.backbone
        index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        vision = {}
        for shard in sorted({v for k, v in index.items() if k.startswith("model.visual.")}):
            with safe_open(str(base / shard), framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key.startswith("model.visual."):
                        vision[key[len("model.visual."):]] = weights.get_tensor(key)
        assert set(vision) == set(self.mm.visual.state_dict())
        self.mm.visual.load_state_dict(vision, strict=True, assign=True)
        # Preserve the native FP32 nonpersistent rotary buffer.
        self.mm.visual.to(device=device)
        for key, tensor in vision.items():
            assert torch.equal(tensor, self.mm.visual.state_dict()[key].detach().cpu()), key
        del vision
        self.mm.eval().requires_grad_(False)
        assert self.mm.language_model is self.decision.backbone
        assert all(p.device.type == "cuda" and not p.requires_grad for p in self.mm.parameters())
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        self.initial_stamp = stamp(self.mm) + stamp(self.decision.head)
        self.identity = {"live_lora_equal_released": True, "live_head_equal_released": True,
                         "shared_language_object": True, "max_sequence": MAX_SEQUENCE,
                         "resize_policy": SIZE, "receiver_hidden_dim": self.decision.head.in_features}

    def prompts(self, question, candidates, state=""):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        if not 2 <= len(candidates) <= 255 or any(not isinstance(c, str) or not c for c in candidates):
            raise ValueError("Supply 2 to 255 nonempty candidate strings")
        return self.candidate_prompts(dict(state=state, question=question, kind="choice", options=candidates))

    @torch.no_grad()
    def prepare_external(self, question, candidates, image_size, state=""):
        from PIL import Image
        blank = Image.new("RGB", image_size, "white")
        proc = self.processor.image_processor(images=[blank], size=SIZE, return_tensors="pt")
        grid = proc["image_grid_thw"].to(self.device)
        replacement = self.processor.replace_image_token(proc, image_idx=0)
        batches = []
        for prompt in self.prompts(question, candidates, state):
            assert "<|image_pad|>" not in prompt
            text = self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>\n" + prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = self.processor.tokenizer(text.replace(self.processor.image_token, replacement),
                                           return_tensors="pt")["input_ids"].to(self.device)
            if ids.shape[1] > MAX_SEQUENCE:
                raise ValueError("Input exceeds the recorded 2048-token policy; no silent truncation")
            assert int(ids.eq(self.mm.config.image_token_id).sum()) == int(grid.prod() // 4)
            batches.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=grid,
                                mm_token_type_ids=torch.tensor(
                                    self.processor.create_mm_token_type_ids(ids.tolist()), device=self.device)))
        return batches, grid[0].tolist()

    @contextmanager
    def external_only(self):
        original = self.mm.visual.forward
        def denied(*args, **kwargs):
            raise RuntimeError("Native visual forward called during external-only scoring")
        self.mm.visual.forward = denied
        try:
            yield
        finally:
            self.mm.visual.forward = original

    def logits(self, batches, feature, order=None):
        """Input gradients remain available when fitting an external interface."""
        out = []
        for index in (range(len(batches)) if order is None else order):
            batch = batches[index]
            ids = batch["input_ids"]
            embeddings = self.mm.language_model.get_input_embeddings()(ids)
            mask, _ = self.mm.get_placeholder_mask(ids, inputs_embeds=embeddings, image_features=feature)
            embeddings = embeddings.masked_scatter(mask, feature.to(embeddings.dtype))
            positions, _ = self.mm.get_rope_index(
                ids, batch["mm_token_type_ids"], image_grid_thw=batch["image_grid_thw"],
                attention_mask=batch["attention_mask"])
            hidden = self.mm.language_model(inputs_embeds=embeddings, position_ids=positions,
                                            attention_mask=batch["attention_mask"], use_cache=False,
                                            return_dict=True).last_hidden_state
            out.append(self.decision.head(hidden[:, -1].float()).reshape(()))
        result = torch.stack(out)
        assert torch.isfinite(result).all()
        return result

    def no_image(self, question, candidates, state=""):
        out = []
        for prompt in self.prompts(question, candidates, state):
            text = self.decision.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = self.decision.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
            if ids.shape[1] > MAX_SEQUENCE:
                raise ValueError("Input exceeds 2048 tokens; no silent truncation")
            hidden = self.mm(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=False, return_dict=True).last_hidden_state
            out.append(self.decision.head(hidden[:, -1].float()).reshape(()))
        return torch.stack(out)

    def unchanged(self):
        assert stamp(self.mm) + stamp(self.decision.head) == self.initial_stamp, "Frozen receiver changed"
