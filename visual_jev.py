"""Visual-JEV V1: separated image/text features with the existing JEV head.

The baseline deliberately keeps preprocessing and orchestration outside the
normal ``jev.vl_smoke`` serving path.  It extracts the same final non-padding
language hidden state used by ``DecisionModel``/``VisionDecisionModel`` before
their scalar head, obtains one image feature through the existing
``VisualAlgorithm``, fuses both with ``LayerNorm(text + image)``, and reuses the
already-loaded scalar decision head.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import torch
from torch import nn

from visual_algorithm import VisualAlgorithm


def describe_tensor(name: str, value: torch.Tensor) -> None:
    """Print the shape, dtype, and device needed for wiring diagnostics."""
    print(
        f"{name}: shape={tuple(value.shape)} "
        f"dtype={value.dtype} device={value.device}"
    )


class VisualJEV(nn.Module):
    """Align visual tokens, fuse candidate features, and emit scalar logits."""

    def __init__(self, vision_dim, hidden_dim, decision_head=None, debug=False):
        super().__init__()
        self.vision_dim = int(vision_dim)
        self.hidden_dim = int(hidden_dim)
        self.visual_algorithm = VisualAlgorithm(
            vision_dim=self.vision_dim,
            hidden_dim=self.hidden_dim,
            debug=debug,
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.decision_head = (
            decision_head if decision_head is not None
            else nn.Linear(self.hidden_dim, 1, dtype=torch.float32)
        )
        if self.decision_head.in_features != self.hidden_dim:
            raise ValueError("decision_head input dimension must equal hidden_dim")
        if self.decision_head.out_features != 1:
            raise ValueError("decision_head must return one scalar per candidate")

    def align_visual(self, visual_tokens):
        """Map ``[batch, visual_tokens, vision_dim]`` to hidden-size vectors."""
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.vision_dim:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, vision_dim]"
            )
        parameter = next(self.visual_algorithm.parameters())
        visual_tokens = visual_tokens.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.visual_algorithm(visual_tokens)

    def fuse(self, text_feature, visual_feature):
        """Apply the V1 fusion: ``LayerNorm(text_feature + visual_feature)``."""
        if text_feature.ndim != 2 or visual_feature.ndim != 2:
            raise ValueError("text_feature and visual_feature must both be rank-2")
        if text_feature.shape != visual_feature.shape:
            raise ValueError("text_feature and visual_feature must have equal shapes")
        if text_feature.shape[-1] != self.hidden_dim:
            raise ValueError("feature dimension must equal hidden_dim")
        parameter = next(self.fusion_norm.parameters())
        text_feature = text_feature.to(device=parameter.device, dtype=parameter.dtype)
        visual_feature = visual_feature.to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return self.fusion_norm(text_feature + visual_feature)

    def score(self, fused_feature):
        """Reuse the JEV FP32 scalar-head boundary and return one logit each."""
        if fused_feature.ndim != 2 or fused_feature.shape[-1] != self.hidden_dim:
            raise ValueError("fused_feature must have shape [candidates, hidden_dim]")
        parameter = next(self.decision_head.parameters())
        logits = self.decision_head(
            fused_feature.to(device=parameter.device, dtype=parameter.dtype)
        )
        return logits.squeeze(-1)

    def forward(self, visual_tokens, text_feature):
        visual_feature = self.align_visual(visual_tokens)
        fused_feature = self.fuse(text_feature, visual_feature)
        return self.score(fused_feature)


@dataclass(frozen=True)
class VisualJEVOutput:
    """Intermediate features and normalized candidate probabilities."""

    text_features: torch.Tensor
    image_feature: torch.Tensor
    fused_features: torch.Tensor
    scores: torch.Tensor
    probabilities: torch.Tensor


class VisualJEVBaseline:
    """Inference-only V1 wiring around an existing ``VisionDecisionModel``.

    The image is encoded once and shared across candidates.  Candidate text is
    encoded without image tokens so visual evidence enters exactly once through
    ``VisualAlgorithm``.  This helper does not replace or modify
    ``VisionDecisionModel.forward`` and therefore cannot break ``jev.vl_smoke``.
    """

    def __init__(self, decision_model, *, debug=True):
        required = ("processor", "backbone", "head")
        missing = [name for name in required if not hasattr(decision_model, name)]
        if missing:
            raise TypeError(
                "decision_model is missing required attributes: " + ", ".join(missing)
            )
        if not hasattr(decision_model.backbone, "visual"):
            raise TypeError("decision_model.backbone must expose the Qwen3-VL visual tower")
        if not hasattr(decision_model.backbone, "language_model"):
            raise TypeError("decision_model.backbone must expose the Qwen3-VL language model")

        self.decision_model = decision_model
        self.processor = decision_model.processor
        self.visual_model = decision_model.backbone.visual
        self.language_model = decision_model.backbone.language_model
        self.debug = bool(debug)

        vision_dim = int(self.visual_model.config.hidden_size)
        hidden_dim = int(decision_model.head.in_features)
        self.bridge = VisualJEV(
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            decision_head=decision_model.head,
            debug=debug,
        ).eval()

        visual_parameter = next(self.visual_model.parameters())
        language_parameter = next(self.language_model.parameters())
        self.bridge.visual_algorithm.to(
            device=visual_parameter.device,
            dtype=visual_parameter.dtype,
        )
        self.bridge.fusion_norm.to(
            device=language_parameter.device,
            dtype=language_parameter.dtype,
        )

    @property
    def weight_status(self) -> dict[str, str]:
        return {
            "visual_backbone": "pretrained and frozen",
            "language_backbone": "pretrained and frozen",
            "visual_algorithm": "random/untrained projector",
            "fusion_norm": "default initialization; untrained",
            "decision_head": getattr(
                self.decision_model,
                "head_status",
                "status not declared by model",
            ),
        }

    def extract_text_features(self, prompts: Sequence[str]) -> torch.Tensor:
        """Return the original JEV final-token hidden feature for each prompt."""
        prompts = list(prompts)
        if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("prompts must contain one or more nonempty strings")
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for prompt in prompts
        ]
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(-1)
        max_length = getattr(self.decision_model, "max_length", None)
        if max_length is not None and lengths.max().item() > max_length:
            raise ValueError(
                f"Input length {lengths.max().item()} exceeds max_length={max_length}; "
                "no silent truncation"
            )
        input_device = next(self.language_model.get_input_embeddings().parameters()).device
        input_ids = input_ids.to(input_device)
        attention_mask = attention_mask.to(input_device)
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_device = output.last_hidden_state.device
        row = torch.arange(len(prompts), device=hidden_device)
        text_features = output.last_hidden_state[
            row, lengths.to(hidden_device) - 1
        ]
        if self.debug:
            describe_tensor("text input_ids", input_ids)
            describe_tensor("text attention_mask", attention_mask)
            describe_tensor("JEV text feature (final non-padding token)", text_features)
        return text_features

    def extract_image_feature(self, image) -> torch.Tensor:
        """Run one PIL image through Qwen3-VL visual tokens + VisualAlgorithm."""
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract visual evidence."},
            ],
        }]
        encoded = self.processor.apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        )
        pixel_values = encoded["pixel_values"]
        image_grid_thw = encoded["image_grid_thw"]
        if image_grid_thw.shape[0] != 1:
            raise ValueError("Visual-JEV V1 smoke path currently accepts exactly one image")
        visual_parameter = next(self.visual_model.parameters())
        pixel_values = pixel_values.to(visual_parameter.device)
        image_grid_thw = image_grid_thw.to(visual_parameter.device)
        vision_output = self.visual_model(
            pixel_values.to(dtype=visual_parameter.dtype),
            grid_thw=image_grid_thw,
            return_dict=True,
        )
        visual_tokens = vision_output.last_hidden_state
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        image_feature = self.bridge.align_visual(visual_tokens)
        if self.debug:
            describe_tensor("pixel_values", pixel_values)
            describe_tensor("image_grid_thw", image_grid_thw)
            describe_tensor("visual tokens (pre-merger)", visual_tokens)
            describe_tensor("VisualAlgorithm image feature", image_feature)
        return image_feature

    def score_prompts(
        self,
        image,
        prompts: Sequence[str],
        *,
        temperature: float = 1.0,
    ) -> VisualJEVOutput:
        """Score all candidates together, then normalize once with softmax."""
        if not isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        text_features = self.extract_text_features(prompts)
        image_feature = self.extract_image_feature(image)
        candidate_image_features = image_feature.expand(text_features.shape[0], -1)
        fused_features = self.bridge.fuse(
            text_features,
            candidate_image_features,
        )
        scores = self.bridge.score(fused_features)
        probabilities = torch.softmax(scores / temperature, dim=0)
        if self.debug:
            describe_tensor("candidate image features", candidate_image_features)
            describe_tensor("Visual-JEV fused features", fused_features)
            describe_tensor("JEV candidate scores", scores)
            describe_tensor("candidate probabilities", probabilities)
        return VisualJEVOutput(
            text_features=text_features,
            image_feature=image_feature,
            fused_features=fused_features,
            scores=scores,
            probabilities=probabilities,
        )


# Backward-compatible spelling used by the first local prototype.
VisualJev = VisualJEV
