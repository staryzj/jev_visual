"""Trainable candidate-aware Visual-JEV V2 adapter.

The Qwen3-VL backbone is intentionally kept outside this module.  The adapter
consumes post-merger visual tokens and one text feature per candidate, so its
checkpoint contains only the small trainable Visual-JEV components.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VisualJEVV2Config:
    """Dimensions and regularization for the trainable V2 adapter."""

    vision_dim: int
    text_dim: int
    adapter_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.vision_dim, self.text_dim, self.adapter_dim, self.num_heads) <= 0:
            raise ValueError("all dimensions and num_heads must be positive")
        if self.adapter_dim % self.num_heads:
            raise ValueError("adapter_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class VisualJEVV2Output:
    """Scores plus the intermediate values needed by diagnostics."""

    scores: torch.Tensor
    attention: torch.Tensor
    projected_visual_tokens: torch.Tensor
    projected_text_features: torch.Tensor
    attended_visual_features: torch.Tensor
    gates: torch.Tensor
    fused_features: torch.Tensor


class VisualJEVV2(nn.Module):
    """Candidate-aware cross-attention and gated scalar scoring head.

    For candidate feature ``c`` and post-merger tokens ``V``, the module uses
    ``q = W_q c``, ``K = W_k V`` and ``U = W_v V`` through multi-head
    attention.  Its feature-wise gate is

    ``g = sigmoid(W_g [q; attended])``

    and the fused feature is ``LayerNorm(g * q + (1-g) * attended)``.  One
    shared linear head maps every fused candidate representation to a scalar.
    """

    checkpoint_format_version = 1

    def __init__(self, config: VisualJEVV2Config):
        super().__init__()
        self.config = config
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(config.vision_dim),
            nn.Linear(config.vision_dim, config.adapter_dim),
            nn.GELU(),
        )
        self.text_projector = nn.Sequential(
            nn.LayerNorm(config.text_dim),
            nn.Linear(config.text_dim, config.adapter_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.adapter_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(config.adapter_dim * 2, config.adapter_dim)
        self.fusion_norm = nn.LayerNorm(config.adapter_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.adapter_dim, config.adapter_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.adapter_dim * 2, config.adapter_dim),
        )
        self.output_norm = nn.LayerNorm(config.adapter_dim)
        self.scalar_head = nn.Linear(config.adapter_dim, 1)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _prepare_inputs(
        self,
        visual_tokens: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [batch, tokens, vision_dim]"
            )
        if text_features.ndim != 2:
            raise ValueError("text_features must have shape [batch, text_dim]")
        if visual_tokens.shape[-1] != self.config.vision_dim:
            raise ValueError(
                f"expected vision_dim={self.config.vision_dim}, got {visual_tokens.shape[-1]}"
            )
        if text_features.shape[-1] != self.config.text_dim:
            raise ValueError(
                f"expected text_dim={self.config.text_dim}, got {text_features.shape[-1]}"
            )

        candidate_count = text_features.shape[0]
        if visual_tokens.shape[0] == 1 and candidate_count > 1:
            visual_tokens = visual_tokens.expand(candidate_count, -1, -1)
        elif visual_tokens.shape[0] != candidate_count:
            raise ValueError(
                "visual batch must equal candidate batch, or contain one shared image"
            )

        if visual_mask is None:
            visual_mask = torch.ones(
                visual_tokens.shape[:2], device=visual_tokens.device, dtype=torch.bool
            )
        else:
            if visual_mask.ndim == 1:
                visual_mask = visual_mask.unsqueeze(0)
            if visual_mask.shape[0] == 1 and candidate_count > 1:
                visual_mask = visual_mask.expand(candidate_count, -1)
            if visual_mask.shape != visual_tokens.shape[:2]:
                raise ValueError("visual_mask must have shape [batch, tokens]")
            visual_mask = visual_mask.to(device=visual_tokens.device, dtype=torch.bool)
        if not visual_mask.any(dim=1).all():
            raise ValueError(
                "every sample must contain at least one valid visual token"
            )

        parameter = next(self.parameters())
        return (
            visual_tokens.to(device=parameter.device, dtype=parameter.dtype),
            text_features.to(device=parameter.device, dtype=parameter.dtype),
            visual_mask.to(device=parameter.device),
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_features: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
    ) -> VisualJEVV2Output:
        visual_tokens, text_features, visual_mask = self._prepare_inputs(
            visual_tokens, text_features, visual_mask
        )
        projected_visual = self.visual_projector(visual_tokens)
        projected_text = self.text_projector(text_features)
        attended, per_head_attention = self.cross_attention(
            query=projected_text.unsqueeze(1),
            key=projected_visual,
            value=projected_visual,
            key_padding_mask=~visual_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended.squeeze(1)
        attention = per_head_attention.mean(dim=1).squeeze(1)
        gates = torch.sigmoid(self.gate(torch.cat((projected_text, attended), dim=-1)))
        fused = self.fusion_norm(gates * projected_text + (1.0 - gates) * attended)
        fused = self.output_norm(fused + self.fusion_mlp(fused))
        scores = self.scalar_head(fused).squeeze(-1)
        return VisualJEVV2Output(
            scores=scores,
            attention=attention,
            projected_visual_tokens=projected_visual,
            projected_text_features=projected_text,
            attended_visual_features=attended,
            gates=gates,
            fused_features=fused,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save adapter/head weights only; the frozen backbone is never embedded."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": self.checkpoint_format_version,
            "model_type": "visual-jev-v2",
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "step": step,
            "metadata": metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)
        return path

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[VisualJEVV2, dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("model_type") != "visual-jev-v2":
            raise ValueError("checkpoint is not a Visual-JEV V2 checkpoint")
        if payload.get("format_version") != cls.checkpoint_format_version:
            raise ValueError(
                f"unsupported checkpoint format {payload.get('format_version')}"
            )
        model = cls(VisualJEVV2Config(**payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model, payload


def candidate_pair_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    loss_type: str = "cross_entropy",
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute listwise cross entropy or pairwise margin ranking loss.

    ``positive_scores`` has shape ``[batch]``.  ``negative_scores`` may be
    ``[batch]`` or ``[batch, negative_count]`` and therefore directly supports
    multiple hard negatives per image.
    """
    if positive_scores.ndim != 1:
        raise ValueError("positive_scores must have shape [batch]")
    if negative_scores.ndim == 1:
        negative_scores = negative_scores.unsqueeze(1)
    if (
        negative_scores.ndim != 2
        or negative_scores.shape[0] != positive_scores.shape[0]
    ):
        raise ValueError("negative_scores must have shape [batch, negatives]")
    if negative_scores.shape[1] == 0:
        raise ValueError("at least one negative score is required")
    if loss_type == "cross_entropy":
        logits = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1)
        targets = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, targets)
    if loss_type == "ranking":
        positives = positive_scores.unsqueeze(1).expand_as(negative_scores)
        targets = torch.ones_like(negative_scores)
        return F.margin_ranking_loss(positives, negative_scores, targets, margin=margin)
    raise ValueError("loss_type must be 'cross_entropy' or 'ranking'")


def pad_visual_tokens(
    token_sequences: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length ``[tokens, dim]`` tensors and return a valid mask."""
    if not token_sequences:
        raise ValueError("token_sequences must not be empty")
    if any(tokens.ndim != 2 for tokens in token_sequences):
        raise ValueError("each visual token tensor must have shape [tokens, dim]")
    feature_dim = token_sequences[0].shape[-1]
    if any(tokens.shape[-1] != feature_dim for tokens in token_sequences):
        raise ValueError(
            "all visual token tensors must have the same feature dimension"
        )
    max_tokens = max(tokens.shape[0] for tokens in token_sequences)
    if max_tokens == 0:
        raise ValueError("visual token tensors must not be empty")
    reference = token_sequences[0]
    padded = reference.new_zeros((len(token_sequences), max_tokens, feature_dim))
    mask = torch.zeros(
        (len(token_sequences), max_tokens), device=reference.device, dtype=torch.bool
    )
    for index, tokens in enumerate(token_sequences):
        tokens = tokens.to(device=reference.device, dtype=reference.dtype)
        padded[index, : tokens.shape[0]] = tokens
        mask[index, : tokens.shape[0]] = True
    return padded, mask


class Qwen3VLFeatureExtractor:
    """Frozen Qwen3-VL feature boundary used by training and diagnostics."""

    def __init__(self, decision_model: Any):
        required = ("processor", "backbone")
        missing = [name for name in required if not hasattr(decision_model, name)]
        if missing:
            raise TypeError("decision_model is missing: " + ", ".join(missing))
        if not hasattr(decision_model.backbone, "visual"):
            raise TypeError("Qwen3-VL backbone must expose .visual")
        if not hasattr(decision_model.backbone, "language_model"):
            raise TypeError("Qwen3-VL backbone must expose .language_model")
        self.decision_model = decision_model
        self.processor = decision_model.processor
        self.backbone = decision_model.backbone
        self.visual_model = self.backbone.visual
        self.language_model = self.backbone.language_model
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    @property
    def backbone_is_frozen(self) -> bool:
        return not any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )

    @torch.no_grad()
    def encode_image_stages(self, image: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact visual features used by three-stage V3 training.

        Qwen's ``last_hidden_state`` is the output of the final vision block
        immediately before ``visual.merger``.  The V3 cache groups each native
        2x2 spatial-merge unit by its mean, yielding ``pre_tokens`` with the
        same token count as ``pooler_output`` but the vision width (normally
        ``[N, 1024]``).  No cached tensors participate in this path.

        Returns:
            ``(grouped_pre_merger, post_merger)`` on the visual tower device.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract visual evidence."},
                ],
            }
        ]
        encoded = self.processor.apply_chat_template(
            [conversation],
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        )
        parameter = next(self.visual_model.parameters())
        pixel_values = encoded["pixel_values"].to(
            device=parameter.device, dtype=parameter.dtype
        )
        grid_thw = encoded["image_grid_thw"].to(parameter.device)
        if grid_thw.shape[0] != 1:
            raise ValueError("encode_image_stages accepts exactly one image")
        output = self.visual_model(pixel_values, grid_thw=grid_thw, return_dict=True)

        native_pre = output.last_hidden_state
        post = output.pooler_output
        if native_pre.ndim == 3:
            if native_pre.shape[0] != 1:
                raise ValueError("unexpected batched pre-merger output")
            native_pre = native_pre[0]
        if post.ndim == 3:
            if post.shape[0] != 1:
                raise ValueError("unexpected batched post-merger output")
            post = post[0]
        if native_pre.ndim != 2:
            raise ValueError("pre-merger output must have shape [tokens, dim]")
        if post.ndim != 2:
            raise ValueError("post-merger output must have shape [tokens, dim]")
        if post.shape[0] <= 0 or native_pre.shape[0] % post.shape[0]:
            raise ValueError("pre/post token counts are not integer aligned")

        group_size = native_pre.shape[0] // post.shape[0]
        expected_group_size = getattr(self.visual_model, "spatial_merge_unit", None)
        if expected_group_size is not None and group_size != expected_group_size:
            raise ValueError(
                f"expected spatial merge group={expected_group_size}, got {group_size}"
            )
        grouped_pre = native_pre.reshape(post.shape[0], group_size, -1).mean(dim=1)
        return grouped_pre, post

    @torch.no_grad()
    def encode_image_pre_merger(self, image: Any) -> torch.Tensor:
        """Return live grouped pre-merger tokens used as the V3 adapter input."""
        return self.encode_image_stages(image)[0]

    @torch.no_grad()
    def encode_image(self, image: Any) -> torch.Tensor:
        """Return Qwen3-VL post-merger tokens with shape ``[tokens, dim]``."""
        return self.encode_image_stages(image)[1]

    @torch.no_grad()
    def encode_candidates(self, candidates: Iterable[str]) -> torch.Tensor:
        """Return the final non-padding language state for each candidate."""
        candidates = list(candidates)
        if not candidates or any(
            not isinstance(item, str) or not item for item in candidates
        ):
            raise ValueError("candidates must contain nonempty strings")
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": item}]}]
            for item in candidates
        ]
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        )
        embedding_device = next(
            self.language_model.get_input_embeddings().parameters()
        ).device
        input_ids = encoded["input_ids"].to(embedding_device)
        attention_mask = encoded["attention_mask"].to(embedding_device)
        output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        lengths = attention_mask.sum(-1)
        row = torch.arange(len(candidates), device=output.last_hidden_state.device)
        return output.last_hidden_state[
            row, lengths.to(output.last_hidden_state.device) - 1
        ]


def attention_statistics(
    attention: torch.Tensor, *, top_k: int = 5
) -> list[dict[str, Any]]:
    """Create JSON-ready attention diagnostics for each candidate."""
    if attention.ndim != 2:
        raise ValueError("attention must have shape [candidates, tokens]")
    results = []
    for row in attention.detach().float().cpu():
        count = min(top_k, row.numel())
        values, indices = torch.topk(row, count)
        entropy = -(row.clamp_min(1e-12) * row.clamp_min(1e-12).log()).sum()
        results.append(
            {
                "sum": row.sum().item(),
                "max": row.max().item(),
                "entropy": entropy.item(),
                "top_indices": indices.tolist(),
                "top_weights": values.tolist(),
            }
        )
    return results
