# Visual-JEV selected visual interfaces

## Model and intended use

These four visual interfaces connect frozen external visual features to
the frozen Open-Jev 27B receiver. They support research on visual transfer,
spatial correspondence and candidate decisions. Each input supplies an
image, question and finite candidate set; the receiver scores candidates
independently. Alternative encoders are separate routes.

| Route | Feature tap | Input width | Receiver width | Interface |
|---|---|---:|---:|---|
| LLaVA-OneVision 0.5B | Native projector output, padded to interface width | 1,152 | 5,120 | Spatial mapping + affine |
| InternVL3.5 1B HF | Native projector input | 4,096 | 5,120 | Spatial mapping + affine |
| Qwen3-VL 4B | Native merged image features | 2,560 | 5,120 | Linear mapping + affine |
| SigLIP2 Base/16-224 | Unpooled vision features | 768 | 5,120 | Spatial mapping + affine + residual |

The LLaVA projector emits 896 channels on the selected path; zero padding
to 1,152 occurs before the saved LayerNorm. Qwen's source and receiver grids
match under the recorded processor policy, making its flat mapping an
identity in token positions. Matching these details is necessary when
loading the original checkpoints.

## Checkpoints and provenance

The release retains the original `spec`, `bridge`, `optimizer` and `step`
checkpoint fields. `torch.load(..., weights_only=True)` loads them. Exact
sizes, SHA256 hashes and selected taps appear in
[`visual_jev_interfaces.json`](../release_metadata/visual_jev_interfaces.json).
Model and processor revisions appear in the
[asset lock](../release_metadata/visual_jev_assets.lock.json).

The language receiver, LoRA, scalar head and calibration temperature come
from the third-party [Open-Jev project](https://github.com/Zefan-Cai/Open-Jev).
Visual-JEV contributes the external visual interfaces, spatial controls and
experimental analysis. These interfaces are not a replacement release of
the original language receiver.

## Evaluation scope

The main study evaluates multiple-choice visual questions and controlled
spatial/evidence interventions. Historical development questions occur in
the complete evaluation population; foundation-model exposure and all
near-duplicates are not fully audited. System comparisons retain different
receiver capacities and preprocessing policies. The release's rendered
smoke test checks code parity, not benchmark accuracy or generalization.

The interfaces are encoder-, feature-tap- and receiver-specific. Their
behavior on video, multi-image inputs, other receiver families or open-ended
generation is not established. Candidate softmax values are scores over the
provided alternatives, rather than guarantees of answer correctness.

## Licenses

Visual-JEV interface weights and additions use this repository's MIT
license. The downloaded Open-Jev receiver packages and Qwen/vision model
weights retain their respective upstream licenses, recorded as Apache-2.0
in the pinned manifests. The upstream source snapshot retains its MIT
license and copyright. See [third-party notices](../THIRD_PARTY_NOTICES.md).
