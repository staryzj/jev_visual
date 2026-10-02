# Backbone adaptation evidence and follow-up protocol

## What is measured

Three historical target-backbone runs use seed 20260928, 512 training, 80 validation, 80 calibration and 80 custom held-out COCO examples. They are not official full test results. All use base checkpoint SHA256 `8e4f9703d4629e222fbdb2e6007c43ae224e7e64137ce8e880ea2ecf39e366ad` and a fixed external VisualJEVV3 decision module. Only backbone-specific adapters are trained; the common text features are cached Qwen features. This is not the official Visual Jev answer-SFT scorer.

Checkpoint, adapter-file hashes, parameter counts and current manifest identity are recorded in `audit.json`. Current base decision-state SHA256 is `24d464f33041f6c1741c3472bf75e6b31e3ab6443ff920ed8fb76180cd365055`. The historical script freezes decision parameters and puts only adapter parameters into its optimizer. No post-training whole-head snapshot was saved, so the audit identifies the reused source checkpoint rather than inventing an after-training hash.

The existing target runner trains CE plus 0.1 blank/noise uniformity from epoch two. It does not use Qwen teacher-token matching for alternative encoders. Pooling is deterministic adaptive 1-D average resampling to 196 tokens. A matching tensor width is an input contract, not semantic-equivalence proof.

Original accuracies are 86.25/88.75/87.50%; blank accuracies are 86.25/90.00/88.75%. These controls do not establish beneficial visual information. Historical wrong controls are a cyclic shift, not verified distinct-image or semantic pairs. Missing exact-protocol Qwen, zero-aligned-token, seeded random-adapter and trained-linear results remain unmeasured.

The source head's preparation archive uses a separate 64-example mixed-domain diagnostic. The main paper's 128-example controlled ablation and six official/released full-split runs use a different controlled-COCO checkpoint `b7589ce7...`. These identities and denominators must not be combined into a shared-head or four-backbone performance claim.

## Full custom-manifest continuation commands (not executed)

`run_backbone_adapter_full.py` consumes EVERY matching row of the declared custom manifests. Current counts are 2048/91/82/83, not 512/80/80/80. This is still a custom COCO protocol, not official full COCO. It refuses an existing output directory, checks that all required text-feature shards exist, and never shrinks the declared membership. Run with `--plan-only` first to see counts and exact commands without training.

From `/home/jiezuo/projects/Open-Jev`:

```bash
.venv/bin/python scripts/run_backbone_adapter_full.py --backbone siglip2 --model models/SigLIP2-Base-Patch16-224 --base-checkpoint experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt --output experiments/results/backbone_generality_full_v6/siglip2 --plan-only
.venv/bin/python scripts/run_backbone_adapter_full.py --backbone internvl --model models/InternVL3_5-1B-HF --base-checkpoint experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt --output experiments/results/backbone_generality_full_v6/internvl --plan-only
.venv/bin/python scripts/run_backbone_adapter_full.py --backbone llava --model models/LLaVA-OneVision-0.5B --base-checkpoint experiments/results/benchmark_v1/fix_negative_transfer_local_rerun/best_model_results/visual_jev_v3_best.pt --output experiments/results/backbone_generality_full_v6/llava --plan-only
```

Remove `--plan-only` only when authorizing these new full-custom-manifest training runs. The wrapper reuses the historical adapter recipe; it does not implement or claim completed random/linear/zero-token baselines. Source model files and text features must already be present. Native official benchmark training/evaluation is a different protocol and must preserve its declared full membership.

## Still needed before an effective-adaptation claim

- A directly comparable Qwen reference on exactly the same evaluation IDs, candidate ordering and fixed scorer.
- Seeded random adapters, trained linear adapters and zero aligned visual tokens under the same head; a blank photograph is not zero tokens.
- Beneficial image-dependent task improvement, not only output variation, and verified semantic-pair decisions for each target adapter.
- Full declared coverage on broader datasets, repeated training seeds and a genuinely matched online timing harness.
- Saved prediction IDs, candidate lists, source revision/fingerprint, split membership hashes, text-feature identity and before/after scorer tensor hashes for new runs. The full-run wrapper now archives current membership but does not by itself repair the older runner's absent score-level provenance; extend that record before a final submission experiment.
