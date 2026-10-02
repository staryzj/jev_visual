# Recorded cross-backbone results

Adapter-only cross-backbone study: the same frozen VisualJEVV3 scorer and cached Qwen candidate text features are reused. Target rows use 512/80/80/80 custom COCO train/validation/calibration/held-out examples, not official benchmark splits. Only the target adapter is trained. Qwen is the source architecture reference; no comparable 80-example result is recorded. Extraction timing excludes text encoding and is not end-to-end latency.

| Backbone | Adapter M | Accuracy % | Blank % | Wrong flip % | Extract ms |
|---|---:|---:|---:|---:|---:|
| SigLIP2 | 3.42 | 86.25 | 86.25 | 1.25 | 9.25 |
| InternVL3.5 | 3.68 | 88.75 | 90.00 | 0.00 | 23.84 |
| LLaVA-OneVision | 3.81 | 87.50 | 88.75 | 0.00 | 99.95 |
