# Frozen Open-Jev 27B visual experiments

This code covers the later 27B receiver experiments in this repository. It retains the released Open-Jev candidate readout and swaps image-slot embeddings using one external encoder and its locked adapter at a time. The earlier V3 study uses a different scorer and protocol.

## Protocol

- A-OKVQA validation (1,145), ScienceQA test (4,241), IconQA text (6,316), IconQA image (11,535), and AI2D (3,088). ScienceQA image (2,017) is a reporting subset, counted once in the overall result.
- `protocol.json` records the four feature taps, adapter specifications and locked checkpoint SHA-256 values. Historical development examples are included in the full evaluation population.
- The efficiency comparison uses one visible GPU, batch size one, BF16 model weights, math SDPA, one timed repeat per UID after warm-up, and the full candidate set. Native generators use the original short-answer prompt (64 new tokens) or the prompted rationale arm (1,024 new tokens). SigLIP2 uses image-text matching with a 64-token text limit and marks image-free questions unsupported.
- The direct scorer emits zero autoregressively generated tokens. It still encodes images, processes input positions, and scores each candidate. The benchmark records generated tokens, processed input positions, synchronized wall time and peak allocated GPU memory separately. Local model timings are distinct from API or published latency numbers.

## Run with local artifacts

From the repository root, with Python 3.11+ and compatible CUDA PyTorch, NumPy and Pillow:

```bash
python -m scripts.evaluation.open_jev_27b.run --help
python -m scripts.evaluation.open_jev_27b.summarize --help
python -m unittest discover -s tests -p test_open_jev_27b_protocol.py
```

Provide a shard JSON list with a `cluster` and a `row` per item. Each row has `uid`, `dataset`, `question`, `candidates`, zero-based `label`, and `images` (paths to the original image or option canvas). The local runtime factory receives `(encoder_name, kind)` and returns:

- For `external`, `model` and `encoder`. The model exposes the frozen `mm`, processor, reference prompt methods, `logits(batches, feature)` and `no_image(row)`. The encoder exposes its frozen `model` and `encode(image, tap)`, returning the source tensor and geometry metadata. The receiver prompt builder imports `release_loader.api.candidate_prompts` from the matching local Open-Jev runtime.
- For `short`, `rationale`, or `matching`, a frozen `model` and its `processor`. The runtime loads the matching model revision and places the model on `cuda:0` with BF16 parameters.

The archived model loader, feature extraction dependencies, licensed datasets, and bridge weights are local inputs. This directory publishes the adapter, training objectives, external slot preparation, evaluation loop and statistics. Exact GPU reproduction also requires the matching runtime, shards, hardware and checkpoint bytes. The original server queue and archived runs are separate.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluation.open_jev_27b.run \
  --encoder qwen --kind external --runtime my_runtime:load \
  --items shard0.json --checkpoint qwen27b.pt --output external0.jsonl

python -m scripts.evaluation.open_jev_27b.summarize efficiency \
  --route external*.jsonl --reference short*.jsonl --expected expected-uids.jsonl
```

For the offline summarizer, the expected JSONL has `uid`, `dataset`, `cluster`, `label`, `candidate_N`, and `has_image`. Accuracy mode also accepts archived prediction rows. Paired summaries use supported common UIDs only and are emitted once both routes cover the manifest without unresolved execution errors. Invalid generated answers count as wrong. Cluster bootstrap intervals and exact McNemar tests preserve paired outcomes, with unadjusted p-values.
