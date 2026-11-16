# Reproduce the paper accuracy

This release supplies the four selected interface weights, the fixed 26,325
question population, original predictions, and the code used for training,
evaluation and analysis. Frozen base models are fetched at the revisions
recorded in the model lock. Feature caches can be regenerated.

## 1. Install and download

Use Python 3.12 and a BF16-capable CUDA device. The 27B receiver was tested
on an 80 GB H100. The runtime uses batch-one scoring and SDPA math attention.

```bash
python -m pip install -e '.[transfer,reproduce]'
python -m visualjev.assets --destination /path/to/assets \
  --assets base27 openjev27 qwen4 llava internvl siglip2 interfaces
python -m visualjev.reproduce fetch --destination /path/to/reproduction --include-source
```

The model download command fetches the pinned 27B dependencies and the original
four interface checkpoints. For a Qwen-only run, use
`--assets base27 openjev27 qwen4 interfaces` with `visualjev.assets`.
The paper bundle contains:

| File or folder | Purpose |
|---|---|
| `paper/benchmark.jsonl` | Exact UIDs, questions, ordered candidates, labels, image hashes and original shard assignments |
| `paper/dataset_sources.json` | Dataset repositories, immutable revisions and source-file hashes |
| `paper/reference_predictions/` | Original per-question predictions for nine routes |
| `paper/expected_summary.json` | Reported complete-split and per-task measurements |
| `original_experiments/` | Original deployed scripts, training plans and result records, in their original project layout |

## 2. Rebuild the evaluation inputs

```bash
python -m visualjev.reproduce prepare \
  --bundle /path/to/reproduction/paper \
  --output /path/to/prepared
```

This downloads the four pinned dataset snapshots and the official IconQA
archive, reconstructs the image-choice canvases, and checks the RGB hash of
every evaluation image. It retains all 26,325 questions, including the
2,224 ScienceQA questions without images. Questions and candidate order come
from the sealed paper manifest rather than a newly sampled population.

Existing dataset resources can be reused with `--source-cache` pointing to
folders named `dataset_name/dataset`, and `--iconqa-root` pointing to the
official extracted `iconqa_data` directory. If the canonical images have
already been prepared, use `--prepared-images` pointing to the folder that
contains `aokvqa/images`, `scienceqa/images`, and the other task folders.
Image hashes are still checked. Dataset downloads retain the original
dataset licenses and attribution recorded in `dataset_sources.json`.

The output contains `questions.jsonl` and the original eight shard files.

## 3. Run the four trained interfaces

```bash
python -m visualjev.transfer.infer \
  --assets /path/to/assets --encoder qwen \
  --input /path/to/prepared/questions.jsonl \
  --output /path/to/results/qwen.jsonl

python -m visualjev.reproduce score \
  --bundle /path/to/reproduction/paper \
  --predictions /path/to/results/qwen.jsonl \
  --output /path/to/results/qwen-accuracy.json
```

Repeat with `--encoder llava`, `internvl` or `siglip2` and distinct output
files. To distribute one route over several GPUs, assign a separate
`shard0.jsonl` through `shard7.jsonl` to each worker, then pass all eight
prediction paths to `score`. Every UID must appear exactly once. Invalid
answers count as errors; incomplete coverage and changed candidate order
are rejected.

The Qwen route's recorded overall result is 23,008 correct answers out of
26,325 (87.40%). The full set of per-task targets is in
`expected_summary.json`. The ScienceQA image result is a subset of ScienceQA
all and is not added again to the overall denominator.

## 4. Check the archived results without a GPU

```bash
python -m visualjev.reproduce verify-reference \
  --bundle /path/to/reproduction/paper \
  --output /path/to/results/reference-accuracy.json
```

This recomputes all nine recorded routes from per-question predictions and
checks them against the original summary. It is separate from running new
model inference. The SigLIP2 source baseline supports only image-bearing
questions, while Visual-JEV supports the complete population.

## Training and controlled experiments

The source archive includes the original `train.py`, `fit.py`, feature
extraction, data preparation, evaluation, queue and analysis scripts, with
their plans and result records. Start with:

- `visual-jev-full-benchmark-20261027`: complete-split evaluation and aggregation.
- `visual-jev-external-token-transfer-20261025`: linear interface fitting.
- `visual-jev-tap-relational-ablation-20261025`: selected LLaVA and InternVL interfaces.
- `visual-jev-residual-bridge-20261025`: selected SigLIP2 residual interface.
- `visual-jev-unified-bridge-confirm-20261026`: selected checkpoint provenance and training recipes.
- `visual-jev-processor-geometry-20261106` and `visual-jev-review-evidence-20261109`: spatial controls.
- `visual-jev-round2-controls-20261110` through `visual-jev-round9-20261110`: matched fits and evidence/probe experiments.
- `visual-jev-full-efficiency-20261104`: matched timing and output-length measurements.

The archived scripts retain the original server layout and integrity checks.
Use the portable commands above to evaluate the published main checkpoints
on a new machine. For new fitting experiments, copy the archived plans to a
fresh project, update resource/output paths, and regenerate feature caches;
the original training manifests and seeds remain available for comparison.

The four released interfaces are the fixed checkpoints used by the paper.
Historical checkpoints from unrelated architectures and intermediate feature
caches are not required to reproduce these accuracy measurements.

## Release checks

All 24,101 image-question inputs passed the sealed RGB-hash check. Accuracy
recomputed from the original predictions matches all nine reported routes.
On one real image question per task and external encoder, all 20 candidate
score vectors match the archived values exactly; the shared text-only path
also passes. This release check does not rerun all 26,325 questions on GPUs.
The tested package versions and checks are recorded in
`release_metadata/reproduction_validation.json` and
`release_metadata/paper_gpu_verification.json`.
