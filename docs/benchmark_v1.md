# Visual-JEV benchmark_v1 interface and calibration

## Scope and current status

`benchmark_v1` uses one closed-ended schema for every main dataset:

```json
{
  "schema": "open-jev-benchmark-v1",
  "id": "dataset:source-id",
  "dataset": "aokvqa|scienceqa_image_only|iconqa_choice|coco_hard_negative",
  "image": "local/path/to/image.jpg",
  "question": "...",
  "candidates": ["...", "..."],
  "label": 0,
  "group_id": "image-or-question-group",
  "metadata": {"source_split": "..."}
}
```

The implementation lives in `jev/benchmark_v1.py`. It validates labels and
candidates, supports variable candidate counts, provides dataset conversion
functions, defines common `DatasetAdapter`/`CandidateScorer` contracts, and
writes deterministic JSONL manifests with SHA-256 provenance. `group_id`
prevents the same image/question group from crossing splits.

The current runnable manifest contains the already complete COCO hard-negative
data only. A-OKVQA, ScienceQA image-only and IconQA choice remain marked as
pending until their downloads are complete; they are not represented by fake
or empty experimental scores.

## Fixed split

Seed: `20260928`.

| Split | Samples | Construction |
|---|---:|---|
| train | 2048 | existing COCO hard-negative train |
| validation | 64 | seed-hashed first half of legacy even validation indices |
| calibration | 64 | seed-hashed second half of legacy even validation indices |
| test | 128 | legacy odd validation indices; unchanged from the paper evaluation |

The files and checksums are in `data/benchmark_v1/manifests/manifest.json`.
Recreate them with:

```bash
uv run --frozen --extra train python prepare_benchmark_v1.py
```

When all main datasets are available, their adapters should yield
`BenchmarkExample` instances and feed the same manifest writer. External-only
sets (SugarCrepe, Winoground, POPE/POPEv2 and MMMU) must use separate manifests
and must never enter training, validation or temperature fitting.

## Independent temperature scaling

`jev/calibration.py` implements a single positive scalar temperature:

\[
p(y=k\mid x;T)=\operatorname{softmax}(z(x)/T)_k,
\]

where only `T` is optimised with LBFGS on the 64-sample calibration manifest.
Model parameters and logits are fixed. The 128-sample test manifest is not read
during fitting. The implementation supports fixed or variable candidate counts.

Run all existing checkpoints with:

```bash
uv run --frozen --extra train python calibrate_visual_jev_v3.py
```

Outputs:

- `experiments/results/calibration.json`: split indices, fitted temperatures,
  and before/after calibration and test metrics.
- `experiments/results/calibration.csv`: paper-table-friendly summary.

Actual held-out results from the current run:

| Model | Temperature | Test NLL before | Test NLL after | Test ECE before | Test ECE after |
|---|---:|---:|---:|---:|---:|
| random untrained head | 20.0000 | 1.1409 | 1.1007 | 0.2251 | 0.2092 |
| MeanPool + MLP + JEV | 3.2055 | 0.8162 | 0.4162 | 0.1306 | 0.0335 |
| V2 candidate-aware | 1.1446 | 0.3491 | 0.3439 | 0.0520 | 0.0453 |
| V3 simple post-merger | 0.1650 | 0.6376 | 0.3257 | 0.3181 | 0.0682 |
| V3 without Stage A | 0.1787 | 0.6756 | 0.3570 | 0.3250 | 0.0407 |
| V3 without Stage C | 2.8231 | 0.4632 | 0.3333 | 0.0964 | 0.0640 |
| Full V3 | 0.1944 | 0.6697 | 0.3673 | 0.3360 | 0.0775 |

Temperature scaling does not change class rankings, so accuracy and macro-F1
remain unchanged. The random baseline reaches the configured upper temperature
bound, as expected for an uninformative overconfident head; it is not a
substantive baseline result.

### Limitation of legacy checkpoints

The current Stage-B/C checkpoints selected epochs using the full legacy even
validation subset. Therefore the new calibration manifest is disjoint from
test but is not historically independent of legacy checkpoint selection. This
is recorded in `calibration.json`. New benchmark_v1 training must select
checkpoints on `validation.jsonl`, fit temperature only on `calibration.jsonl`,
and report once on `test.jsonl`; this removes the limitation in the final rerun.
