# Visual-JEV: a visual interface for bounded JEV decisions

Visual-JEV is a research extension of [Open-Jev](https://github.com/Zefan-Cai/Open-Jev), not an official release of TypeSafe's Jev or the concurrent Visual Jev answer-SFT model. It maps frozen visual representations through an alignment adapter to an external candidate-conditioned scorer. One selected visual encoder is used per run; alternative encoders are not an ensemble.

中文说明：本工程严格区分完整公开 benchmark、matched 自定义对照和 sample/debug；没有结果的项目不填造数值。代码与结果仓库为 [staryzj/jev_visual](https://github.com/staryzj/jev_visual)。

## Evidence boundaries

- **Full benchmark protocol:** every example in the explicitly named official split/native task, with public labels. Official validation is called validation, never test accuracy.
- **Matched protocol:** identical custom held-out examples and candidate order for our model and the official Visual Jev comparator. These are not full official test results.
- **Controlled backbone adaptation:** three recorded adapter-only runs with the same frozen external scorer, each on 512/80/80/80 custom COCO train/validation/calibration/held-out examples. These 80-example results are preliminary, not full benchmark evidence.
- **Debug:** smoke tests and capped runs remain separate and never replace declared full splits.

See [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) and [backbone protocol/next steps](reports/backbone-adaptation-v6/PROTOCOL_AND_NEXT_STEPS.md).

The research claim is an explicit, inspectable visual-to-decision interface. The present records do not establish leaderboard superiority, universal backbone compatibility, matched end-to-end speedups, or beneficial visual dependence for the target-adapter study.

## Recorded complete-split results

These scores are rounded from each dataset's actual `experiments/results/submission_full/<dataset>/result.json`. They use the fixed controlled-COCO checkpoint with SHA256:

`b7589ce737351b88beb398b9027075dfc2cdff3fc5ba637a9f78a453dbc58493`

| Dataset/task | Evaluated split | N | Primary score (%) |
|---|---|---:|---:|
| ScienceQA, all items | official test | 4241 | 28.13 |
| A-OKVQA, native four-choice | official validation | 1145 | 36.33 |
| IconQA, native select-txt task | official test task | 6316 | 30.19 |
| AI2D | official test | 3088 | 26.55 |
| VSR, random split | official test | 2195 | 53.53 |
| SugarCrepe++, strict ITT | released evaluation suite, HF storage split `train` | 4757 | 43.77 |

The total is 21,742 task instances, not a pooled accuracy denominator. SugarCrepe++ uses strict two-positive-over-negative ranking, not ordinary choice accuracy. ScienceQA includes source-defined text-only items; IconQA selects a declared native task, not a random sample. These runs have a different checkpoint from the target-backbone study.

Incomplete GQA, SNLI-VE, TextVQA, TallyQA, NLVR2 and Winoground measurements are not reported as completed results. Current resource/status records and exact follow-up commands belong in the engineering inventory, not a performance claim.

## Recorded target-backbone study

Base scorer checkpoint SHA256:

`8e4f9703d4629e222fbdb2e6007c43ae224e7e64137ce8e880ea2ecf39e366ad`

| Target visual encoder | Custom held-out N | Original accuracy (%) | Blank-image accuracy (%) |
|---|---:|---:|---:|
| SigLIP2 | 80 | 86.25 | 86.25 |
| InternVL3.5 | 80 | 88.75 | 90.00 |
| LLaVA-OneVision | 80 | 87.50 | 88.75 |

Only the target alignment adapter is trained; the external scorer and cached Qwen text path are fixed. The original image does not outperform blank controls in any row. Consequently these results establish recorded execution/reuse, not useful visual contribution. Historical wrong-image shifts are not verified semantic counterfactuals. Same-protocol Qwen, zero-token, random-adapter and trained-linear baseline scores are unmeasured.

See `reports/backbone-adaptation-v6/audit.json`, `backbone_tables.csv`, and the original `experiments/results/backbone_generality/<backbone>/results.json`. The audit does not invent a post-training scorer snapshot or claim historical prediction-level IDs were saved.

## Code map

| Path | Function |
|---|---|
| `visual_jev_v3.py`, `visual_jev_v3_pipeline.py` | Candidate scorer and adapter/pipeline modules |
| `train_visual_jev_v3.py`, `scripts/run_visual_jev_v3_experiment.py` | Source-model training and controlled diagnostics |
| `scripts/submission_benchmarks.py` | Twelve-dataset registry, labelled full-split preparation and candidate manifests |
| `scripts/evaluate_submission_benchmarks.py` | Complete-split evaluation, coverage/identity guards, resumable predictions |
| `scripts/vqa_official_scoring.py` | VQA answer normalization/consensus scoring |
| `scripts/audit_submission_sources.py` | Annotation/source integrity audit |
| `scripts/run_official_visual_jev_matched.py` | Official Visual Jev comparison on custom matched sets |
| `scripts/run_backbone_adapter_fast.py` | Historical capped target-adapter protocol, NOT official full evaluation |
| `scripts/run_backbone_adapter_full.py` | All-row custom-manifest continuation, with `--plan-only` |
| `scripts/audit_backbone_adaptation.py` | Existing checkpoint/adapter provenance audit |
| `scripts/generate_submission_tables.py` | Result-backed CSV, Markdown and LaTeX benchmark/diagnostic tables |
| `scripts/build_backbone_paper_tables.py` | Two target-backbone tables from audited real result records |
| `tests/test_submission_benchmarks.py` | Protocol guards, official scoring and complete-denominator tests |

## Environment and installation

Linux/WSL with Python 3.10+ is the project baseline. GPU feature extraction/evaluation requires compatible PyTorch/CUDA, enough memory, model weights and the declared dataset resources. A code checkout does not contain those resources.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[train,vl]' pytest pyarrow huggingface_hub
```

Use `pyproject.toml` and `requirements.txt` for declared dependency versions. Model/checkpoint files are intentionally not part of this source publication; the recorded SHA256 identifies the files used locally, but is not itself a downloadable checkpoint. Reproducing exact checkpoint-dependent scores requires obtaining those exact weights through a separately authorized release. Without them, the repository supplies code, measured-result records and provenance, not turnkey exact inference reproduction.

## Complete-split evaluation

After the exact checkpoint and frozen Qwen3-VL-4B-Instruct model are present, prepare the declared full resources and evaluate without a sample cap:

```bash
.venv/bin/python scripts/submission_benchmarks.py prepare \
  --datasets scienceqa aokvqa iconqa ai2d vsr sugarcrepe_pp --fetch-images
.venv/bin/python scripts/evaluate_submission_benchmarks.py \
  --datasets scienceqa aokvqa iconqa ai2d vsr sugarcrepe_pp \
  --checkpoint experiments/visual_jev_v3_paper/checkpoints/v3_full.pt \
  --model models/Qwen3-VL-4B-Instruct --controls
.venv/bin/python scripts/generate_submission_tables.py
```

Any missing required image, duplicate identity, invalid label or incomplete prediction coverage blocks the declared full split. Partial progress is not a `complete` score. Do not shrink data to finish a run. Table regeneration requires local candidate manifests/prediction files and their hashes; an aggregate-only publication cannot satisfy that verification by itself.

## Manual access and expensive follow-up work

Winoground requires access approval; SNLI-VE needs authorized Flickr30K images; NLVR2 needs its original image-use authorization. Authentication and accepting terms remain the user's actions. Public mirrors do not grant new image rights.

```bash
.venv/bin/python scripts/resume_submission_benchmark.py snli_ve \
  --flickr30k-root data/flickr30k-images
.venv/bin/python scripts/resume_submission_benchmark.py winoground
.venv/bin/python scripts/resume_submission_benchmark.py nlvr2 --nlvr2-authorized
```

These recovery commands are to be run only after the corresponding access prerequisites are satisfied. GQA/TextVQA vocabulary preparation and full-custom target training commands are in the protocol files. They preserve full declared membership; training is not triggered by generating tables or reviewing records.

## Data availability and release scope

This source-and-results deposit includes experimental source, tests, configuration, aggregate `result.json` records, source/checkpoint hashes, benchmark protocol and generated numerical tables. The six complete-split `result.json` and `predictions.jsonl` files are byte-identical to the local records after SHA256 and unique-UID/count checks (21,742 predictions in total). The prediction exports contain numeric scores, label indices, dataset UIDs and measured timing, not source question/caption/image content. `result.json` embeds dataset revision/fingerprint, named split, sample count and completeness metadata. Historical records preserve absolute local paths for provenance; those paths are not portable resources.

Original benchmark images, question/answer corpora, candidate caches, feature tensors, gated resources, downloaded model weights, checkpoint tensors and paper illustration photographs are excluded unless a separately verified redistribution route is established. Third-party datasets are retrieved from the sources recorded in the registry/manifests under their own licenses. No blanket data license or access permission is invented by this README. A separate checkpoint distribution route and archival release/DOI remain unresolved; no DOI is claimed.

Historical matched/custom logs containing source examples are published only as explicit aggregate exports: each file's `public_export_metadata` lists omitted fields and the original file SHA256. Retained metric fields are unchanged; exported file bytes/hashes are different from the originals and must not be confused with them. `PUBLICATION_MANIFEST.json` maps the source and export hashes separately. These historical aggregates cannot reconstruct deleted examples or repair absent historical prediction-level provenance. Dataset metadata snapshots are under `release_metadata/benchmarks/`; they document acquisition status and source identities, not redistributed datasets. Table regeneration still needs the original local candidate manifests to verify their hashes.

See [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md) for the exact publication boundary. A 37-test protocol/scorer/history/statistics smoke suite passed in the clean publication checkout; it does not rerun the benchmark inference or validate scientific superiority.

## Tests and result generation

```bash
.venv/bin/python -m pytest -q tests/test_submission_benchmarks.py
.venv/bin/python scripts/audit_backbone_adaptation.py
.venv/bin/python scripts/build_backbone_paper_tables.py --paper /path/to/paper
```

The audit needs the exact local checkpoint/adapter tensors and manifest files. The table builder uses existing result JSON, not synthetic data. Unmeasured cells remain `--`.

## Attribution and licenses

Open-Jev upstream attribution and its MIT `LICENSE` are retained. The name Visual-JEV here refers to this research extension; official Visual Jev is a separate comparator. Upstream model/dataset licenses remain separate from the software license, and our own generated results do not relicense third-party images or annotations. Authors should add a final manuscript/release citation once a public record exists; none is invented here.
