# Visual-JEV: adapting visual backbones to one fixed JEV scorer

A selected frozen visual backbone maps through its own trainable adapter into the **same frozen external JEV decision scorer**. Alternative backbones are replaceable choices, not parallel ensemble branches. This research project builds on [Open-Jev](https://github.com/Zefan-Cai/Open-Jev); it is not an official Jev release.

中文：代码按功能归入文件夹，根目录只保留说明、依赖和许可证。完整 benchmark、自定义 held-out 实验和 debug 结果严格分开。

## Repository layout

| Folder | Contents |
|---|---|
| `visualjev/` | Models, adapters, pipeline and shared metrics |
| `jev/` | Required Open-Jev components only |
| `scripts/training/` | Training and custom-set experiments |
| `scripts/evaluation/` | Full-split evaluation and diagnostics |
| `scripts/data/` | Dataset preparation and source audits |
| `scripts/backbones/` | Adapter-only backbone transfer and audits |
| `scripts/tables/` | CSV, Markdown and LaTeX table generation |
| `scripts/demo/` | Prediction and demonstrations |
| `scripts/workflows/` | Multi-step shell workflows |
| `scripts/maintenance/` | Read-only release verification |
| `configs/`, `tests/` | Configurations and regression tests |
| `experiments/` | Real results, complete predictions and table exports |
| `reports/`, `docs/` | Evidence audits and protocols |
| `release_metadata/` | File hashes and migration provenance |

No Python files remain at the root. Old prototypes, unrelated upstream tasks, comparator-specific artifacts and duplicated documentation were removed from this compact tree. Previous commits preserve them; history was not rewritten.

## Installation and checks

Linux/WSL and Python 3.10+ are the experiment baseline. Run commands from the repository root with a compatible PyTorch/CUDA installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[train,vl,test]"
python -m pytest -q
python -m scripts.maintenance.verify_publication
```

Weights, feature caches and raw benchmark images are **not distributed here**. Exact historical reproduction requires the recorded checkpoint bytes and original data/model resources. Users must obtain gated access themselves; the workflow does not bypass restrictions or accept licenses.

## Primary evidence: adapter-only backbone transfer

The external scorer and common cached Qwen text features are fixed. Only the target adapter is trained. Each run uses custom COCO partitions of 512/80/80/80 train/validation/calibration/held-out instances, not official full tests.

| Target backbone | Held-out N | Original accuracy (%) | Blank-image accuracy (%) |
|---|---:|---:|---:|
| SigLIP2 | 80 | 86.25 | 86.25 |
| InternVL3.5 | 80 | 88.75 | 90.00 |
| LLaVA-OneVision | 80 | 87.50 | 88.75 |

Source scorer SHA256: `8e4f9703d4629e222fbdb2e6007c43ae224e7e64137ce8e880ea2ecf39e366ad`.

These records show functional reuse, not leaderboard superiority or beneficial visual dependence: no target row outperforms its blank control. Zero aligned tokens, random adapters, trained linear adapters and verified target-backbone semantic pairs are unmeasured.

See [core table](reports/frozen-scorer-v7/core_backbone_table.csv), [source identities](reports/frozen-scorer-v7/table_sources.json), [checkpoint audit](reports/backbone-adaptation-v6/audit.json) and [continuation protocol](reports/backbone-adaptation-v6/PROTOCOL_AND_NEXT_STEPS.md).

Plan full custom-manifest transfer without reducing coverage:

```bash
python -m scripts.backbones.run_backbone_adapter_full \
  --backbone siglip2 --model /path/to/SigLIP2 \
  --base-checkpoint /path/to/recorded-source-scorer.pt \
  --manifest-root /path/to/custom/manifests \
  --qwen-feature-root /path/to/qwen/features \
  --output experiments/reproduction/siglip2 --plan-only
```

Remove `--plan-only` only when resources are available and full training is intended. This remains custom-manifest transfer, not official benchmark evaluation.

## Secondary evidence: complete declared benchmark splits

These use a separate fixed controlled-COCO checkpoint with SHA256 `b7589ce737351b88beb398b9027075dfc2cdff3fc5ba637a9f78a453dbc58493`, not the target-backbone study checkpoint.

| Dataset/task | Declared split | N | Primary score (%) |
|---|---|---:|---:|
| ScienceQA, all items | official test | 4241 | 28.13 |
| A-OKVQA, four-choice | official validation | 1145 | 36.33 |
| IconQA, select-txt task | official test task | 6316 | 30.19 |
| AI2D | official test | 3088 | 26.55 |
| VSR, random split | official test | 2195 | 53.53 |
| SugarCrepe++, strict ITT | released evaluation suite; HF storage split `train` | 4757 | 43.77 |

The six tasks contain 21,742 unique per-task prediction records: a coverage total, **not pooled accuracy**. SugarCrepe++ uses strict two-positive-over-negative ranking. ScienceQA includes intrinsic text-only items; IconQA covers a declared native task.

Values come from [real result files](experiments/results/submission_full). GQA, SNLI-VE, TextVQA, TallyQA, NLVR2 and Winoground are incomplete here. [Pending commands](experiments/tables/submission/pending_benchmarks.json) are an engineering inventory, not scores. TextVQA must use labelled **validation**, never hidden-test accuracy.

## Common commands

```bash
python -m scripts.data.submission_benchmarks prepare \
  --datasets scienceqa aokvqa iconqa ai2d vsr sugarcrepe_pp \
  --checkpoint /path/to/recorded-full-evaluation.pt --fetch-images

python -m scripts.evaluation.evaluate_submission_benchmarks \
  --datasets scienceqa aokvqa iconqa ai2d vsr sugarcrepe_pp \
  --checkpoint /path/to/recorded-full-evaluation.pt \
  --model /path/to/Qwen3-VL-4B-Instruct --controls

python -m scripts.tables.generate_submission_tables
```

Table regeneration requires complete matching local data manifests; exported results do not replace coverage checks. Use a fresh result root for changed checkpoints/protocols. Published results are evidence snapshots, not output folders to overwrite casually.

After satisfying manual access/image prerequisites:

```bash
python -m scripts.data.resume_submission_benchmark snli_ve --flickr30k-root data/flickr30k-images
python -m scripts.data.resume_submission_benchmark winoground
python -m scripts.data.resume_submission_benchmark nlvr2 --nlvr2-authorized
python -m scripts.training.train_visual_jev_v3 --help
python -m scripts.demo.predict_visual_jev_v3 --help
```

Recovery commands default to recorded checkpoint/model locations; `--checkpoint` overrides the checkpoint. To use another model location, run the evaluator directly with `--model`. A missing image blocks the declared split rather than reducing its denominator.

See [BENCHMARK_PROTOCOL](docs/BENCHMARK_PROTOCOL.md) and [publication notes](docs/PUBLICATION_NOTES.md). The [MIT license](LICENSE) covers repository code; model and dataset licenses remain separate.
