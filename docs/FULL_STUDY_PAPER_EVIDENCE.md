# Complete Visual-JEV paper evidence

## Records and interpretation

The published server snapshot contains exact original aggregate JSON bytes,
selected identities and checkpoint hashes from the author's
`visual-jev-results-32-native.tar.gz`. `release_metadata/FULL_STUDY_SNAPSHOT.json`
hashes every exported member and the source archive. It is an allowlist export,
not the entire runtime. Raw images, weights, feature caches, credentials and
target prediction traces are excluded.

Current study: 27 adapters (3 backbones × 3 seeds × 3 arms), 94,244 full
training examples and 3 complete epochs. Final adapters remain fixed over
all evaluation tasks. The common scorer-state SHA256 is
`80e230c5e5cc1afffb47d78ec9025870fcd391ea605f6ab18db155296bc1086c`.
The source checkpoint file SHA256 is
`b7589ce737351b88beb398b9027075dfc2cdff3fc5ba637a9f78a453dbc58493`.
These are distinct identities: scorer tensor state versus whole checkpoint.

Twelve tasks are fully evaluated by JEV: A-OKVQA validation (1,145),
ScienceQA test/all (4,241), IconQA text/image/fill test (6,316/11,535/3,638),
AI2D test (3,088), VSR random test (2,195), SugarCrepe++ released suite (4,757),
TextVQA validation (5,000), TallyQA test (38,589), GQA balanced test-dev
(12,578), and SNLI-VE test (17,901). NLVR2 and Winoground remain access/resource
gaps and have no reported result. Complete coverage refers to the declared
candidate protocol, not unrestricted open-answer benchmark scoring.

MLP+Null system scores use mean ± sample SD over seeds 20260928/20260929/20260930.
Native uses one released model with its own similarity or answer-likelihood
head. Inputs and candidates match; supervision and text processing differ.
Matched Linear+CE/MLP+CE/MLP+Null arms control the adapter training comparison.
Completed uncached reruns replace numerically rejected cached native results.
No tolerance was relaxed. Four native cases remain missing, not zero:
InternVL and LLaVA on GQA and IconQA fill-in-blank.

## Table generation, no GPU needed

From the repository root:

```bash
python -m scripts.tables.build_paper_evidence
python -m scripts.tables.build_paper_evidence --paper /absolute/path/to/overleaf
```

The generator verifies member hashes, complete status/N, split/candidate and
backbone identities, seeds, shared scorer state, 36/32 record counts and
110,983 evaluation rows. It generates CSV, Markdown and LaTeX tables directly
from `result.json`, including both gains and losses. Ten tasks with complete
three-backbone native comparisons form the main accuracy table. The complete
twelve-task matrix, all-arm timings and source identities remain available.

Cached latency is per-example adapter/scorer wall time with GPU
synchronization. The reported SD is between three seed-level mean times,
not per-example jitter. Native end-to-end latency and generation-token counts
are absent from the aggregate archive; no native speedup is inferred.

## Full training/evaluation on a prepared server

Use the compatible Linux PyTorch/torchvision/transformers environment and
the author's prepared H100 resource bundle. The configuration names required
training inventories, manifests, model resources and source checkpoint.
Those assets are not downloaded implicitly or distributed by this repository.
Check their hashes and permissions. For a new run, use a copied config with
fresh output/report/cache paths; do not overwrite the published snapshot.

```bash
python -m scripts.workflows.run_adapter_study_parallel \
  --config /path/to/new-study-config.json --gpus 0 1 2 3 4 5 6 7 \
  --vision-batch 32 --decode-workers 4 --cpu-threads 2 --prefetch 8
python -m scripts.workflows.run_native_jev_verified \
  --config /path/to/new-study-config.json --gpus 0 1 2 3 4 5 6 7 \
  --native-batch 32 --cpu-threads 2
```

Schedulers print progress and task ETA, preserve complete outputs, and use
explicit uncached fallbacks when the numerical cache guard fails. Training
and native scheduling code is published in functional folders. Import paths
were adapted to the organized package; original source/published hashes are
recorded in `release_metadata/SERVER_CODE_MIGRATION.json`. Historical result
code hashes are retained, not rewritten to pretend these are byte-identical
server execution sources.

The regression tests use synthetic CPU models and do not create new benchmark
results. Before double-blind submission, prepare an anonymized code release
according to the current conference policy; the manuscript's public URL is
an author-requested working-draft link.
