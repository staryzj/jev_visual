# V7 frozen-scorer manuscript revision audit

## Scope and argument

This is an evidence-preserving rewrite, not a new training or evaluation run.
The primary question is whether heterogeneous frozen vision representations,
through backbone-specific adapters, can drive one unchanged external scorer.
Official Visual Jev uses a different decision head and training provenance;
its accuracy cannot isolate this variable.

Shortest evidence chain: shared input contract → recorded adapter-only reuse
of the same checkpoint → missing direct adapter controls and non-beneficial
image controls → preliminary functional compatibility, not demonstrated useful
visual dependence. Full-split dataset transfer is a separate secondary question.

## Result allocation

| Evidence | Canonical location | Boundary |
|---|---|---|
| SigLIP2 / InternVL3.5 / LLaVA-OneVision adapter-only accuracies | Main Table 1; §4.1 | 80 custom held-out examples, not official test |
| Zero-token, random and trained-linear controls | Main Table 1; §4.2 | Unmeasured under exact protocol; `--` |
| Target-backbone original/blank/noise/wrong controls | Main Table 2; §4.3 | Original never exceeds blank; wrong shift is not verified semantic intervention |
| Six complete dataset splits | Main §4.4; full-result and metric tables | Different checkpoint; secondary transfer, not target-backbone proof |
| Historical Qwen semantic pairs and ablations | Main §4.5 | Different checkpoint/membership; motivation only |
| Official Visual Jev matched reference table | Supplement Table S1 | Original numbers retained; different head and training provenance |
| Dataset access gaps, detailed records and qualitative examples | Main remaining experimental subsections | Retained per earlier all-main preference; not completed benchmark results |

The main paper keeps conclusion-changing controls, negative results and
missing baselines visible. Only the unrelated official-system comparison is
relocated, following the latest explicit request for a Supplement.

## Change/deletion log

- Removed Official Visual Jev comparator language from Abstract and the
  secondary-evidence list in Introduction; removed its Conclusion mention.
- Related Work contains the single main-text distinction and Supplement route.
- Removed the dedicated official-reference experimental subsection and main
  table inclusion. Complete numeric rows moved unchanged to Supplement S1.
- Removed the official-adapter 100% semantic-pair comparison from main Qwen
  diagnostics; preserved that result in the Supplement table caption.
- Split the core experiment into adaptation, missing adapter controls and
  existing visual-dependence controls; moved dataset transfer into §4.4.
- No result was upgraded into stronger task utility, universal adaptation,
  superiority, full-test coverage or efficiency evidence.
- Approved overall photo-style diagram, all existing figure bytes and
  user-aligned bibliography are unchanged.

## Numerical provenance and repetition

`scripts/build_frozen_scorer_paper_tables.py` regenerates the two core tables
from the three real `experiments/results/backbone_generality/*/results.json`
files. `table_sources.json` records their SHA256 values. Every other numerical
table row is compared to the reference-aligned V6 source by `qa_v7.py`.

Canonical numerical reporting is in tables. Abstract retains the three core
accuracies and controlled N for a self-contained summary; Introduction and
Conclusion interpret the boundary without repeating those percentages.
Exact-protocol missing controls are explicit, not borrowed from 64/128-example
historical runs. The Qwen row is an architectural reference, not a measured
fourth target run. Blank images and zero aligned tokens remain distinct.

## Source length audit

Whitespace-token counts below measure LaTeX source only, not journal word count:

| Section source | Before | After |
|---|---:|---:|
| Abstract | 165 | 158 |
| Introduction | 351 | 344 |
| Related Work + Method | 1761 | 1761 |
| Conclusion | 130 | 126 |
| Experiments outer file | 931 | 743 |
| Core adaptation subsections | 425 | 505 |
| Detailed experimental record | 1615 | 1625 |

## Delivery QA

Main PDF: 16 pages. Standalone reference Supplement: 1 page. Both roots
compile with explicit `placeins` support for `FloatBarrier`. No undefined
commands, citations, references, BibTeX errors or overfull boxes were found.
The installed template reports its existing hyperref/backref warning and
lineno encoding warning; these do not block compilation. All page contact
sheets and the new core-table/Supplement pages were visually inspected.

This remains a working-length manuscript: keeping detailed diagnostics in
the body does not establish compliance with a conference page limit.
No new experimental run or GitHub push was performed in this revision.
