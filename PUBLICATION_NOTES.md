# Publication snapshot: scope and provenance

Target: https://github.com/staryzj/jev_visual. Prepared 2026-10-02 (Asia/Shanghai).

This independent publication checkout retains the target repository's original initial commit. It does not replace the research worktree's origin or commit unrelated local edits there. Source files derive from the local Open-Jev research extension; upstream attribution and the MIT software license are retained.

Local research checkout base commit: `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`. Publication source includes the selected local experiment extensions/modifications on top of that base; per-file identities are in the publication manifest. The original target initial commit is `f5865e7d73e38b834488e8a01ca5345c849ef7b1`.

## Included evidence

- Six byte-identical full-split result JSON files plus 21,742 complete per-instance prediction records: ScienceQA 4241, A-OKVQA 1145, IconQA text-choice 6316, AI2D 3088, VSR random 2195, SugarCrepe++ 4757.
- Dataset protocol/registry, metadata snapshots, preparation/evaluator/training scripts, tests, configuration and result-backed CSV/Markdown/LaTeX tables.
- Three target-backbone result records (80 custom held-out examples each), a frozen-head identity audit and separate follow-up commands. No full official target-backbone evaluation is claimed.
- Official Visual Jev custom matched aggregates and independent/custom/control aggregates, with embedded original-source hash and omitted-field lists where redacted. These are not byte-identical originals. Complete native benchmark and custom matched results remain separate.

`PUBLICATION_MANIFEST.json` records the original source path/hash, exported file hash/size, copy-versus-aggregate mode and omitted field pointers. It excludes itself and these publication-only notes/ignore rules. No new inference, training, calibration or fabricated score was used to prepare the release.

## Excluded resources and incomplete reproducibility

Raw benchmark images, question/caption/answer corpora, candidate manifests, cached features, downloaded model files, trained checkpoint tensors, partial prediction logs, keys/tokens and paper photo/AI-generated raster assets are not published. Source data are obtained through original dataset access routes under their own terms; a mirror does not provide additional authorization. The software license does not relicense third-party data. No new data license is assigned here.

Exact checkpoint-dependent inference reproduction remains incomplete without the recorded trained weight files and authorized source resources. A SHA256 is an identifier, not a weight download. Full-table regeneration also checks local candidate-manifest hashes. Historical redacted aggregates cannot reconstruct omitted examples; source-score hashes and corresponding original checkpoint hashes remain distinct.

SugarCrepe++ is a released evaluation suite stored under the HF `train` split, not a true test split. A-OKVQA is validation. IconQA covers its native text-choice test task, not all IconQA tasks. No pooled accuracy is calculated across these tasks. Missing/gated datasets are metadata gaps, not completed measurements.

## V7 evidence-allocation update

The refreshed README leads with backbone-specific adapters driving the same
frozen external scorer. Dataset transfer remains secondary and uses its own
checkpoint. Official Visual Jev matched comparisons remain historical
reference records; the V7 manuscript locates their complete table in a separate
Supplement. They are not primary cross-backbone evidence.

The update adds `scripts/build_frozen_scorer_paper_tables.py` and
`reports/frozen-scorer-v7/`: numeric source CSV/Markdown, original result
hashes, revision audit and manuscript QA source. The QA source requires the
separate V6/V7 manuscript directories and compiled PDFs; this code/results
repository does not include those manuscript assets. The original-source
hashes in the V7 reports match aggregate-export metadata, not exported JSON
bytes. The publication validator checks both identities and recorded scores.
Zero/random/linear/semantic-pair controls remain unmeasured. Existing six
complete-split prediction files and numerical results are unchanged.

This refresh reruns publication validation and unit checks, not training or
benchmark inference. It updates corresponding files by a normal Git commit,
preserving previous published history. No force push, raw-data release or
checkpoint redistribution is involved.

## Release validation

The export checked full prediction SHA256, unique UIDs and complete counts, allowed field names, credential-pattern matches and file-size limits. This is a bounded release preflight, not an absolute security/privacy certification or a journal-submission compliance certificate.

39 unit tests passed for complete-split protocol guards, VisualJEVV3 scorer/pipeline, conversation routing, independent-statistics helpers and canonical-source identity/missing-control table exports. GPU benchmark inference and training were not rerun during publication.

Run `python scripts/verify_publication.py` for the standalone read-only checksum, prediction-count and table-denominator audit; no GPU, checkpoints or source benchmark images are needed for that audit.

## 中文核对

公开的是代码、真实数值结果和出处记录，不是所有原始数据或可直接运行的权重包。六个完整 split 的预测记录保持原文件字节和哈希；包含源题目的历史文件只提供有省略字段记录的汇总导出。Winoground、SNLI-VE/Flickr30K、NLVR2 的人工授权和资源要求仍需按原协议处理。后续如发布权重、原始数据或论文配图，必须先完成相应授权与再分发检查。
