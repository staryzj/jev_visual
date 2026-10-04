# Compact publication snapshot

This is the retained Visual-JEV implementation and real experiment evidence, not a copy of the entire Open-Jev worktree.

## Organization and exclusions

All formerly root-level Python sources are under `visualjev/` or functional `scripts/` subfolders. Imports, subprocess paths, shell workflow roots and documented commands were updated together. Shared classification metrics were extracted without changing their calculations.

Unrelated upstream tasks, obsolete prototypes, duplicate documentation and comparator-specific artifacts were removed from the current tree. Required `jev/` dependencies, tests, configurations, protocols and current evidence remain. Previous Git commits preserve removals; there was no force push or history rewrite.

Original full-split result JSON and all 21,742 completed prediction records retain their exact bytes. Three target-backbone aggregate results retain their original source identities and exact exported bytes. Unmeasured controls stay missing.

## Evidence boundaries

The 80-example target-backbone protocol, historical custom controlled/dataset-specific experiments and complete official-split evaluations are distinct. Do not pool accuracy or call custom partitions official full tests.

Target-backbone source scorer SHA256:
`8e4f9703d4629e222fbdb2e6007c43ae224e7e64137ce8e880ea2ecf39e366ad`.

Separate full-evaluation checkpoint SHA256:
`b7589ce737351b88beb398b9027075dfc2cdff3fc5ba637a9f78a453dbc58493`.

Weights, feature caches, credentials, raw source images and gated resources are excluded. Historical reproduction needs the matching resources; a model name alone is not checkpoint identity.

## Integrity checks

`release_metadata/PUBLICATION_MANIFEST.json` lists exported hashes, source identities and migration provenance. Byte-identical records, reorganized sources and declared aggregate exports are distinguished. Aggregates disclose omitted large/private per-example fields in `public_export_metadata`.

Run `python -m scripts.maintenance.verify_publication` for hashes, completed prediction coverage and target-backbone table consistency, and `python -m pytest -q` for regression tests. These checks do not constitute new training or evaluation runs.
