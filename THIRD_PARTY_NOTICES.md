# Third-party notices

## Open-Jev source

Source: https://github.com/Zefan-Cai/Open-Jev

Commit: `bd4118882f733574a3250a4b65fe4d884130c08b`

Runtime source: `third_party/open_jev`

License: MIT; copyright 2026 Open-Jev contributors.

The runtime package and its LICENSE are retained unchanged. The complete
tracked-file snapshot is distributed as `open-jev-source-bd411888.zip` in
the release, including the upstream reports and demonstrations.
`release_metadata/open_jev_source.json` records file sizes and SHA256 hashes
for both the runtime subset and complete archive. Visual-JEV imports the
runtime under a separate namespace and implements visual transfer in
`visualjev/transfer`.

## Released receiver weights

- https://huggingface.co/ZefanCai/Open-Jev-27B-v1.1
- https://huggingface.co/ZefanCai/Open-Jev-9B

Receiver LoRA, scalar head and configuration packages are distributed under
their upstream Apache-2.0 license. Copies in the GitHub release include
the original licenses, model cards and metadata. Their names, source
repositories and hashes identify them separately from Visual-JEV's trained
visual interfaces.

## Base and source models

Qwen3.8-27B, Qwen3.5-9B, Qwen3-VL-4B-Instruct,
InternVL3_5-1B-HF, LLaVA-OneVision-Qwen2-0.5B-OV-HF and
SigLIP2-Base-Patch16-224 are downloaded from their original repositories.
The lock file records immutable revisions and license identifiers. Their
original model cards and any supplied license files accompany the download.
These base models are not redistributed as Visual-JEV-authored weights.
