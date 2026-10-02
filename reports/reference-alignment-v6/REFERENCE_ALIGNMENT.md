# V6 参考文献对应清单

核对范围：最新 V6 compilefix 正文实际使用的 citation keys，对照作者于 2026-10-02 提供的 BibTeX 清单。这是本地清单对应与著录替换，不是出版元数据真实性的在线核验。

- 作者清单：21 条；当前正文：22 个唯一引用，均有 BibTeX 条目。
- 对应条目：21 条，正文不缺少作者清单中的任何条目。
- 额外条目：1 条，暂时保留以避免删除相关工作和对照模型的来源。
- 除额外条目外，采用作者提供的字段、作者次序与 publication version。未添加新的论文引用。

## 不在作者清单中的唯一实际引用

`yu2026visualjev` — Guanxu Yu and Yuhang Yao, **Visual Jev: Accurate and Efficient Decisions from Shared Visual Context**, 2026. 工程已有条目写为 `arXiv:2609.25845`，链接 `https://arxiv.org/abs/2609.25845`；此轮没有重新在线验证该条目的出版元数据。

位置：`sec/2_formatting.tex`，Related Work 的 Visual decision interfaces 段。用途：标明被对照的 official Visual Jev 方法来源，区分其 answer-SFT 路线和本文的 external VisualJEVV3 scorer。建议保留；若作者限定只能使用其 21 条清单，需要同时调整相关工作引用与对照模型来源说明，而非仅从 .bib 删除。

可单独复制的 BibTeX 见 `additional_references.bib`。

## 著录替换和最低限度格式清理

| Key | 旧工程 | 本轮采用作者条目 |
|---|---|---|
| liu2023visual | CVPR 2024, 26292--26302 | NeurIPS 2023, 36, 34892--34916 |
| bi2025llava | 2024 arXiv preprint | ACL 2025, 15230--15250 |
| lu2021iconqa | NeurIPS 34 | 2021 arXiv preprint（按作者提供版本） |
| lu2022learn | 旧条目作者 Xia, Tony，缺少页码 | 作者提供 Xia, Tanglin；补作者给定页码 2507--2521 |
| bai2025qwen3 / tschannen2025siglip / li2024llava / wang2508internvl3 | 旧工程的完整或截短作者列表 | 作者提供的作者顺序与 others 形式 |
| wang2508internvl3 | 作者输入没有 year，journal 含损坏 URL 空格 | 补 year=2025（作者 title 已含 2025）；journal 规范为同一 arXiv:2508.18265，不改变文献身份 |

其余字段统一采用作者输入。未使用的 @String 模板定义不进入新的 main.bib，以避免重复宏定义；这不删除任何文献条目。此轮没有改变任何实验数字、图或正文论证。

## 中文核对

本轮没有凭空补 DOI、页码、出版状态或新参考文献。已有额外引用被显式列出，尚待作者决定。论文 PDF 须重新编译检查所有引用都可解析；在线元数据核验不在这份本地对应清单的完成声明中。
