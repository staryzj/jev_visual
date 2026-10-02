# Visual-JEV V3 三阶段论文实验

> 更新：统一 schema、固定四分割 manifest 与独立 Temperature Scaling
> 已整理在 `docs/benchmark_v1.md`。当前校准结果只使用完整的本地 COCO
> hard-negative 数据；未下载完整的数据不计入任何实验结果。

## 1. 实验结论

本实验在不修改 `jev/vl_smoke.py`、`vision_test.py`、现有 V2 checkpoint 和旧 V3 入口的前提下，完成了 Qwen3-VL-4B-Instruct + JEV 的三阶段训练与独立测试。

- Qwen3-VL 原生视觉 backbone 正常。`test.jpg` 的原生 image-text-to-text 输出能够描述图中的“预览不变、目标改变”的视觉演示；在候选顺序按样本确定性打乱的 128 条独立测试上，原生生成式回答为 100.0% accuracy / 100.0% macro-F1。
- Full V3 已经真正依赖视觉。独立测试中原图 accuracy 为 88.28%；blank/noise 的正确候选概率从 53.48% 降至 33.39%/33.33%，接近三分类均匀分布；48 个语义反事实 pair 的双向正确率为 56.25%，预测翻转率为 60.42%。
- Stage A 有正贡献：相对无 Stage A，accuracy +1.56 个百分点，语义反事实双向正确率 +14.58 个百分点，翻转率 +12.50 个百分点。
- Stage B 是主要的闭集决策学习阶段：Stage-A-only 随机决策头的 accuracy 为 9.38%，Stage A+B 为 87.50%，提升 78.12 个百分点。该比较只用于说明 decision training 的必要性，不将随机头作为正式方法基线。
- Stage C 是解决视觉失效的关键：相对无 Stage C，accuracy +0.78 个百分点，双向反事实正确率 +54.17 个百分点，翻转率 +58.33 个百分点；blank KL 从 0.9475 降至 0.00013，noise KL 从 0.9497 降至 0.00016。
- 结果足以开始撰写方法、实验设置和初步结果，但不足以直接支撑最终投稿中的广泛结论。当前缺口主要是实验规模、数据覆盖和统计重复，不是 V3 仍完全视觉失效。

## 2. 数据与划分

数据沿用项目中已有的 MS COCO 2017 hard-negative 数据，避免引入闭源 API。

| 划分 | 样本数 | 语义反事实 pair | 用途 |
|---|---:|---:|---|
| Train | 2048 | 1656 | Stage A/B/C 训练 |
| Validation | 128 | 40 | checkpoint 选择 |
| Test | 128 | 48 | 独立最终评测 |

原有 256 条 validation 按索引奇偶确定性拆成 128 validation / 128 test。反事实 pair 只有 source 和 partner 同时落在同一子集时才保留，因此 validation/test 不共享图像。候选顺序在评测时按 `seed + index` 确定性打乱，排除了“正确答案永远位于 A/索引 0”的位置捷径。

训练样本为一条正确描述和两条同 supercategory 的 object-substitution hard negatives。该数据能测试对象级视觉依赖，但不能覆盖属性、关系、计数、OCR、细粒度空间推理等更广泛能力。

## 3. 模型与三阶段训练

### Stage A：Visual Adapter Alignment

Qwen3-VL vision/text backbone 全部冻结。对每张图像提取：

- pre-merger tokens：1024 维；
- Qwen3-VL 原生 merger 的 post-merger teacher tokens：2560 维。

原生 spatial merge 每四个 pre-merger token 对应一个 post-merger token。先对四个 token 做组内均值，再通过最小两层 adapter 映射到 2560 维：

\[
\hat{V}=\mathrm{Adapter}(\mathrm{MeanGroup}_4(V_{pre})).
\]

对齐损失为：

\[
\mathcal{L}_{A}=\lambda_{cos}(1-\cos(\hat{V},V_t))
+\lambda_{mse}\frac{\|\hat{V}-V_t\|_2^2}{\mathbb{E}[V_t^2]}
+\lambda_{norm}\,\mathrm{SmoothL1}(\|\hat{V}\|,\|V_t\|).
\]

配置：3 epochs，AdamW，lr `5e-4`，weight decay `1e-2`，`lambda_cos=1`、`lambda_mse=1`、`lambda_norm=0.1`。

| Stage A 指标 | Epoch 1 | Epoch 2 | Epoch 3/best |
|---|---:|---:|---:|
| Validation cosine similarity | 0.8402 | 0.8685 | 0.8805 |
| Validation normalized alignment loss | 0.4745 | 0.3975 | 0.3648 |
| Student/teacher norm ratio | 0.8231 | 0.8792 | 0.8938 |

### Stage B：Candidate-aware JEV Decision

Stage A adapter 初始化后，加载现有 V2 candidate-aware decision 权重。主要结构保持不变：

1. candidate text 形成 query；
2. visual tokens 形成 key/value；
3. multi-head cross-attention 选择候选相关视觉证据；
4. feature-wise gate 融合 text query 与 attended visual evidence；
5. 所有候选共享一个 scalar JEV head。

候选损失为：

\[
\mathcal{L}_{candidate}=-\log\frac{\exp(s_y)}{\sum_k\exp(s_k)}.
\]

配置：4 epochs，AdamW，lr `1e-4`，weight decay `1e-2`；vision/text backbone 保持冻结。最佳 validation accuracy 出现在 epoch 1，脚本自动恢复最佳权重。

### Stage C：Visual Dependency / Anti-shortcut

Stage C 从最佳 Stage B checkpoint 开始，以 `5e-5` 小学习率联合微调 alignment adapter 与 decision adapter。总损失为：

\[
\mathcal{L}=\mathcal{L}_{candidate}
+\lambda_{cf}\mathcal{L}_{counterfactual}
+\lambda_{img}\mathcal{L}_{invalid}
+\lambda_{txt}\mathcal{L}_{text-null}
+\lambda_{rank}\mathcal{L}_{cross-image}.
\]

其中：

- `counterfactual`：同一候选集合在语义匹配图 I1 上要求 T1>T2，在 partner 图 I2 上要求 T2>T1；
- `invalid`：blank、noise 和非语义匹配 wrong image 上最小化 `KL(p || Uniform)`；
- `text-null`：候选文本被替换为同一 null/mean feature 时要求均匀；共享 scalar head 的候选置换对称性也从结构上保证该约束；
- `cross-image`：matched image 上正确候选得分应高于 wrong image 上同一候选得分。

权重为 `lambda_cf=1.0`、`lambda_img=5.0`、`lambda_txt=0.25`、`lambda_rank=0.5`，训练 4 epochs。

最初将 Stage A adapter 在 Stage C 完全冻结时，test 语义反事实双向正确率只有 27.08%。诊断后改为以小学习率联合微调，最终提升至 56.25%。冻结版本保存在 `v3_full_frozen_alignment.pt`，避免覆盖失败证据。

## 4. 主结果

独立 test 共 128 条，所有 scalar 模型报告 accuracy、macro-F1、NLL、multiclass Brier 和 10-bin ECE。

| 模型 | Acc. | Macro-F1 | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL native generative | **100.00** | **100.00** | — | — | — |
| Random/untrained scalar head（wiring sanity） | 12.50 | 12.37 | 1.141 | 0.694 | 0.225 |
| MeanPool + MLP + JEV | 83.59 | 83.51 | 0.816 | 0.273 | 0.131 |
| V2 candidate-aware | **88.28** | 88.32 | **0.349** | **0.191** | **0.052** |
| V3 simple post-merger | **88.28** | 88.31 | 0.638 | 0.348 | 0.318 |
| V3 without Stage A | 86.72 | 86.64 | 0.676 | 0.371 | 0.325 |
| V3 without Stage C | 87.50 | 87.49 | 0.463 | 0.206 | 0.096 |
| Full V3 three-stage | **88.28** | **88.34** | 0.670 | 0.369 | 0.336 |

解释：Full V3 保持了 V2 的 hard accuracy，但 NLL/Brier/ECE 明显差于 V2。原因是 Stage C 主动降低无效视觉下的置信度，并把原图概率从过度尖锐的分布拉回较软分布。后续需要在独立 calibration split 上做 temperature scaling，不能用 test 调温。

Qwen3-VL native 100% 表明 backbone 识图正常，同时也说明当前 COCO object-substitution test 对 4B 原生生成模型偏容易；它不能作为 V3 已达到“通用视觉推理”水平的证据。

## 5. 视觉依赖诊断

`KL-U` 为 `KL(p || Uniform)`，越接近 0 表示候选越均匀。`Pair-both` 要求同一语义 pair 的两张图均选对对应候选。

| 模型 | Original Acc. | Blank KL-U | Noise KL-U | Pair-both | Pair flip |
|---|---:|---:|---:|---:|---:|
| V2 candidate-aware | 88.28 | 0.82676 | 0.82478 | 0.00 | 0.00 |
| V3 without Stage C | 87.50 | 0.94752 | 0.94966 | 2.08 | 2.08 |
| V3 simple post-merger | 88.28 | 0.00004 | 0.00007 | 43.75 | 56.25 |
| V3 without Stage A | 86.72 | 0.00020 | 0.00023 | 41.67 | 47.92 |
| **Full V3** | **88.28** | **0.00013** | **0.00016** | **56.25** | **60.42** |

Full V3 的条件诊断：

| 条件 | Acc. | 正确候选概率 | Margin | 相对 original flip | JS(original, condition) |
|---|---:|---:|---:|---:|---:|
| Original | 88.28 | 53.48 | 0.688 | 0.00 | 0.00000 |
| Blank | 39.84 | 33.39 | -0.008 | 60.16 | 0.03971 |
| Noise | 33.59 | 33.33 | -0.010 | 69.53 | 0.03997 |
| Wrong image | 82.03 | 51.92 | 0.608 | 11.72 | 0.00561 |
| Image swap | 83.59 | 51.54 | 0.596 | 8.59 | 0.00634 |

普通 wrong/swap 条件不保证替换图片恰好对应当前三个完整句子中的另一个答案，因此其“准确率下降”只能作为敏感性指标。真正判断排序是否按语义翻转，应使用 48 个显式 semantic counterfactual pairs：source accuracy 70.83%，counterfactual accuracy 81.25%，双向正确 56.25%，翻转率 60.42%。

缓存审计也排除了复用错误。32 个样本中 original-pre 与 blank-pre 的 pooled L2 最小值为 56.78，original-pre 与 noise-pre 的最小值为 49.68，没有零距离或别名复用。旧 post-merger 缓存与本次重新提取的 Stage-A teacher 抽查逐元素一致，最大绝对误差为 0。

## 6. Native 视觉本体测试

`test.jpg` 绕开 JEV adapter/head，直接调用本地 Qwen3-VL-4B-Instruct 的 `generate`。输出为：

> The image presents a visual demonstration showing that a model’s preview remains unchanged while its actual target changes, highlighting a discrepancy between what’s shown and what’s selected.

另在 128 条 test 上直接进行 image-text-to-text 多选生成，并确定性打乱 A/B/C 顺序；target 位置分布为 A=39、B=43、C=46，结果为 128/128。由此可以把 V2 的视觉失效定位到 adapter/fusion/head/训练目标，而不是 Qwen3-VL vision backbone 无法识图。

## 7. 工程记录

- Seed：`20260928`。
- Effective batch size：1（视觉 token 长度可变；无梯度累积）。
- GPU：NVIDIA GeForce RTX 5060 Ti，16 GB。
- Backbone：Qwen3-VL-4B-Instruct，vision/text 全程冻结。
- 可训练参数：Stage A adapter 3,680,768；candidate-aware decision 831,361；Full V3 4,512,129；MeanPool baseline 830,209。
- 初始完整 suite 训练时间：1573.95 s；修复 Stage-C joint fine-tuning 约 606 s；不含一次性特征缓存约 36.3 min。
- 峰值训练显存（不含 backbone cache generation）：约 406 MB；缓存生成时加载完整 4B backbone。
- 所有 best checkpoint 均按 validation 指标选择，test 只在训练完成后评测。
- LoRA text-backbone 对照没有执行：16 GB 单卡下会牺牲主实验完成度，而且当前核心问题已由冻结 backbone 的 adapter/head 实验隔离。后续可在更大显存机器上补充。

## 8. 产物与复现

关键文件：

- `train_visual_jev_v3.py`：缓存生成与 Stage A/B/C、MeanPool、消融训练；
- `eval_visual_jev_v3.py`：主指标、baseline、native generative baseline；
- `diagnostic_visual_dependency.py`：Original/Blank/Noise/Wrong/Swap 与缓存审计；
- `visual_jev_v3_pipeline.py`：Stage-A adapter、三阶段 wrapper、MeanPool baseline；
- `configs/visual_jev_v3_paper.json`：全部超参；
- `experiments/results/metrics.json` / `metrics.csv`：论文表格源数据；
- `experiments/results/visual_dependency.json`：视觉依赖验收报告；
- `experiments/results/qwen3_vl_native_answers.json`：128 条原生生成结果；
- `experiments/results/native_vision_smoke.json`：`test.jpg` 原生识图结果。

从头运行：

```bash
cd /home/jiezuo/projects/Open-Jev
bash scripts/run_visual_jev_v3_paper.sh
```

已有缓存时单独评测：

```bash
.venv/bin/python eval_visual_jev_v3.py --native
.venv/bin/python diagnostic_visual_dependency.py
```

## 9. 失败案例与限制

1. Full V3 仍有 43.75% semantic pairs 不能双向全对，说明视觉依赖已建立但语义反事实学习尚不充分。
2. 普通 wrong/swap 的分布变化仍弱于 blank/noise；Full V3 的 semantic JS 为 0.00561，而 invalid-image JS 约 0.0398。
3. 只使用一套 2048/128/128 划分和单随机种子，尚无均值±标准差或置信区间。
4. 数据只覆盖 COCO caption object substitution，Qwen3-VL native 已达 100%，存在 benchmark ceiling。
5. Full V3 calibration 较差，需要独立 calibration set 和 temperature scaling。
6. 尚未加入 SugarCrepe 全类别、Winoground、ARO、属性/关系/计数/OCR 等更强 compositional benchmarks。

## 10. 论文准备判断

可以开始写论文，尤其是方法、三阶段训练、loss、视觉失效诊断、缓存排错和当前消融部分。当前方法层面的核心问题已经从“完全不看图”改善为“可测量且显著的语义视觉依赖”。

但在投稿前仍应补：至少 3 个随机种子、更多 semantic pairs、一个或两个公开 compositional benchmark、独立 calibration、统计置信区间和失败案例分类。换言之，当前不足主要是实验规模/数据覆盖/统计重复，而不是方法仍未解决视觉失效。
