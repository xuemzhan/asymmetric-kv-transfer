# Revision Plan for audit2 — split by machine

**Date:** 2026-09-12
**Source:** `paper/audit/audit2.md` (independent re-review; verdict 5.5--6/10 borderline, "补齐前四个关键问题 → 7/10")
**Working rule:** 所有新数字必须先落 `reports/*.json`，再进正文；不得先写数字后补实验。

---

## 0. 分工总览

| 机器 | 内容 | 产出 |
|---|---|---|
| **GPU 机器** | 5 处代码改动 + 5 组实验（B1--B4, C1, C4） | `reports/phaseB_*.json` |
| **编辑机器** | 论文文字、图、PDF（本机已完成 II-A） | `paper/main.tex`, `paper/figures/*`, arXiv bundle |

编辑机器上**不需要**任何实验结果就能做的修改，已经在这一轮全部完成（见 II-A）。

---

# PART I — GPU 机器：需要补充的实验

## I.0 先要写的 5 处代码

| # | 文件 | 改动 | 对应审稿编号 |
|---|---|---|---|
| I-1 | `phaseB_evalcheck.py`（**新建**） | cache-injection vs full-prefill 逐样本比对 | §3 |
| I-2 | `phaseB_adapter.py` | `eval_arms` 增加 `extra_arms: dict[str, KV]` 参数 | §6 |
| I-3 | `phaseB_alignment.py` | 增加 `--mapper {affine,outaware}` 开关 | §9 |
| I-4 | `phaseB_common.py` | `score_arm` 一并返回/保存生成文本 `gen` | §12 |
| I-5 | `phaseB_common.py` | `summarize_rows` 增加 `_em` 指示向量的聚类 bootstrap 分支 | §11 |

I-2 到 I-5 都是局部改动。I-1 是新文件，但可以复用现成组件：`load_student`、`capture_all`、`build_cache`、`greedy_answer_fixed`、`query_of`。

## I.1　B1 — Evaluator 的 gold-standard 验证（§3，最高优先）

**目的：** 当前整篇论文的前提是"旧 evaluator 有 bug、新 evaluator 正确"，但验证只有 6 个样本。这条不闭合，其余结论都悬空。

**做法：** 对每个 student 规模的完整 Self test set（56 题）逐样本跑两条路径：

- (a) cache injection：`build_cache(student_kv)` + query 前缀 → `greedy_answer_fixed`；
- (b) full prefill reference：`ids = concat(doc_ids, query_ids)`，不传 cache → 同一套贪婪解码。

**上报三个量：**

1. token 级贪婪生成一致率（一致样本数 / 总数）；
2. 首个生成 token 的 logits max abs error；
3. 归一化 EM 一致率。

**范围：** 至少覆盖三个 student 规模（0.6B / 1.7B / 4B）的 seed 0；资源允许则做 6 pair × 3 seeds。

**预先登记的判定门禁：**

- 一致率 ≈ 100% → 论文核心前提成立，正文用完整结果替换 "6/6"。
- 一致率明显 < 100% → **先修 evaluator**，修好之前不得提交其余结论。

**命令：** `python3 phaseB_evalcheck.py --seed 0 --n-eval 56 --output reports/phaseB_evalcheck_seed0.json`

## I.2　B4 — 换掉无效的 token-shuffle control（§4，零新代码）

**已核实的结论：** `phaseB_controls.py:85-89` 把**同一个置换**同时作用于 K 和 V，而正文引用的是 `Shuf_KV`（`:148`）。该组合下 attention 恒等，EM≈0.89 是数学必然，不构成内容控制。

**修复（不需要新代码）：** 同一循环里已经算好了两个真正破坏 K/V 对应的 arm：

```python
arms["Shuf_K"] = KV(k=shuf.k, v=sv)   # 打乱 K，保留 V
arms["Shuf_V"] = KV(k=sk, v=shuf.v)   # 保留 K，打乱 V
```

它们已进入 `keys`（`:158`）并写入 JSON。重跑后**改报 `Shuf_K` / `Shuf_V`** 即可。

**命令：** `python3 phaseB_controls.py --pair 8B_0.6B --seed 0 --n-calib 70 --n-eval 56 --output reports/phaseB_controls_8B_0.6B_seed0_fixed.json`

**判定：**

- `Shuf_K`/`Shuf_V` 的 EM 明显低于 0.899 → 拿到有效的"evaluator 对内容破坏敏感"正对照。
- 若仍接近 0.899 → 承认目前**没有**有效的敏感性正对照，只能依赖 wrong-doc/random/zero 的负向证据，并在 Limitations 明写。

## I.3　B2 — Adapter 内容因果（§6）

**目的：** 排除"adapter 学的是任务而不是读取被转移的状态"。

**做法（I-2 改动后）：** 保持 content-matched（teacher Joint KV）adapter **完全不变**，在 test 上分别注入：

- correct teacher KV（已有）
- wrong-document teacher KV
- moment-matched random KV
- zero KV

**判定门禁：** $EM_{correct} \gg EM_{wrong} \approx EM_{random} \approx EM_{zero}$ 才成立。

**预警：** shuffled-document-trained adapter 已达 Joint 0.458（1.7B）/ 0.131（8B），说明 generic teacher-state adaptation 贡献相当大，本实验有真实失败风险，须如实报告。

**命令：** `python3 phaseB_adapter.py --pair 1.7B_0.6B --seed 0 --n-calib 70 --n-eval 56 --output reports/phaseB_adapter_causal_1.7B_0.6B_seed0.json`（8B 同样各跑一次）

## I.4　B3 — 补齐 8B→0.6B adapted Self（§7，零新代码）

**已核实：** `eval_arms` 无条件评测 Self 臂（`phaseB_adapter.py:128,134`），因此该数字是被算出来过、只是没进归档。

**命令：** `python3 phaseB_adapter.py --pair 8B_0.6B --seed 0 --n-calib 70 --n-eval 56 --output reports/phaseB_adapter_8B_0.6B_seed0.json`

**注意：** 补这个数不只是填空。若 adapter 改了 `o_proj`，"K=0.792 恢复了大半"的参照系必须是**同条件的 adapted Self**，而不是原模型的 0.899。参照系错了结论方向可能反转。
## I.5　C1 — Repaired-regime layer scrambling（§9）

**目的：** 当前"layer alignment 不重要"的结论是在 affine mapper 下得到的，而 affine 下 V-only EM 本来就接近地板，存在 floor effect。

**做法（I-3 改动后）：** 在 V-only EM 已被 OutAware mapper 提到 ~0.93 的条件下，重做 offset / permutation scrambling。

**命令：** `python3 phaseB_alignment.py --pair 1.7B_0.6B --seed 0 --n-calib 70 --n-eval 56 --mapper outaware --output reports/phaseB_alignment_repaired_1.7B_0.6B_seed0.json`

**判定：** 仍不掉分 → 现结论加强；V EM 从 0.93 跌到 ~0.1 → **必须改写结论**为"layer alignment 确实重要，只是被 mapper 缺陷掩盖"。

## I.6　C2 — 1.7B Self 反常的诊断（§12）

**现象：** corrected Self 为 0.6B→0.899、1.7B→0.440、4B→0.815，非单调。

**做法（I-4 改动后）：** 保存生成文本，报告 normalized EM、token F1、抽取式答案 EM、生成长度分布，并对 1.7B 的错误样例分类（完全答错 vs "The answer is X." 被 raw EM 判错）。

**判定：** 若主要是格式问题 → 0.44 属 metric artifact，所有以 Self 为参照的绝对数值叙述都要调整。鉴于本文主题就是"评价指标有问题"，这一点不能含糊。

**产出：** `reports/phaseB_selfdiag_seed0.json`（新脚本或 `phaseB_evalcheck.py` 的附加输出，二选一）

## I.7　C3 — 四臂主表的 document-clustered CI（§11）

**已核实：** `phaseB_common.py:386-391` 已实现聚类 bootstrap，但**只作用于 LL**；EM 只有裸均值（`:383`）。

**做法（I-5 改动后）：** 对 `_em` 指示向量做 cluster bootstrap（cluster = document）。test 只有 8 个文档，CI 会明显变宽，须如实呈现。

**命令：** `python3 phaseB_fourarm.py --seed 0 --n-calib 70 --n-eval 56`

**附加：** Wilcoxon 补 document-level paired test 或 clustered permutation 作稳健性检验。主结论 `0.899 vs 0` 不依赖 p 值，聚类不改变判断方向。

## I.8　C4 — Constructive fix 的跨域验证（§13）

**做法：** 用 synthetic 上训练的 OutAware mapper / adapter **直接在 SQuAD 上评测，不重新训练**。

**命令：** `python3 phaseB_squad.py --seed 0 --n-calib 70 --n-eval 30`

**判定：** 非零（哪怕 0.10--0.20）→ 支持修复可跨任务；全 0 → 明确写"修复是 task-distribution-specific"，结论严格限定为 repairability of consumption, not task generalization。

## I.9　运行顺序与注意事项

| 顺序 | 实验 | 阻塞关系 |
|---|---|---|
| 1 | **B1** evaluator 验证 | 其余全部结论的前提；失败则停止 |
| 2 | **B4** controls（Shuf_K/Shuf_V） | 高优先级、零新代码 |
| 3 | **B3** 8B adapted Self | 高优先级、零新代码 |
| 4 | **B2** adapter 内容因果 | 需先改 I-2 |
| 5 | C1 / C3 | 需先改 I-3 / I-5 |
| 6 | C2 / C4 | 需先改 I-4 |

**注意：** 脚本的 `--output` 默认写 `/workspace/v3/reports/`（Linux 路径），请显式传 `--output` 到实际 report 目录。所有产物必须进仓库并在 `METRIC_CORRECTION.md` 登记，否则后续无法交叉校验。

**附带债务：** `scripts/verify_corrected_paper.py` 仍在校验旧字符串且依赖缺失的 `phaseB_*.json`。**开工前先把它对齐到新产物**，否则"数字门禁"形同虚设。
---

# PART II — 编辑机器：论文修改

## II-A　已完成（不需要任何新数据，本轮已落地）

### II-A1　一一对应表：审稿意见 → 方案条目 → 落地位置 → 验证

每一行都可机械核验。`verify_all.py` 对全部 23 个断言逐条检查，当前 **0 失败**；实例侧另用 TeX 结构校验（引用/文献/交叉引用）与 PDF 含量比对。

| audit2 编号 | 审稿意见 | 方案条目 | 落地位置 | 验证断言 |
|---|---|---|---|---|
| §4 | token-shuffle 可能置换不变 | A1 / A4 / A5 / A6 + X1 / X3 | Abstract；Intro 贡献 1；Results §5；`tab:controls`；`fig:controls` 子图 (b) + 图注；Exp Setup §4.3 | `token-shuffled` 全文 0 次；`Token-shuffled` 0 次；`attention is invariant to a shared` 存在 |
| §16 | adapter 被误读为 label-free | A2 + X2 | Abstract；Intro 贡献 3 | `trained only on` 全文 0 次；`cross-entropy on calibration answers under injected teacher KV` 出现 2 次 |
| §9 | layer-alignment 结论有 floor effect | A3 + X4 + A8 | Abstract；Intro 贡献 2；Analysis §6.1 标题 | `is not the controlling` 全文 0 次；`Layer alignment alone does not explain the failure` 出现 2 处 |
| §8 | adapter 是否 arm-specific 不明确 | A7 | Method §3.5 | `trains its own adapter from scratch` 存在 |
| §14 | projection ablation 未完成 | A9 | Analysis §6.4 | `still running` 全文 0 次；`This ablation is preliminary (seed 0, two of` 存在 |
| §5 | “failure is consumer-side” 过强 | X5 + X6 + X7 | Intro 结尾；Discussion §7.1；Abstract | `separates two levels of compatibility` 存在；state-space / consumption-space 分层句存在 |
| §3 | evaluator 仅 6-case 验证 | — | 未改（等 I.1 / B1） | 现保留“六条目端到端探针”的如实表述 |
| §10 | routing 措辞克制 | — | 保留原文 | 审稿人明确要求保留 |
| §15 | 压缩旧 K/V asymmetry 叙事 | — | 未做（编辑性、可选） | 不影响审稿结论 |
| §16 | 标题偏强 | — | **待你决定** | 见 II-B 末 |

### II-A2　重要修正：本轮发现的漏改

第一轮只改了 Abstract，漏掉了同一问题在 Introduction 与 Experimental Setup 中的另外三处：

- Intro 贡献 1 仍写“A token-shuffled student cache keeps EM at 0.89±0.04”（X1）
- Intro 贡献 3 仍写“trained only on calibration examples”（X2）
- Exp Setup §4.3 仍把“a token-shuffled student cache”列为 control（X3）

三处已一并修正。这正是“一一对应”检查的价值：同一个 claim 在全文出现多次时，只改一处会留下自相矛盾。现已可机械证明旧陈述在全文中归零。

## II-B　待 GPU 结果回来后改（按依赖排列）

| # | 位置 | 依赖 | 预期改法 |
|---|---|---|---|
| B1 | Method §3.3 + Limitations | I.1 (B1) | 用完整一致率替换 "six-item end-to-end probe"；Limitations 的 Evaluator provenance 条改写为已逐样本对齐 |
| B2 | Results §5 + `tab:controls` | I.2 (B4) | 新增 `Shuf_K` / `Shuf_V` 两行与正文；若无效则改写道 Limitations |
| B3 | `tab:adapter` + Analysis §6.4 | I.3 (B3) | 补 8B→0.6B 的 Self 列；把"恢复大半"的参照系从 0.899 改为同条件 adapted Self |
| B4 | Analysis §6.4 + 新表 | I.4 (B2) | 增加 correct/wrong/random/zero 因果表；按结果调整"content-specific"的强度 |
| B5 | Analysis §6.1 | I.5 (C1) | 若 repaired regime 下 scrambling 掉分，改写 layer-alignment 结论 |
| B6 | Exp Setup §3.4 + Limitations | I.6 (C2) | 若 1.7B 低分主要是格式问题，补充 normalized EM / F1 并调整 metric 说明 |
| B7 | `tab:main` + §3.4 | I.7 (C3) | 主表 EM 改报 document-clustered CI95；统计协议补 document-level 检验 |
| B8 | Results / Analysis | I.8 (C4) | 增加跨域修复结果；按结果限定 positive claim 的适用范围 |

**待决：标题。** 现标题 "Needs a Compatible Consumer" 偏强，因为 consumer-aware mapper 在 **consumer 权重完全不变**的情况下就把 1.7B→0.6B 的 V-only 从 0.113 提到 0.935。建议改为
`Cross-Model KV Transfer Requires Consumer-Aware Alignment: Evaluation, Diagnosis, and Repair`。
本轮**未改标题**，等你确认后再动（需同步 `main.tex`、`\hypersetup{pdftitle}`、`paper/arxiv_metadata.md`）。

---

## III. 门禁

任何新数字进正文前必须同时满足：

1. 对应 `reports/*.json` 已进仓库；
2. 通过数字交叉校验脚本（**需先修好 `verify_corrected_paper.py`**）；
3. 已登记到 `paper/audit/METRIC_CORRECTION.md`。

---

## IV. 附：代码级取证结论（本轮核实，含行号）

| 审稿意见 | 取证位置 | 结论 |
|---|---|---|
| §4 token-shuffle 可能置换不变 | `phaseB_controls.py:85-89` | **审稿人正确**，且替代 control 已存在（`:146-147`） |
| §7 8B adapted Self 缺失 | `phaseB_adapter.py:128,134,193-204` | 纯归档缺失，零新代码 |
| §8 adapter 是否 arm-specific | `phaseB_adapter.py:196-204` | 每条件独立训练、四臂共用；纯文档问题 |
| §11 主结果缺 cluster CI | `phaseB_common.py:369-393` | 聚类已实现但只作用于 LL；EM 需补分支 |
| §3 evaluator 仅 6-case 验证 | 全仓库 grep | 无参考实现，需新写 |
| §16 adapter 是否 label-free | `phaseB_adapter.py:103-114` | 使用 gold answer 的 next-token CE，摘要原表述属事实性误导（已改） |

---

## V. 本机遗留

`paper/main.pdf` 被 Codex 进程占用，本轮未能就地刷新（仍为 22:04 版本）；最新版已写入 `paper/arxiv_submission/main.pdf`（16 页，全部修改已包含）。关闭占用后可重跑 `tools/compile_paper.ps1` 同步。