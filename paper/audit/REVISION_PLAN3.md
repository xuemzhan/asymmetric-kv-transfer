# Revision Plan for audit3 — 从 6/10 到 7/10

**Date:** 2026-09-13
**Source:** `paper/audit/audit3.md`（独立三审：当前约 **6/10**，Weak Accept / Borderline；
audit3 §19 明确"解决前三个技术问题 → 7/10 Accept"）
**Peer baseline:** `paper/audit/SELF_REVIEW_W21.md`（本机自审，已修 9 处数值/归属错误）

**Working rule（沿用前两轮）：**

1. 任何新数字先落 `reports/*.json`，再进正文；
2. 数字进正文前登记到 `paper/audit/METRIC_CORRECTION.md`（新增 §9）；
3. 由 `paper/audit/verify_audit3_edits.py` 的断言守卫，断言**先写、后改正文**；
4. 门禁在跑实验**之前**写死在本文件里（pre-registered），跑完不得下调阈值。

---

## 0. audit3 §18「只做 6 件事」→ 本方案条目

| audit3 §18 | 条目 | 机器 | 产物 | 论文落点 |
|---|---|---|---|---|
| 1 evaluator residual | **A1** | GPU | `reports/phaseB_evalcheck_residual_*.json`、`phaseB_evalcheck_attrib.json`、`phaseB_evalcheck_squad.json` | §3.3、Fig. 2、Limitations「Evaluator provenance」 |
| 2 K/V 机制拆分 | **T1** | 本机 | 无（纯文本） | §6.3、Discussion §7.1、Abstract/Intro |
| 3 20-epoch causality | **A2** | GPU | `reports/phaseB_adapter_causal20_<pair>_seed{0,1,2}.json` | Table 5（causal）、§6.4、Abstract/Intro |
| 4 第二域 within-domain repair | **A3** | GPU | `reports/phaseB_squadwithin_split{0,1,2}.json` | Table 7（squadrepair）、§5.3/§6.4、Abstract |
| 5 state vs consumption 误差表 | **A4** | GPU | `reports/phaseB_errorbudget_<pair>_seed{0,1,2}.json` | §6.3 新表（插在 `tab:mappers` 后） |
| 6 全 seed 聚类推断 | **A6** | 本机 | `reports/cluster_stats_audit3.json` | Table 1 表注、§5.1、Limitations |

**audit3 未列入清单、但同样必须处理的文本项**（§14/§15/§3 措辞、§16 结论结构）：

| audit3 | 条目 | 说明 |
|---|---|---|
| §15 | **T2** | 「standard protocol / released code」归属必须拆成"实现层发现（我们自己旧实现）"与"方法论发现（teacher-forced LL ≠ functional transfer）" |
| §3 | **T3** | "end-to-end equivalent to full prefill" 措辞过强 |
| §14 | **T4** | 标题仍然偏强（改不改需作者决策） |
| §9 | **A5（P2，可选）** | test 只有 8 个 document；audit3 §9 建议 30–50 documents |

**优先级与依赖**

**执行状态（2026-09-13，W22）**

| 条目 | 状态 | 证据 |
|---|---|---|
| T1 / T1b / T2 / T3 | **已完成**（本机文本） | `paper/main.tex`；守卫 `verify_audit3_edits.py` ALL PASS |
| A6 | **已完成**（本机统计，零 GPU） | `reports/cluster_stats_audit3.json`、`scripts/cluster_stats_audit3.py`；论文 §5.1 + Table 1 表注 + Limitations 已改写 |
| A1–A4 | **已排队**（GPU） | 代码 + `scripts/run_audit3_gpu_queue.sh` 已推送；等待 `reports/*.json` |
| T4（标题）、A5（文档宽度） | 待决策 | 见下 |

| 优先级 | 条目 | 依赖 | 阻塞谁 |
|---|---|---|---|
| P0 | T1 / T2 / T3（本机文本） | 无 | 不阻塞；使审稿意见 #2 当场关闭 |
| P0 | A1（evaluator residual） | 无 | **阻塞全部**：若 residual 非数值噪声，其余结论悬空 |
| P0 | A2（20-epoch causality） | A1 通过 | Abstract/Intro/Table 5 的因果措辞 |
| P1 | A6（聚类统计全 seed，本机） | A2 的 `--dump-rows`（仅 adapter/causal 部分） | Table 1 表注、Limitations |
| P1 | A3（within-SQuAD） | 无 | §5.3 的跨域叙事 |
| P1 | A4（误差预算表） | A1 通过 | §6.3 的核心机制句 |
| P2 | T4（标题）、A5（文档宽度） | 作者决策 / GPU 预算 | — |

---

# PART I — 本机（无 GPU，可立即执行）

## I.1 纯文本修复（不依赖任何新数据）

### T1 — 把 K-side routing 与 V-side consumption 拆开（audit3 §4/§5，最重要）

**问题（审稿人的技术性指控，非措辞问题）：** `paper/main.tex:702-706` 用
"a value mapper cannot repair the addressing perturbation introduced by mapped keys"
解释 V-only 的 residual gap。但四臂定义在 `paper/main.tex:279-283`：
V-only = **student K** + mapped teacher V，根本没有 mapped teacher K，其 routing 就是
student 自己的 routing。因此该解释与本文自己的实验设计矛盾。

**改法：** 删除 `main.tex:702-706`，替换为下面这段（K/V 两条线各自独立）：

```latex
The value arm and the key arm fail for different reasons, and the four-arm design
keeps them separate. In the value arm the student reads its own keys, so its
routing is the student's routing: the failure is that the mapped value is close
to $V_S$ in representation space while the quantity the receiver actually
consumes is $A_S V W_O$. Fitting the mapper in that consumption space recovers
most of the arm on the equal-depth pair ($0.113\to0.935$) and about a quarter of
it on the flagship pair, and the shuffled-target control stays at the floor, so
what is recovered is document content rather than a generic re-scaling of the
cache. The residual gap on the flagship pair is therefore an alignment gap in
the receiver's consumption space, not an addressing effect: this arm never sees
a mapped key.

The key side is a different question, and we report it separately: under mapped
teacher keys the student's routing does move (top-1 agreement with its own
routing is $0.70$ on $1.7$B$\to0.6$B and $0.59$ on $8$B$\to0.6$B, with
attention cosine $0.97$ and $0.93$). We present routing divergence as a
diagnostic for the key side and consumption-space alignment as the operative
variable on the value side; neither is identified as a unique mechanism by an
intervention, and we no longer use one to explain the other.
```

**配套（同一轮一起改，否则仍然自相矛盾）：**

1. §6.3 小节标题（`main.tex:679`）由
   "Routing diverges, and consumer-space mapping recovers values"
   改为 "Two separate failures: mapped keys perturb routing, mapped values must be consumable"。
2. Abstract `main.tex:81-85` 与 Intro 贡献 2 `main.tex:165-174`：现有句子把
   "top-1 agreement 0.59–0.70" 与 "consumer-space mapper 恢复 V" 相邻叙述；
   需各加半句限定（"a separate, address-side diagnostic"），避免读者把两者读成因果。
3. Discussion §7.1（见下面 T1b）改为 audit3 §16 的四层结构。

**守卫断言（先写）：** `addressing perturbation` 全文 0 次；
`this arm never sees a mapped key` 存在；§6.3 小节标题替换为目标字符串；
Abstract 与 Intro 各出现一次 `separate` + `address`（K/V 分层声明），否则视为漏改。

### T1b — Discussion §7.1 按 audit3 §16 重构为四层

`main.tex:958-971` 现在用一句 "the failure is a compatibility failure" 把
evaluation / value / key / joint 全部概括。audit3 §16 要求拆成四层（**这是它认为
比"compatible consumer"更有技术深度的表述**）。替换文本：

```latex
The evidence supports four statements, and we keep them separate rather than
folding them into a single claim about compatibility.
\emph{(i) Evaluation.} Teacher-forced likelihood does not establish functional
cache reuse: content-free caches (wrong document, moment-matched random,
all-zero) reproduce $89$--$105\%$ of the K-only likelihood gain while answering
no question, and the two defects we identify in the legacy evaluation
implementation inflated greedy EM in our own earlier measurements. The second
point is implementation-specific; the first holds independently of any
implementation detail.
\emph{(ii) Value transfer.} Alignment in raw representation space is
insufficient: the affine mapper minimises representation error and leaves the
value arm unusable, whereas a mapper fit where the student consumes value
recovers it (Table~\ref{tab:mappers}). What tracks task performance is alignment
in the receiver's consumption space.
\emph{(iii) Key transfer.} Mapped teacher keys can perturb the receiver's
routing (top-1 agreement $0.59$--$0.70$ against attention cosine
$0.93$--$0.97$); this is a diagnostic for the key side, established on the
points where the key state is the one being injected.
\emph{(iv) Joint handoff.} A full handoff has to satisfy both, which is why the
joint arm is the weakest arm everywhere.
```

**注意：** (ii) 与 (iii) 的顺序与 audit3 §16 一致；若 A4 的误差表显示
$W_O$-space 误差并不随 EM 同向，(ii) 的措辞必须按 II.A4 的降级版本改。

### T2 — 把「实现层发现」与「方法论发现」拆开（audit3 §15）

**问题：** 正文四处把两个 evaluator bug 说成 "the published evaluation code" /
"the released code"（`main.tex:57`、`:121`、`:291`、`:961`），但仓库证据
（`paper/audit/METRIC_CORRECTION.md:12`：*The published evaluation helper is
`phase0_g0.answer_loglik` + `greedy_answer`*）表明**这两个 bug 只在我们自己的旧实现里**。
全仓库检索没有第三方公开实现被逐行核验的记录。

**改法（安全版，先按此改）：**

```latex
The evaluation protocol we used in earlier versions of this work overstates
transfer. Its greedy exact-match (EM) metric is computed on a cache that has
already been advanced through the query and the gold answer, and it re-feeds the
final query token as the first generation input; both defects are in that
implementation, and we withdraw the numbers it produced.
```

§3.3（`main.tex:286`）改为 "The implementation we used, and which the earlier
versions of this line of work inherit, advances a single cache through ..."，
并在 §3.3 或 Limitations 明写 "we did not audit third-party implementations"。
方法论那一句保持不变、且要独立成立：

> teacher-forced likelihood is not evidence of functional cache reuse
> （由 wrong-document / random / zero 控制支撑，与实现无关）

**可选加强（需要新证据）：** 若要在正文写"更广的实现也存在该缺陷"，必须先在
`METRIC_CORRECTION.md` 里登记被核验的公开仓库 + 文件 + 行号；没有这份取证就不写。

**守卫断言：** `published evaluation code` 全文 0 次；`released code` 全文 0 次；
`we did not audit third-party implementations` 存在（或等价句）。

### T3 — 「equivalent」降级为「closely agrees」（audit3 §3）

`main.tex:302` 现写 "the corrected path is end-to-end equivalent to full prefill on
the quantity we report"。改为：

```latex
the two paths closely agree on the quantity we report (normalized answer EM,
agreement $0.929$--$0.982$; at most two items differ in Self EM for any student
size), while token-level decoding diverges on $7$--$21\%$ of items and the
first-token logits differ by up to $1.344$.
```

（A1 落地后在同一句补上 KL 中位数与 first-token top-1 agreement，用来收口这个差异。）

### T4 — 标题（audit3 §14，需作者决策）

现标题 `Cross-Model KV Transfer Needs a Compatible Consumer` 会让人以为必须改 consumer 权重，
而论文最强结果恰恰是"改 mapper 目标、consumer 权重一个不动"（V-only $0.113\to0.935$）。

audit3 给的两个候选，本方案建议第二个：

1. `Cross-Model KV Transfer Requires Consumer-Space Alignment: Diagnosis and Repair`
2. **`Diagnosing Cross-Model KV Transfer: Evaluation Artifacts and Consumer-Space Alignment`**（推荐）

改标题需要**同步 4 处**：`paper/main.tex:2`（注释）、`:30`（`pdftitle`）、`:37`（`\title`）、
`paper/arxiv_metadata.md`，以及重新拷贝 `paper/arxiv_submission/main.tex`。

### T5 — Limitations 的两处必须与新数据一起改（别先改）

| 位置 | 现状 | 改法 |
|---|---|---|
| `main.tex:1011-1016` | "document-clustered EM intervals exist only in the seed-0 reports" | **已被证伪**（见 I.2）：A6 落地后改为"三个 seed 全部给出 document-clustered CI95" |
| `main.tex:1042-1046` | "Four results are seed 0 only"（含两条 causality） | A2 之后只剩 repaired-regime scrambling 与 projection ablation 是 seed 0 |

### T6 — 守卫脚本先建

新增 `paper/audit/verify_audit3_edits.py`（或扩展 `verify_audit2_edits.py` 的 PART III）：

- **文本断言**：T1/T1b/T2/T3/T4 的"归零字符串 + 新增字符串"
  （例：`The evidence supports three statements` → 0 次，`four statements` 存在；
  `addressing perturbation` → 0 次；`published evaluation code` / `released code` → 0 次）；
- **数字断言**：A1–A6 落地后各自的数字必须在正文出现、且与 JSON 一致；
- 在动正文之前先跑一次，把失败清单当作待办基线。**本轮实跑基线：**
  `python paper/audit/verify_audit2_edits.py` → `OK: all audit2 paper-side assertions hold`
  （即 audit3 的改动必须在**不打破**这组断言的前提下进行）。

## I.2 A6 — 全 seed 文档级统计（本机可做，零 GPU）

**已核实的可行性与障碍（本轮逐文件检查结果）：**

| 报告 | 逐样本行 | document 标识 | 结论 |
|---|---|---|---|
| `phaseB_fourarm_seed{0,1,2}.json` | 有（56 行） | **行内自带 `doc_id`** | 三个 seed 都能**现在**重算 clustered EM |
| `phaseB_controls_{8B,1.7B}_..._seed{0,1,2}.json` | 有 | 无 `doc_id`，但 `id` 可 join `data/test_v2_seed{seed}.json` | 可重算 |
| `phaseB_outaware_*_seed{0,1,2}.json` | 有 | 同上（`id` join） | 可重算 |
| `phaseB_alignment_repaired_1.7B_0.6B_seed0.json` | 无 rows，但 `results[map].summary` 已含 `EM_ci95_clustered` | — | 已有，直接引用 |
| `phaseB_squad_*.json` | 有（30 行） | 30 个 doc 各 1 题 | 无需聚类 |
| `phaseB_adapter_*.json`、`phaseB_adapter_causal_*.json` | **无 rows** | — | **必须补 `--dump-rows` 并随 A2 重跑** |

> 因此论文现在这句 "clustered intervals exist only in the seed-0 reports" 是**过时的**，
> 不是不可克服的限制。审计意见 §10 要求的修法成本接近 0。

**新脚本** `scripts/cluster_stats_audit3.py`：

1. 输入（逐一列明，避免漏 pair）：
   `reports/phaseB_fourarm_seed{0,1,2}.json` +
   `reports/phaseB_fourarm_8B_4B_seed0.json`（seed 0 的 8B→4B 单独成文件）+
   `reports/phaseB_controls_{8B,1.7B}_0.6B_seed{0,1,2}*.json` +
   `reports/phaseB_outaware_*_seed{0,1,2}.json` +
   `reports/phaseB_alignment_repaired_1.7B_0.6B_seed0.json` +
   `reports/phaseB_squad_*.json` + `data/test_v2_seed{0,1,2}.json`；
2. 对四个 arm（Self / K-only / V-only / Joint）与 controls、outaware 的每个 Seed 计算：
   document-clustered bootstrap CI95（复用 `phaseB_common.bootstrap_ci95_clustered`）、
   document 级配对检验（把每个 document 的样本先聚合，再做 Wilcoxon signed-rank 或 sign test；
   `stats_utils.paired_wilcoxon_test` 只作用于逐样本，需新增 `doc_level_paired_test`）、
   以及 cluster 数（=8）；
3. 输出 `reports/cluster_stats_audit3.json`（含每 seed 每 pair 每 arm 的
   `EM`、`EM_ci95_clustered`、`n_docs`、`p_clustered`），并汇总 3-seed 均值±std；
4. 论文侧：Table 1 表注加一行"clustered CI95 in Appendix/§5.1"，§5.1 主句改为
   以 clustered CI 陈述"$0.899$ vs $0.000$"以外的中间结果（例如 $0.351$、$0.304$ vs $0.161$）。

**验收：** 脚本可重复运行、结果与 `phaseB_fourarm_seed0` 已存的 clustered 数值一致（回归锚点），
JSON 进仓库，`METRIC_CORRECTION.md §9` 登记。

---

# PART II — GPU 机器

**一次跑完（推荐）：** 代码已随本方案提交，GPU 机器 `git pull` 后执行

```bash
bash scripts/run_audit3_gpu_queue.sh 2>&1 | tee reports/audit3_gpu_queue.log
```

队列按 A1 → A2 → A3 → A4 依赖顺序执行，已存在的产物自动跳过（可断点续跑），
A1 失败即停止（它是停止线）。跑完把 `reports/*.json` 提交回来，本机再跑
`scripts/cluster_stats_audit3.py` 与 `paper/audit/verify_audit3_edits.py`。
下面的章节是每一步的规格与预登记门禁，队列脚本只是它们的命令行载体。

## II.A1 — evaluator residual 追查 + SQuAD validator（audit3 §3）

**审稿人要求的三件事：** 解释 $7$–$21\%$ token 分歧与最大 $1.344$ logit 误差；
补 $\operatorname{mean}|\Delta\logits|$、$p_{95}|\Delta\logits|$、$KL(p_{\text{full}}\|p_{\text{cache}})$、
first-token top-1 agreement；并在 SQuAD Self 上重复 full-prefill validator。

### 代码改动

**I-1a（扩指标）** `phaseB_evalcheck.py` 每条样本新增：

| 字段 | 定义（$d_v=|\ell^{\text{inj}}_v-\ell^{\text{full}}_v|$，float32 计算） |
|---|---|
| `mean_abs_logit_err` | $\frac{1}{|V|}\sum_v d_v$ |
| `p95_abs_logit_err` | $d$ 的 95 分位 |
| `max_abs_logit_err` | 已有 |
| `frac_abs_err_gt_0.5` | $d_v>0.5$ 的比例（判断是否只是极少数 logits） |
| `kl_full_to_cache` | $KL(\mathrm{softmax}(\ell^{\text{full}})\,\|\,\mathrm{softmax}(\ell^{\text{inj}}))$，单位 nats |
| `first_top1_agree` | $\arg\max$ 是否相同 |

聚合层新增：三档 `mean/p95/max` 的跨样本统计、KL 的中位数与均值、
`first_token_top1_agreement` 比例，以及 `max_err_items_diverged`（$d$ 最大的样本是否恰好是 token 分歧的样本）。

**I-1b（SQuAD 域）** 新增 `--domain {synthetic,squad}`：`squad` 用
`data/squad_test_seed0.json`（30 题 / 30 文档），student 固定 `0.6B`，
reference 与 injected 两条路径都改为 8B teacher 无关的 Self 路径（本题只有学生自注入）。
输出 `reports/phaseB_evalcheck_squad.json`。

**I-1c（归因探针，新增 `reports/phaseB_evalcheck_attrib.json`）** 对 0.6B 的 8 题各跑 4 条路径，
把 $1.344$ 归到具体原因：

1. **确定性探针**：同一 doc 连续两次 `capture_kv`，比较 cache 是否逐位相同；
2. **回环探针**：用 reference 前向自己的 `past_key_values` 直接构造 cache（不经过
   `numpy→bf16` 回环）vs `build_cache(capture_kv(...))`，隔离 bf16 回环；
3. **kernel/掩码路径探针**：把 query-only + cache 前向与 doc+query 全序列前向分别在
   `attn_implementation="sdpa"` 与 `"eager"` 下各跑一次；若差值在 eager 下消失，
   则 residual 是 SDPA/cache-掩码路径差别，而非实现 bug；
4. **position 探针**：显式传 `position_ids=arange(S, S+Lq)` 与不传比较。

### 命令

```bash
python3 phaseB_evalcheck.py --seed 0 --n-eval 56 --students 0.6B 1.7B 4B \
  --output reports/phaseB_evalcheck_residual_seed0.json
python3 phaseB_evalcheck.py --domain squad --n-eval 30 --students 0.6B \
  --output reports/phaseB_evalcheck_squad.json
python3 phaseB_evalcheck.py --attrib --students 0.6B --n-eval 8 \
  --output reports/phaseB_evalcheck_attrib.json
```

### 预登记门禁

| 观测 | 结论 → 论文写法 |
|---|---|
| `first_token_top1_agreement = 1.000` 且 KL 中位数 $\le 0.01$ nats，最大 $d$ 只出现在少数 vocab 维度 | residual 属数值/核路径噪声 → 保留 "closely agrees"，Limitations 引用 KL/top-1，审稿意见 #1 关闭 |
| top-1 不一致或 KL 明显偏大（中位数 $>0.1$ nats） | **停止**：先修 evaluator，再谈其他结论（audit2 B1 的停止规则继续有效） |
| eager 下差值消失 | 正文加一句"the residual is an attention-kernel path difference"，并在 Limitations 说明 |
| SQuAD validator 的 normalized-EM 一致率 $\ge 0.93$ / top-1 agreement $\ge 0.97$ | §5.3 SQuAD 段落加一句"the SQuAD Self numbers use the same validated evaluator" |

**成本：** 3 students × 56 题（含 doc 重算）+ 30 题 SQuAD + 归因探针 ≈ 0.5–1 GPU day。

## II.A2 — 用最终 20-epoch adapter 重做 causality（audit3 §6/§7）

**问题：** Table 5 的 causality 是 `rank 8, 10 epochs, seed 0`（
`reports/phaseB_adapter_causal_{1.7B,8B}_0.6B_seed0.json`，报告内 `epochs=10`），
而 headline（Table 8）是 `20 epochs`；审稿人指出两者不能画等号，且 20 epoch 的 adapter
更可能吸收 task template。8B 更是单 seed 且 margin 很小。

### 代码改动

- **I-2a**：无需新逻辑——`phaseB_adapter.py` 的 `--causal` 已经把 correct / WrongJoint /
  RandKV / ZeroKV 四臂挂在**同一个冻结 adapter** 上，只需把 `--epochs 20` 与
  `--conditions joint` 组合（只训 1 个 adapter，省 2/3 训练时间）。
- **I-2b（必须）**：新增 `--dump-rows`，把逐样本 `id / 各臂 LL / 各臂 EM` 写进报告。
  当前 adapter 与 causal 报告**不含 rows**，导致 A6 无法给 causality 与 adapter 结果算聚类区间。

### 命令（6 个 run）

```bash
for S in 0 1 2; do
  python3 phaseB_adapter.py --pair 1.7B_0.6B --seed $S --rank 8 --epochs 20 \
    --conditions joint --causal --n-calib 70 --n-eval 56 --dump-rows \
    --output reports/phaseB_adapter_causal20_1.7B_0.6B_seed$S.json
  python3 phaseB_adapter.py --pair 8B_0.6B --seed $S --rank 8 --epochs 20 \
    --conditions joint --causal --n-calib 70 --n-eval 56 --dump-rows \
    --output reports/phaseB_adapter_causal20_8B_0.6B_seed$S.json
done
```

### 预登记门禁

| 对 | 判定 | 结果 → 论文措辞 |
|---|---|---|
| $1.7$B$\to0.6$B（equal-depth） | correct-KV EM $>$ 所有破坏臂 **3/3 seeds**，且 (correct − max destroyed) 的 document-clustered CI95 排除 0 | "content causality established on the equal-depth pair" |
| $8$B$\to0.6$B（flagship） | 3-seed margin $m=\mathrm{EM}^{\text{correct}}-\max \mathrm{EM}^{\text{destroyed}}$；$m\ge0.10$ 且 clustered CI95 排除 0 → resolved；$0<m<0.10$ 或 CI 含 0 → 保持 "suggestive / unresolved" | **不得**升级为 "demonstrates content-specific transfer on both pairs" |

**报告规范：** 逐 seed 列表 + 3-seed pooled 估计；**不跨 pair 平均**（AGENTS.md 约定 K/V 分开报）。

**成本：** 1.7B 每 run 约 40–60 min，8B 每 run 约 1.5–2 h（W20 期间曾遇 GPU 争用），
合计约 0.5–1 GPU day；串行即可，注意 `OMP_NUM_THREADS=8` 与先跑通一个 seed 再排队。

## II.A3 — 第二域 within-domain repair（audit3 §8）

**动机：** 现有 SQuAD 结果只回答"fix 不 cross-domain generalize"。
audit3 指出还需回答"repair mechanism 能否在第二 domain **内部**复制"，
两者是不同问题；若 within-domain 恢复，故事升级为
"机制跨域可复制，但 repair 参数是 domain-specific"。

**数据事实（已核）：** `data/squad_test_seed0.json` = 30 题 / **30 个不同 document**（每题一文档）；
`data/nq_test_seed0.json` 的 `doc` 为空字符串（NQ 是 open 任务，**没有 document state**），
所以第二域**只有 SQuAD 可用**，且天然无 clustering 问题。

### 新脚本 `phaseB_squad_within.py`

- 文档级划分：`perm = RandomState(split_seed).permutation(30)`，前 15 为 calibration、后 15 为 held-out；
  `split_seed ∈ {0,1,2}` 充当 3 个"seed"（数据只有一个 seed，用划分重采样代替）；
- 复用：`load_teacher/load_student/capture_all/fit_mapper`、`phaseB_mechanism.fit_output_aware_mapper`、
  `phaseB_outaware.fit_wo_aware_mapper`、`phaseB_adapter.train_adapter`（rank 8, 20 epochs）；
- 报告内容：held-out Self 天花板、K-only / V-only / Joint × {affine, outaware, W_O-aware}、
  `adapter(joint, SQuAD-calib)`、`shuffled-target` 对照、逐样本 rows（便于 clustered/document bootstrap）。

```bash
for D in 0 1 2; do
  python3 phaseB_squad_within.py --split-seed $D --n-calib 15 --n-eval 15 \
    --v-mappers affine,outaware,wo --adapter --epochs 20 --rank 8 \
    --output reports/phaseB_squadwithin_split$D.json
done
```

### 预登记门禁（两个方向都能写，但写哪个由数据决定）

| 结果 | 结论 |
|---|---|
| held-out V-only（或 Joint）$\ge 0.5\times$ Self$_{\text{held-out}}$，且 document bootstrap CI95 排除 0 在 $\ge2/3$ 个划分成立，且 shuffled-target 对照 $\le$ 真 mapper 的一半 | "repair mechanism replicates within a domain; the parameters are domain-specific" |
| 各臂 $\approx 0$ | 更强的负结果："consumer compatibility is bound to the task distribution"，正文把 SQuAD 段从"仅不跨域"升级为"域内也不复制" |
| 中间 | 如实报告为 partial，**不给**方向性结论 |

**必须同时声明的混淆：** SQuAD 只有 15 个 calibration 文档（合成域是 70），
且 SQuAD Self 天花板只有 $0.367$。若预算允许，补一个
"synthetic calibration 只用 15 文档"的等量对照，否则在 Limitations 明写该混淆。

**成本：** 3 splits ×（8B teacher 30 文档 capture + 15 文档 adapter 训练 20 epoch + 15 题多臂评测）≈ 0.5 GPU day。

## II.A4 — state-space vs consumption-space 误差预算表（audit3 §11）

**动机：** 论文核心命题是 "where you align matters"，但 `tab:mappers` 只有 EM。
审稿人明确要求一张把 **raw V 误差 / $A_SV$ 误差 / $A_SVW_O$ 误差 / EM** 放在一起的小表。

### 新脚本 `phaseB_errorbudget.py`

对 5 个 mapper 变体（`raw`(无仿射、仅层选择平均)、`affine`、`outaware`、`woaware`、`outaware-shuffled`）
在同一个 pair/seed 上计算三列误差（每样本先在 (layer, head, token) 上求 Frobenius 比，再跨样本取均值±std）：

```text
e_raw  = ||V̂ − V_S||²_F / ||V_S||²_F
e_attn = ||A_S V̂ − A_S V_S||²_F / ||A_S V_S||²_F
e_wo   = ||(A_S V̂ − A_S V_S) W_O||²_F / ||A_S V_S W_O||²_F
```

外加该变体的 V-only corrected EM。$A_S$ 用 `phaseB_mechanism.get_attn_map`（student 自己的 cache），
$W_O$ 用 `student.model.layers[l].self_attn.o_proj.weight` 的 query-head 切片（同 `phaseB_outaware.fit_wo_aware_mapper` 的取法）。

```bash
for S in 0 1 2; do
  python3 phaseB_errorbudget.py --pair 1.7B_0.6B --seed $S --n-calib 70 --n-eval 56 \
    --output reports/phaseB_errorbudget_1.7B_0.6B_seed$S.json
  python3 phaseB_errorbudget.py --pair 8B_0.6B --seed $S --n-calib 70 --n-eval 56 \
    --output reports/phaseB_errorbudget_8B_0.6B_seed$S.json
done
```

### 预登记判定

- 计算三个误差列与 EM 在"变体 × pair"上的 Spearman $\rho$（$n=10$，须标注样本量小、不给 p 值结论）；
- 预期（若核心命题成立）：`e_raw` 最小的是 affine，而 affine EM 最低（二者不同向）；
  `e_attn` / `e_wo` 与 EM 同向；
- 若 `e_wo` 与 EM 不同向 → **必须**把 §6.3 "consumption space 才 trace 性能"降级为
  "in our setting, alignment in the attention-output subspace tracks task performance better than raw
  representation error"，并如实写反例。

**成本：** 每 pair/seed 需 70+56 个 doc 的 attention map（Attention 输出最贵），≈ 0.5 GPU day 全量。

## II.A5 —（P2，可选）文档宽度扩展（audit3 §9）

**现状：** 56 test 题来自 **8 个 document**（每 doc 约 7 题），`data/test_v2_seed*.json` 已核实。
audit3 认为这是"顶会短板"，但把它放在"投资回报"语境、不在 6 条必做清单里。

**做法：** 用同一生成器扩宽 test 文档数，**不要覆盖现有数据**：

```bash
python3 data/build_ood.py --seeds 0,1,2 \
  --n-train-projects 20 --n-val-projects 8 --n-test-projects 30 \
  --output-dir /workspace/v3/data_wide
```

→ test 约 210 题 / 30 文档。需要给 `phaseB_common.load_data` 加 `--data-dir`（或用环境变量
`V3_DATA_DIR`）覆盖硬编码的 `/workspace/v3/data`。

**建议范围（按 ROI）：** 只重跑**旗舰对 8B→0.6B** 的 four-arm + controls（3 seeds），
以及等深对 1.7B→0.6B 的 mapper/alignment 关键臂；作为"wide split (30 docs)"稳健性表报告，
主表仍保留原划分（并在表注写明两者差异），避免全表重录。

**成本：** 约 1–2 GPU day（含 capture 与 greedy 解码）。

**决策点：** A1–A4 完成后再决定是否做；若时间紧，宁可把这条写进 Limitations 并引用 A6 的
document-level 统计，也不要用半成品替换主表。

---

# PART III — 条件式正文改写（数据回来后再动）

| # | 位置（当前行号） | 依赖 | 结果分支 → 改法 |
|---|---|---|---|
| W1 | `main.tex:302`、`:1028-1030`、Limitations「Evaluator provenance」 | A1 | KL/top-1 小 → 写 "closely agrees"，点明 residual 来源；KL 大 → 停止线，先修 evaluator |
| W2 | §5.3 Second domain（`:574-592`）、§6.4 的 SQuAD 段（`:802-813`）、Table 7 | A1 + A3 | A1 通过 → 加"同一验证过的评测器"；A3 复制 → 加 within-domain 行并改写结论；A3 全 0 → 升级为"域内亦不复制" |
| W3 | Table 5（`:815-838`）、§6.4（`:791-800`）、Abstract `:91-95`、Intro `:183-187` | A2 | 20-epoch + 3 seeds 替换 10-epoch 单 seed；按门禁决定 "established / unresolved" |
| W4 | §6.3（`:679`、`:690-706`）、Discussion §7.1（`:958-986`） | T1（无数据依赖） | 按 I.1 的替换文本；A4 落地后再把 "consumption space tracks performance" 换成带数字的版本 |
| W5 | 新表（插在 `tab:mappers` 之后，`:725`） | A4 | 三列误差 + EM；表注写清误差定义与聚类方式 |
| W6 | Table 1 表注（`:468-471`）、§5.1、Limitations「Corpus effects」（`:1007-1016`） | A6 | 改为"三个 seed 全部给出 document-clustered CI95"；中间结果的区间写进正文 |
| W7 | Limitations「Partial seed coverage」（`:1042-1046`） | A2 | 收缩为"两处仍为 seed 0" |
| W8 | 标题 4 处 | T4 决策 | 按决定统一替换 |
| W9 | `paper/audit/METRIC_CORRECTION.md` 新增 §9 | A1–A6 | 每条新结果一段：命令 + 产物 + 关键数字 + 对旧叙述的影响 |

---

# PART IV — 完成定义（DoD）与停止规则

**完成定义（audit3 的 3 个 main concern 全部关闭）：**

1. A1 给出 KL / top-1 / logit-error 分布，且 SQuAD Self 有同源 validator；
2. T1 落地且守卫断言证明 "mapped-key routing" 不再被用来解释 V-only；
3. A2 用最终 20-epoch adapter、3 seeds 重做 causality，8B 按门禁如实定位；
4. A3、A4、A6 三条 P1 完成并进正文；
5. `verify_audit3_edits.py` ALL PASS；`main.pdf` 重新编译（0 undefined）；
   `paper/arxiv_submission/` 同步；`ITERATION_LOG.md` 记录本轮。

**停止规则（沿用 audit2 的纪律）：**

- A1 的 top-1 agreement 显著 $<1$ 或 KL 中位数 $>0.1$ nats → 先修 evaluator，其余实验不作数；
- A2 的 8B margin 不达标 → **不**下调阈值、不换指标，直接在正文保持 "unresolved"；
- A3/A4 出现与叙述相反的结果 → 改结论，不改门禁；
- 任何 K/V 结论必须分开报告，不得平均。

---

# PART V — 取证表（audit3 每条意见的仓库级证据）

| audit3 | 意见要点 | 证据位置 | 本轮核实结论 |
|---|---|---|---|
| §3 | evaluator 仅"EM 一致"不够，需 KL/top-1/logit 分布；SQuAD 也要 validator | `phaseB_evalcheck.py`（只有 max/mean abs err）、`paper/main.tex:294-302` | **成立**：需 I-1a/1b/1c |
| §4 | V-only 没有 teacher K，不能用 mapped-K routing 解释 | `paper/main.tex:279-283`（四臂定义）vs `:702-706`（解释） | **成立，必须改** |
| §5 | 应拆成 K-side routing / V-side consumption 两个问题 | `phaseB_mechanism.py`（routing 诊断）、`phaseB_outaware.py`（W_O-aware） | 证据已足够，纯写作 |
| §6 | causality 用 10-epoch adapter，headline 用 20-epoch | `reports/phaseB_adapter_causal_*.json`（`epochs=10`）、`paper/main.tex:798-800` | **成立**：I-2b + A2 |
| §7 | 8B causality 单 seed、margin 小 | `reports/phaseB_adapter_causal_8B_0.6B_seed0.json` 仅此一份 | 成立；门禁已要求 pooled 3-seed |
| §8 | 缺第二域 **within-domain** repair | `reports/phaseB_squad_{outaware,adapter}_seed0.json`（calib 域=OOD） | 成立；NQ 无 doc，只能 SQuAD 内部划分 |
| §9 | test 只有 8 个 document | `data/test_v2_seed{0,1,2}.json`：56 题 / 8 docs（本轮实测） | 成立；A5（P2） |
| §10 | 聚类区间只有 seed 0 | **本轮实测：fourarm 三个 seed 的 rows 都带 `doc_id`**；controls/outaware 可用 `id` join | **审稿人判断过时**：A6 可全量重算；但 adapter/causal 无 rows，需 I-2b |
| §11 | 缺 state vs consumption 误差表 | `paper/main.tex:708-725`（`tab:mappers` 只有 EM） | 成立；A4 |
| §12 | projection ablation 只有 seed 0，措辞保持 suggestive | `paper/main.tex:843-861`（o_proj/v_proj 均 0.964） | 与论文现措辞一致，保持 |
| §13 | 1.7B Self 低分不是格式问题，但建议 error taxonomy | `reports/phaseB_selfdiag_seed0.json`（raw==norm EM） | 已解决；taxonomy 为可选 |
| §14 | 标题偏强 | `paper/main.tex:2,30,37` | 成立；T4 待决策 |
| §15 | 「standard protocol / released code」归属不清 | `paper/main.tex:57,121,291,961` vs `METRIC_CORRECTION.md:12`（bug 在 `phase0_g0` 自身） | **成立且偏事实性风险**；T2 |
| §16 | 结论应重构为 evaluation / value / key / joint 四层 | `paper/main.tex:958-971`（现为 "The evidence supports three statements"） | 成立；T1b 的替换文本已按此重写 |

---

# PART VI — 不做的事（明确记录，避免下一轮重复讨论）

- **不再扩模型对**：audit3 §9 明确"不会建议现在去加 20 个 model pairs"；现有 6 对足够支撑负结果。
- **不为了让 8B causality 好看而换 rank/epoch/指标**：门禁写死，不达标即如实报 unresolved。
- **不把 clustered CI 当作把 8B→4B V-only $0.440$ 变成正结果的手段**：Self $0.815$ 仍是天花板。
- **不把 "standard protocol is broken" 写成对社区的指控**：除非补齐第三方实现的逐行取证。

---

## VII. 本轮新增/修改的文件清单

| 文件 | 动作 | 机器 |
|---|---|---|
| `paper/audit/REVISION_PLAN3.md` | 新增（本文件） | 本机 |
| `paper/audit/verify_audit3_edits.py` | 新增（T6，先写断言） | 本机 |
| `scripts/cluster_stats_audit3.py` | 新增（A6） | 本机 |
| `phaseB_evalcheck.py` | 扩指标 + `--domain squad` + `--attrib`（I-1a/b/c） | GPU |
| `phaseB_adapter.py` | 新增 `--dump-rows`（I-2b） | GPU |
| `phaseB_squad_within.py` | 新增（A3） | GPU |
| `phaseB_errorbudget.py` | 新增（A4） | GPU |
| `phaseB_common.py` | `load_data` 支持 `--data-dir`/环境变量（仅 A5 需要） | GPU |
| `paper/main.tex`、`paper/arxiv_submission/main.tex`、`paper/arxiv_metadata.md` | T1–T5 + W1–W8 | 本机 |
| `paper/audit/METRIC_CORRECTION.md` | 新增 §9 登记 | 本机 |
| `ITERATION_LOG.md` | 本轮条目 | 本机 |
