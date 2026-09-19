# Revision Plan for audit4 — 从 7/10 Accept 到 8/10 候选

**Date:** 2026-09-13
**Source:** `paper/audit/audit4.md`（独立审稿：**7/10 Accept, confidence 4.5/5**；
审稿人明确"不应再以补漏洞为主要目标，而应把 audit + repair 提升为
functional compatibility framework"）
**Peer baseline:** `paper/audit/audit3.md` → `REVISION_PLAN3.md`（全部条目已闭合，
见 W26/W27；当前正文即 audit4 审阅的 23 页版本）

**Working rule（沿用前三轮 + 本轮新增第 5 条）：**

1. 任何新数字先落 `reports/*.json`，再进正文；
2. 数字进正文前登记到 `paper/audit/METRIC_CORRECTION.md`（本轮新增 §11）；
3. 由守卫脚本断言，**断言先写、后改正文**；
4. 门禁在跑实验**之前**写死在本文件里，跑完不得下调阈值；
5. **（新增）相关系数类结论必须同时报告 pooled 值与逐 run 值**。单 run 只有 5 个点的
   秩相关不得作为唯一支撑（PART 0.1 的复算证明了这条的必要性）。

---

## 0. audit4 建议 → 本轮条目映射与裁量

| audit4 | 条目 | 机器 | 产物 | 论文落点 | 我的判断 |
|---|---|---|---|---|---|
| §3 | **T3** Functional Compatibility Decomposition | 本机 | 无（纯文本 + 公式） | 新 §3.3 | **全盘采纳**，零算力、收益最高 |
| §11 | **T2** 四条贡献重构 | 本机 | 无 | §1 贡献列表 | **全盘采纳**（措辞按 PART 0.3 的归属纪律改写） |
| §10 | **T1** 标题重构 | 本机 | 无 | 标题 + PDF 元数据 + README + arXiv 元数据 | **采纳**（audit3 T4 的悬置项一并收口） |
| §9 | **T5** 两处措辞修正 | 本机 | 无 | §7.1(iv)、Related Work | **全盘采纳**；PART 0.2 给出反例证据 |
| §5 | **T6** "task-conditioned compatibility" 命名 | 本机 | 无 | Abstract / §5.3 / §7 / Limitations | **部分采纳**：命名 + 限定条件，见 T6 的纪律表 |
| §8 | **T4** 减法：factorial、cost 两段移 Appendix | 本机 | 无 | Appendix A / B | **采纳，但 cost 必须先修正**（PART 0.5） |
| §7 | **A2** mapper objective sweep（α/λ/rank） | GPU | `reports/phaseB_mappersweep_*.json` | §6.3 新表 + 图 | **采纳**（2–3 h），关闭"5 点相关"的漏洞 |
| §4 | **A1** routing-aware K mapper | GPU | `reports/phaseB_routingkey_*.json` | §6.3 新段落 + Fig 4 面板 | **采纳**（45–60 min），形成 K/V 对称 |
| §6 | **A4** compatibility frontier（D0–D5） | GPU | — | Future Work（可选 pilot） | **降级为 Future Work 的预登记设计**，见第二部分的 A4 |
| §12–5 | cross-family pair | GPU | — | — | **不做**（本轮）；保留在 Limitations |
| §12–6 | 再加公开 benchmark | GPU | — | — | **不做**（与 audit4 一致） |
| PART 0.1（本文） | 秩相关只报 pooled 值 | 本机 | 复算脚本 | §6.3、Table 4 表注 | **audit4 未提，必改** |
| PART 0.4（本文） | 4 份 v1 文档仍留在 `paper/` | 本机 | 归档 + stale 头 | — | **audit4 未提，必改** |
| PART 0.5（本文） | Cost 段的 mapper 参数化与 §3.4 矛盾 | 本机 | 复算 | Appendix B | **audit4 未提，必改** |
| PART 0.6（本文） | 8 处硬编码 `Section~N` | 本机 | label 化 | 全文 | 结构调整的前置条件 |

**本轮总判断。** audit4 的四个"提升价值"建议里，**三个不需要 GPU**（T1/T2/T3/T5），
一个需要（A1/A2）。审稿人给的排序（K mapper > decomposition > sweep > frontier）我同意，
但要补一句工程判断：**先做 T 系列再做 A 系列**，因为 T2/T3 决定了 GPU 实验要测什么、
门禁怎么判；反过来先跑实验，很容易又跑出"没有叙事落点"的 JSON。

另一个需要写明的分歧：audit4 说"这一版不应该继续补漏洞"，我同意方向，但本方案 PART 0 里
有 3 条**必须先补**的事实性问题（秩相关披露、joint 措辞、cost 参数化），它们都属于
"审稿人一查仓库就能发现"的类型，成本 <1 小时。**框架升级与事实修正不是二选一。**

**执行状态（2026-09-13，W28）**

| 条目 | 状态 | 证据 |
|---|---|---|
| T1–T7、T9、T10 | **已完成**（本机文本，零 GPU） | `paper/main.tex`；`paper/audit/verify_audit4_edits.py` ALL PASS |
| T8 守卫 | **已完成** | 新守卫 + 被取代的 4 条旧断言（audit2 的 B3/B4、audit3 的 T1/A4）已重指向 |
| A1–A3 | **未开始**（GPU） | 见 PART II；队列脚本与门禁待写 |
| A4 frontier | 按计划降级为 Future Work 设计 | 见 PART II A4 |

**执行中的三处偏离（已记录，供后续复核）**

1. **T1 标题**取 "Why Representation Alignment Is Not Enough"，未用 PART I 里的
   "Representation Alignment Is Not a Transfer Metric"：后者容易被读成
   "任何 alignment 都不预测任务表现"，与本文自己的正向结果（消费空间误差与 EM
   一致）冲突。
2. **Appendix B 不出现 2050 这个数字**。原计划在附录里写"旧版曾报 2050"，但把已知
   错误的数字留在论文里只会制造混淆；改为在正文明确当前参数化给出 ≈260 tokens，
   旧数字的去向登记在 `METRIC_CORRECTION.md §11`（含 `phase3_rate_law.py` 的
   per-layer 参数化溯源）。守卫因此断言全库无 `2050`。
3. **T6 命名按 A3 条件式落地**：正文先启用 *task-conditioned*，但同时写明"单一第二域、
   15 份 calibration 文档、held-out Self 0.20–0.40、calibration volume 未被排除"。
   A3 若显示是体量问题，则按 PART III 的条件表撤下该命名。

**验证（W28）**：`verify_audit4_edits.py` / `verify_audit3_edits.py` /
`verify_audit2_edits.py` / `scripts/verify_corrected_paper.py` /
`tests/test_stats_utils.py` 全部 exit 0；`main.tex` 与
`arxiv_submission/main.tex` 逐字节一致；pdflatex 两遍编译 0 error、
0 overfull、0 underfull，24 页。

---

# PART 0 — 独立核查：audit4 前提的仓库级证据

## 0.1 `ρ = −1.00` 的支撑面比正文披露的更窄

审计4 §7 的质疑（"只有 5 个 mapper variants，Spearman −1.0 没有那么有说服力"）成立，
而且比它说的更值得处理：正文的 `−1.00 / −0.80 / −0.10` 是**把 6 个 run 的均值先
pooled、再对 5 个 variant 求秩相关**。逐 run 复算（`reports/phaseB_errorbudget_*.json`，
本机 `python` 复算，见 PART V 取证表 E-1）：

| run | ρ(e_raw, EM) | ρ(e_attn, EM) | ρ(e_WO, EM) |
|---|---|---|---|
| 1.7B→0.6B s0 | −0.40 | −1.00 | −1.00 |
| 1.7B→0.6B s1 | −0.60 | −0.90 | −0.90 |
| 1.7B→0.6B s2 | −0.60 | −0.80 | −0.80 |
| 8B→0.6B s0 | −0.20 | −0.80 | −0.80 |
| 8B→0.6B s1 | −0.20 | −0.80 | −0.90 |
| 8B→0.6B s2 | −0.10 | −0.90 | −0.90 |
| **pooled（正文数值）** | **−0.10** | **−1.00** | **−0.80** |

结论有两层：**(a)** 方向性结论（e_attn/e_WO 追踪 EM、e_raw 不追踪）在 6 个 run 里
**全部成立**，不需要降级；**(b)** 但正文只给 pooled 值，而报告 JSON 自己的 note 已经写着
`"Spearman over 5 mapper variants in one run; n is too small for a p-value"`。
审稿人若打开 `reports/`，会算出 `−0.80` 而不是 `−1.00`，这属于**披露不足**而非数据问题。
→ 落 **T7**（本机，30 分钟）：正文同时给 pooled 与逐 run 范围，并把 Table 4 表注的
"over the five variants"改成"five variants × six runs (pooled means; per-run range …)"。

## 0.2 "joint arm is the weakest arm everywhere" 与本文自己的表冲突

正文 `paper/main.tex:1117`（§7.1(iv)）：

> A full handoff must satisfy both, which is why the joint arm is the weakest arm everywhere.

反例（全部取自论文自己的表）：

| 来源 | 反例 |
|---|---|
| Table 1 `4B→0.6B` | Joint **0.006** > K-only 0.000 = V-only 0.000 |
| Table 1 `8B→1.7B` | Joint **0.012** > K-only 0.000 = V-only 0.000 |
| Table 1 `8B→0.6B` | 三臂同为 0.000（"weakest" 无区分度） |
| Table 5 `8B→0.6B`（adapter 后） | V-only **0.357** < Joint 0.470 |

→ audit4 §9 的第一条成立，落 **T5**。这不是措辞洁癖：该句是 Discussion 四层结论的
第 (iv) 条，一旦被审稿人当作"作者把自己表格读错了"的证据，会伤到整节的信任度。

## 0.3 Related Work 的归属句仍是旧写法

`paper/main.tex:272`：

> we show that **the standard evaluation** can report transfer where none exists functionally

而 §3.3 与 §7.1(i) 已经按 audit3 §15 写成"缺陷在**我们自己早先的实现**里，没有审计
第三方实现"。同一篇论文里两种归属并存。守卫 `verify_audit3_edits.py` 的 T2 只在 4 个
具体站点做 `present`/`absent` 断言，Related Work 这一句没有被覆盖——**这是守卫设计
的盲区，不是偶发遗漏**。

→ 落 **T5**（改句）+ **T8**（把 T2 类断言从"站点级 present"升级为"全库 absent 扫描"）。

## 0.4 `paper/` 下仍有 4 份文档在陈述已撤回的 v1 结论

`paper/BRIEF.md`、`paper/project_context.md`、`paper/architecture.md`、`paper/v3_data_baseline.md`
仍以 v1 标题《Addressing Transfers, Content Does Not》陈述：K-only "6 对全部 EM ≥ 0.71"、
"addressing geometry is composable"、CCA ρ≈1 的归因、2050-token crossover。
`paper/architecture.md` 里甚至留着未完成的自我注释 `(wait, need to recalculate)`。
W23 只清理了 `paper/archive/*.tex`，没有处理这四份。

风险：论文正文声明"all previously published EM figures for this project are withdrawn"，
而仓库根目录下的 paper/ 读者第一眼看到的就是这些数字。→ 落 **T9**（归档到
`paper/archive/` 并在文件头加 `STALE v1 — claims withdrawn`；`README.md` 只保留当前主张）。

## 0.5 【新发现】Cost 段的 mapper 参数化与 §3.4 自相矛盾

`paper/main.tex:1070`（§6.6 Cost）写：`2050` tokens 对应 `58.8M-parameter fp32 mapper (224.2 MB)`。
溯源到 `reports/archive/phase3_rate_law_seed0.json`：

```python
# experiments/phase3_rate_law.py:50-55,173
kv_dim = KV_HEADS * HEAD_DIM          # 8*128 = 1024
params_per_mapper = kv_dim*kv_dim + kv_dim      # 单个 (1024,1024) 仿射 + bias
total_params = 2 * params_per_mapper * n_layers # K + V，各层共享形状
```

即 2 × 28 × (1024² + 1024) = **58,777,600**（224.2 MiB）→ 224.2 MiB / 112 KiB = 2050。
但 §3.4 定义的是 **per-(layer, head)** 的 128×128 仿射映射（`fit_mapper` →
`AffineMapper`，`D=128`；`phaseB_errorbudget.apply_oa` 的 `W[(l,h)]` 也是 128×128）。
按当前定义复算（本机复算，见 PART V E-2）：

```
per-head K+V params = 2 × 28 × 8 × (128² + 128) = 7,397,376  (28.2 MiB fp32)
KV bytes/token      = 28 × 2 × 8 × 128 × 2 B   = 112 KiB
crossover           = 28.2 MiB / 112 KiB       ≈ 258 tokens
```

顺带核对：消费适配器 `rank-8 on o_proj`（0.6B：o_proj 2048×1024）
= 28 × (1024×8 + 2048×8) = 688,128 ≈ **0.7M**，与正文一致 —— 也就是说，
**只有 cost 这一处的参数化是孤例**。

→ 处置落 **T4**：该段按 audit4 移入 Appendix 时，必须同步改成当前 mapper 的参数化
（≈260 tokens），或直接删除该段、只在 Future Work 保留一句"systems 测量待做"。
**不接受原样搬去附录**：把一处已知与 Method 矛盾的数字藏进附录，比放在正文更危险。

## 0.6 结构调整的机械风险（先处理，再动正文）

`paper/main.tex` 里有 8 处硬编码章节号：`Section~5.3`(349)、`6.1`(466)、`6.4`(644)、
`6.1`(1122)、`4.1`(1147)、`5.1`(1165)、`6.1`(1209)、`4.5`(1228)。
本轮要在 Method 插新小节、把 §6.2/§6.6 移进附录，**这些编号全部会失效**。→ 落 **T10**：
给每个 subsection 加 `\label{sec:...}`、把 8 处改成 `\ref{}`，并在守卫里断言正文中不再出现
`Section~` + 数字的字面量。

---

# PART I — 本机（零 GPU，可立即执行）

## T1 — 标题与元数据（audit4 §10 = audit3 T4 的收口）

候选与我的评估：

| 方案 | 标题 | 评估 |
|---|---|---|
| A（audit4 首选） | *Beyond Representation Alignment: Functional Compatibility for Cross-Model KV Transfer* | 概念最清楚；"Beyond" 略套话，但可接受 |
| **B（我推荐）** | *Cross-Model KV Transfer Needs Functional Compatibility: Representation Alignment Is Not a Transfer Metric* | 保留原句式的辨识度（"Needs …"），把新概念前置，并直接给出可传播的否定式结论 |
| C | *Diagnosing Cross-Model KV Transfer: From Representation Alignment to Functional Compatibility* | 最保守，保留 diagnosis 风格 |

**理由：** 论文最强的可引用结论（audit4 §2 归纳）是"representation similarity / teacher-forced
LL 都不是 functional transfer 的度量"，而标题里的 *Compatible Consumer* 只覆盖了
"需要一个消费者"这一层，漏掉了 K/V 的分解与 task-conditioned 边界。B 同时点出对象
（functional compatibility）与否定式结论（alignment 不是 transfer metric）。

**必须同改的位置（漏一处就会出现元数据不一致）：**

1. `paper/main.tex:2` 注释、`:30` `pdftitle`、`:37-38` `\title`；
2. `paper/arxiv_submission/main.tex`（同文件副本，编译后同步）；
3. `paper/arxiv_metadata.md:24-25`；
4. `README.md:4`；
5. Abstract 第一段的落点：标题若采用 B，Abstract 收尾必须出现一次 "functional
   compatibility"，否则标题与摘要的承诺不匹配。

**验收：** `rg -n "Compatible Consumer|Consumption-Side Fix" --glob '!*.pdf'` 只剩
`ITERATION_LOG.md` 与 `paper/audit/*`（历史记录允许保留）。

## T2 — Introduction 四条贡献重构（audit4 §11）

现贡献是 3 条，audit4 建议 4 条。采纳，但**第 1 条的写法要按 audit3 §15 的归属纪律
拆成两句**（通用原则 / 我们自己的实现），否则会把 T2 刚修好的归属问题再带回来。

```latex
\begin{enumerate}
\item \textbf{An evaluation principle.} Teacher-forced answer likelihood is not
  evidence of functional transfer: content-free caches (wrong document,
  moment-matched random, all-zero) reproduce $89$--$105\%$ of the K-only
  likelihood gain while answering no question. This claim is independent of any
  implementation. Separately, the greedy-EM path we used in earlier versions of
  this work contained two defects that inflated our own measurements; we fix
  them and validate the corrected evaluator against a full-prefill reference
  (normalized-EM agreement $0.929$--$0.982$, first-token top-1 agreement
  $0.929$--$0.982$, median $KL$ $0.001$--$0.005$ nats).
\item \textbf{A functional compatibility decomposition.} A cross-model handoff
  perturbs the receiver's output through two terms, a routing term
  $(\hat A-A_S)V_SW_O^S$ and a consumption term $\hat A(\hat V-V_S)W_O^S$; the
  four-arm protocol isolates them, and the key and value arms fail through
  different terms.
\item \textbf{Consumer-space alignment.} Raw representation error does not rank
  mapper variants by task performance (rank correlation $-0.10$); error measured
  after the student's attention output and after its $o$-projection does
  ($-1.00$ and $-0.80$, pooled over five variants and six runs). Fitting the
  value mapper where the student consumes value raises V-only EM from $0.113$ to
  $0.935$ on the equal-depth pair without touching a consumer weight, and from
  $0.000$ to $0.27$ on the flagship pair.
\item \textbf{Repairability and its boundary.} A rank-8 correction of the
  student's output projections ($\approx0.7$M parameters) restores K/V/joint EM
  to $0.94/0.94/0.83$ on the equal-depth pair, and a frozen-adapter causal test
  separates correct teacher state ($0.905\pm0.083$) from destroyed content
  (at most $0.179$) there; on the flagship pair the same margin ($+0.071$) stays
  inside our pre-registered noise band, and on a second domain neither repair
  reappears even when both are refit inside that domain. Compatibility is
  therefore \emph{task-conditioned}, not a property of the model pair alone.
\end{enumerate}
```

**验收：** 每条贡献至少有一个守卫断言 pin 住其最强数字；"task-conditioned" 的限定词
按 T6 写。

## T3 — 新增 §3.3 Functional Compatibility Decomposition（audit4 §3，最高 ROI）

**位置：** Method 内、`Injection protocol`（现 §3.2）之后、`Corrected evaluation`（现 §3.3）
之前——四臂协议刚定义完就给出它的代数含义。插入后原 §3.3–3.5 自动后移（见 T10）。

```latex
\subsection{Functional compatibility decomposition}
Let $O_S=A_SV_SW_O^S$ be the student's attention output on its own cache and
$\hat O=\hat A\hat VW_O^S$ its output on a handed-off cache, with $W_O^S$ frozen.
The perturbation splits exactly,
\begin{equation}
\hat O-O_S
= \underbrace{(\hat A-A_S)V_SW_O^S}_{\text{routing term}}
+ \underbrace{\hat A(\hat V-V_S)W_O^S}_{\text{consumption term}} .
\label{eq:decomp}
\end{equation}
The four arms switch terms on and off. K-only sets $\hat V=V_S$ and leaves the
routing term alone; V-only sets $\hat A=A_S$ and leaves the consumption term
alone; Joint carries both and their interaction. \eqref{eq:decomp} is an
identity about the perturbation, not a claim that task performance decomposes
additively: EM is a discrete function of the output, and
Section~\ref{sec:factorial} shows the likelihood-scale interaction changes sign
across pairs. We use it as the frame that names the two compatibility
requirements, and we test them by intervention rather than by arithmetic.
```

**三条使用纪律（必须写进正文或脚注，否则会被当作过度声称）：**

1. **不声称定理**：这是扰动恒等式 + 实验框架，$\hat O$ 的定义与四臂定义绑死；
2. **不主张可加性**：EM 不可加，§6.2 的 likelihood 交叉项也不稳定——恒等式只用来
   *命名* 两个项，不用来预测幅度；
3. **不与 routing 诊断混用**：§6.3 已声明 routing divergence 只是 key 侧的诊断量、
   没有做干预。加了 (eq. 1) 之后更要说清楚：分解给出的是"哪一项被打开"，
   不是"哪一项是瓶颈"——瓶颈由 A1（routing-aware K mapper）来回答。

**收益：** 贡献从"观察到两个 failure mode"升级为"提出并识别一个 decomposition"，
正是 audit4 §3 认为能把 7 抬到 8 的那一步；成本接近零。

## T4 — 减法与附录（audit4 §8 + PART 0.5）

| 内容 | 现在 | 改为 | 必须同步 |
|---|---|---|---|
| §6.2 `K and V likelihood effects and their interaction`（Table 3） | 正文 1 页 | Appendix A，正文留一句 | §4.5 exploratory 列表、`\label{sec:factorial}`（**必须随小节一起搬**，否则 T3 的 `\ref` 会悬空）、守卫 `verify_audit2_edits.py` E1 |
| §6.6 `Cost`（2050 tokens） | 正文 4 行 | Appendix B，**并修正参数化** | §0.5；若改为删除，则 Future Work 补一句 systems 测量 |

正文腾出的空间按 audit4 的建议分配给：T3 新小节、Table 4（误差预算）、K/V 示意图
（见 A1 的 Fig 4 面板）、causality 结果。

**附录开法：** `paper/main.tex` 目前没有 `\appendix`。在 Reproducibility statement 之后加
`\appendix` + `\section{...}`，两个新附录都要在开头注明"moved from the main text;
numbering follows the main text"。`verify_corrected_paper.py` 与两个 audit 守卫要重跑。

**Appendix B 的修正文本（替换 2050 那句）：**

```latex
For completeness: with the mapper actually used here---one affine map per
(layer, head), $L_S\cdot H\cdot(D^2+D)$ parameters per side and both sides
stored in fp32---the parameter count is $7.40$M ($28.2$ MiB) for a $28$-layer
student against $112$ KiB per token of fp16 KV, giving a byte crossover near
$260$ tokens. The number is a heuristic: it counts mapper parameters rather
than transfer time, ignores quantization and caching, and predates the current
split. An earlier version of this text quoted $2050$ tokens for a per-layer
$1024\times1024$ map, which is not the mapper we use.
```

## T5 — 两处措辞修正（audit4 §9 + PART 0.2/0.3）

**T5a（`main.tex:1117`）：**

```latex
\emph{(iv) Joint handoff.} A full handoff must satisfy both forms of
compatibility and is therefore consistently fragile; it does not reliably
combine the gains of the two components. The joint arm is the lowest of the
three teacher arms on four pairs, is at or above a single arm on the remaining
two, and after adaptation it falls below V-only on the flagship pair.
```

**T5b（`main.tex:272`）：**

```latex
Our contribution is diagnostic and constructive: our legacy task evaluator
could report apparent transfer where none exists functionally; more generally,
we show that teacher-forced likelihood alone is insufficient evidence of
functional transfer. We localize the failure to the consumer and recover
transfer by adapting the consumer rather than the state.
```

（注意 T5b 保留原来的后半句"localize the failure to the consumer..."，且必须先核对
`"the standard evaluation"` 不再出现在全文任何位置。）

## T6 — "task-conditioned compatibility" 的命名纪律（audit4 §5）

采纳命名，但按 data 的实际强度加限定：

* Abstract / Intro 用 **task-conditioned functional compatibility**（形容词 + 概念），
  不用 "Transferability(T,S,D)" 这种函数式写法（那会暗示已做跨分布扫描，A4 未做）；
* §5.3 与 Limitations 必须并列写出三条限定：SQuAD 只有 15 份 calibration 文档、
  held-out Self 仅 0.20–0.40、且这是**一个**额外分布；
* 命名句建议：`Compatibility is therefore conditioned on the task distribution:
  the same recipe that recovers 0.94 V-only EM on the synthetic domain answers
  0.000 of held-out SQuAD questions when refit inside SQuAD. We name this
  task-conditioned compatibility and note that our evidence for it comes from a
  single second domain.`
* **顺序依赖：** 若 A3（校准量对照）结果显示 15-synthetic 与 15-SQuAD 的失败同源，
  说明是**校准量**而非分布，则本命名必须降级为 "distillation-volume-limited" 假设。
  T6 因此标为 **A3 条件式**。

## T7 — 相关系数披露（PART 0.1；audit4 §7 的前置修复）

1. Table 4 表注增加一句：`Rank correlations pool six runs (two pairs, three seeds)
   across five mapper variants: pooled $\rho$ and per-run range are
   $-0.10$ [$[-0.60,-0.10]$] for $e_{\text{raw}}$, $-1.00$ [$[-1.00,-0.80]$] for
   $e_{\text{attn}}$, $-0.80$ [$[-1.00,-0.80]$] for $e_{W_O}$.`
2. §6.3 正文补一句说明 pooled 与 per-run 的关系，并明确 `n=5` 不足以给 p 值。
3. A2 落地后，把这段替换为 sweep 结果（散点 + per-configuration ρ），T7 作为过渡版本，
  保证"即使 A2 没跑，也没有披露漏洞"。

> **W31 更新**：第 1 条里的逐 run 区间 `$[-0.60,-0.10]$` 是序数秩口径的旧值。W31 把
> 全仓库换成平均秩（与 `scipy.stats.spearmanr` 一致）后为 `$[-0.60,+0.05]$`，pooled
> 三值不变。本文件保留当时的预登记数值不改写；现行口径与全部 old→new 见
> `METRIC_CORRECTION.md` §14。

## T8 — 守卫 `paper/audit/verify_audit4_edits.py`（先写断言，后改正文）

至少 14 条：

| tag | 断言 |
|---|---|
| T1 | `present` 新标题；`absent` 旧标题字符串（在 `main.tex` / `arxiv_submission/main.tex` / README / arxiv_metadata 四处都查） |
| T2 | `present` 四条贡献的标题词（`An evaluation principle` / `A functional compatibility decomposition` / `Consumer-space alignment` / `task-conditioned`） |
| T3 | `present` `\label{eq:decomp}`、`routing term`、`consumption term`、三条纪律中的两条关键句 |
| T4 | `absent` `2050`；`present` `\appendix`；`present` `7.40`M 与 `260` tokens（Appendix B 新数字）；正文不再出现 `\subsection{Cost}`；`\label{tab:factorial}` 仍存在（随附录移动） |
| T5 | `absent` `weakest arm everywhere`；`absent` `the standard evaluation can report transfer`；`present` 新句 |
| T6 | `present` `task-conditioned`；`present` `fifteen SQuAD calibration documents`（限定词不能丢） |
| T7 | `present` `per-run range`，并与 `reports/phaseB_errorbudget_*.json` 交叉核对 6 个 run 的 ρ |
| T10 | 正则断言 `Section~[0-9]` 在 `main.tex` 中出现 0 次 |
| A1/A2/A3 | 数据回来后再加数字断言（先留 `TODO(audit4)` 占位，禁止留空通过） |

同时修 `verify_audit2_edits.py` / `verify_audit3_edits.py` 里被本轮取代的断言：
T5 会打破 T1b 的 `present("T1b", …, "\\emph{(iv) Joint handoff.}")`（该句保留，
但同一断言组里若 pin 了旧措辞需更新）；T4 会打破任何指向正文 Table 3 位置的断言
（E1 的 `$-1.22{\pm}0.05$` 同时在 Table 1，预计仍通过，需实跑确认）。

## T9 — 仓库卫生（PART 0.4）

1. `paper/{BRIEF.md,project_context.md,architecture.md,v3_data_baseline.md}` → `git mv` 到
   `paper/archive/`，并在每个文件首行加 `> **STALE (v1).** Claims below are withdrawn;
   see paper/main.tex and paper/audit/METRIC_CORRECTION.md.`；
2. 若 `AGENTS.md` / `paper/` 内任何脚本引用这四个文件，同步改路径（先 `rg` 确认）；
3. `ITERATION_LOG.md` 追加 W28 条目：本轮完成项、A1–A3 状态、本文件的优先级表。

## T10 — 交叉引用 label 化（PART 0.6）

给 §5.x/§6.x/§7.x 的每个 subsection 加 `\label{sec:...}`，把 8 处 `Section~N` 改为
`Section~\ref{}`；Objective 是让后续任何结构调整都不再产生"编号漂移"这类低级错误。

---

# PART II — GPU（按 ROI 排序，全部先在本文件写死门禁）

**成本基准（W26 实测，RTX 4090，`reports/audit3_gpu_queue.log`）：**
A2-causal20 ≈ 13 min/run；A3-squadwithin ≈ 4.4 min/run；A4-errorbudget ≈ 6.3 min/run
（5 个 variant × 56 样本，含误差预算 + EM 评分）。下述估算以此为标尺。

## A1 — Routing-aware K mapper（audit4 §4，最高 ROI）

**目的：** 把 key 侧从"诊断"升级为"干预"，形成
`K: routing-space alignment` / `V: consumption-space alignment` 的对称结构。

**脚本：** 新增 `experiments/phaseB_routingkey.py`（复用 `capture_all` / `get_attn_map` /
`score_arm` / `bootstrap_ci95_clustered`）。

**默认方案（与 V 侧 outaware 同构，闭式、可复用现有代码）：**
对每个 student layer $\ell$、KV head $h$，在**与 `AffineMapper` 相同的 de-RoPE 约定**
下拟合 $Z$（$\hat K = K_T Z$），但把岭回归的行按学生的注意力质量加权：

$$\min_Z \sum_j w_j\,\lVert \tilde K_{T,j}Z-\tilde K_{S,j}\rVert^2,\qquad
w_j=\textstyle\sum_q\sum_i A_S[q,i,j]\ \ (\text{对 query head 与 position 求和}),$$

即"学生真正去看的文档位置，映射要准"。实现上只需把
`fit_output_aware_mapper` 的 `X=A@V_T, Y=A@V_S` 换成
`X=√w·K_T, Y=√w·K_S`（同样的 per-(layer, head) ridge、同样的 `lam=1e-3`）。
**自洽性门禁（必须先过）：** 令 $w_j\equiv1$ 时，该拟合必须复现 affine K 的 routing TV /
top-1 数值（同一 seed、同一文档），否则说明 de-RoPE 或位置约定接错了。

**备用方案（仅当默认方案降不动 routing TV 时启用）：** 用 torch 直接最小化
$\mathrm{KL}(A_S\|\hat A)$（对 $Z$ 做梯度下降，每个 (layer, head) 单独一轮，
几百步、秒级），结果必须标注为 gradient-fitted。不要一上来就写梯度版：
噪声、步数、初始化都会变成审稿人追问点，而默认方案没有这些自由度。

**对照：** (i) affine K（现基线）；(ii) routing-aware K；(iii) routing-aware K，
targets 取自**另一篇文档**（shuffled-target 控制）；(iv) identity/raw layer selection。
**指标：** routing TV、top-1 agreement、attention cosine、K-only EM（+ document-clustered
CI95）、以及 coupling 值（EM 与 routing TV 的逐 run 关系）。

**预登记门禁（三支，全部可发表，禁止事后挑支）：**

| 结果 | 判定 | 论文写法 |
|---|---|---|
| routing TV 相对 affine 下降 ≥30%，且 K-only EM 的 clustered CI95 排除 0 | **intervention 成立** | "routing-space alignment partially repairs the key arm"；Abstract/贡献 2 升级 |
| routing TV 明显下降，但 K-only EM 仍在 floor | **更强结论** | "closing the routing gap alone is insufficient — the consumption term must also be aligned"；这支持而非削弱本文主线 |
| routing TV 未下降 | **mapper 家族限制** | 如实写成 negative intervention + Limitations 一条（不得改门禁） |

**范围与成本：** 2 pairs（1.7B→0.6B、8B→0.6B）× 3 seeds；预估 ≈7 min/run
（capture + 3 次 fit + 4 臂 × 56 样本评分）→ **45–60 min**，加 seed0 smoke 约 1.5 h。

**论文落点：** §6.3 新段落 + Fig. 4 新面板（routing TV vs K-only EM）+ 贡献 2 + Abstract 一句。

## A2 — Mapper objective sweep（audit4 §7）

**目的：** 把"消费空间误差预测任务表现"从 5 个点做成 20–30 个配置的经验规律，
同时消除 §0.1 的披露缺口。

**设计：**

1. $\alpha \in \{0,0.1,\dots,1\}$（11 点），目标
   $\min_M (1-\alpha)\|MV_T-V_S\|^2+\alpha\|A_SMV_T-A_SV_S\|^2$。
   **实现提示：** 把行堆叠成
   $X=[\sqrt{1-\alpha}\,V_T;\ \sqrt{\alpha}\,A_SV_T]$、
   $Y=[\sqrt{1-\alpha}\,V_S;\ \sqrt{\alpha}\,A_SV_S]$，直接复用
   `fit_output_aware_mapper` 的 ridge 求解（$\alpha=0$ 退化为 affine，$\alpha=1$
   退化为 attention-output-aware，可作为实现自洽性检查）；
2. ridge $\lambda \in \{10^{-4},10^{-3},10^{-2}\}$；
3. mapper rank（PCA 截断）$\in \{8,32,128\}$（可选，若时间允许）；
4. 每个配置同一 run 内同时算 $e_{\text{raw}},e_{\text{attn}},e_{W_O}$ 与 V-only EM
   （复用 `phaseB_errorbudget.py` 的 `error_budget`）。

**统计：** pooled 散点 + 每配置的 $\rho$ 范围 + 配置级 bootstrap（对 20–30 个配置重采样，
给出 $\rho$ 的区间）。**必须**报告 $\rho_{\text{raw}}$ 的上界，否则"raw 不预测"仍是弱断言。

**门禁：**

| 结果 | 判定 |
|---|---|
| $\rho_{\text{attn}},\rho_{W_O}\le-0.7$ 且 $\rho_{\text{raw}}\ge-0.3$（配置级 bootstrap 区间不重叠） | 保留 "the distance that predicts transfer is consumption-space distance" 的强表述 |
| 方向一致但区间重叠 | 保留方向，改为 "consistent with"，不给数字式的普遍性断言 |
| 方向翻转（某个 $\alpha$ 段内 raw 预测更好） | 如实报告该段，并把结论限定为"在 $\alpha$ 的两端" |

**成本：** 约 20 配置 × 2 pairs × 3 seeds ≈ 120 次 variant 评估 ≈ **2–3 h**。

## A3 — 校准量 / 分布对照（audit4 未提，但我建议做；T6 的前置）

**目的：** 把"task-conditioned"与"只有 15 份 SQuAD calibration 文档"这两个解释分开。

**三格设计（同一 held-out Self，同一评分路径）：**

| 条件 | calibration 组成 | 预测 |
|---|---|---|
| C1 | 15 份 SQuAD 文档（= 现 A3） | 迁移臂全 0（已知） |
| C2 | 15 份**合成**文档（体量对照） | 若同样全 0 → 体量不是主因 |
| C3 | 15 份 SQuAD + 70 份合成 | 若仍全 0 → 分布是主因，T6 命名成立；若恢复 → 改为"calibration 覆盖不足" |

**成本：** 复用 `experiments/phaseB_squad_within.py` 的骨架加一个 `--calib-mix` 参数，
≈30 min GPU。**收益：** 这一步直接决定 Abstract 能不能写 "task-conditioned"。

## A4 — Compatibility frontier（audit4 §6）→ 本轮降级为 Future Work 的预登记设计

采纳其**设计**，但不在本轮执行。理由：论文已有 4 条支柱（T2），第 5 条需要 D0–D5 的
shift ladder（每格都要重训 mapper + adapter），按 W26 实测的同类 run（4–13 min/run，
而 ladder 的每格都是"一次完整重训 + 全臂评分"）至少 3–5 h GPU 加一轮新的写作；
而 audit4 自己也说"这很可能成为下一篇论文"。折中方案：

* 正文 Future Work 写清 ladder 设计（D0 paraphrase → D1 question template →
  D2 relation vocabulary → D3 answer format → D4 document schema）、要测的量
  （$V$-mapper EM、adapter EM、$e_{\text{attn}}$、$e_{W_O}$）与预登记判据；
* 如果 T 系列与 A1/A2 提前完成且 GPU 空闲，再补一个 **D0–D2 pilot**（只做 1.7B→0.6B，
  3 seeds），作为 §6.3 的补充面板；pilot 不通过就不写进正文（避免"空图"）。

---

# PART III — 条件式正文改写（数据回来后按行落）

| 结果 | 改哪句 | 措辞方向 |
|---|---|---|
| A1 routing 干预成立 | 贡献 2、§6.3 末段、Abstract | "the key side is repairable in routing space, at least partially, and the value side in consumption space" |
| A1 routing 改善但 EM 不动 | §6.3、Discussion (iii) | "routing mismatch is necessary but not sufficient; the two compatibility terms are both required" |
| A1 无改善 | Limitations + Future Work | mapper family 限制，不做机制推断 |
| A2 支持消费空间 | §6.3 + Table 4 + Abstract | 用 sweep 替换五点结论，τ 报告 pooled 与 bootstrap 区间 |
| A2 不支持 | §6.3 + Limitations | 结论限定为"我们测试的五个 mapper 变体上" |
| A3 分布是主因 | Abstract/§5.3/T6 | 启用 "task-conditioned compatibility" 命名 |
| A3 体量是主因 | Abstract/§5.3/T6 | 改名为 "calibration-coverage-limited"，并撤下 task-conditioned 措辞 |

---

# PART IV — 完成定义（DoD）与停止规则

**DoD：**

1. `python paper/audit/verify_audit4_edits.py`、`verify_audit3_edits.py`、
   `verify_audit2_edits.py`、`scripts/verify_corrected_paper.py`、`tests/test_stats_utils.py`
   全部 exit 0；`python -m py_compile experiments/*.py scripts/*.py` 通过；
2. `main.tex` 与 `arxiv_submission/` 同步编译（tectonic，0 error / 0 undefined）；
3. `METRIC_CORRECTION.md §11` 登记本轮所有新数字（含 pooled/per-run ρ、A1/A2/A3 门禁判定）；
4. T1–T10 全部落地或明确标注 `deferred + reason`；
5. `ITERATION_LOG.md` 追加 W28 条目；四个 stale v1 文档已归档。

**停止规则（不允许放宽）：**

* A1 的 K-only EM 若在 floor 上，**不得**把"routing TV 下降"写成"K 可迁移"；
* A2 的 bootstrap 区间若重叠，**不得**保留数字式的普遍性表述；
* A3 若 C2/C3 显示是体量问题，**不得**继续使用 task-conditioned 命名；
* 任何新数字未落 `reports/*.json` 前，禁止进正文（沿用前三轮规则）。

---

# PART V — 取证表（本方案每条事实的来源）

| 编号 | 事实 | 仓库证据 | 复现命令 |
|---|---|---|---|
| E-1 | pooled ρ 与逐 run ρ | `reports/phaseB_errorbudget_{1.7B,8B}_0.6B_seed{0,1,2}.json` | 见下方 E-1 复算 |
| E-2 | per-head mapper 7.40M / 258 tokens vs 58.78M / 2050 | `experiments/phase3_rate_law.py:50-55,173`；`reports/archive/phase3_rate_law_seed0.json` | 见 E-2 命令 |
| E-3 | joint 不是"everywhere"最弱 | Table 1 / Table 5 自身数字 | `rg -n "0\.006|0\.012" paper/main.tex` |
| E-4 | Related Work 归属句位置 | `paper/main.tex:272` | `rg -n "standard evaluation can report transfer" paper/main.tex` |
| E-5 | 4 份 v1 文档仍在陈述撤回结论 | `paper/BRIEF.md`、`project_context.md`、`architecture.md`、`v3_data_baseline.md` | `rg -n "EM ≥ 0.71|Addressing Transfers" paper/*.md` |
| E-6 | GPU 每 run 实测用时 | `reports/audit3_gpu_queue.log` | `rg "START|DONE" reports/audit3_gpu_queue.log` |

E-1 复算（本机，只读）：

```bash
python -c "
import json,glob,numpy as np
rows={}
for f in sorted(glob.glob('reports/phaseB_errorbudget_*.json')):
    d=json.load(open(f)); rows[(d['pair'],d['seed'])]=d
names=['raw','affine','outaware','outaware-shuffled','woaware']
def sp(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx,ry)[0,1])
p={n:{k:np.mean([rows[r]['summary'][n][k] for r in rows]) for k in
      ['e_raw_mean','e_attn_mean','e_wo_mean','EM']} for n in names}
em=np.array([p[n]['EM'] for n in names])
for k in ['e_raw_mean','e_attn_mean','e_wo_mean']:
    print(k, 'pooled', round(sp(np.array([p[n][k] for n in names]),em),3),
          'per-run', [round(sp(np.array([rows[r]['summary'][n][k] for n in names]),
                               np.array([rows[r]['summary'][n]['EM'] for n in names])),2)
                      for r in rows])
"
```

E-2 复算：

```bash
python -c "
print('per-head K+V params:', 2*28*8*(128*128+128))
print('fp32 MiB:', 2*28*8*(128*128+128)*4/1024/1024)
print('KV KiB/token:', 28*2*8*128*2/1024)
print('crossover tokens:', (2*28*8*(128*128+128)*4/1024/1024)/(28*2*8*128*2/1024/1024))
print('legacy per-layer:', 2*28*(1024*1024+1024), 2*28*(1024*1024+1024)*4/1024/1024)
"
```

---

# PART VI — 本轮不做的事（明确记录）

1. **不加公开 benchmark**（HotpotQA / TriviaQA / Llama / Mistral 大表）：与 audit4 §12 的排序一致
   （它把这列为最低优先级），且会稀释 framing；
2. **不做 cross-family**：保留在 Limitations；它是"边界测试"而非"概念创新"；
3. **不把 2050 crossover 修好后留在正文**：它不承担任何主张（audit4 明确说
   "does not bear on the state–consumer findings"）；
4. **不再改动任何已闭合的 audit1–audit3 结论**：本轮只做 framing 升级 + 三处事实修正；
5. **不引入新的评测指标**：EM 仍是主指标，新增量（routing TV、e_*）只作为机制量。

---

# PART VII — 文件清单与执行顺序

**执行顺序（依赖关系）：**

```
T10 (label 化) ─┬─> T9 (仓库卫生)
                ├─> T8 (先写守卫断言)
                └─> T1/T2/T3/T5/T6/T7 (正文改写) ─> T4 (附录搬家) ─> 三守卫 + 编译
                                                         │
                                  A1 ─┐                  ▼
                                  A2 ─┼─> PART III 条件式改写 ─> 守卫加数字断言 ─> W28 登记
                                  A3 ─┘
```

**本轮新增/修改的文件：**

| 文件 | 动作 |
|---|---|
| `paper/audit/REVISION_PLAN4.md` | 新增（本文件） |
| `paper/audit/verify_audit4_edits.py` | 新增（T8） |
| `paper/main.tex`、`paper/arxiv_submission/main.tex` | 改写（T1–T7、T10） |
| `paper/arxiv_metadata.md`、`README.md` | 标题同步（T1） |
| `paper/archive/{BRIEF,project_context,architecture,v3_data_baseline}.md` | 归档 + stale 头（T9） |
| `experiments/phaseB_routingkey.py` | 新增（A1） |
| `experiments/phaseB_mappersweep.py` | 新增（A2） |
| `experiments/phaseB_squad_within.py` | 加 `--calib-mix`（A3） |
| `scripts/run_audit4_gpu_queue.sh` | 新增（A1→A2→A3，可续跑，沿用 W26 队列模式） |
| `paper/audit/METRIC_CORRECTION.md` | 追加 §11（本轮数字与门禁判定） |
| `ITERATION_LOG.md` | 追加 W28 条目 |
| `scripts/verify_corrected_paper.py` | 若附录搬家影响断言则同步 |
