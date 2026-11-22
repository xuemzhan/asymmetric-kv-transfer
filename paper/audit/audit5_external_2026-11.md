# audit5：2026-11 外部文献复审（novelty / positioning / 证据强度）

**触发**：W33（2026-09-21）冻结后两个月未动、未投稿。本审计重新做一次全网检索，
判断论文的四条主张在新文献面前是否仍然成立、新颖性窗口是否还在、负结果是否站得住。

**审计对象**：`paper/main.tex`（1675 行，27 页，11 表 8 图；与
`paper/arxiv_submission/main.tex` 逐字节一致）。

**审计前提**：本文所有实验数字未做任何改动，本文件只做定位与证据强度判断。
本审计不重跑实验。

---

## 0. 检索方法与可复现性

- 数据源：arXiv OAI API（`export.arxiv.org/api/query`），2026-11-22 执行。
- 时间窗：`submittedDate:[202606010000 TO 202611300000]`（覆盖论文冻结前后）。
- 查询式（8 组宽扫 + 4 组定向）：

```
all:"KV cache transfer"
all:"cross-model" AND all:"KV cache"
all:"KV cache" AND all:"reuse" AND all:"across models"
all:"KV cache translation"
all:"latent communication" AND all:"language model"
all:"cache reuse" AND all:"prefill"
all:"KV cache" AND all:"alignment" AND all:"models"
all:"cross-model" AND all:"activation"

all:"functional compatibility"
all:"KV cache" AND all:"nonlinear"
all:"task-conditioned" AND all:"compatibility"
ti:"Platonic Representation Hypothesis"
ti:"relative representations"
all:"model stitching" AND all:"representations"
all:"Qwen3" AND all:"technical report"
```

- 引用核实：用 `id_list` 对四条核心引用（`heo2026crossmodel`、`fu2026c2c`、
  `lee2026translators`、`dery2026latentalign`）逐条回查。**四条全部真实存在**，
  作者、年份、arXiv 号均可对上；唯一偏差见 §3.2-Mo5。

### 0.1 引用核实明细

| 本文 bibitem | arXiv | 实际题名 | 结论 |
|---|---|---|---|
| `heo2026crossmodel` | 2608.03893v1（2026-08-04） | Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse | 一致 |
| `fu2026c2c` | 2510.03215v2（v1 2025-10-03） | Cache-to-Cache: Direct Semantic Communication Between Large Language Models | 一致 |
| `dery2026latentalign` | 2601.06123v1（2026-01-04） | Latent Space Communication via K-V Cache Alignment | 一致 |
| `lee2026translators` | 2607.28979v2（v1 2026-07-31，v2 2026-10-03） | KV Cache Translation across Heterogeneous Large Language Models | **题名偏差**（见 Mo5） |

---

## 1. 论文自述的四条贡献（用于逐条比对）

- **C1 评估原则**：teacher-forced likelihood 不是 transfer 的证据；content-free
  cache（错文档 / moment-matched 随机 / 全零）复现 89–105% 的 K-only LL 增益而
  EM≈0；并修掉自家 greedy-EM 路径的两个缺陷，用 full-prefill 参考做验证。
- **C2 功能兼容性分解**：routing 项 $(\hat A-A_S)V_SW_O^S$ 与 consumption 项
  $\hat A(\hat V-V_S)W_O^S$；四臂开关两项；key 侧可诊断不可修，value 侧是消费空间失配。
- **C3 消费空间对齐**：对齐的**目标**比对齐的**精度**更重要；raw representation
  error 在每次 run 内与 EM 正相关，attention-output / o-projection 空间误差强负相关
  （198 配置扫描）。
- **C4 可修复性与其边界**：rank-8 输出投影修正（≈0.7M）+ 冻结 student 与 mapper，
  恢复 0.94/0.94/0.83（1.7B→0.6B）；旗舰对只到 0.79/0.36/0.47 且因果检验未过门；
  换域（SQuAD）重训亦不复活 → 命名 task-conditioned functional compatibility。

---

## 2. 检索结果：必须处理的新文献

### 2.A 与 C3/C4 核心论断重合，且规模 / 配方更强

| arXiv | 日期 | 题名 | 与本文的关系 |
|---|---|---|---|
| 2610.15509 | 2026-10-11 | CacheLink: Functional KV Transport for Zero-Prefill Cross-Model Context Handoff | **最强优先权威胁**。原文结论「reconstruction fidelity is a poor proxy for receiver utility」与本文 C3 同一论断；但证据是 4 对独立训练模型 × 7 个评估面板，外加 7.27× 延迟收益 |
| 2609.32610 | 2026-09-26 | KV-Lingo: Learning KV-Cache Translators with Distillation | 用 distillation 直接最小化 receiver 预测分布 divergence 来训练 per-layer linear maps；small→large 与 large→small 都保持强下游性能。**这是本文 C3 那条「在 receiver 消费处拟合」的更强实现** |
| 2611.11761 | 2026-11-09 | BridgeKV: Large-Model Prefill and Small-Model Decoding via KV Cache Translation | 学习式 KV-stitching + **K/V 各自的非线性 map** + **第二阶段 low-rank adapter**；6 对 × 5 数据集，闭合最多 80% 的质量差。**与本文「mapper + rank-8 adapter」的配方几乎同构** |
| 2609.00891 | 2026-09-01 | CacheBridge: Efficient Cross-Model KV Transfer for Multi-Model LLM Serving | 三族五对，保留 re-prefill 质量的 97.94% / 102.94%。**日期早于本文冻结（W33），属漏引** |
| 2610.42116 | 2026-10-30 | ACCT: Asymmetric Cross-LLM Context Transfer | 同族跨规模，KV 压缩 75% 时「matches or exceeds the target model's original inference」 |
| 2608.30963 | 2026-08-31 | A Universal Context-Reuse Layer for Cross-Model KV Sharing | Qwen2.5-7B→1.5B 上 LongBench2 27.59%→34.48%；跨族 handoff 接近原生。**日期早于本文冻结，属漏引** |
| 2610.11358 | 2026-10-08 | RaReCache: ... Rank disagreement-based Selective Recomputation | 承认「transfer accuracy degrades as the model-size gap widens」，用选择性重算补位。与本文「size gap 越大越差」同向 |

### 2.B 与 C1 独立同向 —— 反过来是本文的背书

| arXiv | 日期 | 题名 | 与本文的关系 |
|---|---|---|---|
| 2611.06976 | 2026-11-04 | On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer | 受控实验显示增益可由 **uninformative sender signal / self-communication / receiver-only training** 复现；提出 PACT-Check 量化真实通信贡献。与本文 content-free 控制**同构** |
| 2607.26773 | 2026-07-29 | Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM | 在 sender→receiver 边界做受控 message replacement 的因果审计。**方法论同源且早于本文，本文未引** |
| 2611.14051 | 2026-11-10 | Can KV Cache Be Reused Across Model Updates? A DeepSeek-V4-Flash Case Study | 提出 Functional Cache Divergence；核心结论「caches can retain useful task information without faithfully reproducing the updated model's behavior」——与本文 LL/EM 解离同构 |
| 2611.06397 | 2026-11-04 | Where Should a KV-Cache Repair Budget Go? | 提出 **contextual compatibility**；「recomputing the target against the reused cache beats inserting its full-prefill KV ... although the recomputed KV stays far from the full-prefill KV」——独立支持「KV 空间距离 ≠ 功能可用性」 |
| 2610.15721 | 2026-10-11 | To Cache or Not to Cache: Safety Incompatibility in Cross-Model KV Cache Reuse | 「value reuse causing substantially larger deviations than key reuse across all evaluated pairs」——与本文 V 侧是瓶颈的判断同向 |

### 2.C 回答了本文列为 Future Work / 未做的事

| arXiv | 日期 | 题名 | 对应本文哪条 |
|---|---|---|---|
| 2609.32259 | 2026-09-26 | Prefill-Free Cross-Family KV Cache Transfer (HeteroFold) | 本文 Future work (ii)「cross-family evaluation」——已被做完（6 个方向，4 个长上下文基准） |
| 2610.15095 | 2026-10-10 | Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families | 同上，17 基准 / 12 模型 / 6 族 |
| 2610.09058 | 2026-10-06 | Learning Cross-Model Activation Alignments with Explicit Many-to-Many Layer Maps (MATCHA) | **直接挑战本文的层映射设定**：42 对上证明层对应是 many-to-many 的「band」，而非 proportional / identity |
| 2405.07987 | 2024-05 | The Platonic Representation Hypothesis | 本文口号「representation alignment is the wrong target object」字面否定这一整条线，**全文未引** |
| 2609.27252 | 2026-09-23 | What Converges in the Platonic Representation Hypothesis? Structure over Geometry | 正好支持本文「几何坐标不重要、结构才重要」——**最强的理论盟友，未引** |
| 2209.15430 | 2022-09 | Relative representations enable zero-shot latent space communication | 跨模型潜空间通信的经典线，未引 |
| 2505.09388 | 2025-05 | Qwen3 Technical Report | 四个模型全部来自 Qwen3，**正文未引** |
| 2608.22462 | 2026-08 | Functional compatibility as a determinant of persistent neural learning | 「functional compatibility」一词已被占用（神经科学语境） |

---

## 3. 总体判断

**一句话**：论文的**诊断**被同期文献独立复现（有利），论文的**口号**已被他人抢先发表（不利），
论文的**负结果**与多篇正结果直接冲突且冲突点落在本文自己承认的软肋上（致命）。

### 3.1 MAJOR（投稿前必须处理）

**M1　新颖性窗口已关闭一半，贡献声明必须重写。**
C3（消费空间对齐优于表示空间对齐）的独立发现权被 CacheLink（10-11）与 KV-Lingo
（09-26）占据，两者证据规模都更大；C4（mapper + 小 adapter 修复）被 BridgeKV（11-09）
以非线性 map + 两阶段 adapter 的更完整配方覆盖。同时 C1 被 PACT（11-04）与因果审计
（07-29）独立复现 —— 这是本文目前**最值钱的资产**，因为它不再是"我们的实现有 bug"，
而是"这条文献线的评估范式有系统性缺陷"。
**处置**：贡献顺序改为 C1（评估范式）→ C2（分解）→ C3（定位，降级为"我们给出机械解释"）
→ C4（边界条件）。

**M2　负结果与多篇正结果冲突，不正面处理会被判 artifact。**
本文：8B→0.6B 三条 teacher 臂在三个 seed 上全为 0.000 EM；无任何 pair 的 3-seed 均值
超过 0.149（K）/ 0.065（Joint）。
同期：CacheBridge 保留 ~98–103% re-prefill 质量；ACCT「matches or exceeds」；
Universal Context-Reuse Layer 在 LongBench2 上 +6.89pp；KV-Lingo「strong downstream
performance」。
三条可检验的差异解释（本文目前只在 Limitations 里含糊承认 (a)，(b)(c) 完全没提）：

- **(a) mapper 家族线性**。本文最强 mapper 仍是线性（affine / $A_S$ 空间 ridge /
  $W_O$-aware 线性）。BridgeKV 用非线性 map，KV-Lingo 用 distillation。
  **在线性之外已被证明可行的情况下，「线性 mapper 不行」不支持「表示对齐不够」**，
  只支持「线性不够」。这是 C3 最严重的逻辑缺口。
- **(b) 任务与指标**。合成 KG 检索 + 严格 EM，$n=56$、仅 8 个文档；
  竞争工作是 LongBench2 / $\tau$-bench / RAG / GSM8K 等。EM 对错实体零容忍，
  系统性低估"部分可用"的状态。
- **(c) K 侧从未用行为目标拟合**。本文只对 V 做消费空间拟合，K 始终是
  「按注意力质量加权的 ridge 回归」——**仍是表示空间回归**。CacheLink 是 coupled
  K/V correction，BridgeKV 给 K 单独的非线性 map。本文把 K 侧失败读作
  「necessary but not sufficient」（§6.2、Discussion (iii)），更保守的读法是
  **「K 侧没试过本文自己主张的那种目标」**。
  → 三者叠加足以让审稿人认为负结果是方法学产物。必须在 Discussion 显式承认并
  给出可证伪的边界表述（例如只主张「linear mapper + 严格 EM + K 未修」这一条件组合）。

**M3　「表示对齐不够」缺少与 Platonic Representation Hypothesis 的对话。**
本文标题与口号否定的是 PRH / relative representations / model stitching 这一整条线，
但 Related Work 只引了 Bansal 的 stitching，未引 Huh et al. 2024（2405.07987）、
Moschella et al. 2022（2209.15430），也未引 2609.27252（结构 over 几何）。
其中 2609.27252 是**支持本文立场**的：它把 PRH 的收敛对象定位为结构而非几何，
与本文「raw representation error 不是正确目标」完全一致。漏引等于放弃了最强的理论盟友。

**M4　层映射实验设计与新基线脱节。**
本文只测 proportional / $\pm3$ offset / random permutation / calibration-learned
top-1 source layer。MATCHA（2610.09058）在 42 对上证明层映射是 many-to-many 的 band，
并且这正是它提升 retrieval 的关键。「permutation 使 V-only 从 0.911 掉到 0.107」这个
结论方向是站得住的，但文中隐含的「proportional 最好」在 MATCHA 的基线下未必成立。
§6.1 的"repaired-regime scrambling"只有 seed 0，且只覆盖置换类扰动。

### 3.2 MODERATE

- **Mo1 术语冲突**：「functional compatibility」已被 2608.22462 使用；
  「contextual compatibility」已被 2611.06397 使用；「reconstruction fidelity」
  是 CacheLink 的用词。建议要么加一句显式界定，要么改名为更具体的
  *consumer-space compatibility*。
- **Mo2 单一严格指标 + 单一合成域**。同期工作全部用任务级指标。本文的
  answer log-likelihood 已经在手，可零成本补出 answer-margin / token-F1 型统计，
  回应「EM 太严」这一必然质疑（2611.14051 正是用 answer-margin 解释"检索决策在
  内部状态大幅变化下仍能存活"）。
- **Mo3 SQuAD 域的判别力不足**。Self 天花板只有 0.20–0.40，$n=15$/$n=30$，
  本身贴近地板。在这种情况下「transfer arm 全 0.000」几乎是必然的，
  用它支撑主命名 *task-conditioned* 证据偏弱。需要 power 论证，或换一个
  ceiling 更高的第二域。
- **Mo4 旗舰因果检验仍 unresolved**（margin $+0.095$ < 预注册门 $0.10$），
  且作者自报同 seed 重训得到 0.679 / 0.429，训练方差吞掉效应。
  C4 在旗舰对上**实际上未被证实**，文中已诚实标注，但摘要的措辞仍偏正面。
- **Mo5 `lee2026translators` 题名偏差**：本文写作
  "Mixture-of-Translators: Translating KV Caches across Heterogeneous Large Language
  Models"，arXiv 实际题名是 "KV Cache Translation across Heterogeneous Large Language
  Models"（MoT 是方法名）。另该文 v2 于 2026-10-03 更新，本文引的是 v1 口径。
  本文对该文只作定性引用，风险低，但题名需改。
- **Mo6 未引 Qwen3 技术报告（2505.09388）**，而四个模型全部来自 Qwen3。

### 3.3 MINOR

- `paper/arxiv_metadata.md` 的 pre-submission checklist 仍有三项未勾：license、
  repo URL（或明确不公开）、primary category 的 endorsement。
- 一旦新增引用，页数 / 表数会变，Comments 字段的 "27 pages, 11 tables, 8 figures"
  需重算。
- 中文版 `paper/main_zh.tex` 的 `verify_zh_parity.py` 断言 `\cite` 键与 `\bibitem`
  顺序一一对应；英文版新增引用后必须同步中文版，否则守卫会失败。

---

## 4. 改进意见（按投入产出排序）

### P0 —— 投稿前必做，1–2 天，不需要新实验

1. **重写 §2 Related Work**，补三组：
   (i) 同期跨模型 KV 传输系统 —— CacheBridge / ACCT / CacheLink / RaReCache /
   Universal Context-Reuse Layer / HeteroFold / SwaT / BridgeKV / KV-Lingo；
   (ii) 同期评估批判 —— PACT-Check(2611.06976) / Do Latent Channels Actually
   Communicate(2607.26773) / Functional Cache Divergence(2611.14051)；
   (iii) 表示对齐理论 —— PRH(2405.07987) / Structure over Geometry(2609.27252) /
   relative representations(2209.15430) / MATCHA(2610.09058)。
2. **重排贡献声明**（§1 与摘要）：C1 升为首要贡献并明确写"同期工作独立复现"；
   C3 从"我们发现"降为"我们给出机械解释与误差分解"。
3. **Discussion 新增一节** "Why our negative result coexists with positive systems
   results"，把 §3.1-M2 的 (a)(b)(c) 三条逐条写出，并把主张边界收缩到
   「linear mapper + 严格 EM + K 侧未用行为目标」这一条件组合。
4. 修 `lee2026translators` 题名；补 Qwen3 技术报告引用；补术语界定。
5. 把 C4 的主张范围全部收窄到 equal-depth pair，旗舰对统一写成 open question
   （摘要、贡献 4、结论三处口径要一致）。

### P1 —— 投稿前强烈建议，各需 1–2 个 GPU-day

6. **给 K 侧补上消费空间目标（最高价值的一条）**。本文自己主张"在 receiver 消费处
   拟合"，但 K 侧只做了注意力质量加权的 ridge。对称的做法是把目标换成 **logit /
   pre-softmax 空间的最小二乘**：求 $M_h$ 使
   $Q_S K_T^\top M_h^\top \approx Q_S K_S^\top$。
   给定标定集这是闭式解（对 $M_h$ 是线性最小二乘），成本与现有 ridge 同量级。
   如果它在 K 臂上仍然失败，"key 侧必要非充分"才真正成立；如果它有效，
   本文的负结果会被显著收窄 —— 无论哪个结果都提升论文。
7. **补一个非线性 mapper 对照**（per-head 两层 MLP 或 kernel ridge 即可）。
   作用是**把「线性不够」和「表示对齐不够」分开**，这是 reviewer 必问的第一问题。
8. **补一个 many-to-many 层映射对照**（MATCHA 式 band map），检验
   「proportional 最优」是否只是线性 mapper 家族的产物。
9. **加一个非严格任务指标**（answer-margin 或 token-F1），从现有 LL 数据零成本计算。

### P2 —— 提升接收概率

10. seed 增到 5，或对旗舰 causal test 补 3–5 次同 seed 重训给出训练方差区间
    （现已有 0.679 / 0.429 两个点）。
11. 换或加一个 ceiling 更高的第二域（LongBench2 子集），替换/补充 SQuAD。
12. 补延迟 / 显存测量，替代 Appendix B 的参数量启发式。同期工作全都有
    系统数字（CacheLink 7.27×、BridgeKV 3.0×、ACCT 6.5–21.2×、KV-Lingo 9.6–29×、
    CacheBridge −30%…−98% TTFT）。缺系统数字会被 systems 侧审稿人减分。
13. 完成 arXiv 提交清单三项。

### 投稿策略

- **首选 TMLR**。现形态（诊断 + 分解 + 边界）与 TMLR 的口味（接受负结果、
  接受复现、无页数压力）匹配；COLM 需要更强的"方法"成分，而方法部分已被
  BridgeKV / KV-Lingo 抢先。
- **时间紧时做"最小可投"版本**：只保留 C1 + C2 + C3 的分解，C4 收缩为
  equal-depth 结论、旗舰写成 open question。篇幅可从 27 页压到约 18 页，
  同时回避 M2 中 (a)(c) 两条最锋利的攻击面。

---

## 5. 本审计未做的事（如实记录）

- 未重跑任何实验，未改动任何实验数字。
- 未对 2.A/2.B 表中所列论文做全文精读，判断依据是 arXiv 摘要与题名；
  引用其具体数值前需回原文核对（尤其 CacheBridge 的 97.94%/102.94% 与
  Universal Context-Reuse Layer 的 27.59%→34.48%）。
- P1 / P2 的八条建议**一条也未执行**：本机无 GPU（`nvidia-smi` 缺失、
  `torch 2.14.0+cpu`、`cuda_available=False`），无法跑任何需要模型推理的实验。
  稿件里的实验数字一个也未改动。
  （编译方面：W35 已在本机装入 Tectonic 0.17.0 并重编译了三份 PDF，见
  `ITERATION_LOG.md` 的 W35 段；P0 的写作改动已全部落地。）
- 未检索非 arXiv 渠道（OpenReview、ACL Anthology、NeurIPS/ICML 正式论文集）。
  CacheLink / CacheBridge 等工作可能已有正式版本，投稿前应补查一次。
