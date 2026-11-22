我重新按 **2026 年 11 月当前相关工作格局、ICLR/NeurIPS 主会标准**审查了这版 30 页稿件，并额外核对了过去两个月新出现的 CacheBridge、CacheLink、KV-Lingo、PACT-Check 等工作。

我的结论是：**论文技术质量比两个月前明显提高，但由于同期工作快速出现，“consumer-space alignment 本身”的新颖性已经被明显稀释。当前最合理的审稿结论是 6/10 Weak Accept，偏向接收；如果把创新定位进一步收紧并修正一个关键逻辑表述，我会给 7/10 Accept。**

## 一、总体审稿结论

| 维度 | 9 月版本 | 当前版本 | 当前评价 |
|---|---:|---:|---|
| 问题重要性 | 9 | **9** | 很重要 |
| 原创性 | 8.5 | **7.5–8** | 论文增强，但并发工作增多 |
| 技术严谨性 | 8 | **8.5** | 很强 |
| 实验设计 | 8.5 | **9** | 诊断链基本完整 |
| 因果证据 | 7 | **8** | 显著提升 |
| 机制解释 | 8 | **8.5** | decomposition 是重要进步 |
| 泛化性 | 6 | **6** | 仍是主要弱点 |
| 写作可信度 | 9 | **9** | 边界控制非常好 |
| 当前综合评分 | 7 | **6–7** | **Weak Accept → Accept** |
| 审稿信心 | 4.5/5 | **4.5/5** | |

这里有一个看起来反常但很重要的判断：

> **论文自身比两个月前更强了，但领域里的竞争也明显更强了。**

所以不能简单因为实验变多就把评分从 7 提到 8。

---

# 二、这两个月最重要的提升，是论文终于形成了真正独立的理论框架

最新版本新增的 **Functional Compatibility Decomposition** 是我认为最有价值的变化。

现在论文正式写出了：

\[
\hat O-O_S =
(\hat A-A_S)V_SW_O^S+
\hat A(\hat V-V_S)W_O^S
\]

并明确把前一项定义成 routing term，后一项定义成 consumption term；K-only、V-only、Joint 四臂分别通过干预把两个问题拆开。论文也非常正确地强调：这是 output perturbation 的恒等式，而不是声称 task EM 可以线性分解。

这比此前单纯说：

> K 是 routing，V 是 consumption

强很多。

现在可以把整篇论文归纳成：

\[
\boxed{\text{K-side: routing compatibility}}
\]

\[
\boxed{\text{V-side: consumption compatibility}}
\]

\[
\boxed{\text{Joint: both must hold}}
\]

我认为**这个 decomposition 才应该成为当前论文第一机制贡献**，而不是 adapter 本身。

---

# 三、198 个 mapper configuration 的实验非常重要，显著加强了论文

这是当前版本另一个真正值得保留在主文的新增实验。

之前 reviewer 很容易说：

> “你只有 5 个 mapper 点，Spearman −1 可能只是巧合。”

现在已经变成：

- 2 个 model pair；
- 3 seeds；
- 每 run 33 个 \((\alpha,\lambda)\) configurations；
- 共 **198 个点**。

而结果是，raw representation error 在每个独立 run 内反而与 EM **正相关**，而 attention-output / \(W_O\)-space error 与 EM 强烈负相关。论文因此把结论收敛为：

> **预测 transfer 的不是 mapped state 离 student state 有多近，而是 receiver 在实际消费位置上的 functional error。**

这个结论现在已经有明显说服力。论文在 Introduction 中也明确将这一结果提升为 consumer-space alignment 的关键证据。

从研究价值看，我认为这是全篇目前**最值得引用的实验结果之一**。

---

# 四、但两个月后最大的变化，是“functional alignment”已经不再是你们可以独占的新颖点

这是本轮必须面对的现实。

9 月以后出现的 **CacheLink** 已经非常明确地提出：

> reconstruction fidelity is a poor proxy for receiver utility

并直接根据 receiver behavior 学习 coupled K/V correction。[arXiv](https://arxiv.org/abs/2610.15509?utm_source=chatgpt.com)

**KV-Lingo** 更进一步，直接通过 target model prediction distillation 训练 KV translators，并在多种架构、大小模型之间取得接近 re-prefill 的性能。[arXiv](https://arxiv.org/abs/2609.32610?utm_source=chatgpt.com)

**CacheBridge** 也使用 attention-sensitive calibration，并在多个 model family 上取得接近原生 target 的效果；最新版本已经报告多 family、agent/chat/reasoning/RAG 和真实 TTFT 指标。[arXiv](https://arxiv.org/abs/2609.00891?utm_source=chatgpt.com)

与此同时，PACT-Check 和 latent communication causal audit 也已经开始强调：

> task improvement 不能自动证明 receiver 使用了 sender 的 example-specific information。 [Hugging Face](https://huggingface.co/papers/2611.06976?utm_source=chatgpt.com)


好的一点是，当前论文已经**非常正确地承认了这一现实**：明确将这些工作称为 concurrent，并声明“不对 consumer-space insight 本身主张优先权”。

这是对的，千万不要撤回这种克制。

---

# 五、因此当前论文必须重新回答：“既然 CacheLink/KV-Lingo 已经存在，你们独特在哪里？”

我认为现在仍然可以形成非常清楚的差异化，而且其实不弱。

你们真正独有的组合不是：

> “consumer space 比 representation space 更好。”

而是下面四件事一起成立：

**第一，functional compatibility decomposition。**  
不是简单学习一个更好的 translator，而是用四臂干预把 key-side routing 与 value-side consumption 分开。

**第二，proxy failure 的 causal audit。**  
wrong-document、random、zero cache 可以复制 89%–105% likelihood gain，却没有 task performance，说明 teacher-forced likelihood 本身不能作为 transfer evidence。

**第三，198-point error-budget evidence。**  
不仅说 reconstruction error 不好，而且系统证明了“raw-space error 与 functional error”存在方向性的错配。

**第四，task-conditioned functional compatibility。**  
同一个 producer-consumer pair 在一个 distribution 可以 repair，在另一个 distribution 即使重新训练也无法 repair。

我建议把这四条作为论文真正的 novelty。

---

# 六、Routing-aware K mapper 是很好的新增实验，但目前有一个必须修改的逻辑问题

这版加入了 key-side intervention，这是正确方向。

routing-aware mapper 能把：

\[
TV:0.167\rightarrow0.139
\]

以及：

\[
0.254\rightarrow0.153
\]

top-1 routing agreement 也明显提高。

但是 K-only EM 仍基本在 floor。

这个实验实际上证明的是：

\[
\boxed{\text{better routing alignment is not sufficient}}
\]

但它**没有证明**：

\[
\boxed{\text{routing alignment is necessary}}
\]

目前摘要中有一句：

> “closing the routing gap is necessary but not sufficient”

这个“necessary”我认为在逻辑上仍然站不住。

因为要证明 necessary，你需要证明：

> 没有 routing compatibility → functional transfer 不可能；

或者针对一个已经能够 transfer 的 condition，主动破坏 routing compatibility，并观察性能崩溃。

当前做的是：

> routing compatibility 改善了很多 → transfer 仍失败。

只能推出 **not sufficient**。

而且更关键的是，论文自己在 Discussion 已经承认：

> 当前 routing-aware mapper 只是 attention-mass reweighted representation regression，并不是与 V-side 完全对称的 functional objective；真正对称的 key-side 实验应该直接拟合 receiver 的 pre-softmax logits。这个实验尚未做。

这实际上是当前论文**最重要的技术 limitation**。

### 我建议必须修改

把摘要、Introduction 中类似：

> “necessary but not sufficient”

全部改为：

> **“routing alignment alone is not sufficient.”**

至于 necessary，留给未来实验。

这个修改成本为零，但会避免一个很容易被强 reviewer 抓住的逻辑漏洞。

---

# 七、如果还准备补一个实验，我仍然最推荐真正对称的 Functional K Mapper

这一点已经不是“锦上添花”，而是最自然的下一步。

论文现在 V-side 优化：

\[
A_S\hat V_T \rightarrow A_SV_S
\]

非常成功。

那么 K-side 应直接优化：

\[
Q_S\hat K_T^\top
\rightarrow
Q_SK_S^\top
\]

或者：

\[
KL(\hat A\|A_S)
\]

而不是仅仅 attention-mass weighted raw-K regression。

如果这个 experiment 得到：

### 情况 A

routing logits 对齐后 K-only EM 仍然 0：

那是非常重要的 negative result：

> **即使 functional routing itself 对齐，teacher K 仍然无法 transfer。**

此时说明 Eq. (1) 中的 routing term 仍不足以描述 K-side compatibility，还隐藏了其它 query/key interaction。

### 情况 B

K-only 恢复：

那整篇论文形成最漂亮的闭环：

\[
\boxed{K:\ routing-space alignment}
\]

\[
\boxed{V:\ consumption-space alignment}
\]

这会明显提升到更强的论文。

所以如果还有算力，我认为这仍然是 **P0 级最值得做的新实验**。

---

# 八、“Task-conditioned Functional Compatibility”是新版本非常有潜力的新概念，但现在只能作为 evidence-backed hypothesis

当前实验已经比两个月前更完善。

不仅：

\[
Synthetic \rightarrow SQuAD
\]

失败；

即使：

\[
SQuAD calibration\rightarrow SQuAD heldout
\]

重新训练 mapper/adapter，所有 transfer arms 仍然为 0。

然后作者又补了：

- 15 synthetic docs，与 SQuAD calibration volume 相同；
- 15 SQuAD + 70 synthetic = 85 docs；

结果仍无法恢复，因而把“数据量不足”和“distribution mismatch”拆开。论文由此正式命名：

> **task-conditioned functional compatibility**。

我认为这个概念有价值。

但现在只有：

\[
\mathcal D_1=\text{synthetic project-status}
\]

和

\[
\mathcal D_2=\text{SQuAD}
\]

两个非常不同的 domain。

因此从两个点直接上升到：

> compatibility 是 task-distribution-conditioned property

还稍微有一点快。

更准确的表述应是：

> **“our results provide evidence that functional compatibility can be task-conditioned.”**

不要写成普遍定律。

如果下一版本愿意继续做，最有价值的不是第三个随机 benchmark，而是前面我曾建议过的 **controlled distribution shift**：

\[
D_0:\text{原模板}
\]

→ paraphrase

→ question-style shift

→ schema shift

→ relation-vocabulary shift

→ SQuAD。

这样会形成真正的：

> **Compatibility Frontier**

这很可能本身就值得下一篇论文。

---

# 九、当前 title 比两个月前明显更好，但仍略微过宽

现在：

> **Cross-Model KV Transfer Needs Functional Compatibility: Why Representation Alignment Is Not Enough** main(20261122-074120)

方向是对的。

但“Representation Alignment Is Not Enough”容易被解读为：

> 所有 representation alignment 方法都不够。

而现在 concurrent work 已经证明更复杂的 nonlinear / behavioral / distillation alignment 可以取得很好结果；论文自己也承认，当前 negative evidence 主要约束的是 linear mapper family。

所以我更推荐稍作调整。

例如：

> **Functional Compatibility in Cross-Model KV Transfer: Why Representation Fidelity Is Not Enough**

我认为把：

**Representation Alignment**

改成：

**Representation Fidelity**

非常重要。

因为你们真正击破的是：

> state reconstruction distance / fidelity

而不是“alignment”整个研究范式。

或者：

> **Beyond Representation Fidelity: Functional Compatibility for Cross-Model KV Transfer**

这是我目前最推荐的标题。

---

# 十、与两个月前相比，我反而认为不应再把“adapter repair”放得太核心

原因是最新 related work 已经有：

- trainable correction；
- nonlinear mapper；
- distillation；
- receiver behavioral objectives；
- cross-family translation；
- near-native systems results。

所以从方法竞争上，rank-8 adapter 已经不算很强的新方法。

但它作为**科学验证工具**仍然非常有价值：

> 如果 receiver-side adapter 能使原本 unusable 的 teacher state 变 usable，说明失败不是 teacher state 完全没有信息，而是 producer-state 与 consumer computation 不兼容。

尤其 1.7B→0.6B 中 frozen adapter 的 causality：

\[
correct=0.905\pm0.083
\]

而 destroyed controls ≤0.179，

这是强证据。

所以我建议 framing 从：

> “we propose a fix”

降低成：

> **“we use a lightweight consumer adaptation as an intervention showing repairability.”**

这样反而更学术，也更不容易被 CacheLink/KV-Lingo 抢 novelty。

---

# 十一、当前稿件最大的优点已经是“科学边界控制”

这一版我非常认可的一点，是作者对 concurrent positive systems 的处理。

论文专门增加了：

> **Why a negative result coexists with positive systems**

并区分：

- mapper family；
- key-side objective；
- task / metric。

这是非常好的 Discussion。

当前稿件没有说：

> 别人的系统都是错的。

而是承认：

> 你们 negative result 约束的是当前线性 mapper + 当前任务/指标，不能推出所有 cross-model KV transfer 都失败。

这让整篇论文可信度提升很多。

从审稿角度，这一点甚至比再加一个 2% 的性能提升更加分。

---

# 十二、我现在认为还有三个主要审稿 concern

除了前面“necessary”逻辑必须修之外，我会在正式 review 中留下三项 concern。

第一是**并发工作显著压缩 novelty**。CacheLink 已经明确提出 reconstruction fidelity ≠ receiver utility，并用 receiver behavior 直接优化；KV-Lingo 也直接以 receiver prediction distillation 做 translation。[arXiv](https://arxiv.org/abs/2610.15509?utm_source=chatgpt.com) 因此论文不能再把 consumer-space alignment 本身作为主要 novelty。

第二是**key-side 仍没有真正对称实验**。当前 intervention 只能证明 reducing routing TV 不足以恢复功能，而不是证明 routing 的完整 functional compatibility 已经被测试。

第三是**task-conditioned compatibility 证据仍来自一个特殊 synthetic corpus + 一个低 Self-ceiling 的 SQuAD setting**。这足以作为重要 observation，但还不足以作为普遍规律。

这三个问题都不会让我直接 reject，但阻止我给 Strong Accept。

---

# 十三、我建议投稿前只做非常有限的改动

此时已经不应该继续堆大量实验。

最值得做的是：

**第一优先级：** 修改 key-side 的“necessary but not sufficient”为“not sufficient”，或者完成真正 symmetric pre-softmax key mapper。

**第二优先级：** 把论文 novelty 重写成：

> functional compatibility decomposition + proxy audit + consumer-space error law + task-conditioned boundary

而不是 consumer-space mapper / adapter。

**第三优先级：** 把标题中的 “Representation Alignment” 改为 “Representation Fidelity” 或 “Raw Representation Alignment”。

除此之外，我不会要求为了当前投稿增加更多模型 family。

---

# 十四、正式审稿结论

如果我是现在的 ICLR / NeurIPS reviewer，我会给：

**Overall: 6/10 — Weak Accept**

**Confidence: 4/5**

我的正式判断大致会是：

> This work provides a careful diagnostic study of cross-model KV transfer. Its strongest contribution is not a new cache translator, as recent concurrent work already demonstrates receiver-aware translation and stronger systems results, but a functional decomposition and causal evaluation framework showing why reconstruction fidelity and teacher-forced likelihood can fail to predict whether a receiving model can actually use a handed-off state. The 198-configuration mapper sweep substantially strengthens the representation-vs-consumption-space argument, and the task-conditioned repair failure is an interesting boundary result. The work is technically careful and unusually transparent about withdrawn claims and unresolved cases. The main remaining weakness is that the key-side intervention is not the symmetric functional objective advocated by the paper, so the claim that routing alignment is “necessary but not sufficient” is stronger than the evidence supports. The broader novelty is also narrowed by recent concurrent systems such as CacheLink and KV-Lingo.

### 我的最终倾向

**接收概率：约 65%–75%。**

如果把 key-side 逻辑修正、重新突出 decomposition 而不是 mapper novelty，我会倾向：

**7/10 — Accept。**

目前我不会给 8/10 Strong Accept，主要不是论文质量不够，而是**过去两个月相关领域发展过快，最直观的 consumer-aware alignment insight 已被多篇并发工作覆盖；你们现在最需要保护的是“为什么它成立、怎样正确验证、边界在哪里”这一层原创价值。**