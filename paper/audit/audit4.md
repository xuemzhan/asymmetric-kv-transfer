我重新以独立审稿人的角度完整审查了最新 23 页版本，并重点对照上一轮提出的 evaluator、K/V 机制拆分、最终 adapter 因果性、第二域验证、聚类统计等问题。我的判断是：**这一版已经从“边缘可接收”进入了“有明确接收理由”的状态。**

如果按 ICLR / NeurIPS 主会尺度，我目前会给 **7/10，Accept，Confidence 4.5/5**。更重要的是，我认为现在不应该继续把主要精力放在“补漏洞”上，而应该开始把论文从一篇很严谨的 *audit + repair paper*，进一步提升成一篇有清晰新概念、新规律和后续研究方向的论文。

## 一、当前审稿结论

| 维度       |       当前评价 | 判断                                           |
| -------- | ---------: | -------------------------------------------- |
| 问题重要性    |       9/10 | KV 跨模型复用是明确的系统与模型问题                          |
| 原创性      | **8.5/10** | consumer-space alignment 已形成独立贡献             |
| 技术严谨性    |       8/10 | 关键反例和因果控制基本补齐                                |
| 实验设计     |     8.5/10 | negative → diagnosis → repair → boundary 很完整 |
| 机制解释     |       8/10 | K-side / V-side 已正确拆开                        |
| 统计可信度    |     7.5/10 | document-clustered 分析已明显加强                   |
| 泛化性      |       6/10 | 单家族，且 repair 有明显 distribution dependence     |
| 写作可信度    |       9/10 | 对旧结论的修正非常克制                                  |
| 潜在影响力    | **8.5/10** | 如果 framing 再提升，有成为“评价范式论文”的潜力                |
| **综合评分** |   **7/10** | **Accept**                                   |

上一版最容易被 reviewer 攻击的几个核心问题，现在基本已经闭合。

首先，corrected evaluator 已不再只是“作者自己说正确”。现在不仅对三个 student size 的完整 Self test set 做了 full-prefill 对照，还报告了 normalized-EM、first-token top-1、KL，并进一步追踪 residual 到低精度 attention-kernel path；cache round-trip 和显式 position offset 都能 bit-for-bit 重现 reference，而且 SQuAD 上 validator 也复现。 

其次，上一轮我指出的 K-side / V-side 逻辑混用已经被真正修掉。新版现在明确说明：K-only/Joint 的问题是 mapped K 改变 receiver routing，而 V-only 使用 student 自己的 K，因此 V 的问题是 consumption-space mismatch，不能再用 K routing divergence 解释。 

第三，论文现在最强的新结果已经不仅是“consumer-space mapper 好用”，而是出现了一个很有价值的**度量规律**：raw representation error 与 EM 的 rank correlation 只有 −0.10，而 attention-output space 与 \(W_O\) space 的 error 分别达到 −1.00 和 −0.80；甚至 affine mapper 的 raw error 最低，却几乎不能完成任务。 这已经非常接近一个可以独立命名和传播的研究结论。

第四，adapter 的因果证据现在也真正匹配 headline 配置了：使用最终 20-epoch adapter、3 seeds，在 1.7B→0.6B 上 correct teacher KV 达到 \(0.905\pm0.083\)，而 wrong-document/random/zero 最多 0.179，并且三个 seed 的 document-clustered margin 都排除 0。相比之下 8B→0.6B 的 margin 只有 +0.071，论文没有硬说成功，而是保持 unresolved。 

这四项结合起来以后，我已经不会再以“实验可信性不足”作为拒稿理由。

---

# 二、这一版真正应该强化的创新点，不是“adapter”，而是一个更一般的科学命题

我认为现在最大的机会，是把论文从：

> Cross-model KV transfer needs a compatible consumer.

提升成一个更一般、更有理论味道的命题：

> **Cross-model representation transfer is a functional compatibility problem, not merely a representation-alignment problem.**

这是当前论文最有价值、也最可能被后续工作引用的思想。

你们现在已经有足够实验支持下面三层结论：

$$
\text{representation similarity}
\not\Rightarrow
\text{functional transfer}
$$

$$
\text{teacher-forced likelihood}
\not\Rightarrow
\text{information transfer}
$$

以及：

$$
\text{alignment in the consumer's functional space}
\rightarrow
\text{task usability}
$$

尤其第二条非常强。wrong document、random Gaussian、zero cache 能重现真实 K-only 的 89%–105% likelihood gain，却基本没有 task EM。

这不是单纯“发现一个 evaluator bug”，而是对 latent-state transfer 文献提出一个很有普适性的警告：

**不要用一个 state 看起来像不像另一个 state，或者让 answer likelihood 变好了多少，来判断 state 是否真的被新的模型“理解”。**

这是论文真正的价值所在。

---

# 三、我最建议增加的不是更多 benchmark，而是一个“Functional Compatibility Decomposition”

这是我认为当前最可能把论文从 **7 分提升到 8 分候选** 的改动，而且不一定需要新增大量实验。

现在四臂设计天然已经对应一个非常漂亮的 attention-output decomposition。

设 student 自己的输出为：

$$
O_S=A_SV_SW_O^S
$$

跨模型 handoff 后：

$$
\hat O=\hat A\hat V W_O^S
$$

则可以写成：

$$
\hat O-O_S
=
(\hat A-A_S)V_SW_O^S
+
\hat A(\hat V-V_S)W_O^S
$$

或者使用等价分解。

这样马上得到：

**K-only：**

$$
\hat V=V_S
$$

只剩：

$$
(\hat A-A_S)V_SW_O^S
$$

即 **routing compatibility**。

**V-only：**

$$
\hat A=A_S
$$

只剩：

$$
A_S(\hat V-V_S)W_O^S
$$

即 **value-consumption compatibility**。

**Joint：**

两项同时存在，而且可能产生 interaction。

这几乎就是你们实验现象的数学表达。

论文现在已经用文字把这两个失败分开了， 但如果加这样一个 decomposition，就不再只是：

> “我们观察到两个 failure mode。”

而会变成：

> **我们提出了一个 functional compatibility decomposition，并通过四臂实验分别识别其 routing term 和 consumption term。**

这会显著提高创新感。

我甚至建议把它作为 Method 3.2 后面的一个小节：

### 3.x Functional Compatibility Decomposition

不必声称 theorem，只需要作为分析框架。

这是我目前最推荐的修改。

---

# 四、如果只允许再做一个新实验，我建议做 Routing-aware K Mapper

论文现在 V-side 已经形成了几乎完整的闭环：

$$
\text{raw V mapper fails}
$$

↓

$$
\text{consumer-space V mapper}
$$

↓

$$
EM:0.113\rightarrow0.935
$$

而 K-side 目前只有：

$$
\text{mapped K}
\rightarrow
\text{routing divergence}
$$

还是 diagnostic，没有 intervention。

论文自己也已经把 routing-aware key mapper 写进 Future Work。

如果计算资源允许，我认为应该把它从 Future Work 提到正文。

可以直接优化：

$$
\mathcal L_K=
\left\|
Q_S\hat K_T^{\top}
-
Q_SK_S^{\top}
\right\|^2
$$

或者更直接：

$$
\mathrm{KL}
\left(
A_S\;\|\;\hat A
\right)
$$

其中：

$$
A_S=\mathrm{softmax}(Q_SK_S^\top)
$$

$$
\hat A=\mathrm{softmax}(Q_S\hat K_T^\top)
$$

然后比较：

* raw-K affine mapper；
* routing-aware K mapper；
* shuffled-target control；
* K-only EM；
* routing TV/top-1 agreement。

如果得到：

$$
routing\ mismatch\downarrow
\quad\Rightarrow\quad
K\text{-only EM}\uparrow
$$

那么整篇论文会形成非常漂亮的对称结构：

$$
\boxed{
K:\ routing\text{-}space\ alignment
}
$$

$$
\boxed{
V:\ consumption\text{-}space\ alignment
}
$$

这将不再只是“一个 V mapper + 一个 adapter”的论文，而可能成为：

> **cross-model KV transfer 的 functional alignment framework。**

这是最值得投入的新实验。

---

# 五、SQuAD 的失败结果，不应该被当作弱点，要把它变成一个新的研究发现

新版现在已经做到了一个非常有意思的事情：

不仅：

$$
Synthetic\rightarrow SQuAD
$$

不泛化；

就连在 SQuAD 内重新训练 mapper 和 adapter，15 个 calibration documents → 15 个 held-out documents，所有 transfer arms 仍然是 0。 

这说明一个比“adapter 不泛化”更深的现象：

> **Transferability is not merely model-pair dependent; it can be task-distribution dependent.**

我认为可以把它上升成论文第四个 conceptual contribution。

也就是说，cross-model state transfer 不是：

$$
Transferability(T,S)
$$

而更合理的是：

$$
\boxed{
Transferability(T,S,\mathcal D)
}
$$

甚至：

$$
Transferability(
\text{producer},
\text{consumer},
\text{task distribution},
\text{alignment objective}
)
$$

这是一个很重要的新视角。

我建议论文不要试图“解释掉 SQuAD failure”，而应该把它正式命名为：

> **distribution-conditioned compatibility**

或者更谨慎：

> **task-conditioned functional compatibility**

这样一个 negative result 就变成了研究贡献。

---

# 六、比“多加一个 benchmark”更有创新价值的实验，是做 Compatibility Frontier

如果还有中等规模的实验预算，我不建议简单再加 HotpotQA、TriviaQA、Llama、Mistral，然后做一个大表。

更有学术价值的是做一个**受控分布迁移实验**。

例如保持同一个 hidden knowledge graph，只逐步改变：

| Shift | 改变内容                       |
| ----- | -------------------------- |
| D0    | 原始 project-status template |
| D1    | 文档 paraphrase，语义不变         |
| D2    | question template 改变       |
| D3    | relation vocabulary 改变     |
| D4    | answer format 改变           |
| D5    | document schema 改变         |

然后测：

$$
V\text{-mapper EM}
$$

$$
adapter\ EM
$$

$$
e_{\text{attn}},e_{W_O}
$$

随 distribution distance 怎么衰减。

最终可以得到一个：

> **Functional Compatibility Frontier**

这比“再加 3 个公开 benchmark”更有原创性。

它回答的是：

> consumer-space compatibility 究竟依赖于模型架构，还是依赖于任务表示、文档结构、问题分布？

这很可能成为下一篇论文，但如果能有哪怕一个小型 pilot figure 放入当前论文，影响力会明显提高。

---

# 七、Table 4 是当前论文最可能成为“代表性结果”的表，但还可以再加强

Table 4 非常重要：

* Affine raw error = **0.267**，最低；
* 但 V-only EM = **0.057**；
* OutAware raw error = 0.772，反而更差；
* 但 EM = **0.592**；
* attention-output error 从 3.20 降到 0.073；
* \(W_O\)-space error从 3.75 降到 0.082。

这是非常漂亮的反直觉结果。

目前唯一的问题是：

$$
\rho=-1.00
$$

实际上只有 **5 个 mapper variants**。

强 reviewer 可能会说：

> “五个点上的 Spearman −1.0 没有那么有说服力。”

因此如果想把这个结论变成真正的“law-like result”，非常值得做一个低成本 sweep：

改变：

* ridge λ；
* mapper rank；
* interpolation：

$$
\mathcal L=
(1-\alpha)\mathcal L_{\text{raw}}
+
\alpha\mathcal L_{\text{consumer}}
$$

取：

$$
\alpha=0,0.1,\ldots,1
$$

然后得到 10–30 个 mapper configurations。

画：

$$
e_{\text{raw}}\ vs\ EM
$$

$$
e_{\text{attn}}\ vs\ EM
$$

$$
e_{W_O}\ vs\ EM
$$

如果仍然是：

$$
R^2_{\text{consumer}}\gg R^2_{\text{raw}}
$$

那这个结果会非常强。

论文的核心贡献就可以变成：

> **The distance that predicts transfer is not representation distance, but consumption-space distance.**

这句话很可能是整篇论文最值得传播的结论。

---

# 八、有两个地方现在反而应该适当“减法”

目前稿件已经到 23 页，证据链很充分。继续堆结果有可能让核心创新被稀释。

我建议把以下内容降低权重。

第一，**K×V likelihood interaction**。现在论文已经明确证明 LL 不是 functional transfer metric，那么 Table 3 的 likelihood factorial decomposition 学术优先级明显下降。 可以移 appendix，只在正文一句话说明“interaction is unstable and is not interpreted functionally”。

第二，**2050-token byte crossover**。论文自己已经说它“does not bear on the state–consumer findings”。 我建议直接移 Appendix。

这样省出来的正文空间，应留给：

* Functional Compatibility Decomposition；
* Table 4；
* K-side vs V-side schematic；
* causality result。

论文会明显更加“像一篇新理论框架论文”，而不是“过去几个月所有实验的完整记录”。

---

# 九、还有两处措辞建议修正，避免无谓损失审稿分

一处是 Discussion 中：

> “a full handoff must satisfy both, which is why the joint arm is the weakest arm everywhere.” 

“weakest everywhere”太强，而且从数字上并不严格成立。例如部分 floor 条件下 Joint 会略高于某单臂；adapter 后 8B→0.6B Joint 也高于 V-only。

建议改成：

> **“Full handoff must satisfy both forms of compatibility and is therefore consistently fragile; it does not reliably combine the gains of the two components.”**

这个表述更准确，而且更有科学意义。

另一处是 Related Work 仍然有一句：

> “the standard evaluation can report transfer where none exists functionally.” 

但正文现在已经非常正确地声明：

> evaluator defects 是你们自己 earlier implementation 的问题，没有 audit third-party implementations。

因此 Related Work 也最好保持一致。

可以改为：

> “our legacy task evaluator could report apparent transfer where none exists functionally; more generally, we show that teacher-forced likelihood alone is insufficient evidence of functional transfer.”

这样科学责任边界会非常清楚。

---

# 十、论文标题现在也到了值得重构的时候

当前标题：

> **Cross-Model KV Transfer Needs a Compatible Consumer: Diagnosis and a Consumption-Side Fix** 

已经不错，但现在论文的内容实际上已经超过“compatible consumer + fix”。

我更推荐三个方向。

**最推荐：**

> **Beyond Representation Alignment: Functional Compatibility for Cross-Model KV Transfer**

这个最有“新概念论文”的感觉。

第二选择：

> **Cross-Model KV Transfer Requires Functional Compatibility, Not Just Representation Alignment**

更加直接。

如果希望保持原来的 diagnosis 风格：

> **Diagnosing Cross-Model KV Transfer: From Representation Alignment to Functional Compatibility**

我个人认为第一种最好。

因为现在真正的新东西已经不是：

> “我们给 student 加了一个 adapter。”

而是：

> **representation alignment 是错误的目标对象；正确的目标是 functional/consumption compatibility。**

标题应当围绕这个贡献。

---

# 十一、我建议最终把论文贡献压缩成四条，而不是很多实验点

如果我是作者，我会让 Introduction 的 Contributions 最终变成一个非常清晰的四段结构：

1. **Evaluation principle**：functional transfer 必须用 corrected task generation + content controls 来验证；teacher-forced likelihood 不足以证明 transfer。

2. **Functional compatibility decomposition**：K transfer 对应 routing compatibility，V transfer 对应 value-consumption compatibility，Joint 同时依赖两者。

3. **Consumer-space alignment**：raw representation error 不预测 transfer，而 attention/output-space error 与任务性能一致；consumer-aware mapper 可以在不修改 student weights 的情况下恢复 V transfer。

4. **Repairability and boundary**：小型 consumer adapter 可以在一个 task distribution 内恢复大量 transfer，但 causal recovery 在大 gap pair 更弱，并且 repair 不跨 task distribution；因此 compatibility 本身是 task-conditioned 的。

这四条非常像一篇成熟论文。

---

# 十二、从“继续补实验”的投资回报率看，我的排序已经变化

如果现在还有算力，我的优先级是：

1. **Routing-aware K mapper**——最高价值，能形成 K/V 对称闭环。
2. **增加 Functional Compatibility Decomposition**——几乎不需要算力，但能显著提升理论创新。
3. **mapper objective sweep / interpolation**——验证 consumer-space error 是普遍预测量，而不是五个 mapper 点的偶然现象。
4. **controlled distribution-shift frontier**——非常创新，甚至可作为后续论文主线。
5. **cross-family pair**——有价值，但我不再认为是当前论文必须项。
6. **更多普通 benchmark**——优先级最低。

也就是说，我现在不建议为了“审稿安全”盲目堆 benchmark。

**应该把资源用于让已有发现变成一个更一般的规律。**

---

# 十三、如果我是本轮正式审稿人，我的最终 Review 会这样写

**Recommendation: 7/10 — Accept**

这篇论文重新审视了 cross-model KV transfer 的评价与机制。最有价值的结果不是简单证明某些 KV components 可以或不可以迁移，而是表明 state-space similarity 与 functional usability 可以显著解耦：content-free caches 可以重现大部分 teacher-forced likelihood gain，却无法回答问题；相反，raw reconstruction 更差的 consumer-space mapper 可以大幅恢复 task performance。论文进一步将 K-side routing mismatch 与 V-side consumption mismatch 分开，并通过 mapper、adapter 和 content-destruction controls 对后者提供了较强因果证据。尤其在 1.7B→0.6B 上，最终 adapter 在 correct teacher state 上达到 \(0.905\pm0.083\)，而破坏内容后性能大幅跌落。

论文的主要限制是目前只覆盖 Qwen3 family，核心 repair 在第二任务分布上无法复现，而且部分 mechanistic analyses 仍只有单 seed。作者对此没有隐藏，而是明确限定了结论范围。 我认为这些限制不足以抵消论文在 evaluation methodology 和 functional representation transfer 上的贡献。

---

## 最后判断

这版论文我已经**不建议再以“修缺陷”为主要目标**。

现在应该做的是把故事从：

> **“旧的 KV transfer 评价有问题，我们修了一个 mapper/adapter。”**

提升为：

> **“跨模型隐状态迁移存在一个普遍的 Functional Compatibility 问题：表示空间相似不等于可消费；K 与 V 分别受 routing compatibility 和 value-consumption compatibility 约束，而 task distribution 决定这种 compatibility 是否成立。”**

如果按这个方向重构，并最好再完成 routing-aware K mapper，我认为论文有机会从目前的 **7/10 Accept**，进一步走向一篇真正有辨识度、后续容易被引用和延伸的工作。
