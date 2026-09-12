我以“独立审稿人、按 ICLR/NeurIPS 主会标准”的角度完整审查了这篇 16 页论文。论文的核心问题是有价值的，而且四臂 K/V 拆分实验具有潜在原创性；但当前版本存在一个比较关键的问题：**实验现象本身有意思，但论文把“现象”推进为“机制规律”的速度过快，若干核心结论实际上没有被现有实验充分识别出来。**

### 总体审稿结论

| 维度           | 评价                                   |
| -------------- | -------------------------------------- |
| 研究问题重要性 | 8/10                                   |
| 创新性         | 7/10                                   |
| 技术合理性     | 6/10                                   |
| 实验充分性     | 5/10                                   |
| 统计严谨性     | 4.5/10                                 |
| 机制解释可信度 | 4.5/10                                 |
| 写作与结构     | 7.5/10                                 |
| 可复现性       | 4.5/10                                 |
| 当前综合评分   | **5/10，Weak Reject / Major Revision** |
| 审稿信心       | **4/5**                                |

我不会因为实验规模不够而直接否定这项工作。真正影响我当前给出 Weak Reject 的，是下面几个**可以改变论文核心结论的科学问题**。如果把前 4 项解决好，我认为这篇论文很有机会进入 **6–7/10、Weak Accept 到 Accept** 区间。

论文目前的主张非常清楚：K 是可组合的 addressing substrate，近乎普遍可跨模型迁移；V 是 weight-bound content，只在能力/架构较匹配时迁移；同时作者进一步把 CCA 几何对齐与功能迁移区分开，并提出 layer localization、heterogeneous reassembly 和约 2050 token 的成本交叉点。

------

## 一、我认为最重要的问题：四臂实验实际上显示了很强的 **K×V 交互效应**，但论文没有分析它

这是目前论文最值得重视、也是最可能“改写整篇论文”的问题。

作者设计的是：

| K 来源  | V 来源  | Arm    |
| ------- | ------- | ------ |
| Student | Student | Self   |
| Teacher | Student | K-only |
| Student | Teacher | V-only |
| Teacher | Teacher | Joint  |

这实际上天然就是一个 **2×2 factorial design**。

但是文章几乎一直把 K-only 和 V-only 当成两个独立效应讨论，而没有正式分析 **K/V interaction**。

尤其是旗舰实验 8B→0.6B：

- K-only：ΔLL = +2.62
- V-only：ΔLL = −0.23
- Joint：ΔLL = **+4.52**

这意味着：

$Interaction = \Delta Joint-\Delta K-\Delta V =4.52-2.62-(-0.23) =\mathbf{+2.13}$

更直观地说：

**Teacher V 在 Student K 条件下的贡献：−0.23；
Teacher V 在 Teacher K 条件下的边际贡献：4.52−2.62=+1.90。**

这是一个非常大的反转。

因此当前论文的结论：

> “V does not transfer / V is weight-bound”

至少在这个实验上并不是最准确的解释。

更符合数据的解释可能是：

> **K 有更稳定的独立可迁移性，而 V 的可迁移性高度依赖 K–V compatibility。**

甚至可以更进一步：

> **V 并不是没有 transferable information，而是 Teacher V 在 Student K 所定义的 attention routing 下可能失配；Teacher K 恢复后，Teacher V 又重新变得有用。**

这件事对论文 5.1 的机制解释尤其重要。论文现在认为 V 的失败发生在 downstream weight-parameterized consumption pathway。

但要注意：

**V-only 和 Joint 的 downstream student weights 是一样的，mapped Teacher V 也是一样的；二者唯一主要变化恰恰是 K。**

如果 Joint 从 −0.23 变成 +4.52，那么就不能简单把 V-only failure 归因于下游权重消费。至少还存在非常强的 **attention routing × content compatibility**。

SQuAD 又出现反方向现象：K-only 三个 seed 约 +5.24，V-only约 +0.97，Joint约 +3.61。

对应 interaction 大约是：

$3.61-5.24-0.97\approx -2.60$

这意味着 **K×V interaction 甚至可能是 domain-dependent 的**。

### 我会要求作者必须补做

把所有六组 pair 的 Joint 全部报告出来，并做正规的 2×2 factorial analysis：

$LL \sim K_{source}+V_{source}+K\times V$

至少同时报告：

- K main effect；
- V main effect；
- K×V interaction；
- 每个 pair 的置信区间和显著性；
- Synthetic 与 SQuAD 的 interaction 是否一致。

如果这个结果成立，我甚至建议把论文的核心贡献改成：

> **Robust K transfer with interaction-dependent V transfer**

这比“Addressing transfers, content does not”更准确，也更有科学含量。

------

# 二、“capability-gated V transfer”目前并没有被实验真正识别

论文多次把结果解释为：

> V transfer is gated by teacher–student capability matching.

但当前六组实验同时改变了：

- 参数规模；
- layer 数；
- source-target layer mapping；
- 模型实际能力；
- 可能还有 representation distribution。

模型配置是：8B/4B 为 36 层，1.7B/0.6B 为 28 层；两个 equal-layer pair 恰好是 8B→4B 和 1.7B→0.6B，而所有 36→28 都采用 proportional layer mapping。

实际结果又恰好是：

> V 只在两个 equal-layer pair 上表现相对成功。

因此现在数据最多只能说明：

> **V success 与 equal-layer condition 高度相关。**

不能区分到底是：

$\text{Capability gap}$

还是

$\text{Layer-count mismatch}$

还是

$\text{Proportional layer mapping error}$

造成了 V failure。

论文自己其实已经意识到这个 confound，并在 Limitations 中承认。

但问题是：**这个 confound 恰好污染的是论文最核心的结论之一。**

更明显的是，4B→1.7B 的参数比例约 2.35×，实际上比 1.7B→0.6B 的 2.8× 更接近，但前者 V 的 EM 从 1.00 崩到 0.12，而后者从 0.43 上升到 0.80。

所以“capability gap”并不能解释这组反例。

### 应补充一个真正的解耦实验

至少做三类控制：

| 实验                                                     | 目的                                  |
| -------------------------------------------------------- | ------------------------------------- |
| 28→28 pair 人为错位 layer mapping                        | 检验 equal layer 是否真的关键         |
| 36→28 使用 learned layer selection / top-k source layers | 检验失败是否来自 proportional mapping |
| 36→36 pair 人为 proportional/offset mapping              | 将 capability 与 layer alignment 解耦 |

最好建立一个二维矩阵：

**x 轴：capability / parameter gap**
**y 轴：layer-alignment quality**

然后观察 V transfer boundary。

在做到这一点以前，我建议把全文的：

> capability-gated

改为更保守的：

> **alignment-dependent / architecture-dependent**

------

# 三、“K transfers near-universally”也存在定义问题

论文把 K 的 EM ≥ 0.71 作为“near-universal transfer”的重要证据。

但如果以作者自己在 Method 中提出的目标：

> “preserving or improving task performance”

来看，部分实验实际上明显下降。

例如 seed 0：

| Pair    | Self EM | K-only EM |
| ------- | ------- | --------- |
| 8B→4B   | 1.00    | 0.91      |
| 4B→1.7B | 1.00    | 0.95      |
| 8B→1.7B | 1.00    | **0.82**  |



1.00→0.82 是 18 个百分点的下降。如果称之为“preserving task performance”，审稿人很容易质疑。

V 的“successful transfer”也存在同样的问题：

> 8B→4B：Self 1.00 → V-only 0.88

12 个百分点下降仍被论文归类为成功。

这里缺少的是一个**事先定义的 transfer success criterion**。

建议作者采用：

$Retention = \frac{Score_{transfer}}{Score_{self}}$

或者直接做 non-inferiority test，例如：

$EM_{transfer}-EM_{self}>-\epsilon$

其中 ε 事先定义为 3pp、5pp 或任务相关阈值。

否则“0.71 算 transferable，而 0.12 不算”的界限是事后人为定义的。

------

# 四、CCA 这一部分目前不足以支持“geometric equivalence”

这是另一个我认为比较严重的技术问题。

论文以第一 canonical correlation：

$\rho_1 \approx 0.99$

来得出：

> K 与 V 几乎同等 geometrically alignable；

甚至进一步写成：

> “The two models therefore share nearly the same linear subspace.”

表 3 还只报告了 **seed 0**。

这里存在几个问题。

首先，**CCA 的第一 canonical correlation 只说明存在一对线性投影方向高度相关**，并不意味着两个高维 representation space “几乎相同”。

一个 128 维空间即使只有少数方向高度对应，也完全可能得到：

$\rho_1\approx1$

而其他绝大多数 canonical dimensions 对不上。

所以：

$\rho_1 \approx 0.99$

不能推出：

$\mathcal S_K \approx \mathcal S_V$

更不能直接推出“geometrically equivalent”。

其次，如果 CCA 不是严格在 held-out token/sample 上计算，在高维条件下非常容易产生乐观估计。

第三，论文自己的 layer-transfer 结果实际上已经说明 representation geometry 的 functional structure 很不均匀：只有少量 V layer 有用。

### 建议至少补充

报告完整 canonical spectrum，而不是只报告 ρ₁；同时给出：

- mean/top-k canonical correlation；
- SVCCA / PWCCA；
- linear CKA；
- held-out reconstruction $R^2$；
- Procrustes reconstruction error；
- seed 0/1/2 全部结果。

更有价值的是进一步做：

$Representation\ similarity \rightarrow Functional\ transfer$

的预测分析，例如跨 layer/head/pair 计算 correlation。

那样才真正能支持：

> geometric alignment does not predict functional transferability.

------

# 五、现在的“V is weight-bound”更像一个 plausible hypothesis，而不是已经证明的机制

论文将 K 描述为 addressing，将 V 描述为 content，这作为直觉没有问题，但表述过于本体化。

标准 attention 是：

$A=\mathrm{softmax}(QK^\top),\qquad O=AVW_O$

所以：

- K 不是独立的 addressing，它只有相对于 **student Q** 才决定 routing；
- V 也不是纯粹的“semantic content”；
- V-only failure 可能来自 $K_S$ 与 $V_T$ 不协调；
- 也可能来自 mapped $V_T$ 与 student $W_O$ 不协调；
- 还可能来自 residual/MLP downstream dynamics。

现有实验没有区分这些机制。

尤其前面提到的 Joint interaction 已经提示：

> **K–V compatibility 本身可能就是关键变量。**

我建议增加两个机制实验：

$A_S=\mathrm{softmax}(Q_SK_S^\top)$

和

$A_T'=\mathrm{softmax}(Q_S\hat K_T^\top)$

直接比较 attention maps。

然后测试：

$A_S\hat V_T,\quad A_T'\hat V_T,\quad A_T'V_S$

进入 $W_O^S$ 后的表示误差。

如果再进一步做一个 **$W_O$-aware V mapper**，例如直接最小化 attention output-space mismatch，而不是 V-space MSE，就可以真正验证：

> “V failure comes from weight-bound consumption.”

这会显著加强论文的机制深度。

------

# 六、“Heterogeneous reassembly beats both full models”目前属于明显过度表述

这是我会在审稿意见中特别指出的措辞问题。

论文称：

> Teacher K + student V beats both full models.

依据是：

- Teacher full LL = −14.77
- Student full LL = −9.89
- K-only = −7.27

但同一个图里：

- Teacher full EM = **1.00**
- Student full EM = 0.43
- K-only EM = **0.76**



也就是说：

**K-only 的 gold-answer log-likelihood 更高，但任务 exact-match 明显低于 Teacher。**

作者自己随后也承认：

> teacher-forced answer log-likelihood is not a capability measure.

那就不应该再使用：

> “beats both complete models”

这么强的表述。

建议改成：

> **achieves higher gold-answer log-likelihood than both standalone runs**

而 heterogeneous reassembly 真正“超过 teacher”的结论，需要在 EM/F1/任务准确率等 functional metric 上成立。

------

# 七、统计分析还没有直接检验论文的“中心假设”

作者使用了 per-seed paired Wilcoxon，并要求 3 个 seed 都达到 p<0.05；这比很多论文认真，而且明确承认未做 multiple-comparison correction。

但是核心统计问题是：

论文要证明的是：

$K\ transfer > V\ transfer$

当前主要检验的却是：

$K\ vs Self$

以及：

$V\ vs Self$

“一个显著、另一个不显著”并不能证明两者之间显著不同。

应该直接做：

$D_i = (\Delta LL_K)_i-(\Delta LL_V)_i$

对每个样本进行 paired test，并报告 CI。

更进一步，既然已经有四臂，应直接检验：

$K,\quad V,\quad K\times V$

三个效应。

另外还有几个统计报告问题：

论文宣称“all bootstrap CIs are computed and reported per seed”，但主表实际只看到 mean±std 和 sig×3，没有看到完整 CI；而 EM 主表甚至只报告 seed 0。

建议最终主表统一报告：

**mean ± 95% CI over three splits + each-seed result + direct K-vs-V p-value。**

EM 应做 paired bootstrap/McNemar 或 non-inferiority test。

------

# 八、Layer 8 / 12 的热点结果很好看，但目前有明显“selection on test set”的风险

论文在 8B→0.6B 上扫描全部 28 层，然后指出 layer 8 和 12 是 hotspot，再组合 L8+L12 得到：

- L8：+0.31
- L12：+1.24
- L8+L12：+2.15
- All layers：−0.23

这是很有潜力的发现。

但论文没有清楚说明：

**L8/L12 是在 validation set 上选择，然后在 held-out evaluation set 验证，还是看完 evaluation set 后选出来的？**

如果是后者，+2.15 就包含明显的 post-selection bias。

而且扫描 28 层本身就是 28 次 comparison。

还存在一句机制性过度解释：

> “26 of 28 layers add noise.”

现有实验只证明了：

> 所有 28 层同时注入是 harmful。

它并不能证明另外 26 个 layer **每一个**都在“add noise”；可能存在复杂的正负 interaction。

建议用 validation-only layer selection，然后一次性冻结 L8/L12，在测试集和第二个 model pair 上验证。

------

# 九、Synthetic OOD 数据的描述远远不足以支撑可复现性

目前方法只告诉读者：

- 1–4 hop knowledge graph；
- train 70 / val 28 / test 56；
- entity-cluster split；
- 3 seeds。

缺失的内容很多：

- graph 如何生成；
- entity 如何生成；
- relation 类型；
- 每个 hop 的比例；
- document 模板；
- question 模板；
- answer normalization；
- prompt template；
- context length；
- 模型是 Qwen3 Base 还是 Instruct；
- thinking 是否开启；
- decoding 参数；
- calibration token 数；
- affine mapper 到底是 per-layer/per-head 还是 global；
- regularization；
- mapper fitting objective。

其中还有一句很值得作者核查：

> “A no-context probe on 30 nq_open questions confirms the entities are not memorized.”



如果这里真的是 `nq_open` Natural Questions，那么它并不能证明**synthetic entities**没有出现在 Qwen3 pretraining data 中。

而且严格意义上，作者也不可能知道 Qwen3 完整预训练语料，因此最好不要说：

> “entities do not appear in pretraining data”

而应改成：

> “synthetically constructed to minimize prior familiarity, with a no-context empirical probe showing low recall.”

这是更严谨的措辞。

------

# 十、SQuAD replication 是加分项，但目前远不足以支撑“跨域规律”

SQuAD 只使用：

- 30 questions；
- 8B→0.6B 一个 pair；
- 平均 document 119 tokens。

结果确实很清楚：

- Self EM ≈ 0.77
- K-only ≈ 0.78
- V-only ≈ 0.03
- Joint ≈ 0.68。

这个实验很好地表明了 V-only 在该 setting 下会 catastrophic failure。

但严格来说：

> K-only 并没有在 EM 上获得明显 improvement，而是基本“not harmful”。

所以建议论文区分：

**beneficial transfer** 和 **non-destructive transfer**。

另外，如果目标是 KV reuse serving，那么 119-token context 太短。至少需要加入：

- 1K；
- 2K；
- 4K；
- 8K+

不同 context length 的实验。

------

# 十一、2050-token cost crossover 目前更像“算术示意”，还不是 systems result

作者的计算是：

- mapper：58.78M 参数，fp32 = 224.2 MiB；
- KV：112 KB/token；
- 因此约 2050 token 交叉。

图 9 也明确说明量化和 mapper overhead 未考虑。

这里存在三个问题。

第一，为什么 KV 用 fp16/bf16 大小，而 mapper 强制按 fp32？

如果 mapper 用 fp16，交叉点大约直接减半。

第二，实际 serving 中 mapper 通常是**预部署并被大量请求摊销的静态参数**，并不是每个请求都要重新网络传输 224 MB。

所以：

> “send state vs send mapper weights”

并不完全对应典型 deployment decision。

第三，真正重要的是：

$T_{map}+T_{KV\ transfer} \quad\text{vs}\quad T_{student\ re-prefill}$

而不是纯字节数。

因此最好补真实 GPU serving benchmark：

- mapper latency；
- KV transfer latency；
- re-prefill latency；
- TTFT；
- throughput；
- context length scaling。

否则建议把“cost crossover”降级成 appendix/heuristic analysis，而不要作为五大贡献之一。

------

# 十二、缺少几个非常关键的 control experiments

这部分如果我是 reviewer，会明确要求补充以下实验，而不是单纯“希望未来做”。

1. **Wrong-document K / V**：把 sample A 的 teacher K 注入 sample B。若仍然有大幅增益，说明当前 effect 可能是 distribution/calibration effect，而非 example-specific information transfer。
2. **zero / random / moment-matched K/V**：确定性能变化确实来自映射后的真实 teacher state。
3. **identity/no-map baseline**：尤其 Qwen3 各模型 KV head 数和 head dimension 很接近，需要知道 pair-specific mapper 到底贡献多少。
4. **Joint across all six pairs**：做 K×V interaction。
5. **learned layer selection** 与 proportional mapping 对比：直接验证 V failure 是否只是 layer mapping 较差。
6. **held-out layer-hotspot discovery**：避免 L8/L12 的 post-hoc selection。
7. **larger cross-domain benchmark + long context**。
8. **至少一个 cross-family pair**，否则“across LLMs”标题略大，实际证据只是 “across Qwen3 sizes”。

其中 mismatched-cache control 尤其重要。近期独立研究已经专门指出，仅有 benchmark delta 并不足以证明 cache 中发生了 example-specific latent transfer，并采用 deranged-cache、zero-cache 和 moment-matched random-cache 做 causal audit。([arXiv](https://arxiv.org/abs/2608.04893?utm_source=chatgpt.com))

------

# 十三、从最新相关工作的角度，论文需要重新界定自己的 novelty

这是我额外查阅当前公开工作的结论，**不属于论文自身提供的证据**。

2026 年 8 月的 Heo et al. 已经在六个 pair、三个 model family 上研究 cross-model KV transfer，并观察到 K 的线性可预测性明显高于 V，同时报告实际 mapper 比 re-prefill 快 2.7–25×。([arXiv](https://arxiv.org/abs/2608.03893?utm_source=chatgpt.com))

Mixture-of-Translators 也已经覆盖 Qwen2.5、GPT-2、OPT 等 heterogeneous model transfer。([arXiv](https://arxiv.org/abs/2607.28979?utm_source=chatgpt.com))

因此这篇论文真正最有竞争力的 novelty 不是：

> “KV cache can transfer across models”

也不宜主要押在：

> “K and V have different geometry”

而应该明确收缩到：

> **首次系统地以功能干预的方式拆分 K-only / V-only，并揭示 K 比 V 更具有鲁棒的独立可迁移性，以及 V 对 architecture/alignment/K–V coupling 更敏感。**

这个定位反而更独特。

------

# 十四、标题也建议修改

当前：

> **Addressing Transfers, Content Does Not**

与论文自己的结果已经不完全一致：

- V 在 1.7B→0.6B 有明显成功；
- 8B→4B 仍能保持较高 EM；
- L12 单层 V 很强；
- L8+L12 为 +2.15；
- flagship Joint 中加入 Teacher V 反而让 +2.62 变成 +4.52。

所以标题虽然抓眼球，但容易被 reviewer 抓住作为 overclaim。

我更推荐：

**Keys Transfer Robustly, Values Conditionally: Asymmetric KV State Transfer Across LLMs**

或者更学术：

**Asymmetric and Interaction-Dependent KV State Transfer Across Language Models**

第二个标题其实更符合现有数据。

------

## 我认为最合理的论文重新定位

如果作者不想无限扩大实验，我反而建议**主动缩小理论宣称，把实验逻辑做深**。

当前论文的故事是：

> K = model-agnostic addressing
> V = weight-bound content
> capability gap 决定 V 是否能迁移。

我认为目前证据更支持：

> **Mapped K exhibits robust standalone functional portability within Qwen3. Mapped V is substantially less robust and is strongly conditioned on layer alignment, model pairing, and K–V interaction. Geometric linear similarity alone does not explain this functional difference.**

这个结论虽然看起来“弱了一点”，实际上更加扎实，而且后续理论空间更大。

------

# 最终审稿意见

**当前建议：Weak Reject / Major Revision，5/10。**

我认为论文不是“方向不成立”，而是恰恰相反：**作者已经拿到了一个可能比当前论文故事更有意思的现象，但没有按正确的统计和机制框架把它挖出来。**

其中最关键的不是继续堆更多模型，而是优先解决四件事：

**① 把四臂实验真正当成 2×2 factorial design，系统分析 K×V interaction；
② 解耦 capability gap、layer-count mismatch 和 layer-mapping error；
③ 重做 CCA/representation similarity 分析，不能只用 ρ₁≈0.99；
④ 用 non-inferiority、K-vs-V direct test、wrong-cache controls 等方式重新定义“functional transfer”。**

完成这四项后，我认为论文的核心故事有可能从现在较容易被质疑的：

> **“K transfers, V does not”**

升级为一个更有研究价值的结论：

> **“K provides a robust transferable routing substrate, whereas V transfer is interaction- and alignment-dependent; cross-model KV portability is therefore a coupled property of attention routing, content representation, and decoder compatibility.”**

这会比当前版本的理论解释更严谨，也更接近一篇顶会主会论文应有的贡献强度。