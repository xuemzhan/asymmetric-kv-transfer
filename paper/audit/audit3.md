我重新按“独立审稿人、以 ICLR/NeurIPS 主会标准”为尺度，把这版 20 页论文与上一次审稿结论逐项对照了一遍。结论先说：

**这一版已经明显跨过了“核心实验链条不完整”的阶段。上一次我提出的几个 P0 问题，大部分都被正面解决了。现在如果投稿，我会从上一版的 5.5–6/10 上调到约 6/10，属于 Borderline / Weak Accept；但距离比较稳的 7/10 Accept，还剩 3 个真正需要处理的技术问题。**

其中最值得肯定的是：作者没有“维护旧结论”，而是在新证据出现后继续修正自己的解释。特别是 token-shuffle、layer alignment、adapter causal control、adapted Self、cross-domain repair 这些地方，这一版比上一版严谨很多。

---

# 一、总体评分：已经进入“有机会接收”的区间

| 评价维度    |   上一版 |        当前版 | 独立评价                           |
| ------- | ----: | ---------: | ------------------------------ |
| 问题重要性   |     9 |      **9** | 很强                             |
| 新颖性     |     8 |    **8.5** | evaluation audit + repair 有辨识度 |
| 实验设计    |     7 |      **8** | 已形成较完整诊断链                      |
| 技术严谨性   |     7 |    **7.5** | 大幅提高                           |
| 因果证据    |   6.5 |      **7** | equal-depth pair 已较强           |
| 统计严谨性   |     6 |    **6.5** | clustering 仍未完全解决              |
| 跨域/泛化   |     5 |    **5.5** | 诚实证明“fix 不泛化”                  |
| 写作可信度   |     9 |      **9** | 非常克制                           |
| 可复现性    |   6.5 |    **7.5** | provenance 明显加强                |
| 当前综合评分  | 5.5–6 | **约 6/10** | Weak Accept / Borderline       |
| 修完关键问题后 |     — | **约 7/10** | Accept 有竞争力                    |

现在论文最有价值的贡献已经不是“某种 K/V transfer law”，而是：

$$
\boxed{
\text{representation-space alignment}
\not\Rightarrow
\text{functional cache reuse}
}
$$

以及进一步证明：

$$
\boxed{
\text{functional reuse depends on where/how the receiver consumes the state}
}
$$

这条主线比最初版本成熟得多。

---

# 二、上一轮最关键的审稿意见，这一版解决情况很好

上一次我列的几个最关键问题，目前状态如下。

| 上次审稿意见                               | 当前处理                                                    | 我的判断                           |
| ------------------------------------ | ------------------------------------------------------- | ------------------------------ |
| corrected evaluator 只验证 6 个样本        | 已扩展到完整 56-item Self test set、3 个 student size           | **大幅改善**                       |
| shared K/V permutation control 无效    | 已明确承认 permutation invariance，并增加只打乱 K 或 V               | **完全解决**                       |
| adapter 是否真的读取 teacher content       | 增加 frozen-adapter + wrong/random/zero KV causality test | **equal-depth 基本解决**           |
| adapted 8B Self 缺失                   | 已补 adapted Self=0.940±0.041                             | **解决**                         |
| adapter 是否 arm-specific 不清楚          | 明确每个 training condition 独立训练一个 adapter，再测四 arms         | **解决**                         |
| layer-map null result 有 floor effect | 用 consumption-space mapper 后重做 scrambling               | **解决得很好**                      |
| 1.7B Self 低是否只是格式                    | raw / normalized EM 检查后仍低                               | **排除了 formatting explanation** |
| cross-domain repair 未验证              | 已在 SQuAD 上验证，而且结果为失败                                    | **科学上是加分项**                    |
| projection ablation 未完成              | 已补 v_proj，与 o_proj 同样强                                  | **明显改善**                       |

尤其 layer alignment 这一项，是这版非常漂亮的修正。

普通 affine mapper 下，V-only 本来已经接近 floor，所以 offset/random map 看起来“不影响”；但在 consumption-space mapper 把 V-only 拉到 0.911 后，同样的 scrambling 让 EM 跌到 0.107–0.232。

这非常有效地说明：

> 上一版的“alignment 不重要”实际上是 floor effect。

我认为这是当前论文里最扎实的 mechanistic result 之一。

---

# 三、但是 evaluator 问题虽然大幅改善，仍然是当前最大的潜在风险

作者现在已经在完整 56-item Self test set 上，将 cache-injection evaluator 与 full-prefill reference 做比较：

* normalized EM agreement：0.964 / 0.929 / 0.982；
* token-level greedy agreement：0.786 / 0.804 / 0.929；
* first-token logits 最大绝对误差：1.344 / 0.875 / 0.969；
* 最终 Self EM 最多只差 2 个样本。

这比上版的 6/6 probe 强太多。

但是作为 reviewer，我看到这里会产生一个新的问题：

## 理论上，为什么两个路径没有更接近“数值等价”？

如果：

* 是同一个 model；
* 是同一个 document；
* student 自己生成的 cache；
* position/cache position 相同；
* query 相同；

那么“document prefill → 保存 cache → query continuation”和“document+query standard causal forward”原则上应该非常接近。

因此：

> **first-token logit 最大差 1.344，并且 7%–21% 的 greedy token sequence 不一致**

不算特别小。

normalized EM 最后基本一致当然很好，但论文当前写：

> “the corrected path is end-to-end equivalent to full prefill on the quantity we report.”



这里的 **equivalent** 我认为稍强。

更准确的说法应该是：

> “closely agrees with full prefill on normalized answer EM.”

因为 token path 显然并不 equivalent。

### 我会要求作者继续追一下 1.344 的来源

至少检查：

$$
position\_ids
$$

$$
cache\_position
$$

$$
attention\ mask
$$

$$
RoPE\ positions
$$

$$
prefill/query\ boundary
$$

以及 bf16/fp16 numerical path。

最好报告：

$$
\operatorname{mean}|\Delta logits|
$$

$$
p_{95}|\Delta logits|
$$

$$
KL(p_{\text{full}}\|p_{\text{cache}})
$$

和 first-token top-1 agreement。

如果只是极少数 vocabulary logits 出现 1.34，而分布整体 KL 极小，那这个问题就基本关闭了。

如果不是，那 evaluator 可能仍然存在一个 residual implementation discrepancy。

### 还有一个非常便宜但应该补的实验

论文现在用 SQuAD 作为第二域，而且 corrected Self 从 legacy 的 0.767 变成 0.367。

那么我强烈建议：

**也在 SQuAD Self 上做 cache-injection vs full-prefill validator。**

因为 SQuAD 正是论文“第二域负结果”的基础。

这项工作量很小，但能显著降低 reviewer 的疑虑。

---

# 四、当前论文仍有一个很重要的“机制逻辑混用”：K routing 和 V-only recovery 必须拆开

这是我认为这一版需要认真修改的一个技术性问题。

论文在 Section 6.3 中报告：

mapped teacher K 会让 student routing 改变：

$$
top1\ agreement:
0.70\;(1.7B\rightarrow0.6B)
$$

$$
0.59\;(8B\rightarrow0.6B)
$$

同时 attention cosine 仍为 0.97 / 0.93。

这是一个好的 **K-side diagnostic**。

但紧接着文章讨论 consumer-space **V mapper**：

$$
V\text{-only}:
0.113\rightarrow0.935
$$

和：

$$
0\rightarrow0.250
$$



然后解释 residual gap 时写：

> “a value mapper cannot repair the addressing perturbation introduced by mapped keys…”



这里逻辑上有问题。

因为 **V-only arm 没有 mapped teacher K**。

根据论文自己的四臂定义：

$$
V\text{-only}=
K_S+\hat V_T
$$

也就是说 V-only 用的是 **student K**，其 routing 应该仍然是 student routing。

因此：

> 8B→0.6B V-only 的 residual gap 不能由 “mapped teacher K 导致 routing divergence” 来解释。

这是一个真正的技术逻辑错误，不只是措辞问题。

---

# 五、建议把“consumer compatibility”机制正式拆成两个不同问题

我觉得论文会因此反而更漂亮。

## K-side：routing compatibility

对于：

$$
K\text{-only}, Joint
$$

问题是：

$$
Q_S\hat K_T^\top
$$

是否还能生成与：

$$
Q_SK_S^\top
$$

相容的 attention routing。

你们现在的 top-1 / TV / cosine diagnostic 正是在测这个。

所以 K-side 应该明确写成：

$$
\boxed{\text{Mapped K can perturb student routing}}
$$

## V-side：consumption-space compatibility

对于：

$$
V\text{-only}=K_S+\hat V_T
$$

routing 本来就是 student 自己的。

它失败说明的是：

$$
\hat V_T
$$

虽然在 raw representation space 里可能接近 \(V_S\)，但经过：

$$
A_S\hat V_TW_O^S
$$

以后并不等价于：

$$
A_SV_SW_O^S
$$

所以 V-side 的真正故事是：

$$
\boxed{
\text{raw V alignment}
\neq
\text{student-consumable V alignment}
}
$$

consumer-space mapper 的强结果正好证明这一点。

这样一来整个机制结构就会非常清晰：

$$
\textbf{K-side: routing mismatch}
$$

$$
\textbf{V-side: value-consumption mismatch}
$$

$$
\textbf{Joint: both}
$$

比现在用一个“consumer incompatibility”把两者混起来更强。

---

# 六、Adapter causality 实验是巨大进步，但还没有完全验证“headline adapter”

这一版终于补上了我上一次最希望看到的实验：

**adapter 固定，只改变 injected state。**

1.7B→0.6B：

* correct teacher KV = 0.625
* wrong-document = 0
* random = 0.018
* zero = 0

这确实是非常好的 causal evidence。

我认同作者的判断：

> equal-depth pair 上，adapter 确实在读取 transferred teacher state，而不是纯粹学会了任务。

但这里还剩一个不小的问题。

### causality adapter 和 headline adapter 不是同一个训练配置

Causality test：

> rank 8，**10 epochs，seed 0**

而 Table 8 的 headline recovery 是：

> **20 epochs，3 seeds**

作者自己也解释了为什么正确 KV 的结果从 0.625 到 0.827/0.964 存在差异。

这意味着严格来说，目前证明的是：

> 一个 10-epoch adapter 有 content causality。

但论文 headline 用的是：

> 20-epoch adapter。

这两者不能完全画等号。

尤其 20 epoch 后，adapter 更可能吸收更多：

* task template；
* corpus regularity；
* answer distribution。

### 最干净的处理

直接拿 **Table 8 使用的最终 20-epoch adapter**，冻结以后做：

$$
correct
$$

$$
wrong\ document
$$

$$
random
$$

$$
zero
$$

最好跑 3 seeds。

这会一次性把这个问题彻底关闭。

我甚至认为，这比再加更多模型 pair 更值得做。

---

# 七、8B→0.6B 的 causality 仍然不能支持强机制结论

论文在这一点上已经非常克制：

8B→0.6B：

* correct = 0.304
* wrong = 0.143
* random = 0.161
* zero = 0.161

而且只有一个 seed。

文章正确地写：

> “flagship causality as unresolved.”



这个判断我完全赞同。

因此 Abstract 里这部分一定不能写成：

> “The adapter demonstrates content-specific transfer on both pairs.”

目前只能说：

> strong evidence on 1.7B→0.6B, suggestive but unresolved on 8B→0.6B.

实际上摘要已经基本这么做了，这一点处理得不错。

---

# 八、Cross-domain repair 失败不是论文缺陷，反而应该成为更明确的结果

这是这版很好的变化。

Synthetic-trained mapper 和 adapter 直接应用到 SQuAD：

| Condition                 | Own cache | K-only | V-only | Joint |
| ------------------------- | --------: | -----: | -----: | ----: |
| Default                   |     0.367 |      0 |      0 |     0 |
| Consumer mapper           |     0.367 |      0 |      0 |     0 |
| Synthetic-trained adapter |     0.600 |      0 |  0.033 |     0 |



这说明：

$$
\boxed{
\text{repairability}
\neq
\text{cross-domain generalization}
}
$$

我反而建议进一步强化这点，不要把它藏在 limitation 里。

这是一个很有价值的 negative result：

> consumer compatibility 很可能不仅是 architecture-specific，还是 distribution-specific。

这会使论文的结论更成熟。

不过，如果想冲更稳的 Accept，我建议新增一个更有判别力的实验：

## 在 SQuAD 内重新训练 repair，再测 held-out SQuAD

即：

$$
SQuAD_{calibration}
\rightarrow mapper/adapter
\rightarrow SQuAD_{heldout}
$$

这个实验回答的是：

> repair mechanism 是否能在第二个 domain 内复制？

它与现有：

$$
Synthetic\rightarrow SQuAD
$$

测试的是两个不同的问题。

当前只能得出：

> fix 不 cross-domain generalize。

如果加入 within-SQuAD repair，再得到恢复，则故事会变成：

$$
\text{mechanism replicates across domains}
$$

但：

$$
\text{repair parameters are domain-specific}
$$

这是一个非常有价值且非常合理的结论。

---

# 九、论文的一个潜在“顶会短板”仍然是：真正独立的 document 太少

虽然每 seed 有：

$$
n=56
$$

questions，但实际上只有：

$$
8\ unique\ documents
$$

平均每文档约 7 个问题。

对于 negative result：

$$
0.899\rightarrow0
$$

这不是大问题。

但对于：

* mapper improvement；
* adapter training；
* layer selection；
* causality；
* content-specificity，

8 个独立 document 就显得比较少。

论文自己也已经承认了 corpus template 与 clustering 问题。

### 从“投稿投资回报率”看

我不会建议现在去加 20 个 model pairs。

我更愿意看到：

> 56 questions 来自 30–50 个不同 document。

哪怕总 question 数不增加太多，也比继续增加 question/document 比例更有价值。

因为论文现在关注的恰恰是：

> **document state 是否真的 transferable。**

那么 independent unit 最自然应该是 document，而不仅是 question。

---

# 十、统计聚类问题已经被认识到，但最好不要继续只放在 Limitations

现在 seed 0 有 document-clustered CI，但 seed 1/2 因为报告生成时没有保存 identifiers，没有完整重算。

这种诚实说明很好。

但如果原始 dataset/split 还在，其实没有必要接受这个 limitation。

完全可以重新：

$$
question\rightarrow document\_id
$$

进行 reconstruction，然后为三个 seed 全部算：

$$
cluster\ bootstrap
$$

或者直接以 document 为单位 aggregation。

因为 test set 只有 8 documents，计算成本几乎为零。

这个问题对：

$$
0.899\ vs 0
$$

不会改变结论。

但是对：

$$
0.304\ vs 0.161
$$

或者：

$$
0.351
$$

这类中间结果会明显影响 uncertainty。

所以最好修。

---

# 十一、“state-space vs consumption-space”的核心概念还缺一张真正决定性的表

现在论文的核心理论说法是：

> affine mapper 在 representation space 拟合得很好，但 task performance 很差；consumer-space mapper 才真正恢复功能。

这个故事很好。

但目前 Table 4 只报告：

> 最终 EM。



我建议加一张非常重要的小表：

| Mapper          | raw-V reconstruction error | \(A_SV\) error | \(A_SVW_O\) error |    EM |
| --------------- | -------------------------: | -------------: | ----------------: | ----: |
| Affine          |                      **低** |              高 |                 高 | 0.113 |
| OutAware        |                       可能较高 |          **低** |                 — | 0.935 |
| \(W_O\)-aware   |                          — |              — |             **低** | 0.899 |
| Shuffled-target |                          … |              … |                 … | 0.077 |

这样论文一句话就能证明：

$$
\text{representation error}
\not\sim
\text{task performance}
$$

但：

$$
\text{consumption-space error}
\sim
\text{task performance}
$$

这会比现在纯靠 conceptual explanation 强很多。

我认为这是一个**非常低成本、高收益**的改进。

---

# 十二、Projection ablation 现在已经有价值，但建议继续保持“suggestive”，不要升级为机制证明

当前 seed 0：

* o_proj teacher KV：Joint 0.964
* v_proj teacher KV：Joint 0.964
* q/k teacher KV：明显更低

同时 student-trained adapter 在 o_proj/v_proj 上基本崩掉，而 q/k 更像 generic adaptation。

因此论文得出的：

> “value-consumption path 比 addressing path 更有效”

作为 **suggestive evidence** 是合理的。

但因为只有 seed 0，所以不要写成：

> “we localize the bottleneck to o_proj / V path.”

目前文章已经比较克制：

> “consistent with the bottleneck…”

这个措辞是正确的，应保持。

---

# 十三、1.7B Self 的异常低分已经不是 evaluator 的直接红旗，但仍值得解释

新版做了很好的排查：

* 0.6B normalized/raw = 0.911；
* 1.7B = 0.429；
* 4B = 0.804；
* normalization 没有把任何 wrong answer 翻成 right answer。

所以 1.7B 的 dip 不是简单 formatting artifact。

这一点已经比上一版好很多。

现在它可以留在 limitation，不再是 P0。

如果还有精力，可以做简单 error taxonomy：

* hallucination；
* wrong entity；
* wrong hop；
* refusal；
* reasoning verbosity；
* answer omission。

不用很大规模，抽 20 个 1.7B errors 就够。

目的不是解释 scaling law，而是让 reviewer 确认：

> 0.44 的确是 task accuracy，而不是另一个 pipeline artifact。

---

# 十四、当前标题仍然稍微偏强，我依然建议改

现在是：

> **Cross-Model KV Transfer Needs a Compatible Consumer: Diagnosis and a Consumption-Side Fix** 

“compatible consumer”容易让人理解成：

> 必须修改 consumer weights。

但论文自己已经证明：

1.7B→0.6B 的 V-only 可以从：

$$
0.113\rightarrow0.935
$$

而完全不改 consumer weights，只需要换 mapper objective。

所以我仍然更推荐：

> **Cross-Model KV Transfer Requires Consumer-Space Alignment: Diagnosis and Repair**

或者更保守、更像顶会论文：

> **Diagnosing Cross-Model KV Transfer: Evaluation Artifacts and Consumer-Space Alignment**

我个人更推荐第二个。

因为它准确覆盖了论文真正最强的两个贡献：

**evaluation audit + consumer-space alignment。**

---

# 十五、还有一个必须收紧的 framing：“standard protocol”到底指谁？

这是新版仍然容易被 reviewer 追问的问题。

论文多次说：

> “The standard protocol overstates transfer.”

同时又写：

> “we confirm both defects in the published evaluation code.” 

但后文又明确：

> “All EM numbers reported in the previous version of this work … are withdrawn.” 

因此读者必须非常清楚：

**这两个 evaluator bug 到底是：**

* 你们自己旧版本代码中的 bug？
* 某篇 prior work [8] 的公开实现？
* 多篇论文共同采用的实现？
* 一个 community baseline？

这是涉及 novelty 和科学责任的重要问题。

如果这两个 bug 只是你们自己的旧 evaluator，那么就不能写：

> “the standard protocol is broken”

应写：

> “the legacy evaluation implementation used in our previous version is broken.”

而：

> teacher-forced LL 不能代表 functional transfer

才是可以向更广泛文献推广的结果，因为有 wrong/random/zero cache controls 支撑。

建议把这两层明确分开：

$$
\textbf{Implementation-specific finding: evaluator defects}
$$

和：

$$
\textbf{Methodological finding: likelihood is not functional transfer evidence}
$$

这样会非常安全，也避免 reviewer 认为文章在夸大“纠错整个领域”。

---

# 十六、我现在认为论文最准确的核心科学结论应该重新写成这样

我建议不要再用一个笼统的：

> “state–consumer compatibility problem”

覆盖所有现象。

更精确的三层结构是：

### 第一层：Evaluation

$$
\boxed{
\text{Teacher-forced likelihood does not establish functional cache reuse.}
}
$$

wrong-document / random / zero controls已经很好地支持这一点。

### 第二层：Value transfer

$$
\boxed{
\text{Raw representation alignment is insufficient;
value transfer depends on alignment in the student's consumption space.}
}
$$

OutAware / \(W_O\)-aware mapper 是最直接证据。

### 第三层：Key transfer

$$
\boxed{
\text{Mapped teacher keys can perturb the student's routing.}
}
$$

top-1 / TV / cosine 是 diagnostic evidence。

然后：

### Joint handoff

$$
\boxed{
\text{Full KV handoff must satisfy both routing and value-consumption compatibility.}
}
$$

这个逻辑非常干净，而且完全由现有数据支持。

我认为这比目前的统一“compatible consumer”表述更有技术深度。

---

# 十七、如果我是正式 reviewer，我现在会给出的审稿意见

### Strengths

论文现在最大的优势是有一个真正完整的“发现错误—否定旧证据—定位失败—提出修复—测试边界”的研究链条。尤其 wrong-document/random/zero cache 对 LL 的反例非常强，consumer-space value mapper 的恢复结果也非常有说服力。论文还主动报告修复无法跨域泛化、flagship causality unresolved，这种结果边界控制值得肯定。

### Main concern

我目前最大的技术 concern 已经不再是“论文结论是否成立”，而是：

> **论文把 K-side routing mismatch 与 V-side consumption mismatch 在部分论证中混在了一起，而这两个现象在四臂设计里其实可以严格区分。**

特别是 V-only 没有 teacher K，因此不能用 teacher-K routing divergence 解释其 residual failure。这一处必须改。

第二个 concern 是 evaluator 尚未达到 numerical equivalence；normalized answer EM 很接近，但 7–21% token-level divergence 和最大 1.344 logit error 值得解释。

第三个 concern 是 causal content test 与 headline adapter 使用不同训练 schedule 且主要是单 seed，因此目前最强的 causality evidence 尚未完全覆盖最终 20-epoch结果。

---

# 十八、投稿前我建议只优先做 6 件事，不再大面积扩实验

1. **追清 evaluator 的 residual discrepancy。**解释 0.786–0.929 token-level agreement 和最大 1.344 logit error，补 KL/top-1/logit-error distribution，并在 SQuAD Self 上重复 full-prefill validator。

2. **重写 K-side / V-side mechanism。**不要再用 mapped-K routing divergence 解释 V-only residual；明确拆成 routing compatibility 与 value-consumption compatibility。

3. **用最终 20-epoch headline adapter 重跑 causality control。**同一个 frozen adapter 下测试 correct / wrong-doc / random / zero，最好 3 seeds；尤其补强 8B→0.6B。

4. **增加第二域的 within-domain repair。**用 SQuAD calibration 训练、SQuAD held-out 测试，判断 repair mechanism 是否可在第二 domain 重现；现有 Synthetic→SQuAD failure 保留。

5. **补 state-space / consumption-space loss 对照表。**把 raw reconstruction error、attention-output error、\(W_O\)-space error 与 EM 放在一张表中，直接支撑论文最核心的“where you align matters”。

6. **把 document-clustered inference 扩展到全部 seeds。**至少主结果和 adapter/causality 结果统一用 document-level bootstrap 或 clustered inference，避免 n=56 实际只有 8 个独立 document 的统计疑问。

---

# 十九、最终审稿判断

如果这一版今天直接投，我倾向于：

**6/10，Weak Accept / Borderline Accept，Confidence 4.5/5。**

严格 reviewer 仍可能因为：

* evaluator 数值路径尚非严格一致；
* synthetic corpus 只有 8 个独立 test documents；
* fix 不 cross-domain generalize；
* flagship causality 不够强；

给 5 分。

但与前两个版本最大的区别是：

**现在已经没有一个明显证据可以直接推翻论文主结论。**

当前剩下的主要问题是：

> 结论边界和机制拆分还需要更精确，而不是整个实验故事需要重来。

如果把前面前三项尤其是 **evaluator residual、K/V mechanism separation、20-epoch causality** 解决，我会比较有信心给：

**7/10，Accept。**

而且我认为最终最有竞争力的论文不是强调“KV transfer needs consumer adaptation”，而是强调下面这句话：

> **Cross-model KV transfer should be judged in the receiving model’s functional consumption space. Representation similarity and teacher-forced likelihood can both suggest transfer even when the transferred cache is unusable; functional reuse requires preserving the receiver’s routing and value-consumption compatibility.**

这已经是一条相当成熟、也比最初“K transfers, V does not”更有研究价值的结论。
