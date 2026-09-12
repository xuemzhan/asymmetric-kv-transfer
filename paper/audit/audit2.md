我重新按“独立审稿人、以 ICLR/NeurIPS 主会标准”为尺度审了一遍最新 16 页版本。与上一版相比，这已经不是简单修补，而是**研究问题和论文主线发生了实质性重构**：从“K transferable、V weight-bound”的现象论文，变成了“原评估存在缺陷 → 传统 transfer 证据并不成立 → failure 与 state–consumer compatibility 有关 → 通过 consumer-aware mapper / adapter 可以部分恢复”的诊断型论文。这个方向明显比上一版稳健，也更有研究价值。

但我目前仍不会给出明确 Accept。原因已经不是上一版那些明显的 K/V 解释问题，而是集中到了三个更深层的问题：**corrected evaluator 是否足够可信、所谓 consumer-side diagnosis 是否被真正因果识别、adapter 恢复的究竟是 transferred content 还是 task-specific adaptation。**

## 一、最新版本的总体评价

| 维度      | 上一版 |            最新版 | 我的评价                                    |
| ------- | --: | -------------: | --------------------------------------- |
| 研究问题重要性 |   8 |          **9** | 很有价值                                    |
| 新颖性     |   7 |          **8** | “evaluation audit + consumer repair”更独特 |
| 技术严谨性   |   6 |          **7** | 大幅改善                                    |
| 实验设计    |   5 |          **7** | controls 明显加强                           |
| 统计严谨性   | 4.5 |          **6** | 有改善，但仍有 cluster 问题                      |
| 机制解释    | 4.5 |        **6.5** | 明显收敛，但还没完全因果化                           |
| 写作与诚实度  | 7.5 |          **9** | 主动撤回旧结论是强项                              |
| 可复现性    | 4.5 |        **6.5** | 有 provenance，但 evaluator 验证仍弱           |
| 当前综合评分  |   5 | **5.5–6 / 10** | Borderline / Weak Reject～Weak Accept    |
| 审稿信心    | 4/5 |      **4.5/5** |                                         |

如果现在直接投稿，我大概率会给 **5 或 6 分**，取决于会议审稿人的风格。如果把下面前四个关键问题补齐，我认为论文完全有希望进入 **7/10 Accept** 区间。

---

# 二、上一轮我提出的主要问题，这一版解决得怎么样

这部分其实值得肯定，因为不少关键问题已经真正被处理，而不是只在文字上回避。

| 上一版问题                              | 最新版状态                             | 评价        |
| ---------------------------------- | --------------------------------- | --------- |
| K×V interaction 未分析                | 已增加 factorial decomposition       | **基本解决**  |
| 原 EM 是否真的表示 transfer               | 发现并修复 evaluator 两个缺陷              | **核心突破**  |
| LL 是否等价于 transfer                  | wrong-doc/random/zero controls 否定 | **解决得很好** |
| layer 8/12 是否 test-selection       | validation scan 选出 12/16          | **基本解决**  |
| capability gap 与 layer mismatch 混淆 | 增加 layer-map scrambling           | **部分解决**  |
| 机制表述过强                             | 明确区分 evidence / interpretation    | **明显改善**  |
| V“绝不迁移”的过度结论                       | 已完全撤回                             | **解决**    |
| “beats both models”的不当表述           | 已删除                               | **解决**    |
| cost crossover 被过度强调               | 降级为 completeness heuristic        | **正确处理**  |

尤其值得注意的是，新版明确承认之前所有 defective-path EM 均撤回，同时把 confirmatory、exploratory 和 withdrawn claims 分开写出来。 这实际上增强了论文可信度，而不是削弱。

更重要的是，修正后的六组实验完全改变了故事：例如 8B→0.6B 的 Self EM 是 0.899，而 K/V/Joint 全部变成 0；8B→4B 也只有 V-only 达到 0.440，仍远低于 Self 0.815。

这比上一版的“K transferable”结论科学上健康得多。

---

# 三、我目前认为最严重的问题：Corrected evaluator 还没有被验证到足以支撑整篇论文

这是我现在最担心的一点。

这篇论文当前最重要的结论全部建立在：

> 原 evaluator 有两个 defect，而 corrected evaluator 才是真正正确的。

方法部分描述得很清楚：旧路径在 scoring 时已经把 query 和 gold answer 推进 cache，之后又在同一个 mutated cache 上 greedy generation；同时 generation 的首 token 还重复输入了最后一个 query token。新版改成 LL 和 generation 分别从 fresh cache 开始，并从 prefill logits 取第一个生成 token。

这个 diagnosis 很有说服力。

问题在于，目前 corrected evaluator 只和 full-prefill reference 做了：

> **6 个样本，6/6 exact-match agreement**

而且论文自己承认：

> 这 6 个 item 没有保留下来，不能构成 broad validation。

到 Limitations 又再次承认这一点。

这对普通实验来说可以接受，但对于这篇论文不够。

因为整篇文章最重大的主张就是：

> “过去看到的 transfer 主要是 evaluator artifact。”

所以审稿人很容易问：

**为什么我要相信第二个 evaluator，而不是第一个 evaluator？**

尤其新版中出现了一个很值得警惕的现象：

* 0.6B student Self ≈ **0.899**
* 1.7B student Self ≈ **0.440**
* 4B student Self ≈ **0.815**

也就是说 1.7B 比 0.6B 的任务准确率低一半左右。作者明确说：

> “Our sources do not explain this gap.” 

这个现象非常不自然。

它未必表示 evaluator 还有 bug，但它意味着必须排除：

* chat template 差异；
* thinking mode；
* EOS 行为；
* answer formatting；
* tokenizer；
* generation stopping；
* answer extraction；
* Qwen3 不同 size 的生成模式差异。

### 我认为这是投稿前必须补的 P0 实验

不是再抽 6 个，而是直接把：

$$
\text{cache injection evaluator}
$$

和

$$
\text{full concatenated prefill reference}
$$

在**完整 Self test set**上逐样本比较。

至少应该报告：

$$
\text{token-level generation agreement}
$$

$$
\text{first-token logits max error}
$$

$$
\text{final normalized EM agreement}
$$

最好做到全部 6 pair × 3 seeds 或至少所有 student model。

如果 corrected evaluator 真正确，就应该基本达到接近 100% agreement。

这一个实验，会极大提高整篇论文的可信度。

---

# 四、Token-shuffled student cache 这个 control 很可能在数学上是无效的

这是新版中我认为最容易被强审稿人抓住的技术点之一。

论文写道：

> token-shuffled student cache 保持 EM 0.890±0.040，而 intact student cache 为 0.899±0.010，因此 evaluator 能识别 usable context。

问题是：

如果所谓 token shuffle 是对 K 和 V 的 token 维度做**相同排列**，

$$
K' = PK,\qquad V'=PV
$$

那么 attention：

$$
\mathrm{softmax}(qK'^T)V'
$$

等价于：

$$
\mathrm{softmax}(qK^TP^T)PV
=
\mathrm{softmax}(qK^T)V
$$

换句话说：

**attention 对 K/V pair 的共同 permutation 本来就是 permutation-invariant 的。**

即使 K 中已经包含 RoPE，只要 positional information 跟着 K vector 一起移动，改变 cache tensor 中行的存储次序，理论上也不会破坏其内容。

所以如果代码里做的是：

```text
K = K[:, permutation, :]
V = V[:, permutation, :]
```

那么 EM 仍然 0.89 不仅“不意外”，而且这个实验**几乎不能作为“context is still usable”的控制证据**。

这需要作者立即核查。

如果其实使用的是：

* K 和 V 独立 shuffle；
* V-only token shuffle；
* K/V correspondence 打乱；
* positional assignment 重排；

那必须在正文写清楚。

如果确实是 joint permutation，我建议把这个 control 删除，替换成真正破坏内容的：

$$
K_t,\;V_{\pi(t)}
$$

或者：

$$
K_{\pi_1(t)},\;V_{\pi_2(t)}
$$

其中 \(\pi_1\neq\pi_2\)。

这个问题本身不会推翻论文主要 negative result，因为 wrong-document/random/zero controls 更重要，但它会削弱作者目前用来验证 evaluator sensitivity 的论证。

---

# 五、“failure is consumer-side”比上一版合理很多，但现在仍然没有被完全因果识别

这是论文目前最重要的理论问题。

新版主张是：

> cross-model KV transfer failure 是 state–consumer compatibility problem。

这个表述比上一版的“K addressing universal / V weight-bound”好很多。

但当前证据其实更精确地支持的是：

> **standard representation-space mapping is insufficient; consumer-aware alignment greatly improves usability.**

两者不是完全一回事。

因为论文最强的 constructive result 之一并没有修改 consumer：

attention-output-aware value mapper：

$$
0.113\rightarrow0.935
$$

WO-aware：

$$
0.113\rightarrow0.899
$$

在 1.7B→0.6B 上已经恢复到了 Self≈0.899。

也就是说：

**只改变 mapper fitting objective，就几乎完全解决 V-only。**

consumer weights 根本没有变化。

因此这个结果同样可以解释为：

> 原来的 affine mapper 优化了错误的 representation-space objective。

而不是：

> consumer 本身“不兼容”。

我建议把论文中的机制分成两个层次：

$$
\text{state-space compatibility}
\neq
\text{consumption-space compatibility}
$$

第一层 conclusion 可以很强：

> Matching raw KV representations is insufficient; what matters is alignment in the space where the student consumes them.

第二层才是：

> If mapping alone is insufficient, a small consumer adaptation can further repair the mismatch.

这样逻辑会更严谨。

从这个角度，我甚至认为标题可以比现在更准确：

> **Cross-Model KV Transfer Requires Consumer-Aware Alignment: Diagnosis and Repair**

会比：

> *Needs a Compatible Consumer*

更贴合实验。

---

# 六、Adapter 是目前论文最有价值的正结果，但也最需要补关键 control

现在 adapter 的结果很漂亮：

1.7B→0.6B：

$$
K/V/J:
0.030/0.113/0
\rightarrow
0.940/0.940/0.827
$$

8B→0.6B：

$$
0/0/0
\rightarrow
0.792/0.357/0.470
$$



这是足够成为论文第二核心贡献的。

但目前最大的替代解释是：

> adapter 可能学习的是任务，而不仅仅是在学习如何读取 transferred state。

因为 adapter 是通过：

> next-token cross-entropy on calibration split

训练的。

而 synthetic corpus 又高度规则化：

* project-status template；
* 70 training questions；
* 8 个 test documents；
* 平均每 document 约 7 个 questions。

论文已经加入了 student-KV adapter 和 shuffled-document teacher-KV adapter，这很好。

但是现在最关键的 control **还缺一个**：

### 用 content-matched teacher-KV 训练好的 adapter，在 test 时喂 wrong-document / random / zero KV

也就是保持：

$$
A_{\text{teacher-trained}}
$$

完全不变，然后测试：

$$
\text{correct teacher KV}
$$

$$
\text{wrong-document teacher KV}
$$

$$
\text{random KV}
$$

$$
\text{zero KV}
$$

如果结果是：

$$
EM_{\text{correct}}
\gg
EM_{\text{wrong}}
\approx
EM_{\text{random}}
\approx
EM_{\text{zero}}
$$

那么才能真正说明：

> adapter 恢复的是 transferred document information。

这是比“用 shuffled teacher states 训练另一个 adapter”更干净的 causal test。

现在 shuffled-trained adapter 仍然达到 Joint 0.458（1.7B）和 0.131（8B）。

这已经表明 generic teacher-state adaptation 占了相当一部分效果。

所以 inference-time content destruction test 非常必要。

---

# 七、Table 5 缺失 8B→0.6B 的 adapted Self，是当前结果表一个明显漏洞

Table 5 明确写：

> 8B→0.6B 的 Self column 没有保留下来，所以省略。

这个数据必须重跑。

因为如果 adapter 修改了 student 的 o_proj，那么应该比较的不是：

$$
EM_{\text{adapted transfer}}
$$

和原模型的：

$$
EM_{\text{unadapted Self}}=0.899
$$

而应该是：

$$
EM_{\text{adapted transfer}}
$$

对：

$$
EM_{\text{same adapted model, Self cache}}
$$

例如如果 adapted model 的 Self 只有 0.75，那么 K-only=0.79 已经完全恢复甚至超过；

如果 adapted Self=0.98，那么 0.79 仍然有明显 gap。

没有这个数，8B→0.6B 的 “recovers most of the K arm” 其实缺少正确 reference。

这是一个很容易修，但投稿前必须修的问题。

---

# 八、Adapter 到底是不是“一个 adapter”，当前正文有歧义

这部分需要作者非常明确。

正文写：

> a rank-8 correction ... restores K/V/joint EM ...

容易理解为：

> 训练一个 adapter，然后同时适配 K/V/Joint。

但后面又出现：

> “joint-trained condition”

例如 o_proj joint-trained joint arm 为 0.964。

这让我怀疑实际上可能是：

* K-only adapter 在 K-only cache 上训练；
* V-only adapter 在 V-only cache 上训练；
* Joint adapter 在 Joint cache 上训练。

如果是这样，必须明说。

因为两个结果的工程含义完全不同。

一个 universal adapter：

$$
A(T,S)
$$

能兼容多种 transferred state，非常强。

三个 arm-specific adapter：

$$
A_K,\quad A_V,\quad A_{KV}
$$

仍然有价值，但成本、部署和科学结论都不同。

建议 Table 5 增加一列：

**Training injection arm**

并在 Method 明确：

> Each cell uses a separately trained adapter

或者

> The same adapter is shared across all three evaluation arms.

不能让 reviewer 猜。

---

# 九、“Layer alignment is not the controlling variable”这一结论目前还有 floor-effect 问题

新版为了解决上一版 layer alignment/capability confound 做了很好的尝试。

1.7B→0.6B 中：

* identity V EM：0.05–0.14；
* offset ±3：0；
* random permutation：0。

8B→0.6B 中不同 mapping 也基本没有产生 task transfer。

问题在于：

**baseline 本来就几乎已经在 floor。**

从：

$$
0.05\rightarrow0
$$

不能很强地证明：

> layer mapping doesn't matter。

因为它可能只是：

> 所有 mapping 都差到低于任务阈值。

论文同时用 LL 来支持 scrambling 不伤害，但这篇论文自己已经证明 LL 不是 transfer metric。

所以这个 argument 存在一点内部张力。

一个非常好的补实验是：

先使用已经恢复 transfer 的：

> attention-output-aware mapper

使 V-only 达到：

$$
EM=0.935
$$

然后重新做 layer scrambling。

如果：

$$
0.935
$$

在 random layer map 下仍接近：

$$
0.9
$$

那才能真正证明 layer alignment 不重要。

如果突然跌到 0.1，那么说明 layer alignment 其实仍然重要，只是原始 mapper 条件下看不出来。

因此我建议把 Section 6.1 当前标题：

> Layer alignment is not the controlling variable

稍微降调为：

> **Layer alignment alone does not explain the failure**

在补完 repaired-regime experiment 前，这个版本更加准确。

---

# 十、Routing divergence 是很好的 diagnostic，但还不能叫 mechanism

这版在这一点上已经非常克制，正文明确写：

> 这是两个 co-varying quantities 的解释，并不是 independently identified mechanism。

这个表述很好，应保留。

目前看到：

1.7B→0.6B：

$$
top1=0.70,\ TV=0.165,\ cosine=0.97
$$

8B→0.6B：

$$
top1=0.59,\ TV=0.245,\ cosine=0.93
$$

同时后者 mapper recovery 更差。

这是有启发性的。

但如果希望把论文进一步提高一个档次，我会建议做：

$$
routing\ divergence_i
\rightarrow
task\ failure_i
$$

的 sample-level 或 layer-level predictive correlation。

更强的是直接优化：

$$
\|Q_SK_T'^T-Q_SK_S^T\|
$$

做一个 attention-logit-aware K mapper。

论文 Future Work 已经提出这一方向。

如果能在投稿前完成，它可能会成为很强的机制闭环：

$$
K\ mapper
\rightarrow routing\ recovery
\rightarrow EM\ recovery
$$

比现在单纯 observation 强很多。

---

# 十一、统计问题比上一版好，但还没有完全过关

最新版已经意识到 test set 实际只有：

> 56 questions / 8 unique documents

所以 question 并非独立样本。

并且 control suite 开始使用 document-clustered bootstrap。

这是正确方向。

但 main four-arm EM 表仍然只有 seed std，而作者自己承认：

> clustering 后 uncertainty 很可能更宽，因为 archived summary 没有 document identifiers。

这里我建议不要留作 limitation，直接重跑。

因为数据规模本来就不大。

主结果至少应该报告：

$$
95\%\ cluster\ bootstrap\ CI
$$

cluster unit = document。

另外当前不少 Wilcoxon 是：

$$
n=56
$$

的 sample-level paired test。

如果同一 document 下 7 个问题高度相关，这个 p-value 会过于乐观。

更合适的是：

* document-level paired means；
* clustered permutation；
* cluster bootstrap；
* 或 mixed-effects model。

好消息是很多关键结论不依赖 p-value——例如：

$$
0.899\ vs\ 0
$$

无论怎么聚类都不会改变判断。

但为了论文规范，最好全部统一。

---

# 十二、Self baseline 的非单调现象必须解释

我会把这个单独列出来，因为 reviewer 很可能注意到。

三类 student 的 corrected Self 是：

$$
0.6B:0.899
$$

$$
1.7B:0.440
$$

$$
4B:0.815
$$



这个顺序非常反常。

论文现在说它“不影响 within-pair comparison”，逻辑上没错。

但它可能揭示的是：

> EM metric 受到 model-specific answer style 影响。

特别是 Qwen3 这类模型，generation formatting / reasoning style 很可能随模型尺寸改变。

建议至少加入：

* normalized EM；
* token F1；
* answer extraction 后的 EM；
* short-answer accuracy；
* generation length statistics。

最好给出 1.7B 错误案例分析：

到底是：

> 完全回答错；

还是：

> “The answer is X.” 被 raw EM 判错？

如果主要是格式问题，那么当前的 0.44 并不是模型能力差，而是 metric issue。

鉴于这篇论文的主题本身就是“评价指标有问题”，这一点尤其不能含糊。

---

# 十三、SQuAD 的 negative replication 是强项，但 constructive fix 还没有跨域验证

SQuAD 当前结果：

$$
Self=0.367
$$

$$
K=V=Joint=0
$$

而 K-only LL 仍然 +5.24。

这个结果非常有价值：

> LL/EM dissociation 不是 synthetic task 特有现象。

所以 negative half 已经有第二域 replication。

但 positive half：

> consumer-aware mapper / adapter restores transfer

还只在 synthetic project-status corpus 上验证。

而 adapter 又明确：

> trained and evaluated on the same task distribution. 

因此目前论文实际上证明的是：

> **failure generalizes better than the fix.**

如果资源允许，我认为最值得新增的一个实验是：

**synthetic 上训练 mapper/adapter → 直接测试 SQuAD，不做 task-specific retraining。**

即使结果只恢复到：

$$
0.10\sim0.20
$$

也很有信息量。

如果完全为 0，则说明 adapter 主要是 task-distribution-specific。

两种结果都值得发表。

---

# 十四、Projection ablation 现在不应该以“实验还在跑”的状态出现在正式稿

正文写：

> v_proj condition was still running when this version was prepared. 

正式投稿版最好绝对不要保留这句话。

要么把：

* o_proj；
* q_proj/k_proj；
* v_proj；
* qkv；
* MLP control

完整做完；

要么直接删掉不完整 ablation，只保留：

> preliminary seed-0 evidence suggests o_proj is stronger.

否则会给 reviewer 一种：

> submission is not experiment-complete

的感觉。

特别是你现在 claim 的机制是 consumption stage，那么 projection ablation 正好是最重要的 mechanistic evidence，不应该半完成。

---

# 十五、论文现在最大的优点，其实是“negative result + repair”结构已经形成了

我认为最新版本真正有潜力的故事不是：

> “K/V 哪一个 transferable？”

而是下面这个三段式：

$$
\boxed{
Raw\ state\ similarity
\not\Rightarrow
functional\ cache\ reuse
}
$$

然后：

$$
\boxed{
Teacher\text{-}forced\ LL
\not\Rightarrow
content\ transfer
}
$$

最后：

$$
\boxed{
Functional\ reuse
requires
consumer\text{-}aware\ alignment
}
$$

其中第二条证据尤其强。

8B→0.6B：

* real K：+2.62；
* wrong-doc K：+2.61；
* random K：+2.33；
* zero KV：+2.75；
* 但 corrected EM 全部约 0。

这是非常干净、非常值得保留在论文核心位置的一张表。

相比之下，上一版 CCA=0.99 那条故事反而没那么重要了。

我甚至建议整篇文章再进一步压缩：

**少讨论旧的 K/V asymmetry，多突出“functional validation of cache reuse”。**

---

# 十六、标题和摘要还可以再收敛一点

现在标题：

> **Cross-Model KV Transfer Needs a Compatible Consumer: Diagnosis and a Consumption-Side Fix**

已经比上一版好很多。

但“Needs a Compatible Consumer”仍然稍微偏强，因为 consumer-aware **mapper** 本身已经能恢复 1.7B→0.6B 的 V transfer，而无需修改 consumer。

我更倾向：

> **Cross-Model KV Transfer Requires Consumer-Aware Alignment: Evaluation, Diagnosis, and Repair**

或者：

> **When Mapped KV States Fail to Transfer: Consumer-Aware Alignment Across Language Models**

第一个更准确。

摘要中这句：

> “A rank-8 correction ... trained only on calibration teacher states”

也建议修改。

因为实际上使用了：

> next-token cross-entropy

也就是用了 supervised targets。

建议改成：

> “trained with next-token cross-entropy on calibration examples under injected teacher KV”

避免让人误解成纯 state-only / label-free adaptation。

---

# 十七、如果我是正式 reviewer，我现在会给出的 Major Concerns

我会把需要修改的优先级排成下面这样：

| 优先级    | 问题                                             | 投稿前建议                                                             |
| ------ | ---------------------------------------------- | ----------------------------------------------------------------- |
| **P0** | corrected evaluator 仅 6-case 验证                | **完整 test-set 对 full-prefill reference 验证**                       |
| **P0** | token-shuffle control 可能 permutation-invariant | **核查代码，必要时换 K/V mismatch control**                                |
| **P0** | adapter 是否真正利用 document content                | **matched-trained adapter + wrong/random/zero inference control** |
| **P0** | 8B adapted Self 缺失                             | **重跑补齐**                                                          |
| **P1** | adapter 是否 arm-specific 不明确                    | **方法和表格明确说明**                                                     |
| **P1** | layer-map 结论有 floor effect                     | **在 OutAware/recovered regime 重做 scrambling**                     |
| **P1** | 1.7B Self 反常低                                  | **解释并增加 normalized EM/F1**                                        |
| **P1** | document clustering                            | **所有 main result 重算 cluster CI**                                  |
| **P1** | constructive fix 仅 synthetic                   | **至少做一次跨域 recovery**                                              |
| **P2** | routing 仅相关性                                   | 加 routing-aware K mapper                                          |
| **P2** | projection ablation 未完成                        | 做完或删除                                                             |
| **P2** | 系统价值仍只有 byte heuristic                         | 后续补 TTFT / latency                                                |

---

# 十八、我建议作者现在不要继续大规模“堆模型”，而是补最有判别力的 4 个实验

现在已经不需要再补十几个 pair。

真正可以决定审稿结果的是：

**第一个：Evaluator gold-standard validation。**
完整 Self set 与 full-prefill reference token-by-token 对齐。

**第二个：Adapter content causality。**
teacher-trained adapter 在正确 / wrong-doc / random / zero KV 下测试。

**第三个：Repaired-regime layer ablation。**
OutAware mapper 已经把 V EM 提到 0.935 后，再 scramble layer mapping。

**第四个：Cross-domain repair。**
至少验证 OutAware mapper 或 adapter 是否能在 SQuAD 上恢复非零 functional transfer。

如果这四个结果都支持当前故事，论文会非常完整：

$$
\text{Evaluation artifact}
\rightarrow
\text{negative transfer result}
\rightarrow
\text{diagnosis}
\rightarrow
\text{causal repair}
\rightarrow
\text{out-of-domain validation}
$$

这是一条很漂亮的顶会故事线。

---

# 十九、我的最终审稿结论

**当前评分：5.5–6/10，Borderline，明显高于上一版。**

我不会再因为“核心 hypothesis 本身站不住”拒稿。现在论文的核心方向已经是成立的，而且有几个非常好的实验，尤其是：

* corrected evaluator；
* wrong-document/random/zero controls；
* LL 与 functional task performance 的明确解耦；
* consumer-aware value mapping；
* rank-8 consumer adaptation；
* 对旧结论的主动撤回和 confirmatory/exploratory 区分。

这些使论文从一篇容易被 reviewer 质疑的“现象归纳论文”，变成了一篇真正有可能形成影响力的 **methodological audit + mechanism/repair paper**。

但目前我仍会保留一个核心审稿意见：

> **论文已经有了一个很强的 diagnosis，但目前“corrected evaluator is correct”以及“recovery really comes from reading transferred content”这两件事，还没有达到与论文结论强度相匹配的证据标准。**

一旦把这两个链条闭合，我认为这篇论文最合理的最终核心结论应该是：

> **Cross-model KV reuse should be evaluated by whether a receiving model can functionally consume the transferred state, not by representation similarity or teacher-forced likelihood. Raw state-space alignment is insufficient; transfer becomes usable when alignment or adaptation is performed in the consumer’s functional space.**

这个表述比上一版的“K transfers, V does not”成熟很多，也更可能经得起顶会审稿。
