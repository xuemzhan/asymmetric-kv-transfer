# response_to_audit6 — 对 audit6.md 审稿结论的处置记录

**日期**：2026-11-22（W36）
**审稿文档**：`paper/audit/audit6.md`（6/10 Weak Accept；要求三条优先级改动）
**处置原则**：只落地 audit6 §13 明确要求的三类改动；不新增实验（本机无 GPU，
且审稿人明确说「此时已经不应该继续堆大量实验」）；不改动任何实验数字。

---

## 一、逐条对应

### 第一优先级：key-side「necessary but not sufficient」必须改（audit6 §6、§12、§13）

**审稿意见**：proving necessity 需要「在本来能迁移的条件下破坏 routing 并观察崩溃」，
本文做的是「routing 改善很多但迁移仍失败」，只能推出 not sufficient。

**处置：全部改为「not sufficient」，并把 necessity 明确写成未检验。**

| 位置 | 改前 | 改后 |
|---|---|---|
| Abstract | `closing the routing gap is necessary but not sufficient` | `routing alignment alone is not sufficient` |
| Contribution 2 | `routing term is necessary and not sufficient` | `Since reducing routing divergence leaves the task arm at the floor, routing alignment alone is not sufficient. We test no converse, so we do not claim routing compatibility is necessary---only that it is not sufficient.` |
| Discussion 7.1 (iii) | `not repairable by routing-space alignment` | `shrinking it from the routing side does not repair the task arm on this evidence; we claim no converse and cannot say that routing compatibility is necessary, because no experiment here destroys routing compatibility in a condition that otherwise transfers` |
| Chinese edition | 三处对应句 | 逐句镜像 |

**未做**：对称的 pre-softmax key mapper（audit6 §7 的「P0 最值得做的新实验」）。
原因：本机无 GPU（`nvidia-smi` 缺失、`torch 2.14.0+cpu`）。该实验以**明确的局限 +
两种可能结果的解读**写进 Discussion 7.3，而不是被静默略过：

> Either outcome would be informative, and we have run neither. If the symmetric
> objective left the key arm at the floor, ... the decomposition would be hiding a
> further query--key interaction. If it recovered the arm, the two sides would
> close symmetrically: routing-space alignment for keys and consumption-space
> alignment for values.

`paper/main.tex` 的 Discussion 7.3 同时还写了对称实验的闭式形式
$Q_S\hat K^{\top}M^{\top} \approx Q_SK_S^{\top}$，使后续实现无歧义。

注意：本文另有一处 `Layer alignment is therefore necessary but not sufficient`
（Analysis 6.1），指的是**层映射**而非 routing，证据是打乱层映射使 V-only EM 从
0.911 掉到 0.107--0.232 且区间不重叠——这是「破坏后崩溃」的形态，因此该处保留。

### 第二优先级：novelty 重写为四项，而不是 mapper / adapter（audit6 §5、§10、§13）

**处置**：

1. Introduction 的 enumerate 之后新增一段总结性声明：

> Taken together, what this paper claims is not a new translator. Receiver-aware
> translation already exists in concurrent work, and several systems beat us on
> downstream quality. The claim is a set of four things that hold together: the
> arm decomposition ...; a causal audit of the proxy that this literature
> evaluates with; an error law showing that the state-space and consumer-space
> error measures disagree in sign; and a boundary result ...

2. 贡献 3 改题并改述：**`Consumer-space alignment` → `An error law for the consumer space`**，
   并明确「we do not claim the target as new. What we add is directional evidence
   about the two error measures rather than the idea of fitting the receiver」。
3. 贡献 4 改题并改述：**`Repairability and its boundary` →
   `Repairability, used as an intervention`**，并加上审稿人建议的定位：
   「if a small receiver-side adaptation makes an unusable teacher state usable,
   the failure is producer--consumer incompatibility rather than absent
   information」。
4. Conclusion 同步：新增「Rather than the presence of a receiver-aware
   translator---which concurrent work now provides---the contribution is the
   decomposition, the audit of the proxy, an error law ..., and the boundary」。
5. `README.md` 的 artifact 说明与 headlines 同步；`experiments/phaseB_errorbudget.py`
   的 docstring 从 `raw representation alignment` 改为 `raw representation fidelity`。

**未做**：把 adapter 完全移出正文（审稿人没有这样要求，且 §10 明确说它作为
「科学验证工具」仍然有价值）。

### 第三优先级：标题（audit6 §9、§13）

**处置**：采用审稿人「目前最推荐」的方案。

- 改前：*Cross-Model KV Transfer Needs Functional Compatibility: Why Representation Alignment Is Not Enough*
- 改后：***Beyond Representation Fidelity: Functional Compatibility for Cross-Model KV Transfer***

同步位置：`paper/main.tex`（`\title`、`pdftitle`、`pdfkeywords`、文件头注释）、
`paper/main_zh.tex`（标题块与元数据）、`paper/arxiv_submission/main.tex`（由
`main.tex` 复制）、`README.md`、`paper/arxiv_metadata.md`。

理由：本文真正击破的是 state reconstruction distance / fidelity；改用 "fidelity"
后，标题不再字面否定 "alignment" 这一整条研究线（PRH、relative representations、
model stitching），也就不与 §2.5 和 Discussion 7.1 的立场自相矛盾。

---

## 二、审稿人未要求、但本次一并处理的连带项（audit6 §8）

**task-conditioned 从「定律」降为「证据支持的解释」**：

| 位置 | 改前 | 改后 |
|---|---|---|
| Abstract | `We name this \emph{task-conditioned} functional compatibility.` | `We read this as evidence that functional compatibility is \emph{task-conditioned} rather than a property of the model pair.` |
| §5.5 | `We name the phenomenon \emph{task-conditioned functional compatibility}` | `On the evidence of these two domains we describe the same producer--consumer pair as \emph{task-conditioned} functional compatibility` |
| arxiv_metadata 摘要镜像 | 同步 |

**未做**：controlled distribution shift（D0 → paraphrase → question-style → schema →
relation-vocabulary → SQuAD）。审稿人自己也标注「这很可能本身就值得下一篇论文」，
且属需新实验的 P2，不在本次范围。

---

## 三、守卫的相应更新（`paper/audit/verify_audit4_edits.py`）

两条断言随正文改动而失效，按本仓库既有惯例「被取代的旧断言重指向新措辞并注明原因」
更新，而不是放宽检查：

1. **T1**：原断言要求标题含 `Cross-Model KV Transfer Needs Functional`。新增
   `TITLE_W35 = "Beyond Representation Fidelity"` 与
   `TITLE_W33 = "Cross-Model KV Transfer Needs Functional Compatibility"`，
   改为「四处镜像必须含 W35 标题」且「四处都不许再出现 W33 标题」——检查强度只增不减。
2. **T2**：原断言要求两个旧贡献标题存在。改为新的两个标题，并加注释说明改名原因。

其余守卫（`verify_zh_parity.py`、`scripts/verify_corrected_paper.py`、
`verify_audit2_edits.py`、`verify_audit3_edits.py`）未改动，全部通过。

---

## 四、验证记录

- 编译（Tectonic 0.17.0）：`paper/main.pdf` **30 页**、`paper/main_zh.pdf` **31 页**、
  `paper/arxiv_submission/main.pdf` **30 页**；三份均 0 error、0 undefined
  reference/citation、0 Overfull `\hbox`、0 LaTeX Warning、0 缺字。
- 中文 parity：数字 1334、label 52、ref 72、eqref 1、cite 37、bibitem 36、
  图 8、表 11、章节 10 —— 与英文稿逐项相等。
- 五个守卫全部 exit 0。
- `main.tex` 与 `arxiv_submission/main.tex` SHA256 一致。

## 五、本次未做的事（如实记录）

1. 无新实验。四条 P1（行为式 key 目标、非线性 mapper、many-to-many 层映射、
   answer-margin 指标）与四条 P2 全部未做——本机无 GPU。
2. 未改任何实验数字。
3. 未做 controlled distribution shift（见上）。
4. 未检索非 arXiv 渠道（OpenReview / ACL Anthology / 正式论文集）；
   投稿前应补查 CacheLink / CacheBridge 是否已有正式版本。
