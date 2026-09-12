# V3 论文一审修复与优化方案

**日期：** 2026-08-31
**依据：** `paper/audit/audit1.md`（独立审稿，Weak Reject / Major Revision，5/10）
**执行状态：** 第一阶段（A 类）已完成（见文末 §7 与 `ITERATION_LOG.md` W15 条目）；
第二阶段（B 类 GPU 实验）待执行。
**前提核对：** 已通读 `main.tex`、`v3_data_baseline.md`、`ITERATION_LOG.md` 及
`reports/` 全部 JSON。本方案把审稿意见映射为三类动作：

- **A 类：已有数据可立即完成**（文本/统计/制图，无需 GPU）
- **B 类：需要补做实验**（GPU，可复用现有脚本与数据管线）
- **C 类：建议性增强**（可作为 rebuttal 承诺或在算力允许时完成）

---

## 0. 与审计前提不一致的两处数据现状（先澄清，避免方案落空）

1. **Joint 臂数据其实已经存在**：审稿意见一认为"Joint 没有全量报告"，
   但 `reports/phase4_scaling_law_v2_seed{0,1,2}.json` 已含全部 6 对 × 3 seed
   的四臂均值/EM/Wilcoxon（审计所依据的可能是旧版草稿）。**缺的是：**
   逐样本四臂 LL 行数据（仅 G0 flagship 的 `g0_v2_seed*.json` 有 `rows`），
   以及正式的 2×2 析因分析。补齐逐样本保存后，A 类即可完成。
2. **审计中 1.7B→0.6B 的 K/V 数字与现基线不一致**：审计写 K-only +0.63、
   V-only −0.23；权威基线（`v3_data_baseline.md`）是 K-only **−0.82**、
   V-only **+0.63**。方案以现基线为准；若审计依据旧版草稿，需在 response letter
   中说明数字版本差异，并附核对表。

---

## 1. A 类：论文文本、统计与制图修复（无需新 GPU 实验）

### A1. 把四臂实验正式做成 2×2 析因分析（审计 §1，最重要）

**动作：**
1. 修改 `phase4_scaling_law_v2.py`，在 JSON 中保存逐样本四臂 LL 与 EM（`rows`），
   重跑 3 seeds（或仅重跑并补齐逐样本，因为均值已存在）；
2. 新增 `scripts/factorial_analysis.py`，对每对 × 每 seed 计算：
   - K 主效应 = K-only − Self；
   - V 主效应 = V-only − Self；
   - K×V 交互 = Joint − K-only − V-only + Self；
   - 逐样本 bootstrap CI95（10,000 重采样）与配对显著性；
   - EM 上的对应交互（用 McNemar 或配对 bootstrap）；
   - SQuAD 域的同一套分析（`phase7_second_domain_seed*.json` 已有逐样本？若没有
     则同样补存）。
3. 正文新增 "K×V interaction" 子节 + 表 + 图（每对交互点估计与 CI95）。

**已算出的交互概览（均值口径，3 seeds）：**

| Pair | Seed0 | Seed1 | Seed2 |
|---|---|---|---|
| 8B→0.6B | +2.04 | +2.17 | +2.15 |
| 4B→0.6B | +1.85 | +2.78 | +1.81 |
| 1.7B→0.6B | −2.25 | −2.12 | −1.82 |
| 8B→4B | −2.57 | −2.69 | −2.75 |
| 4B→1.7B | −3.55 | −3.60 | −3.37 |
| 8B→1.7B | −1.15 | −0.76 | −4.03 |

**叙事修正（按审稿人建议但用真实数据）：**
- 大差距 0.6B 学生对：**正向交互**（Joint 显著超过 K-only+V-only 简单加和，
  Teacher V 在 Teacher K 条件下反而有用）→ "V 完全不传"不成立，
  更准确的说法是 **V 的边际贡献依赖 K 条件**；
- 1.7B 学生对与 1.7B→0.6B：**负向交互**（Teacher K + Teacher V 组合不如分开放）。
- 结论改写为：**K 具有稳定的独立可迁移性；V 的可迁移性高度依赖
  K–V compatibility 与 pairing**（审稿人 §1 建议的定位，且与 SQuAD 方向一致的
  负交互共同支持 interaction 是 domain-dependent 的）。

### A2. 直接检验中心假设 "K transfer > V transfer"（审计 §7）

**动作：** 新增统计脚本，对每个样本计算 D_i = ΔLL_K,i − ΔLL_V,i，
配对 Wilcoxon + bootstrap CI95，输出 6 对 × 3 seed 的完整表；
EM 侧用 McNemar（K-only vs V-only）。
正文把 "K 显著 / V 不显著" 的表述升级为 "K 与 V 的差异直接显著"。

### A3. 定义 transfer success criterion（审计 §3）

**动作：**
1. 在 Method 预注册成功判据（回顾性应用并明说）：
   - Retention = EM_transfer / EM_self ≥ 0.85；
   - 或 non-inferiority：EM_transfer − EM_self > −3pp；
   - LL 侧：ΔLL 显著为正 且 CI95 下界 > 0。
2. 用该判据重划 6 对结果的"成功/失败"标签，替换现有 "EM ≥ 0.71 算可传" 的事后
   口径；表 1 加一列 "Success (pre-registered criterion)"。
3. 对 SQuAD 区分 **beneficial**（K-only EM 提升）与 **non-destructive**
   （K-only EM ≈ Self）两种表述（审计 §10）。

### A4. CCA / 表示相似度重做（审计 §4）

现有数据：`w4_cca_perhead.json` 已含 6 对 per-head ρ₁（seed 0）。
**动作：**
1. 补算并报告完整 canonical spectrum（前 k 个典型相关系数、均值、
   SVCCA/PWCCA 摘要量、linear CKA、held-out Procrustes R²/重建误差）；
2. 补 seed 1/2 的 ρ₁（模型确定性高，至少报告 seed 0 全谱 + 3 seeds 的
   mean ρ₁ 稳定性）；
3. 把 "share nearly the same linear subspace" 降级为
   "high first-canonical correlation on a small number of dimensions"；
4. 新增 "representation similarity → functional transfer" 的预测性分析：
   以 per-layer/per-head 相似度（ρ₁、CKA）为自变量、per-layer 功能 ΔLL 为因变量，
   报告相关性与散点图（审稿人点名的"几何对齐不预测功能可迁移性"由此才有直接证据）。

### A5. 措辞与标题修订（审计 §6、§13、§14）

1. **Reassembly 表述**：删除 "beats both full models"，
   改为 "achieves higher gold-answer log-likelihood than both standalone runs"，
   并在同一段报告 EM：K-only 0.76 vs teacher 1.00 / student 0.43，
   明确 teacher-forced LL 不是 capability measure。
2. **"26/28 layers add noise"** 改为 "joint full-layer injection is harmful;
   individual-layer contributions interact, so noise attribution to specific layers
   is not established"（审计 §8）。
3. **novelty probe 措辞**：nq_open 无 context 探针只能说明"低回忆"，
   改口为 "synthetically constructed to minimize prior familiarity, with a
   no-context empirical probe showing low recall"（审计 §9）。
4. **capability-gated** 在解耦实验（B1）完成前全部改为
   **alignment-/architecture-dependent** 或 **pairing-dependent**（审计 §2）。
5. **标题**：推荐
   *Asymmetric and Interaction-Dependent KV State Transfer Across Language Models*；
   备选 *Keys Transfer Robustly, Values Conditionally: Asymmetric KV State Transfer
   Across LLMs*。Abstract/Intro/Conclusion 同步收敛到同一叙事。
6. **新增 contribution 定位**（审计 §13）：
   "首次以功能干预方式系统拆分 K-only/V-only，揭示 K 的独立可迁移性更稳健、
   V 对 alignment/K–V coupling 更敏感"——不与 Heo et al./MoT 撞车。

### A6. 方法/数据可复现性补全（审计 §9）

在 Method 或 Appendix 补：graph 生成规则（实体池、每项目 2 部件、
hop-1..4 模板）、document/question/answer 模板全文、answer normalization、
prompt 模板、decode 参数（greedy、max_new）、Qwen3 是否 instruct、
calibration token 数、mapper 是否 per-layer/per-head、正则化与拟合目标、
数据规模（train 70 / val 28 / test 56 per seed，而非正文残留的 42/14）。

### A7. 统计报告统一（审计 §7）

主表统一为 **mean ± CI95（3 seeds）+ 各 seed 值 + K-vs-V 直接 p 值**；
EM 表报告 3 seeds 而非仅 seed 0；所有 CI 为 per-seed bootstrap 10,000。
增加 multiple-comparison 的说明（28 层扫描 + 12 臂×对）或做
Benjamini–Hochberg FDR 校正。

### A8. 清理旧版文件

`section3_results.tex` 是 v1（n=14）陈旧草稿且未被 main.tex include，
与主文件数字冲突。归档到 `paper/archive/` 或删除，避免审稿人/协作者误用。

---

## 2. B 类：补做实验（GPU，可复用现有管线）

### B1. 解耦 capability gap 与 layer alignment（审计 §2，最关键的新实验）

目标：打破 "V 成功 ⇔ equal-layer" 与 "capability gap" 的混淆。

| 实验 | 操作 | 目的 |
|---|---|---|
| 28→28 错位映射 | 1.7B→0.6B 改用 offset/permuted layer map | 检验 equal-layer 是否真关键 |
| 36→28 learned selection | 8B→0.6B 用 val 集学习 top-k 源层（替代比例映射） | 检验失败是否来自比例映射 |
| 36→36 错位/偏移映射 | 8B→4B 加 offset layer map | 在 capability 固定的条件下操纵 alignment |
| 二维矩阵 | x = 参数比（2.0×–13.3×），y = alignment 质量 | 画 V transfer boundary 相图 |

实现：在 `phase4_scaling_law_v2.py` 的 `layer_map_generic` 加
`--layer-map {proportional,offset,learned-topk}` 分支，复用 mapper/KV 管线。

### B2. held-out 层热点发现（审计 §8）

- 明确协议：在 **val 集**（28 样本）上扫描 28 层 → 冻结 L8/L12 →
  在 **test 集**（56 样本）与第二个 pair（如 4B→0.6B 或 8B→4B）上验证；
- 报告两层选择在 val/test 的一致性与 post-selection 校正；
- 若 val 选择不稳定，改为 ridge/selective 正则化选择。

### B3. 机制实验：routing mismatch vs weight-bound consumption（审计 §5）

1. 比较 attention maps：A_S = softmax(Q_S K_S^T) 与 A_T' = softmax(Q_S K̂_T^T)；
2. 测试 A_S·V̂_T 与 A_T'·V̂_T 进入学生 W_O 后的表示/输出误差；
3. **W_O-aware V mapper**：直接最小化 attention-output 空间（O = A V W_O）的
   mismatch，而非 V 空间 MSE。若 W_O-aware mapper 恢复 V 功能，
   则 "V 失败来自权重绑定消费" 成立；若仍失败，则指向 routing/content 本身。

### B4. control experiments（审计 §12）

| Control | 内容 | 预期区分 |
|---|---|---|
| Wrong-document cache | sample A 的 teacher K/V 注入 sample B | 排除 calibration/分布效应 |
| Zero / random / moment-matched KV | 替换为 0、随机、匹配均值/方差的随机状态 | 证明增益来自真实 teacher state |
| Identity / no-map baseline | 教师原始 state 不映射直接注入 | 量化 mapper 贡献 |
| Joint × 6 pairs | A1 已覆盖 | 交互方向是否 domain-dependent |
| Cross-family pair | Llama/Qwen（可选） | "Across LLMs" 的边界 |

实现建议：新脚本 `scripts/control_cache.py`，复用 `phase0_g0_v2.py` 的
capture/build_cache/answer_loglik/greedy_answer。

### B5. 长上下文与第二域扩展（审计 §10）

- SQuAD 文档扩展至 ~1K / 2K / 4K（当前 119 token），看 K-only 是否保持
  non-destructive、V-only 是否持续 catastrophic；
- 第二域可加 GSM8K 或多文档 QA（算力允许时）。

### B6. Serving 成本测量（审计 §11）

若算力允许：在 GPU 上测 mapper 推理延迟、KV 注入延迟、student re-prefill
延迟与 TTFT/throughput，替换/支撑 2050-token 交叉点；同时补 fp16 mapper
（224.2 MiB → ~112 MiB，交叉点减半）与 mapper 预部署摊销的敏感性分析。
即使不测，也把 cost crossover 降级为 Appendix/heuristic（正文五大贡献删除该项或
改为 "deployment heuristic"）。

---

## 3. C 类：建议性增强（rebuttal 承诺 / 后续轮次）

- Cross-family（Llama 3.2-3B → Qwen3-0.6B 或 Llama 3.2-1B）：RoPE 配置不同，
  直接检验 "within-Qwen3" 边界；
- 更大统计功效：n=56 已可，交互项如需更高功效可扩样本；
- 非线性 mapper（MLP）一档，排除线性映射器容量对 V 边界的影响；
- 每 hop 分解（hop-1..4）报告 transfer 与推理深度的关系。

---

## 4. 建议实施顺序与工作量

### 第一阶段（1–2 天，纯文本/脚本，不依赖 GPU）
1. ~~A5 措辞/标题/叙事收敛（含 Interaction 新定位）~~ ✅ 已完成；
2. ~~A2 + A3 统计脚本与判据（复用现有汇总数据）~~ ✅ 已完成
   （`scripts/factorial_analysis.py` → `reports/factorial_analysis.json`）；
3. ~~A6 可复现性附录、A8 清理~~ ✅ 已完成
   （Appendix + `section3_results.tex` 归档至 `paper/archive/`）；
4. ~~A1 的析因分析脚本（先在 g0_v2 逐样本上跑通，产出旗舰 pair 结果）~~ ✅
   已完成（6 对均值 + 旗舰逐样本显著性）；
5. ~~重编译验证（`tools/compile_paper.ps1`）~~ ✅ 已完成（无 error）。

> 待第二阶段补：6 对逐样本四臂 rows（phase4 落盘）→ 全对交互显著性；
> SQuAD 逐样本 rows；CCA 全谱/相似度-功能预测（A4 的 1–4 项）。

### 第二阶段（3–5 天 GPU，优先级从上到下）
1. `phase4` 补存逐样本四臂 → 完成 6 对析因分析全表；
2. B1 解耦实验（28→28 错位、36→28 learned selection、36→36 offset）；
3. B2 held-out 层热点；
4. B4 控制实验（wrong-document / zero / moment-matched / identity）；
5. B3 机制实验（attention maps + W_O-aware mapper）。

### 第三阶段（可选）
B5 长上下文、B6 serving benchmark、C 类 cross-family。

---

## 5. 重定位后的论文骨架（对应审计 §13/§14 建议）

**主张：**
> Mapped K exhibits robust standalone functional portability within Qwen3.
> Mapped V is substantially less robust and strongly conditioned on layer
> alignment, model pairing, and K–V interaction. Geometric linear similarity
> alone does not explain this functional difference.

**五大贡献（v3 版）：**
1. 四臂分解 + 2×2 析因：K 稳健独立迁移，V 条件迁移（交互依赖）；
2. Correlation ≠ transferability（完整谱 + 相似度→功能预测分析）；
3. V 优势稀疏定位（L8/L12，held-out 验证版）；
4. 异质重组（LL 口径，明确非 capability 声明）；
5. 解耦后的 alignment/capability 相图（B1 产出）。

---

## 6. 验证门禁

- 所有正文数字与 `reports/*.json` 交叉验证（扩展 `scripts/verify_paper_numbers.py`，
  新增 interaction 表、K-vs-V 表、success criterion 列）——✅ 已完成，65+ 断言
  ALL PASS；
- pdflatex 无 error（Tectonic ✅）；禁词 grep（"beats both", "capability-gated",
  "near-universal" 按新口径核对）通过；
- ITERATION_LOG.md 追加本轮方案与执行结果——✅ W15 条目。

---

## 7. 第一阶段交付清单（2026-08-31）

1. `scripts/factorial_analysis.py` + `reports/factorial_analysis.json`：
   2×2 析因（K/V 主效应、交互、直接 K-vs-V、EM retention 判据）；
2. `paper/main.tex`：新标题/摘要/引言/贡献、§4.2 K×V Interaction Analysis
   （含表）、Transfer Success Criterion、CCA 表述修正、reassembly 口径修正、
   Appendix（可复现性）；
3. `paper/audit/response_to_audit1.md`：逐条回应（Phase 1 / Phase 2 标注）；
4. `scripts/verify_paper_numbers.py` 扩展（65+ 断言 ALL PASS）；
5. `paper/archive/section3_results_v1_stale.tex`：v1 陈旧草稿归档；
6. `ITERATION_LOG.md` W15 条目。
