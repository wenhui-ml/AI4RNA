# RNA 逆设计自改进 — 提升解题率 × 论文验证 联合路线图 (Paper C)

> 配套文件:`POSITIONING.md`(内部去碰撞 vs A1/A2/A3/B)、`rna-selfimprovement.tex`(draft)。
> 外部 arXiv 校准见 memory `project-paperC-lit-calibration`。本文件是执行计划,边跑边勾。

## 0. 一句话原则(短期↔长期的和解)

**不追 Eterna100 那个数字本身**(长尾堆算力 = 本末倒置,查实 24 题 headroom 是 solver-hard,SOTA
DesiRNA 要 24h/题)。**改为:测"哪个自改进 mode 能把去脚手架地板上的解题率抬起来、抬多少、为什么"。**
每个 mode 在诚实地板上产生的 Δaccuracy,**既是短期的解题率提升,也是长期论文的实验证据**——同一批
实验,两种读法。

## 1. 双目标映射

| | 短期读法(提解题率) | 长期读法(论文验证) |
|---|---|---|
| 去脚手架地板/天花板 | 诚实的能力基线 | 中心论点:脚手架已编码 procedure → 自改进环测出 null 是**混淆伪影** |
| 记忆 mode(agent 自写) | 记忆能不能抬地板 | M1:null 是真 null(机制=冗余),多种子确认 |
| 工具获取 mode | 撞墙后自主装 solver 能不能抬 | 新增一个**被测 mode**;结果两头都干净 |
| 算法自优化 mode | 进化出更强 design 算法能不能抬 | 唯一可能的**正结果**,平衡 M1 的 null |
| 长尾归因 | (不追)解释为什么抬不动 | designability split:绑死的是搜索收敛,不是策略/推理 |

## 2. 约束与红线(全程生效)

- 实验一律 **tmux + conda `rna`**;共享机做**礼貌 CPU 邻居**(不抢他人 ray/vllm/nemo-rl 作业)。
- **红线**:记忆只能是蒸馏的策略/过程,**绝不能是答案序列**;**禁止 web-search/背诵已发表 Eterna100
  答案**。工具获取放开的是**求解器**(DesiRNA/NEMO 是 de novo solver,不是答案库),**不是答案 fetch**。
- solved = 折叠 oracle(ViennaRNA 2.7.2, Turner2004, 37℃, dangles=2)下设计正确;**不做体内/功能声明**。
- 改进建在 **openclaude 通用框架**上,**不过度特化**;工具获取必须是**通用能力**(agent 为自己的墙找任意
  工具),在 RNA 上演示,而**不是写死"装 DesiRNA"的脚本**。
- **不造 CPU 求解器代替 agent**(本末倒置);成本 ~$0.08/episode,真正约束是**墙钟**。
- OpenRouter key 在 `scratchpad/.orkey`(0600),**绝不打印/暴露**。
- 家族**去碰撞**:Paper C 对 A1/A2/A3/B 的共享量只引用不 claim(见 POSITIONING)。

## 3. 分阶段计划

### Phase 0 — 锁基线 + 装仪表(Day 0–1)
- **落实 draft 里所有 `[verify]` 数字**对 `rna_harness/runs`:76/75/76(frozen/K_θ/A_θ)、
  autonomous 地板 3/35(68/100)vs 带拐杖 11/35(76/100)、56/65 solver-carried 归因。POSITIONING
  明令未核实的数不许进稿。
- 冻结评测集:full Eterna100-V2(100 题,报天花板/地板)+ 35 题 hard-tail(做 mode-Δ 的判别子集)。
- 固化两个协议:**天花板**(flag off,今天的 crutched harness)/ **地板**(`RNA_AUTONOMOUS=1`)。
- 跑 1 个 35-题 epoch 量**精确 per-episode 墙钟**(旧估 ~4.6h/epoch),据此定并发度。
- **产出**:一张核实过的基线表 + 精确墙钟 → 定 Phase 1 的种子/模型规模。

### Phase 1 — 去脚手架地板/天花板:严谨度底座 + 混淆证明(Day 1–5,可与 P2 重叠)
论文实证脊柱,也是补上"single seed/single model"这个 AAAI 头号短板的地方。
- **臂**:crutched-ceiling vs autonomous-floor。
- **种子**:≥3。**模型**:≥2(glm-5.2 + deepseek-v4-pro 或 sonnet-5;A1 的 RNA 行报过这几个)。
- 判别子集(35 hard-tail)多种子做 CI;full-100 每 (模型×臂) 单种子出天花板/地板数。
- **产出**:floor-and-ceiling(带 bootstrap CI)+ Δ = 有多少解题率是脚手架给的、不是模型自己的环。
- **论文声明**:去脚手架归因 + 脚手架-混淆桥,多种子多模型版。

### Phase 2 — 地板上的 mode 效能:三条路当被测 mode(Day 4–9)
"提解题率"与"论文"在这里完全对齐:测每个 mode 把 3/35 地板抬多少。
- **M-memory(地板上复测)**:autonomous_with_memory vs autonomous_frozen,多种子。agent **自写**的
  策略能不能 compound?(注意:这是 agent 自己蒸馏的过程记忆,红线安全。)
- **M-tool-acquisition(Route 2,首要新实验)**:受控放开网络/安装(见下红线 guard),测 agent 撞墙后
  能否**自主识别墙 → 找到 → 装上 → 用起来** DesiRNA/NEMO,解题率动没动。
  - 量两件:**行为**(有没有自主获取到对的工具)+ **结果**(实际预算内数动没动)。
  - 两种结果都干净:真 climb → 工具获取是有效 mode;"装成功但预算内没用"→ 墙是**算力**不是工具
    (强化 oracle-residue 的 judgment-vs-computation 连接)。
  - **红线 guard(必须写死)**:仅此臂放开,且**只准** `pip install` / `git clone` **求解器**包
    (domain allowlist:solver 的 github/pip);**WebSearch/WebFetch 到答案库仍封死**;每个网络动作
    记日志供审计;禁止 fetch 任何 Eterna100 答案文件。此臂是**独立、重日志**的分支,不改默认。
- **M-algorithm-evolution(Route 3,次要,需有能力 subject)**:FunSearch 式在 M2 的 autonomous 集上
  进化 design 算法。**glm-5.2 已证失败**(推理吃光 token)→ 换 **Claude 类 subject** + 先修
  reasoning-cap(`funsearch_rna.py` 的 max_tokens/content=null 问题)。进化出更便宜/更高 crack-rate
  的算法 → 正结果候选。**方法本身已被 AlphaEvolve/ShinkaEvolve 群占,只作"一个 mode"引用不 claim。**
- **长尾(Route 1,不追,只归因)**:出 **designability split**(agent-limited vs oracle-limited 失败),
  证实"绑死解题率的是经典搜索收敛,不是策略/推理召回"。**不堆算力硬啃。**

### Phase 3 — 综合:效能地图 + 论文 reframe(Day 9–12)
- 拼**效能地图**:每个 mode 在地板上的 Δ + 机制解释。
- 按校准把 POSITIONING + abstract/intro **重心挪到"桥 + 去脚手架测量 + 效能地图"**;
  加 ~10 条外部引用(harness-effect 群 / Externalization review / 自进化记忆 benchmark 群 /
  Self-Harness 作反面 / AlphaEvolve 群);**M4 降级**为"crowded 范式的一个 mode"。
- 家族一致性扫一遍(SAT/RNA 坐标数与 A1/A2 完全一致)。

### Phase 4 — 写 + 内审(Day 12–14+)
- 全文 draft;每个数字标 `% src`;跑一遍 simulated review(照 Paper B 的 `reviews/` 套路自审)。

## 4. 实验 → 论文声明 → AAAI 弱点 映射

| 实验 | 支撑的论文声明 | 补的 AAAI 弱点 |
|---|---|---|
| P1 多种子×多模型 地板/天花板 | 去脚手架归因;脚手架-混淆桥 | **单种子/单模型(头号 rigor gap)** |
| M-memory 地板复测(多种子) | M1 null 是真 null,机制=冗余 | n=1 记忆 null |
| M-tool-acquisition(guarded) | mode 效能对比;能力扩展 mode;算力墙 | mode 覆盖太薄;**新增一个实测 mode** |
| M-algorithm-evolution(能力 subject) | 唯一能 compound 的 mode(**正结果**) | "全 null" 弱点;旗舰未跑 |
| Designability split(长尾) | 绑死的是搜索收敛不是召回 | 解题率天花板无解释 |

## 5. Cut line(2 周内的取舍)

- **最小可投稿(MVP)** = P0 + P1(多种子地板/天花板)+ M-tool-acquisition + 长尾 designability split
  + M-memory 复测。**这份就能扛住"single seed/single model/flagship 未跑"的致命挑战。**
- **Stretch** = M-algorithm-evolution 的正结果(有能力 subject 且它真 climb)。跑不出就**诚实降级**成
  related-mode,不当旗舰。
- **Fallback** = 若 2 周 rigor 压不进 AAAI 主榜,MVP 转下一 deadline / workshop,不硬投。

## 6. 风险登记 + go/no-go gate

- **R1 墙钟**:P1 粗算 2 臂×3 种子×2 模型×35 题 ≈ 12 epoch ≈ 55h,~4 并发(礼貌)≈ 2–3 天。
  Gate:P0 精确墙钟出来后,若 >阈值则砍到 2 种子或 1.5 模型。
- **R2 M4 能力 subject**:Claude 类 API 预算 + reasoning-cap 修复。Gate:先小规模 smoke,evolve 不动就走 Fallback,不 sink 时间。
- **R3 工具获取红线**:放开网络是最敏感动作。Gate:先在**离线镜像/allowlist**下 dry-run 审计日志,
  确认无答案 fetch 才正式跑;任何一次越界 → 立即封停该臂。
- **R4 过度特化漂移**:每加一个 mode 自检"这是 openclaude 通用能力还是 RNA 专用 hack";工具获取尤其。
- **R5 deadline vs Paper B**:Paper B 尚在修订。Gate:Paper C 不得挤占 Paper B 的 rigor 资源;
  若冲突,Paper C 让路(它是家族第 5 篇,可延)。

## 7. 时间线(锚定 AAAI-2027:abstract Jul 21 / full Jul 28;今天 Jul 9)

```
Jul 9–10   P0 锁基线+核数字+精确墙钟         → 定规模
Jul 10–14  P1 多种子多模型 地板/天花板(后台) ∥ 开始改稿重心
Jul 13–18  P2 M-tool-acquisition + M-memory 复测 + 长尾归因(后台)
Jul 15–19  P2 M-algorithm-evolution(能力 subject,Stretch,可并行)
Jul 18–21  P3 效能地图 + abstract 定稿  →  ★ Jul 21 abstract 交
Jul 21–28  P4 全文 + 内审               →  ★ Jul 28 full 交
```
后台实验跑,改稿并行推进。每个 Phase 出结果即回来 review,不闷头跑到底。
