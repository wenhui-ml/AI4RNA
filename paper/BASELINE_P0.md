# P0 核实基线表 (Paper C) — 2026-07-09

所有数字来自 `rna_harness/runs/` 产物(非重跑,artifact-grounded)。draft 里所有 `[verify]` 用本表替换。
每行标产物路径 → 可写成 `% src:` 注释。glm-5.2 除非注明。评测集 Eterna100-V2 (n=100)。
solved = ViennaRNA 2.7.2 (Turner2004, 37℃, dangles=2) MFE 折叠 = target,不含体内声明。

## A. 三个 glm 配置(draft 曾混淆 → 必须拆清)

| 配置 | Solved/100 | agent_A | 产物 |
|---|---|---|---|
| glm memory-free base | **58** | 0.860 | `glm5.2_eterna100/summary.json` |
| glm crutched(full toolset + strategy, memON) | **65** | — | `glm52_eterna_memON/summary.json` |
| glm crutched + 4-epoch meta(frozen 臂) | **76** | — | `meta_ladder_frozen/epoch_progress.json` |

⚠️ draft 抽象里的 "GLM-5.2 56" 错 → 应为 **58**(memory-free)。"65"/"76" 是另两个更富配置,别混。

## B. 每模型 memory-free base(A1 共享量,只引不 claim)

| 模型 | Solved/100 | agent_A(=A_model) | 产物 |
|---|---|---|---|
| glm-5.2 | **58** | 0.860 | `glm5.2_eterna100` |
| claude-sonnet-5 | **51** | 0.847 | `sonnet5_eterna100` |
| deepseek-v4-pro | **47** | 0.860 | `deepseek_v4pro_eterna100` |

agent_A (0.860/0.847/0.860) ≈ A1 报的 (0.86/0.85/0.86) ✓ 一致。**A 坐标引 A1,不重报。**

## C. M1 — 记忆 3 臂(在 65 baseline 之上、35-tail 上做 4 epoch;glm-5.2;**n=1/臂**)

| 臂 | e1 | e2 | e3 | e4 | Δ vs 65 | 产物 |
|---|---|---|---|---|---|---|
| frozen(控制) | 70 | 71 | 76 | **76** | +11 | `meta_ladder_frozen` |
| accumulate K_θ | 70 | 72 | 74 | **75** | +10 | `meta_ladder_accumulate` |
| askill A_θ | 71 | 75 | 76 | **76** | +11 | `meta_ladder_askill` |

- **76/75/76 确认** ✓,三臂落在彼此 ±1 → 无 meta-learning 效应。
- **Skill 工具精确调用 = 3/35**(Taraxacum officinale / Snowflake 4 / Bullseye),**这 3 个全部未解 → 0 decisive**。
  技能库最终 2 个(rna-multiloop, rna-tandem-repeat)。→ "skills 从不 load-bearing" 稳。
- ⚠️ n=1/臂,run-to-run 噪声大 → **P1 必须多种子**才能进稿。

## D. M2 — 自主地板(auto_frozen, `RNA_AUTONOMOUS=1`, 35-tail, 1 epoch, glm-5.2)

- 自主 **+3/35**(Repetitious Seq 52 / Loop-next-to-Multiloop 72 / Fractal 3 83)vs crutched frozen **+11/35**。
- hard-tail 脚手架贡献 = (11−3)/11 = **8/11 ≈ 73%**。产物 `meta_ladder_auto_frozen`。
- ⚠️ **"68/100" 不是干净的全 100 自主地板**(= 65 拐杖解 + 3 自主解;那 65 是 crutched 拿的)。
  **P1 必须全 100 题自主重跑**,得真floor。3/35 vs 11/35 的 hard-tail 对比本身严谨,可用。

## E. M3 — 解题归因(65 baseline,glm-5.2,可复现)

两个可复现定义(stale 的 "56/65" 不可复现,已弃):
- **episode-level(自由):61/65 episode 调用过 search solver;仅 4/65 全程纯手设**(无任何 solver 签名)。
- **producer-level(保守):winning seq 直接由 MCP solver `parallel_search` 产出 23/65;由 agent 自写 Bash 产出 32/65;经 verify/read echo 10/65**。
- 两个定义同向:**LLM 编排经典搜索,from-scratch 设计是极小残余(4–9 题)。** 产物 `glm52_eterna_memON` + traces。
- 论文里**明写定义**,建议主打 producer-level 23/65(最保守、最难攻)+ episode-level 61/65 作上界。

## F. 失败模式(支撑"绑死的是搜索收敛,不是推理")

unsolved 中撞 1500s 墙钟上限的比例:**glm 80%(34/42)/ sonnet 95%(47/49)/ deepseek 84%(45/53)**;
且几乎全部 unsolved 都有 best_effort 近似解(agent 在磨、只是没在预算内收敛)。
→ draft 的 "90–98% timeouts" **改成 "80–95%"**。产物:三个 base run 的 records.elapsed_s/best_effort。

## G. 墙钟(P1 定规模用;无需重跑,artifact 已含)

- 每 task 预算上限 **1500s (25min)**;p90=max=1500 → 相当比例 task 撞顶。
- full-100 **串行** ≈ 20–24h/run(glm 20.1h / sonnet 23.5h / deepseek 24.6h)。
- 35-tail 自主 1 epoch = 16511s ≈ **4.6h**(≈472s/task)。
- **P1 规模建议**:全 100 自主 floor 每(模型)1 次 ≈ 22–28h;35-tail 多种子(≥3)× 2 模型 × (floor+ceiling) ≈ 12 run × 4.6h;礼貌并发 4 路 → floor 阶段约 2–3 天墙钟。

## H. draft 必改清单(P0 产出)

1. `56` → **58**(glm memory-free)。
2. "56/65 solver-carried" → **23/65 (producer) / 61/65 (episode)**,明写定义。
3. "90–98% timeouts" → **80–95%**。
4. "68/100 autonomous" → 改述为 "**+3/35 on hard-tail;干净 full-100 floor 待 P1**"。
5. 保留(已确认):3/35 skills invoked & 0 decisive;agent_A≈0.86(引 A1);76/75/76;+11/35 crutched。
