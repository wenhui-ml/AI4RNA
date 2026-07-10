# Eterna100 基准:我们用的是哪个版本 + 要超越的公开 SOTA 到底是多少

## 1. 我们用的是不是 V2?V2 与 V1 的确切区别

**是 V2,而且我们的配置与 V2 是"同一模型族",方向正确。**

- 我们的 harness 从 Eterna100 TSV 读取的是 **V2 目标列**(该 TSV 第 3 列 `Secondary Structure V2`,并配套读 `Sample Solution (V2/Vienna2)`);评测用**已安装的 ViennaRNA 2.7.2 + Turner2004 + T=37 + dangles=2**。这正是 V2 应有的评测模型族。
- **V1**(Anderson-Lee et al., *J Mol Biol* 2016)= 100 个 Eterna 玩家设计的谜题,其目标结构是在 **ViennaRNA 1.8.5 / Turner1999** 折叠引擎下定义的。
- **V2**(Koodli et al., bioRxiv 2021.08.26.457839,Das 实验室)= **同样这 100 个谜题**,但因为有 **19 个谜题**在切换到 Vienna 2 引擎后变得不可解,Eterna 专家对这 19 个目标结构做了**最小改动重新设计**,使它们在 **Vienna 2.x / Turner2004** 下仍可设计。谜题总数仍是 100。

**哪个版本被重新定义:是 V2 被重新定义,以适配 Vienna 2 引擎。**

**V2 是否更难?不是"更难",而是"更适配 Vienna2、因而更可解"。** 在 Vienna 2 引擎下,原始 V1 只有 81/100 可解,而 V2 有 99/100 可解。所以 V2 在 Vienna2 下的实际可解天花板约 **99/100**。两个版本上各算法的排名基本一致。

## 2. 当前公开 SOTA:要超越的"榜首"到底是多少

| 方法 | Solved/100 | 版本/引擎 | 预算 | 出处(置信度) |
|---|---|---|---|---|
| **Montparnasse (MOGNRPALR)** | **100/100** | **V1** (Turner1999) | ~13.6h/50线程 | Cazenave, "Eterna is Solved", arXiv:2505.02110 (2025) / arXiv:2606.07562 (2026)(高) |
| **DesiRNA** | **100/100** | **V1** (Turner1999) | 24h | Wirecki et al., *NAR* 53(2):gkae1306, 2025(高) |
| **DesiRNA** | **97/100** ← **V2 榜首** | **V2** (Turner2004) | 24h(85个<1min,95个<1h) | 同上(高) |
| NEMO | 95 (V1) / 94(V2, 24h);90 MFE/87 uMFE(严格定预算) | V1 & V2 | 24h 或定预算 | Portela, bioRxiv 345587 (2018);V2定预算数见 Gautam et al. arXiv:2602.12470 (2026)(高) |
| GNRPA/GNRPALR | 95 | V1 | — | Cazenave & Touzani 2024(高) |
| ES + eM2dRNAs | 94 | V1 | 24h | Rubio-Largo et al., IEEE 2024(中) |
| aRNAque | 92(V1 combined)/ 93(V2) | V1 & V2 | — | Merleau & Smerlak, *BMC Bioinf* 23:335 (2022)(高) |
| MoiRNAiFold | 91 | V1 | — | DesiRNA 比较表引用(中) |
| SAMFEO | 77 MFE/74 uMFE(V2/Turner2004);83/79(严格定预算 V2) | V2 | 5000步 或定预算 | Zhou et al., *Bioinformatics* 2023;Gautam et al. 2026(高) |
| SentRNA | 78(V1)/ 69(V2) | V1 & V2 | — | Shi et al., arXiv:1803.03146 (2018)(高) |
| RNAinverse-pf | 82 MFE/77 uMFE(严格定预算 V2) | V2 | 定预算 | Gautam et al. 2026(高) |
| LLM(Qwen2.5-0.5B, SL+RL, 受限解码) | 81 MFE/79 uMFE | V2 | — | Gautam et al., arXiv:2602.12470 (2026)(高) |
| Meta-LEARNA | ~65 | V1 | — | Runge et al., ICLR 2019(中) |

**明确结论:**
- **V1 榜首 = 100/100**(Montparnasse 与 DesiRNA 两个独立方法都做到了;Montparnasse 更快)。**V1 基准已被"解决/饱和"。**
- **V2 榜首 = DesiRNA 的 97/100**(24h 预算)。**这是与我们 harness(V2/Turner2004)真正对应、要打的那个数字。**
- **是否有方法解出全部 100?只在 V1 上有(两个方法)。V2 上没有任何已发表方法做到 100/100**;在更严格的"定预算精确 MFE"判定下,V2 榜首降为 NEMO 的 90/100(置信度:97 这个数高;严格判定下 90 高)。#97 与 #100 是 V2 上历史上最难、长期未被自动方法解出的谜题。

## 3. 科学严谨性提醒

- **我们本地 rnainverse ≈ 37% 不是标尺,是"预算受限的地板"。** 跨论文的 solved 数对协议极其敏感:同一个 NEMO 在 24h 预算下报 95(V1),在定步数预算下只有 79(V2)。RNAinverse 纯 MFE 只有 30,加 -pf 能到 70/82。所以"我们跑出 37%"只反映我们给的预算/重启次数很小,**真正的 bar 是公开 SOTA,不是我们自己的地板数**。
- **版本匹配的严谨性坑(确实存在):** 官方 Eterna100 评分脚本(eternagame/eterna100-benchmarking)对 **V2 固定使用 ViennaRNA 2.6.4**,而我们装的是 **2.7.2**。好消息:Turner2004 参数集在 2.6→2.7 之间未变、MFE 算法稳定,所以计数应当**几乎相同**;但**在边界谜题上不保证逐题一致**。若要对外声称"超越榜单",应:
  1. **对齐 ViennaRNA 版本到 2.6.4**(或至少把少数边界谜题在 2.6.4 与 2.7.2 下各折一次,确认 MFE 结构一致);
  2. **对齐参数**:Turner2004、T=37、**dangles=2**(RNAfold CLI 默认);
  3. **绝不用 Vienna1 去评 V2 目标**(那会混用 V2 所针对的模型)。

## 4. 一句话总结

**我们要打的目标数字是 V2 榜首 = DesiRNA 的 97/100(ViennaRNA 2.x / Turner2004,24h 预算)**;而要做出"可对外声称超越 SOTA"的严谨结论,必须满足四项对齐:**同一题集(Eterna100-V2 目标列)、同一 ViennaRNA 版本与参数(建议 pin 2.6.4 + Turner2004 + T=37 + dangles=2)、同一 solved 判定(设计序列 MFE 结构与目标精确一致)、并明确报告计算预算(budget)**——否则任何"超过 97"的数字都不构成与榜单的对等比较。