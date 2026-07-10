RNA逆合成 · 全球合作者名录
说明：本名录按 地区分组、组内按 优先级（top → high → medium → watch）排序。每条含：姓名 / 现任单位（已核实 或  需更正+更正内容）/ 主页 / 邮箱/ 1–2 项最相关近作 / 一句话项目阶段相关性（P1 oracle / P2 基准+基线 / P3 RL / theory 理论 / HITL 人在环 / FM 基础模型）。
> 红线提示：所有“relevance”仅限“计算折叠oracle（ViennaRNA/NUPACK/RNAstructure/EternaFold）下的设计成功”，绝不涉及体内功能。湿实验/化学探测类（Lucks、Weeks、Qiangfeng Zhang）仅作oracle效度的参考验证联系人。
---
1. 美国 (United States)
Rhiju Das — 优先级 top
- 单位：斯坦福大学医学院生物化学系教授；HHMI 研究员（2021/22 起）；Eterna 平台创始人
  - 部分旧页仍写“副教授”——现为正教授。当前 SOTA 模型为 RibonanzaNet2 (2025)。
- 主页：http://daslab.stanford.edu/ ｜ 邮箱：rhiju@stanford.edu（中置信度，来自 Stanford profile 搜索快照，官网本轮 403 未直连复核）
- 近作：RibonanzaNet2 (~100M 参数 RNA 结构基础模型，2025)；Ribonanza 双众包深度学习 (Nature 2024)
- 相关性：P2-benchmark（Eterna100-V2）+ P1-oracle（EternaFold）——项目验证器与基准的双重根基。

Liang Huang (黄亮) — 优先级 top
- 单位：俄勒冈州立大学 EECS 正教授（2023 晋升）+ 生化生物物理系兼聘。“Baidu Research USA”为已卸任旧职。
- 主页：https://web.engr.oregonstate.edu/~huanlian/ ｜ 邮箱：liang.huang@oregonstate.edu（高，主页直见；亦列 liang.huang.sh@gmail.com）
- 近作：SamplingDesign（连续优化+MC 采样，Nat. Commun. 2025）；SAMFEO（ISMB 2023，Eterna100 SOTA 基线）；LinearDesign（Nature 2023）
- 相关性：P2-baseline——SAMFEO 是 harness 显式复现(~300–500 LoC)/直接调用的成熟基线，代理必须超越的“非学习地板”。

Niles A. Pierce — 优先级 top
- 单位：加州理工 MacArthur 应用与计算数学暨生物工程讲席教授；BBE 执行主任
- 主页：https://www.piercelab.caltech.edu/ ｜ 邮箱：niles@caltech.edu（高，piercelab/contact）
- 近作：NUPACK 4 全新代码库（同轴/悬挂堆叠子系综，20–120× 加速，可达 ~30,000 nt），ACS Synth. Biol. 2026
- 相关性：P1-oracle——NUPACK 是项目多oracle后端之一，直接的oracle提供方。

Chaitanya K. Joshi — 优先级 top
- 单位：已从剑桥迁至斯坦福（Das 实验室博后 / Stanford Data Science Fellow，2026-02 起）；剑桥仅为博士母校。
- 主页：https://www.chaitjo.com/ ｜ 邮箱：chaitjo@stanford.edu（高，主页直见）
- 近作：gRNAde（几何 GNN 3D RNA 反向折叠，ICLR 2025 Spotlight）；gRNAde 湿实验验证核酶（bioRxiv 2025-12）
- 相关性：FM / P2-baseline——领先的 3D 反向设计生成模型，且现处 Eterna/EternaFold 谱系，几何设计↔基准生态直接打通。

Michael Lindsey — 优先级 top
- 单位：UC Berkeley 数学系助理教授 & 劳伦斯伯克利国家实验室
- 主页：https://quantumtative.github.io/ ｜ 邮箱：经主页/GitHub 核实（未见邮箱字符串）
- 近作：GoForth：在结构+序列+编码约束下用语言模型设计 RNA（arXiv:2605.07608, 2026）
- 相关性：P2 / FM(LLM-for-design)——几乎正是本项目“混合约束 LLM 代理设计”任务的外部对应物（注：GoForth 作者归属本轮存在单作者标注不一致，出访前需再核实）。

David H. Mathews — 优先级 high
- 单位：罗切斯特大学医学中心 Lynne E. Maquat 讲席教授（生化与生物物理系；RNA 生物学中心），M.D./Ph.D.
- 主页：http://rna.urmc.rochester.edu/ ｜ 邮箱：无官方公开（官网仅电话 (585) 275-1734；常见 David_Mathews@urmc.rochester.edu 本轮未在官方页确认，故不列）——经主页核实
- 近作：memerna 稀疏折叠含同轴堆叠 (JMB 2025)；最小不可设计 RNA motif（RECOMB 2025，与 Huang）；m1Ψ 近邻参数进 RNAstructure (2024)
- 相关性：P1-oracle（RNAstructure 交叉校验）+ theory（不可设计性）+ Turner 参数根基。

Peter Clote — 优先级 high
- 单位：波士顿学院生物系教授（2026 仍在任，非荣休）
- 主页：https://bioinformatics.bc.edu/clotelab/ ｜ 邮箱：peter.clote@bc.edu（高，BC 教师目录）
- 近作：专著《RNA: Computational Methods…》(Routledge 2024)；RNAiFold 2.0 (2015, 奠基)
- 相关性：P2-baseline——RNAiFold 穷举式约束规划反向折叠，混合约束设计的天然对照。（其研究产出约 2020 后放缓）

Ryan K. Krueger — 优先级 high
- 单位：哈佛 SEAS 应用数学（合作者 Max Ward, 西澳大学）
- 主页：https://github.com/rkruegs123/jax-rnafold ｜ 邮箱：经主页/论文核实
- 近作：JAX-RNAfold（可微、可扩展至 ~1250 nt 的 Turner 配分函数，Bioinformatics 2025）
- 相关性：P1-oracle（可微折叠）——可把 harness 的“验证器引导修复”升级为连续梯度优化。

James Zou — 优先级 high
- 单位：斯坦福 生物医学数据科学副教授（兼 CS/EE）
- 主页：https://www.james-zou.com/ ｜ 邮箱：经主页核实
- 近作：Virtual Lab 多智能体设计 SARS-CoV-2 纳米抗体 (Nature 2025)；Paper2Agent (2025)
- 相关性：HITL / 代理自改进——PI+Scientific Critic+专家的多代理验证环，与 harness 的“代理+验证器+子代理”架构直接对应。

Jure Leskovec — 优先级 high
- 单位：斯坦福 计算机科学教授
- 主页：https://cs.stanford.edu/~jure/ ｜ 邮箱：经主页核实
- 近作：Biomni（150 工具通用生物医学代理+技能/记忆模块，bioRxiv 2025）；BioDiscoveryAgent (2024)
- 相关性：FM / 代理+记忆——工具调用+记忆/元学习+子代理应用于科学的领先模板。

Xiaohui Xie (谢晓辉) — 优先级 high
- 单位：UC Irvine 计算机科学教授（Bren 学院）
- 主页：https://ics.uci.edu/~xhx/ ｜ 邮箱：xhx@ics.uci.edu / xhx@uci.edu（高，均官方）
- 近作：UFold（深度学习二级结构预测，NAR 2022）
- 相关性：FM / P1-oracle（学习型折叠器）——非热力学学习oracle候选。 RhoFold+/RNA-FM/RhoDesign/RiboDiffusion 及 RiboPO 均非其工作，勿归于他。

Hannah K. Wayment-Steele — 优先级 high
- 单位： 威斯康星大学麦迪逊分校生化系助理教授（不再是斯坦福博士生/Brandeis 博后）
- 主页：https://hwaymentsteele.github.io/ ｜ 邮箱：wayment@brandeis.edu 已过期（低置信，2024 Brandeis CV）；新的 wisc.edu 地址本轮未核实——请勿假定
- 近作：EternaFold（Nature Methods 2022，仍为参考版）；蛋白/RNA 构象系综+NMR+ML (2024–25)
- 相关性：P1-oracle（EternaFold 作者）/ FM。

Julius B. Lucks — 优先级 medium
- 单位：西北大学化学与生物工程教授；合成生物学中心联合主任；2025 AAAS Fellow
- 主页：https://luckslab.org/ ｜ 邮箱：jblucks@northwestern.edu（高）
- 近作：RNA 折叠动力学控制核糖开关体内敏感度 (Nat. Commun. 2025)；共转录 SHAPE-Seq+R2D2 (Mol. Cell 2021)
- 相关性：P1-oracle 效度参考——共转录折叠动力学是对 MFE oracle的经验现实检验（体内动力学陷阱≠MFE 目标）。属红线外的验证性联系人。

Elena Rivas — 优先级 medium
- 单位：哈佛 分子与细胞生物学教授
- 主页：http://rivaslab.org/ ｜ 邮箱：经主页核实
- 近作：CaCoFold-R3D（3D-motif+二级结构概率文法，Nature Methods 2025）；R-scape 螺旋级协变 (2023–24)
- 相关性：theory / P1-oracle——进化信息型替代验证器，回答“目标是否可信/可设计”。

Joseph D. Yesselman — 优先级 medium
- 单位：内布拉斯加大学林肯分校 化学系助理教授
- 主页：https://chem.unl.edu/person/joseph-yesselman/ ｜ 邮箱：经主页核实
- 近作：NSF CAREER (2024) RNA 三级接触预测 + Foldit-RNA 界面；RNAMake 3D 设计工具
- 相关性：HITL / P2-baseline——RNAMake 经典 3D 设计基线；Foldit-RNA 众包是 Eterna 的直接人在环类比。

Gabe Gomes — 优先级 medium
- 单位：卡内基梅隆 化学与化工助理教授
- 主页：https://www.gomesgroup.org/ ｜ 邮箱：经主页核实
- 近作：Coscientist（自主 LLM 实验代理，Nature 2023）
- 相关性：HITL / 代理闭环——工具调用自主设计代理的架构先例（其环闭合于湿实验，本项目闭合于折叠oracle）。

Wengong Jin & Divya Nori — 优先级 medium
- 单位：Wengong Jin：Broad Institute / 东北大学；Divya Nori：MIT CSAIL
- 主页：https://proceedings.mlr.press/v235/nori24a.html ｜ 邮箱：经主页核实
- 近作：RNAFlow（反向折叠式流匹配共设计序列+结构，ICML 2024）
- 相关性：P2-baseline（生成）/ P3——同族生成基线，可作 RL 策略架构候选。

Kevin M. Weeks — 优先级 watch
- 单位：UNC 教堂山 Kenan 讲席化学教授；2026 NAI 资深会员
- 主页：https://chem.unc.edu/faculty/weeks-kevin/ ｜ 邮箱：weeks@unc.edu（高）
- 近作：RNAvigate（化学探测数据探索，NAR 2024）；单分子相关化学探测 smCCP (2024)
- 相关性：P1-oracle 效度参考（红线外）——SHAPE/DMS-MaP 实验基准真值，用于判断 MFE 验证器保真度。


2. 加拿大 (Canada)
Jérôme Waldispühl — 优先级 top
- 单位： 麦吉尔大学计算机学院 正教授 + 加拿大首席科学家 Tier 1（生物医学公民科学）；D2R 计划成员（旧页“副教授”已过期）
- 主页：https://www.cs.mcgill.ca/~jeromew/（lab: csb.cs.mcgill.ca）｜ 邮箱：jerome.waldispuhl@mcgill.ca（高）
- 近作：Borderlands Science 公民科学改进微生物系统发生 (Nature Biotech 2024)；PERFUMES（Bioinformatics 2024）
- 相关性：HITL——最大规模生物分子公民科学平台（Phylo/Borderlands），是 Eterna100 之外的人在环 RNA 设计数据/轨迹来源。

Anne Condon — 优先级 high
- 单位：UBC 计算机科学教授
- 主页：https://www.cs.ubc.ca/~condon/ ｜ 邮箱：condon@cs.ubc.ca（高）
- 近作：多链核酸复合体 MFE 预测 APX-hard（Condon-Hajiaghayi-Thachuk，ICALP 2025 引用）；DNA 杂交动力学 (2023)
- 相关性：theory——提出“RNA 二级结构设计是否可解”的奠基性开放问题（ICALP 2003），为“启发式搜索+oracle验证”而非精确设计提供理论依据。（近期产出偏向 DNA 动力学）

Hosna Jabbari — 优先级 high
- 单位： 阿尔伯塔大学生物医学工程系正教授（兼 ECE），COBRA Lab 主任，工程研究讲席——不再在维多利亚大学（旧职/LinkedIn 过期）
- 主页：https://thecobralab.com ｜ 邮箱：jabbari@ualberta.ca（高，U of A 目录）
- 近作：Spark（稀疏分层能量最小化，可扩展伪结预测，Bioinformatics/WABI 2025）；SparseRNAFolD (2023)
- 相关性：P1-oracle——伪结可用的 MFE 折叠引擎（HotKnots/Knotty/Iterative-HFold/DinoKnot/Spark），把验证器扩展到伪结之外。

Mirela Andronescu — 优先级 watch
- 单位： BC Cancer（Aparicio 实验室）生物信息学科学家（非教职）；已转向单细胞癌症基因组学，不再从事 RNA 设计
- 主页：https://aparicio.molonc.ca/team/Mirela-Andronescu ｜ 邮箱：未找到官方（ZoomInfo 提示 @bccrc.ca 但被遮蔽，不作确认）
- 近作：单细胞三阴乳腺癌转录组重编程 (Genome Biol. 2024)
- 相关性：P1-oracle 历史权威——HotKnots V2.0 / Turner 型能量参数（2008–2010 博士期），今为历史参考而非现役合作者。



3. 中国 (China) + 香港
Yu Li (李煜) — 优先级 top
- 单位：香港中文大学 CSE 助理教授（ml4bio / AIH 组）；MIT TR35 China 2024
- 主页：https://www.cse.cuhk.edu.hk/people/faculty/yu-li/（个人 yu-li.github.io）｜ 邮箱：liyu@cse.cuhk.edu.hk（高）
- 近作：RhoFold+（RNA-LM 3D 预测，CASP15 冠军，Nature Methods 2024）；RhoDesign（Nat. Comput. Sci. 2024）；RiboDiffusion（扩散反向折叠，ISMB 2024）
- 相关性：P2-baseline / FM（并 P1-oracle via RhoFold+）——一间实验室同时提供折叠oracle候选与两个结构→序列反向设计基线。

Siqi Sun (孙思琦) — 优先级 top
- 单位：复旦大学智能复杂体系研究院副教授；上海 AI Lab（beamlab.ai）
- 主页：https://intersun.github.io/ ｜ 邮箱：siqisun@fudan.edu.cn（高）
- 近作：RhoFold+ (Nature Methods 2024)；OriGene（自演化虚拟疾病生物学家，agentic science，bioRxiv 2025）；PriFold (AAAI 2025)
- 相关性：P1-oracle / FM（+ P2/P3 agentic）——OriGene 的自主 agentic-science 回路正对应本项目的元学习 harness。

Stan Z. Li (李子青) — 优先级 top
- 单位：西湖大学工学院 AI 讲席教授（AI Lab）
- 主页：https://en.westlake.edu.cn/faculty/stan-zq-li.html ｜ 邮箱：Stan.ZQ.Li@westlake.edu.cn（高）
- 近作：R3Design（三级结构 RNA 序列设计，Brief. Bioinform. 2024）；RDesign (ICLR 2023)；RiboPO（结构+稳定性感知的偏好优化/RL 式设计，arXiv 2025）
- 相关性：P2-baseline / P3-RL——RiboPO 是本项目 GRPO-on-traces 目标最接近的外部类比。

Yaoqi Zhou (周耀旗) — 优先级 high
- 单位：深圳湾实验室系统与物理生物学所资深 PI（兼 SMART）；Griffith(澳)为次要旧职
- 主页：https://en.szbl.ac.cn/research/groups/Yaoqi-Zhou.htm（中文 zhouyq-lab.szbl.ac.cn）｜ 邮箱：未确认（本轮 szbl 服务器拒连、论文邮箱被遮蔽）——经主页核实
- 近作：RNA-MSM（基于 MSA 的 RNA 语言模型，NAR 2024）；RNA-LM 零样本基准 (Brief. Bioinform. 2026)；SPOT-RNA/2（奠基）
- 相关性：FM / P1-oracle / P2-benchmark——SPOT-RNA oracle候选，RNA-LM 基准直接指导oracle与表征模型选型。

Qiangfeng Cliff Zhang (张强锋) — 优先级 medium
- 单位：清华大学生命学院教授（结构生物学）
- 主页：http://zhanglab.net/ ｜ 邮箱：未确认（Google Scholar 仅显示 tsinghua.edu.cn 域）——经主页核实
- 近作：RASP v2.0（RNA 结构探测图谱，NAR 2025）；综述“AI meets RNA”(Sci. China Life Sci. 2025)
- 相关性：P1-oracle / theory（红线外参考）——icSHAPE/RASP 实验结构真值，作oracle效度参考，非反向设计贡献（“ML 结构信号恢复”的说法夸大了其 ML 角度）。

Zhen Xie (谢震) — 优先级 medium
- 单位：清华大学自动化系 & 合成系统生物学中心副教授（BioCircuit Lab）
- 主页：http://bioinfo.au.tsinghua.edu.cn/staff-member/zhen-xie/ ｜ 邮箱：未确认（服务器本轮不可达）——经主页核实
- 近作：ERNIE-RNA（结构增强 RNA 语言模型，Nature Commun. 2025；与微软 Tao Qin 合作）
- 相关性：FM——注意力图零样本碱基配对预测的表征骨干候选（其本组为合成生物学/RNA 电路，ERNIE-RNA 属跨组合作）。

Zhi John Lu (鲁志) — 优先级 watch
- 单位：清华大学生命学院副教授/PI（生物信息学 MOE 重点实验室）
- 主页：https://lulab.life.tsinghua.edu.cn/ ｜ 邮箱：未确认——经主页核实
- 近作：无编码 RNA / cell-free exRNA 液体活检生物标志物 (2025)
- 相关性：FM / theory（最弱对齐）——近年偏 biomarker/液体活检，与折叠oracle反向设计回路关联松散。

Yang Yang & Hongbin Shen（DRAG 组）— 优先级 high
- 单位：上海交通大学 CSE / 图像处理与模式识别研究所
- 主页： 占位（Yang Yang SJTU CS 页需出访前核实）｜ 邮箱：未找到（取自论文元数据）
- 近作：DRAG：以分层图+强化学习做二级结构反向折叠（Brief. Bioinform. 26(2), 2025）
- 相关性：P3-RL / P2-baseline——与 harness 同任务同验证器设置的 2025 RL 基线与 P3 先例。

BEACON 团队（Yuchen Ren, Xihui Liu 等）— 优先级 medium
- 单位：上海 AI 实验室 & 香港大学（等）
- 主页：https://github.com/terry-r123/RNABenchmark ｜ 邮箱：经主页核实
- 近作：BEACON：13 项 RNA 任务的 RNA-LM 综合基准（NeurIPS D&B 2024）
- 相关性：P2-benchmark——Eterna100/RNAInvBench 之外可评估 LLM/FM 设计臂的基准。



4. 欧洲 (Europe)
Ivo L. Hofacker — 优先级 top
- 单位：维也纳大学理论化学所正教授；TBI 组长（兼生信与计算生物学组）
- 主页：https://www.tbi.univie.ac.at/~ivo/ ｜ 邮箱：ivo@tbi.univie.ac.at（高）
- 近作：Old Dog New Tricks：精确种子策略提升 RNA 设计（RECOMB 2025）；ViennaRNA 修饰 RNA 预测 (Bioinformatics 2023)
- 相关性：P1-oracle——ViennaRNA 作者/维护者，即项目 Turner2004 MFE oracle与“solved”的定义性验证器。

Ronny Lorenz — 优先级 top
- 单位：维也纳大学理论化学所——ViennaRNA 包首席开发/维护者（u:find 2025-12 确认在职）
- 主页：https://www.tbi.univie.ac.at/~ronny/ ｜ 邮箱：ronny.lorenz@univie.ac.at（高；旧 ronny@tbi.univie.ac.at 仍可用）
- 近作：ViennaRNA 软约束框架、盐校正更新 (2024–25)；ViennaRNA 2.6.x 维护
- 相关性：P1-oracle——项目核心验证器软件的工程负责人，oracle接入首选联系人。

Frederic Runge — 优先级 top
- 单位：弗莱堡大学机器学习实验室博士研究员（导师 Frank Hutter；ELLIS Unit Freiburg）
- 主页：https://ml.informatik.uni-freiburg.de/profile/runge/ ｜ 邮箱：runget@cs.uni-freiburg.de（高）
- 近作：Partial RNA design / libLEARNA（Bioinformatics/ISMB 2024）；RnaBench (bioRxiv 2024)；KinPFN (ICLR 2025)
- 相关性：P2-baseline / P3-RL——最直接的欧洲对应者：LEARNA/Meta-LEARNA（深度 RL 设计）+ 混合约束(libLEARNA) + RnaBench，元学习+RL 议程正是 P3 核心。弗莱堡集群日常最佳联系人。

Ke Li (李轲) — 优先级 top
- 单位： 埃克塞特大学计算机系 AI 正教授（2025-10 起，由 Reader/副教授晋升）；UKRI FLF & Alan Turing Fellow；COLA Lab
- 主页：https://colalab.ai/ ｜ 邮箱：k.li@exeter.ac.uk（高）
- 近作：RNAInvBench（2D+3D 反向设计基准+基线+指标）；RIDER（RL 引导扩散 3D 反向设计，ICLR 2026，学生 Tianmeng Hu 主导）；CocoRNA（合作多智能体 RL，ICML 2026）
- 相关性：P2-benchmark / P3-RL / HITL——覆盖 P1→P3 全栈的最接近外部对应物。

Frank Hutter — 优先级 high
- 单位： Prior Labs 联合创始人兼 CEO（Freiburg/Berlin/NYC）；弗莱堡 ML 正教授（目前休假）；ELLIS Tübingen 兼职 PI。SAP 于 2026-05 宣布收购 Prior Labs。
- 主页：https://ml.informatik.uni-freiburg.de/profile/hutter/ ｜ 邮箱：fh@cs.uni-freiburg.de（中——真实但在休假期公开请求勿直接来信，建议经 Runge 或 Prior Labs）
- 近作：TabPFN v2（Nature 2025）；RnaBench / Partial RNA design（资深作者，2024）
- 相关性：P3-RL / FM——LEARNA 线与 PFN 基础模型的 PI；活跃合作请走 Runge。

Rolf Backofen — 优先级 high
- 单位：弗莱堡大学生物信息学讲席教授（W3）；工程学院院长；CIBSS/BIOSS PI
- 主页：http://www.bioinf.uni-freiburg.de/~backofen/ ｜ 邮箱：backofen@informatik.uni-freiburg.de（高）
- 近作：Partial RNA design 共作者 (2024)；RNAformer（二级结构预测，bioRxiv 2024）
- 相关性：P1-oracle / P2-baseline——Freiburg RNA Tools（IntaRNA/LocARNA）+ RNAformer，锚定折叠oracle层。

Pietro Liò — 优先级 high
- 单位： 剑桥大学计算机系 计算生物学正教授（旧 bio 仍写“Reader”）
- 主页：https://www.cst.cam.ac.uk/people/pl219 ｜ 邮箱：pl219@cam.ac.uk（高）
- 近作：gRNAde（ICLR 2025，资深作者）；RNA-FrameFlow（de novo 3D RNA 骨架流匹配，2024）
- 相关性：FM——几何深度学习生成式 RNA 设计引擎，可作 P2 基线/基础模型。

Christoph Flamm — 优先级 high
- 单位：维也纳大学理论化学所副教授（TBI 创始成员）
- 主页：https://www.tbi.univie.ac.at/team/ ｜ 邮箱：xtof@tbi.univie.ac.at（高）
- 近作：Pathway Realizability in Chemical Networks (J. Comput. Biol. 2025)；ViennaRNA 共同开发
- 相关性：P1-oracle——ViennaRNA 共作者、折叠动力学/能量景观理论，关涉oracle内部与 P2 适应度景观。

Peter F. Stadler — 优先级 high
- 单位：莱比锡大学生物信息学正教授、IZBI 主任；Santa Fe Institute 外聘；Max Planck 外部科学成员
- 主页：https://www.bioinf.uni-leipzig.de/~studla/ ｜ 邮箱：未确认（官网未显示；推测 studla@bioinf.uni-leipzig.de 未证实，不作已找到）——经主页核实
- 近作：RNA 二级结构预测的实验证据极限 (Front. Bioinform. 2024)；计算复杂度/演化 (2025)
- 相关性：theory——RNA 组合学与适应度景观理论，支撑项目的确定性-景观框架与 P1/P2 预测极限推理。

Sebastian Will — 优先级 high
- 单位：巴黎综合理工 LIX 生物信息学正教授（2020-03 起，Palaiseau, 法国）
- 主页：https://www.lix.polytechnique.fr/~will/ ｜ 邮箱：sebastian.will@polytechnique.edu（高）
- 近作：Spark 伪结预测 (Bioinformatics 2026)；Infrared 声明式 RNA 设计框架 (Alg. Mol. Biol. 2024)；LocARNA 2.0
- 相关性：P2-baseline——Infrared 声明式设计工具是成熟 P2 基线，兼顾快速折叠调用效率。

Yann Ponty — 优先级 high
- 单位： CNRS / LIX——CNRS 研究主任（DR）兼 AMIBio 团队负责人（旧称“CNRS researcher”应升级）
- 主页：https://www.lix.polytechnique.fr/~ponty/ ｜ 邮箱：未确认（主页 JS 隐藏；推测 yann.ponty@lix.polytechnique.fr 未证实）——经主页核实
- 近作：Old Dog New Tricks 精确种子 RNA 设计（RECOMB 2025，与 Hofacker）；Infrared 框架 (2024)
- 相关性：P2-baseline——IncaRNAtion/RNARedPrint/Infrared 反向设计组合学工具链，P2 基准应对比的成熟基线。

Holger H. Hoos — 优先级 medium
- 单位：RWTH Aachen 洪堡讲席教授（AI 方法学）+ 莱顿大学 ML 教授 + UBC 兼职；EurAI 主席
- 主页：https://www.aim.rwth-aachen.de/ ｜ 邮箱：未找到——经主页核实
- 近作：AutoSR4EO (Remote Sensing 2024)；历史：RNAsoft/RNA-SSD (Andronescu-Condon-Hoos)
- 相关性：P2-baseline（方法学/工具）——自动算法选择/配置可用于 P2 基线求解器组合选优与 P3 超参搜索；RNA 设计本身为历史工作。

---

5. 其他地区 / 跨区域（日本 · UAE · 新加坡等）
Le Song — 优先级 high ｜非中国区
- 单位：GenBio AI CTO（美）；MBZUAI 教授（阿布扎比，UAE）
- 主页：https://genbio.ai/ ｜ HF: huggingface.co/genbio-ai ｜ 邮箱：未找到——经主页核实
- 近作：AIDO.RNA-1.6B（16 亿参数 RNA 基础模型，含反向折叠+二级结构任务，bioRxiv 2024）
- 相关性：FM / P2-baseline——直接的反向折叠基线与设计回路骨干候选。

Eric Xing (邢波) — 优先级 high ｜非中国区
- 单位：MBZUAI 校长/大学教授（UAE）；CMU 教授（美）；GenBio AI 联创兼首席科学家
- 主页：https://mbzuai.ac.ae/study/faculty/professor-eric-xing/（CMU: cs.cmu.edu/~epxing/）｜ 邮箱：未找到（官网仅联系表单）——经主页核实
- 近作：AIDO “数字有机体”计划（含 AIDO.RNA-1.6B）；AIDO.ModelGenerator (2025)
- 相关性：FM——基础模型基础设施与反向折叠基线的项目/平台 PI（非亲手做 RNA 设计）。

Kengo Sato — 优先级 medium ｜日本
- 单位：庆应义塾大学 生物科学与信息学系
- 主页：https://satoken.github.io/ ｜ 邮箱：经主页核实
- 近作：MXfold2（学习分数+Turner 近邻能量经 Zuker 算法融合，2021）；神经网络碱基配对概率含伪结 (2022–24)
- 相关性：P1-oracle（学习型折叠候选）——热力学锚定的 ML 折叠oracle，加入鲁棒性面板而不易纯 ML 过拟合。

Mile Šikić & Rafael Josip Penić — 优先级 medium ｜新加坡/克罗地亚
- 单位：A*STAR 新加坡基因组研究所 & Ruđer Bošković 研究所 / 萨格勒布大学（LBCB）
- 主页：https://github.com/lbcb-sci/RiNALMo ｜ 邮箱：经主页核实
- 近作：RiNALMo（6.5 亿参数 RNA 语言模型，跨未见家族泛化，Nature Commun. 2025）
- 相关性：FM——最大的通用开源 RNA 基础模型，P3 RL 训练开放模型的强预训练骨干。

Sakana AI（David Ha；Robert Tjarko Lange）— 优先级 high ｜日本
- 单位：Sakana AI，东京
- 主页：https://sakana.ai/（RSI: sakana.ai/rsi-lab/）｜ 邮箱：无（机构条目）
- 近作：AI Scientist-v2（agentic 树搜索，2025；首篇全 AI 论文过评审，Nature 2026-03）；Darwin Gödel Machine（自改进代理，与 UBC/Jeff Clune）
- 相关性：代理自改进 / 元学习（P3-邻接）——正是本项目 Meta-Learning Harness 的核心命题旗舰。

Philippe Schwaller — 优先级 medium ｜瑞士
- 单位：EPFL 助理教授（人工化学智能实验室；NCCR Catalysis）
- 主页：https://schwallergroup.github.io/ ｜ 邮箱：经主页核实
- 近作：ChemCrow（18 工具增强 LLM 代理，Nat. Mach. Intell. 2024）
- 相关性：FM / 代理闭环——openclaude_rna 所专门化的“代理+工具”模式的范式作品（移植到分子设计）。

RiboPO 团队（通讯：Fang Wu, Yang Zhang；含 Minghao Sun, Yejin Choi, Pheng-Ann Heng）— 优先级 top/high ｜ 单位未定
- 单位：❌ 多机构，本轮未从 arXiv/OpenReview 解析出具体单位（出访前须核实作者块）
- 主页/来源：https://arxiv.org/abs/2510.21161 ｜ 邮箱：无
- 近作：RiboPO：以物理反馈的偏好优化/RL（3D 保真+热力学稳定的复合奖励）微调 gRNAde（arXiv 2510.21161, 2025）
- 相关性：P3-RL——最接近本项目“验证器奖励 RL on 设计轨迹”的已发表实例。 归属 Xie/UC Irvine 或 China 组均为误判。
