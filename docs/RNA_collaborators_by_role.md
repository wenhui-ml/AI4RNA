## 一档 · 互补型 RNA 生物 / 湿实验 / 结构 / oracle 领域专家 (role_fit = domain_expert) — 最符合你的需求

> 这一档是你(LLM/agent/RL 一侧)最该主动接触的人:他们几乎不碰你的赛道,却带来你完全没有的资产——湿实验验证、RNA 热力学与自由能参数、折叠 oracle 权威、基准与社区、真实行业网络。按 fit_priority(top → high)排序。

### top 级

**David H. Mathews** · University of Rochester Medical Center(生化与生物物理;RNA Biology 中心)
- 背景:M.D./Ph.D.;一辈子做 RNA 折叠热力学与二级结构算法,RNAstructure 的作者与维护者。纯算法/热力学,不做 ML。实验室:dry_computational。
- 互补价值:RNA 折叠热力学的头号权威——现代二级结构预测底层的近邻自由能参数与配分函数折叠算法就出自他。RNAstructure 本身就是一个可直接用来校准/交叉验证 ViennaRNA verifier 的折叠 oracle 权威。带来 LLM/agent 研究者最缺的物理能量模型,与你几乎零重叠。
- 项目阶段:P1(折叠 oracle / 热力学权威;RNAstructure 作为替代 verifier)。

**Mirela Andronescu** · BC Cancer / UW Genome Sciences(CS PhD,UBC,Condon 一脉)
- 背景:UBC CS 博士(导师 Anne Condon),代表作是 RNA 自由能**参数估计**(Boltzmann-likelihood / constraint-generation,即 Andronescu BL*/CG* 参数集)。现偏基因组/癌症生信,但招牌资产是 RNA 热力学。实验室:dry_computational。
- 互补价值:折叠 oracle 本身背后的热力学/能量参数权威——她的自由能参数被 mfold/ViennaRNA/RNAstructure/HotKnots 广泛采用,能直接校准和批判 verifier 的能量模型,是对一个 LLM/agent 设计者而言最贴 oracle 的互补点。计算型,非湿实验。
- 项目阶段:P1 oracle / 热力学校准。

**Qiangfeng Cliff Zhang(张强锋)** · 清华大学生命科学学院(生物信息学教育部重点实验室;清华-北大生命科学中心)
- 背景:USTC 计算机本硕博 + 哥伦比亚生物物理 PhD;运行湿实验 RNA 结构生物学组。发明 icSHAPE(高通量活细胞内 RNA 结构探测),构建 RASP 转录组结构图谱,绘制 SARS-CoV-2 与哺乳动物 RNA 结构组。实验室:hybrid(实验探测 + 计算)。
- 互补价值:带来你正缺的东西——活细胞内实验 RNA 结构 ground truth(icSHAPE/RASP)、RNA 结构生物学权威、可用于校准折叠 oracle 并让生成设计对标真实结构数据的结构数据集。
- 项目阶段:P1 oracle 校准 / P2 基准(实验结构 ground truth)。

**Yaoqi Zhou(周耀旗)** · 深圳湾实验室 系统与物理生物学研究所(兼深圳医学科学院)
- 背景:物理/生物物理出身的计算结构生物学家(曾任教 SUNY Buffalo、IUPUI、Griffith)。核心:RNA/蛋白结构预测(SPOT-RNA、早于 AlphaFold 的扭转角深度学习)、知识型能量函数、折叠、CASP15 RNA。用 DL 作工具,但**不是** LLM/agent/RL 研究者。实验室:hybrid。
- 互补价值:你校准和压测 ViennaRNA oracle、贡献替代结构 oracle、以及超越二级结构判断设计合理性所需的 RNA/蛋白结构预测、折叠与能量函数权威。
- 项目阶段:P1 oracle(结构/折叠权威)/ P2 基准。

**Ivo L. Hofacker** · University of Vienna 理论化学所(TBI)
- 背景:理论化学出身,维也纳理论化学系正教授,用约 15 年创建并领导 ViennaRNA Package。核心是 RNA 二级结构热力学与折叠算法;不做 ML。实验室:dry_computational。
- 互补价值:ViennaRNA 折叠包的创建者——也就是你项目 verifier 所构建之上的那个折叠 oracle;带来 LLM/agent 研究者缺乏的 RNA 热力学与二级结构权威。
- 项目阶段:P1(折叠 oracle / verifier)。

**Ronny Lorenz** · University of Vienna 理论化学所(TBI)
- 背景:莱比锡 CS 学位 + 维也纳理论化学博士;2008 年起 ViennaRNA Package 的主开发者与维护者,专攻软/硬折叠约束与修饰碱基折叠。dry computational,非 ML。
- 互补价值:ViennaRNA 的首要维护者——他实际拥有/维护你用作 verifier 的那个折叠 oracle;深厚的约束折叠与热力学功底可用于校准 oracle。
- 项目阶段:P1(折叠 oracle / verifier)。

### high 级

**Kevin M. Weeks** · UNC-Chapel Hill(Kenan 杰出化学教授;Lineberger 癌症中心)
- 背景:耶鲁 PhD(1992),UC Boulder 博后;一辈子的湿实验 RNA 化学家,不做 ML。实验室:wet_lab。
- 互补价值:奠基性实验 RNA 结构化学家,发明 SHAPE、SHAPE-MaP 及单分子相关化学探测——生成 RNA 结构 ground-truth 测量的领域标准"化学显微镜"。正好补上 LLM/agent 研究者完全没有的湿实验结构验证与真实试剂资源;与你零重叠。
- 项目阶段:P1(实验结构 ground-truth / oracle 校准);湿实验验证。

**Julius B. Lucks** · Northwestern(化学与生物工程;合成生物学中心联合主任)
- 背景:UNC 化学本科,哈佛化学物理 PhD(Hertz Fellow),UC Berkeley Miller Fellow 期间转向合成生物学。运行湿实验室(带计算分支)。实验室:wet_lab。
- 互补价值:实验 RNA 结构权威,开发 SHAPE-Seq 与共转录 SHAPE-Seq(核苷酸分辨率探测新生/共转录折叠),研究核糖开关折叠路径。带来高通量实验结构数据、折叠动力学/共转录专长、合成 RNA 工程,用于验证与校准 ML 预测——你完全没有的能力。
- 项目阶段:P1(实验结构数据用于 oracle 校准);湿实验验证。

**Elena Rivas** · Harvard(分子与细胞生物学;Eddy/Rivas 实验室)
- 背景:萨拉戈萨理论物理 PhD,UCLA 物理博后,后与 Sean Eddy 一起转向计算 RNA 生物学。做比较基因组学/共变统计——算法型,**明确不做 ML**(且公开对 ML 结构预测持怀疑态度)。实验室:dry_computational。
- 互补价值:基于进化的 RNA 结构权威:R-scape、CaCoFold、Infernal/Rfam 共变方法,判断一个结构在进化上是否真实。以严谨、怀疑的结构评估方法论 + Rfam 社区地位补足你,用来校准/批判 ML 设计产出。注意:她的共变/比较焦点是结构**评估**而非 oracle 下的逆设计,属部分而非直接契合。
- 项目阶段:P1/P2(预测/设计结构的 ground truth 与严格评估)。

**Joseph D. Yesselman** · University of Nebraska-Lincoln(化学系助理教授)
- 背景:罗切斯特物理本科,密歇根生物物理 PhD,Rhiju Das 实验室(斯坦福)博后。运行 wet+dry 混合实验室:RNA 3D 设计、RNA 纳米技术/纳米机器、RNA 热力学与结构预测。实验室:hybrid。
- 互补价值:Das 一脉的 RNA 设计实验家:带来湿实验 RNA 3D 设计/纳米技术构建 + 热力学,且熟悉项目基准来源的 Eterna 生态。为 ML 设计闭环提供实验设计/验证与热力学根基;自身工作是设计+实验而非 ML,与你重叠低。
- 项目阶段:P2(RNA 设计 baseline + 实验验证);设计 + 热力学。

**Zhen Xie(谢震)** · 清华大学自动化系 / 合成与系统生物学中心(前 MIT Weiss 实验室)
- 背景:合成/系统生物学与生物医学工程;在活体哺乳动物细胞中构建并实验测试 RNA 合成逻辑电路(RNAi 逻辑、RNA-IN/OUT 基因电路、CRISPR/Cas 调控),用于基因/细胞治疗。MIT Ron Weiss 合成生物学实验室训练。实验室:wet_lab(功能性 RNA 器件工程)。
- 互补价值:活细胞湿实验 RNA 器件工程与功能验证——一条检验计算设计的 RNA 是否真作为功能部件工作的路径,加上你缺乏的合成生物学应用网络(与折叠/热力学相邻,而非正面重合)。
- 项目阶段:P2 / 下游功能性湿实验验证。

**Zhi John Lu(鲁志)** · 清华大学生命科学学院(生物信息学教育部重点实验室)
- 背景:USTC 生物科学本科,罗切斯特生物物理 PhD(2008),耶鲁生信博后。运行 ncRNA/RNA 生信实验室:非编码 RNA 发现、RNA 二级结构与蛋白-RNA 相互作用预测、exRNA/癌症非编码标志物、数据库与流程。根植 RNA 生物学,ML 只作工具,非 LLM/agent。实验室:dry_computational。
- 互补价值:RNA 生物学(非编码 RNA)领域知识、RNA 结构/相互作用专长,以及可让你突破 Eterna100、把 harness 扎根真实 RNA 功能语境的策展 RNA 数据集、数据库与基准资源。
- 项目阶段:P2 基准/数据集(RNA 生物学 + 策展数据)。

**Christoph Flamm** · University of Vienna 理论化学所
- 背景:理论化学家(Dr.),副教授。做 RNA 折叠动力学、离散能量地形、代数化学、计算 RNA 设计;ViennaRNA 共同作者。dry computational,非 ML。
- 互补价值:RNA 折叠动力学与能量地形权威 + 计算(逆)RNA 设计——用结构/热力学设计理论与 ViennaRNA 血统补足 ML 研究者。
- 项目阶段:P1/P3(oracle + 能量地形奖励信号)。

**Peter F. Stadler** · Leipzig University(生信 / IZBI);Santa Fe Institute 外聘
- 背景:维也纳化学 PhD(1990),Manfred Eigen(马普,哥廷根)博后,理论化学 habilitation。生信正教授。dry computational/理论,非 ML。
- 互补价值:奠基性 RNA 生信与适应度/中性地形理论家,ViennaRNA 生态共同缔造者;带来热力学理论 + 巨大的 RNA 领域网络/社区。
- 项目阶段:P1/P3(热力学 + 用于 RL-on-traces 的适应度地形理论)。

**Rolf Backofen** · University of Freiburg 生信主任;de.NBI RNA 生信中心
- 背景:埃尔朗根 CS,萨尔兰 PhD(计算逻辑/约束编程)。核心:RNA 二级结构、RNA-RNA 相互作用、CRISPR-Cas、RNA 结合蛋白、Freiburg RNA 工具(IntaRNA、LocARNA);生命科学中用一些 ML。实验室:hybrid。
- 互补价值:运营一整套主流 RNA 生信工具/数据库生态与 de.NBI RNA 生信中心——带来单打独斗的 ML/agent 研究者缺乏的 RNA 领域广度 + 基准/社区/基础设施。
- 项目阶段:P1/P2(RNA 工具 + 基准/社区)。

**Sebastian Will** · École Polytechnique, LIX(生信教授)
- 背景:RNA 生信,师从 Hofacker(维也纳)与 Stadler(莱比锡);RNA 结构预测/比较/比对的组合算法与约束(LocARNA)、折叠动力学与假结。dry computational,非 ML。
- 互补价值:RNA 二级结构预测/比较与折叠算法权威(ViennaRNA/莱比锡血统)——带来结构 RNA 与约束折叠深度,用于校准 oracle 与基准。
- 项目阶段:P1/P2(结构算法 + 基准)。

**Yann Ponty** · CNRS / LIX, École Polytechnique
- 背景:巴黎第十一大学 CS PhD;CNRS 终身研究员。用枚举/解析组合学与动态规划做 RNA 结构;RNA 逆折叠/设计理论;Clote 结构生信实验室博后。dry computational。
- 互补价值:RNA 逆折叠/设计理论与结构系综组合学权威——提供与你逆设计任务直接相关的、有原则的 RNA 设计 baseline、采样算法与结构理论(接近 algorithm_tool 但是 RNA 专属)。
- 项目阶段:P2(RNA 设计 baseline / 理论)。

---

## 二档 · Hybrid(生物/化学根基 + 也做 ML)(role_fit = hybrid) — 部分互补,需看具体分工

> 这些人有真实的生物/化学根基**和**折叠/结构资产,但也确实在做 ML,和你有部分重叠。价值取决于能否谈成"他出领域/实验、你出 agent/RL"的分工。

**Rhiju Das** · Stanford(生化;HHMI)— **fit_priority: top,但是把双刃剑**
- 背景:哈佛物理本科、斯坦福物理 PhD、David Baker(UW)计算蛋白折叠博后。现生化教授,实验室同时做自研湿实验化学 mapping(高通量 SHAPE/DMS 探测)与重度计算/ML RNA 建模。
- 互补价值:名单上最高价值的领域资产——他本人领导 Eterna(25 万玩家众包 RNA 设计社区),因而拥有项目所依赖的 Eterna100/Eterna100-V2 基准,外加产出实验化学 mapping 验证(Ribonanza)的湿实验室。⚠️ 警告:他**同时**造 ML RNA 模型(RibonanzaNet、EternaFold),在设计/算法侧与你重叠,可能把你的工作读成竞品。
- 项目阶段:P2(拥有 Eterna100-V2 基准 + 设计社区);P1(化学 mapping oracle 校准)。

**Kengo Sato(佐藤健吾)** · 庆应 / Institute of Science Tokyo(生物科学与信息学系)— high
- 背景:庆应 CS PhD(2003),RNA 信息学。以把深度学习与 Turner 近邻热力学结合做 RNA 二级结构预测著称(MXfold2,热力学积分/正则化)。生信 + ML。
- 互补价值:RNA 二级结构预测与热力学积分专家(MXfold2)——提供替代/互补折叠预测器与深厚热力学知识来校准 verifier;在 ML 侧部分重叠。
- 项目阶段:P1(折叠 oracle 校准 / 替代预测器)。

**Hannah K. Wayment-Steele** · UW-Madison(生化系助理教授)— medium
- 背景:斯坦福化学 PhD;Vijay Pande 组做蛋白动力学深度学习,Rhiju Das 实验室做 RNA 结构 ML(EternaFold、EternaBench、mRNA 稳定化设计)。新 PI,建"数据科学/DL/实验交叉"实验室。
- 互补价值:化学根基 + 正在搭实验生物分子动力学实验室,并拥有深厚 EternaBench/Eterna 基准知识——部分互补。但她已证明的核心强项(EternaFold、基于 DL 的 RNA 结构预测)恰是你已占据的 ML/RNA 建模赛道,互补有限,更像建模侧的 peer。
- 项目阶段:P2(EternaBench 基准专长)/ P3(ML)——但与你的 ML 赛道重叠。

**Pietro Liò** · University of Cambridge(计算机系,AI 组)— medium
- 背景:两个 PhD——复杂系统/非线性动力学(佛罗伦萨)与理论遗传学(帕维亚)。计算生物学教授,但当前工作以 AI/图神经网络与多组学/精准医学深度学习为主。
- 互补价值:部分根植遗传学/计算生物学,但现主要是 AI/GNN 研究者;带来宽的计算生物学 + 深度学习广度(非 RNA 专属、非湿实验、非热力学),部分重叠你的 ML 赛道。
- 项目阶段:P3(ML/GNN 方法论)。

**Philippe Schwaller** · EPFL(人工化学智能实验室 LIAC)— low
- 背景:伯尔尼化学 PhD(Reymond 组)+ 剑桥物理 MPhil,EPFL 本硕;前 IBM Research。做化学的 ML/LLM(Molecular Transformer、逆合成预测、化学 LLM agent)。化学根基但以 ML 为主。
- 互补价值:化学根基但主要是 LLM/ML-for-science 研究者;其领域是小分子/反应化学而非 RNA——与你的 LLM-for-science 方法论重叠,无 RNA 专属领域资产。
- 项目阶段:P3(LLM-for-science 方法论)——重叠/相邻。

---

## 三档 · 算法/oracle 工具专家 (role_fit = algorithm_tool) — 非生物领域专家,但项目基础设施必需(oracle / baseline / 理论)

> 他们不是 RNA 生物学领域专家,给的是工具、baseline、理论,而非湿实验或行业网络。项目基础设施上必需,但不填补"领域"这一维度。

**Anne Condon** · University of British Columbia CS(ACM Fellow, RSC Fellow)— high
- 背景:UCC 本科;华盛顿大学 CS PhD(1987)。计算复杂性/算法;RNA/DNA 二级结构预测、RNA 设计(逆折叠)、分子编程/DNA 计算的长期权威。dry_computational。
- 互补价值:RNA 结构预测与 RNA 设计(逆折叠)算法/理论资深权威 + 分子编程社区地位——提供 ML/agent 研究者缺乏的形式化 RNA 结构/oracle 基础与领域公信力;非湿实验非 ML,接近 oracle-社区领域权威。
- 项目阶段:P1 oracle 理论 / P2 RNA 设计 baseline 与基准。

**Liang Huang(黄亮)** · Oregon State CS(生化/生物物理兼聘)— high
- 背景:UPenn CS PhD(2008),根在计算语言学/NLP(句法分析、SCFG),后把这些方法迁移到 RNA。LinearFold/LinearPartition(线性时间折叠)与 LinearDesign(mRNA 设计,Nature 2023)作者。dry_computational。
- 互补价值:快速 RNA 折叠算法/oracle(LinearFold、LinearPartition)与算法型 mRNA 设计(LinearDesign)权威,兼聘生化——提供生产级折叠 oracle/baseline 与折叠社区地位;与你 LLM/agent 赛道不同,虽 NLP 根子与 ML 相邻。非湿实验生物学家。
- 项目阶段:P1 oracle / P2 算法设计 baseline。

**Niles A. Pierce** · Caltech(应用与计算数学、生物工程教授;Molecular Instruments 联合创始人)— medium
- 背景:普林斯顿机械/航天工程本科(致辞代表),牛津应用数学 DPhil(Rhodes 学者)。应用数学家/生物工程师;NUPACK 作者。
- 互补价值:NUPACK 作者——热力学系综分析与核酸序列**设计**算法。提供折叠/设计 oracle 与强逆设计 baseline(项目必需的基础设施/理论),但他是应用数学家而非 RNA 生物学领域专家,给的是工具/baseline 而非湿实验或行业网络互补。
- 项目阶段:P1/P2(NUPACK 作折叠 oracle 与逆设计 baseline)。

**Hosna Jabbari** · University of Alberta 生物医学工程(前 U Victoria);UBC PhD(Condon 一脉)— medium
- 背景:UBC CS/计算生物学 PhD(师从 Condon)。副教授,做 RNA 二级结构预测算法,尤其假结(HFold、Knotty、CCJ、Spark)与能量模型;有些 RNA 治疗/RBP/疾病框架。dry_computational。
- 互补价值:假结型 RNA 折叠算法与能量模型权威——可把 ViennaRNA oracle 扩展/压测到其无假结基础范围之外,并添折叠社区地位;折叠算法专长,非 ML 非湿实验。
- 项目阶段:P1 oracle(假结扩展)。

**Ryan K. Krueger** · Harvard SEAS,Michael Brenner 组(应用数学博士生)— medium
- 背景:哈佛应用数学博士生(Brenner 组),做生物分子的可微/逆设计;JAX-RNAfold(可微 RNA 折叠)与梯度/过参数化序列设计方法作者。dry_computational。
- 互补价值:构建 JAX-RNAfold(可微 RNA 折叠 oracle)+ 梯度/过参数化设计 baseline——项目必需的 oracle 工具与强的非 LLM 设计 baseline;是应用数学方法学家而非生物领域专家,且资历尚浅(无实验室资源)。
- 项目阶段:P1 可微 oracle / P2 非 LLM 设计 baseline。

**Ke Li(李可)** · University of Exeter, COLA-Lab(UKRI Future Leaders / 图灵 Fellow)— low
- 背景:CS;进化多目标优化、贝叶斯优化,近期有 ML 与 LLM 工作。无 RNA/生物领域。ml_ai(role_fit=algorithm_tool)。
- 互补价值:提供多目标/进化优化方法论与 baseline(及一些 LLM 工作);非 RNA 生物学专家——优化器/基础设施,与你自己的 ML 赛道相邻。
- 项目阶段:P2(多目标优化 baseline)。

**Holger H. Hoos** · RWTH Aachen / Leiden University — low
- 背景:CS PhD;经验算法学、算法选择与配置(ParamILS、Auto-WEKA、SAT 求解)、AutoML 与基准测试。无 RNA/生物领域。
- 互补价值:带来 AutoML/算法配置/基准方法论(优化器与评估基础设施),非 RNA 生物学专长——与你的 ML/工具赛道重叠。
- 项目阶段:P2/P3(算法配置/基准方法论)。

**Michael Lindsey** · UC Berkeley 数学系助理教授(LBNL AMCR)— low
- 背景:UC Berkeley 应用数学 PhD(2019),Courant 博后。数值线性代数、优化、随机化、量子蒙特卡洛、量子化学/凝聚态的张量/神经网络。**未发现 RNA 工作**。
- 互补价值:应用数学/量子蒙特卡洛方法学家,无 RNA 领域资产;其优化/采样理论落在(或紧邻)你自己的方法赛道,而非用生物学补足你。
- 项目阶段:P3(优化/采样理论)——你自己的赛道。

---

## 四档 · 同赛道 ML/LLM/agent/RL (role_fit = ml_agent_peer) — 与你重叠,建议当 peer/竞品/baseline 而非"领域专家合作方"

> 这些人本质上和你在同一条赛道(ML/LLM/agent/RL/生成式设计)。他们是很好的 baseline、竞品、参考,但**不是**你要找的互补型领域专家。逐个点明重叠点:

- **Chaitanya K. Joshi**(Stanford Das 实验室博后;剑桥 CS PhD,几何深度学习)——**重叠**:gRNAde 本身就是深度学习 3D RNA 逆设计流程(同一任务),直接 peer/baseline/竞品。唯一部分互补:Das 实验室/Holliger-LMB 关系可通往湿实验验证。→ P2 baseline(gRNAde)。
- **Wengong Jin(金文光)**(Northeastern CS & Broad;MIT CSAIL PhD)——**重叠**:生成式 AI 分子与 RNA 设计(RNAFlow)是直接 ML RNA 设计 peer/baseline;带来 ML 方法而非湿实验/热力学/oracle 资产。→ P2 baseline(RNAFlow)。
- **Divya Nori**(MIT EECS;与 Wengong Jin、Caroline Uhler 合作)——**重叠**:RNAFlow(ICML 2024)共同第一作者,生成/流匹配 RNA 设计,纯 ML 背景的早期研究者,同赛道 peer。→ P2 baseline(RNAFlow)。
- **Xiaohui Xie(谢晓辉)**(UC Irvine CS)——**重叠**:基因组学深度学习研究者(含 DL RNA 二级结构),ML 赛道 peer/baseline,非湿实验或热力学领域专家。→ P2 DL baseline。
- **James Zou(邹昊)**(Stanford 生物医学数据科学)——**重叠**:主流 ML/AI-for-biomedicine 研究者(含生物中的 LLM/agent);带来 ML 方法而非湿实验/热力学/oracle 资产。→ P3 ML/RL 方法。
- **Jure Leskovec**(Stanford CS,SAIL/CRFM)——**重叠**:奠基性图 ML/GNN 研究者,同 ML 赛道;无 RNA 生物学/湿实验/热力学资产来补足 ML/agent 设计者。→ P3 ML/GNN 方法。
- **Yu Li(李煜)**(CUHK CSE,"AI in Healthcare")——**重叠**:USTC 生物本科但 KAUST(Xin Gao 组)CS 硕博,本质是 ML-for-biology 方法研究者,而非湿实验/结构/热力学领域专家,充当 peer/baseline。→ P3 ML 方法 peer。
- **Siqi Sun(孙思琦)**(复旦智能复杂体系研究所 & 上海 AI Lab)——**重叠(用户自己举的例子)**:数学本科 → CS → 生物分子基础模型/LLM(蛋白 contact map、RhoFold RNA 3D 结构 LM、抗体语言模型),与你占据的赛道完全重合,是 peer/竞品/baseline,而非互补的 RNA 生物学领域专家。→ P3 LLM-for-bio peer。
- **Stan Z. Li(李子青)**(西湖大学 AI 研究中心 CAIRI)——**重叠**:职业 ML/深度学习方法研究者(现 AI4Science),无湿实验、RNA 热力学或折叠 oracle 权威——peer/baseline。→ P3 ML 方法 peer。
- **Yang Yang(杨洋) & Hongbin Shen(沈红斌)**(上海交大;DRAG 作者)——**重叠**:模式识别/ML 教授把 ML 用于生信;DRAG = "把 RNA 设计成分层图 + 强化学习",即基于 RL 的 RNA 逆设计系统,直接撞你的 P3 RL-on-traces RNA 设计工作,是 baseline/竞品而非互补领域专家(但 DRAG 是有用的参考 baseline)。→ P2/P3 直接 RL RNA 设计 baseline。
- **Le Song(宋乐)**(MBZUAI ML 系主任 & GenBio AI CTO)——**重叠**:构建生物分子基础模型与多智能体 AI-for-Science 系统,同一赛道,是 peer/竞品而非湿实验/热力学互补。→ P3 基础模型 + agent peer。
- **Eric Xing(邢波)**(MBZUAI 校长 & CMU CS & GenBio AI 联创/首席科学家)——**重叠**:尽管有分子生物学 PhD(Rutgers),其职业与现实验室是统计 ML、概率图模型、分布式 ML 系统、LLM/基础模型平台(Petuum、GenBio);生物 PhD 只给领域素养,运营赛道是 ML/LLM 系统,是 peer/竞品而非湿实验或折叠 oracle 领域专家。→ P3 ML 系统 + 基础模型 peer。
- **Mile Šikić**(A*STAR GIS & 萨格勒布大学 CS)——**重叠**:计算基因组学 & AI(Racon、Raven、HERRO、RiNALMo——迄今最大的 RNA 语言模型);RiNALMo 是直接的 RNA 语言模型 peer/竞品。可作 baseline/embedding 来源,但他是 CS/AI 构建者,非湿实验/RNA 热力学/折叠 oracle 领域互补。→ P3 RNA-LM peer(可能 baseline/embedding)。
- **Frederic Runge**(University of Freiburg,AutoML/Hutter 组)——**重叠**:Meta-LEARNA 是 RL + 元学习 RNA 逆设计,与你完全同一赛道的 baseline/竞品,而非领域专家互补。→ P2 baseline / P3 RL。
- **Frank Hutter**(University of Freiburg;Prior Labs;ELLIS Tübingen)——**重叠**:AutoML 先驱(Auto-sklearn、Auto-PyTorch、TabPFN),LEARNA 背后的资深 PI;无 RNA 生物学领域资产,元层面上和你同一 ML 赛道。→ P2/P3(AutoML/RL 方法论)。
- **David Ha(Sakana AI 联创/CEO)**(东京)——**重叠**:纯 LLM/agent + 进化 AI 实验室(World Models、进化式模型合并、AI Scientist agent),正是你的赛道;无 RNA/湿实验/热力学资产,peer/竞品而非领域互补。→ P3(LLM/agent 方法论)。

---

## 🎯 首选(互补型)Top-6 outreach 名单

> 只从"一档"里选,每人一句"他能给你的 LLM/agent 工作补上的具体领域资产"。这 6 人共同覆盖了三类你最缺的资产:折叠 oracle 与热力学能量模型权威、实验结构 ground-truth 湿实验、以及结构预测/评估权威。

1. **David H. Mathews(Rochester)** — 补上你没有的物理能量模型:RNAstructure 与近邻自由能参数,可作为独立于 ViennaRNA 的第二折叠 oracle 来交叉校准你的 verifier。
2. **Mirela Andronescu(BC Cancer/UW)** — 补上折叠 oracle 底层的自由能参数本身(BL*/CG* 参数被各大折叠器采用),能直接审查并校准你 verifier 的能量模型——最贴 oracle 的单点互补。
3. **Ivo Hofacker(Vienna, ViennaRNA 创建者)** — 补上你 verifier 所依赖的那个折叠包本人的热力学与二级结构权威,可从源头解释和校准 oracle 行为。
4. **Ronny Lorenz(Vienna, ViennaRNA 首席维护者)** — 补上对你实际使用的那套 verifier 代码的维护级掌控与约束折叠专长,是让 oracle 定制/校准落地的最直接人选。
5. **Qiangfeng Cliff Zhang(清华,icSHAPE/RASP)** — 补上你完全没有的活细胞内实验 RNA 结构 ground-truth 与结构组数据集,让生成设计能对标真实结构而非只对标算法 oracle。
6. **Kevin M. Weeks(UNC,SHAPE/SHAPE-MaP 发明者)** — 补上领域标准的湿实验结构测量与真实试剂/湿实验资源,为你的 ML 设计提供实验闭环验证。

> 备选(同为一档 top/high,视方向替换):**Yaoqi Zhou(深圳湾,结构预测/能量函数)**、**Julius Lucks(Northwestern,共转录 SHAPE-Seq 湿实验)**、**Rolf Backofen(Freiburg,RNA 工具生态 + 社区/基准基础设施)**。若你更看重"设计社区 + 基准所有权",还可谈 **Rhiju Das**,但他属二档 hybrid、在 ML 侧与你重叠、可能视你为竞品,需谨慎定位分工。

---

## ⚠️ 关键提醒

**原名录(按地区组织)最容易把这些 ML 人误当成"领域专家",实则与你同赛道:**

- **Eric Xing** 有 Rutgers 分子生物学 PhD,极易被当成"生物领域专家"——但他的实际运营赛道二十年来是统计 ML / 概率图模型 / 基础模型平台(Petuum、GenBio),生物 PhD 只给素养不给湿实验或 oracle 资产,是同赛道 peer。
- **Siqi Sun** 挂在生命科学机构、做 RNA 3D 结构模型(RhoFold),表面像 RNA 专家——但正如你自己指出:数学→CS→生物大模型,是纯 ML/LLM 路径,与你完全重叠。这是全名录最典型的"看起来像领域专家、其实是你的镜像"案例。
- **Yu Li**、**Stan Z. Li**、**Le Song**、**Mile Šikić** 都挂"AI for bio / 生物基础模型 / RNA 语言模型"招牌,但学术根子与产出都是 ML 方法(CS PhD / 模式识别 / 基础模型),没有湿实验、热力学或折叠 oracle 权威——都是 baseline/竞品。
- **Yang Yang & Hongbin Shen(DRAG)** 与 **Frederic Runge(Meta-LEARNA)** 是最需要警惕的"伪合作方":他们做的正是 RL RNA 逆设计,和你的 P3 是正面竞品,不该被列为"领域专家",但作为 baseline 极有价值。
- **Chaitanya Joshi(gRNAde)/ Wengong Jin & Divya Nori(RNAFlow)** 挂在 Das/Broad 等生物招牌下,容易被误读为"生物侧合作者"——实则是生成式 3D RNA 逆设计的直接同任务竞品。
- **Wayment-Steele** 有真实化学根基和新建实验室,但已证明的核心是 EternaFold/DL 结构预测——落在你已占的建模赛道,互补有限,更像 peer。
- **Hutter / Hoos / Ke Li / David Ha / Lindsey / Schwaller** 都是 ML/AutoML/优化/agent 或非 RNA 化学,给的是方法与工具,不是 RNA 领域互补。

**反过来,真正互补(湿实验 / 结构 / 热力学 / oracle 权威),但在按地区排的旧名录里被稀释、没被凸显的人:**

- **折叠 oracle / 热力学能量模型权威**(你 verifier 的地基,却最缺):Hofacker、Lorenz(ViennaRNA 本体)、Mathews(RNAstructure)、Andronescu(自由能参数本身)、Flamm、Stadler、Sato(MXfold2)、Yaoqi Zhou。
- **湿实验结构 ground-truth(你零能力的一侧)**:Weeks(SHAPE)、Lucks(共转录 SHAPE-Seq)、Q.C. Zhang(icSHAPE/活细胞)、Zhen Xie(活细胞功能 RNA 器件验证)。
- **结构评估 / 进化真实性 / 社区与基准所有权**:Rivas(R-scape/Rfam,且对 ML 结构预测持怀疑——正好当你的严格审稿人)、Backofen(de.NBI RNA 中心 + 工具生态)、Zhi John Lu(策展 RNA 数据集,助你突破 Eterna100)、Will、Ponty、Condon(RNA 设计理论与 oracle 社区公信力)。

一句话:旧名录按"地区"排,把张强锋、周耀旗这些真正的实验/结构互补者与孙思琦、Eric Xing 这些 ML 镜像混在同一"中国/跨区"栏里;按 role-fit 重排后,你该主动接触的是一档的 oracle/热力学/湿实验权威,而四档那批人当 baseline 引用即可,不要误投为"领域合作方"。