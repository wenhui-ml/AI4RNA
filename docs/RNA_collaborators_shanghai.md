# 上海 RNA 合作方补充目录 — InceptLabs(上海 + 硅谷)

> 定位:为你(自建 RNA 逆向设计 meta-learning harness 的 LLM/Agent/RL 侧,verifier = ViennaRNA 折叠 oracle,benchmark = Eterna100-V2,红线 = 只在计算折叠 oracle 下判成功、绝不 in-vivo)寻找**互补型 RNA 生物学领域专家**——湿实验 RNA 结构/探测、RNA 热力学、结构生物学、RNA 治疗、RNA 修饰/ncRNA、折叠 oracle 权威。凡本身主业是 ML/LLM/Agent/RL 的人,记为**同赛道镜像(peer/baseline)**,互补性低。上海本地为加分项(可做面对面协作 + 湿实验闭环 + 产业对接)。
> 数据来源:各校官方页(fudan.edu.cn / ibs.fudan.edu.cn / life.fudan.edu.cn / siidb.fudan.edu.cn / shsmu.edu.cn / life.sjtu.edu.cn 等)、张江 mRNA 国际创新中心(shmitc.org.cn)、本人主页及论文。邮箱只在有官方/自建来源时给出并标置信度,未见即不臆造。

---

## 复旦大学(优先)

### 一档 · 互补型领域专家(domain_expert)— 最该接触

**陈东戎(Dongrong Chen)· 生物医学研究院 IBS**
- 背景 / 实验室类型:研究员、博导;**湿实验**核酸高级结构与功能生物化学家。核酸高级结构与功能、riboswitch(核糖开关)与新型 ribozyme(核酶)的发现与结构-功能、riboswitch 调控抗生素抗性基因、RNA 修饰/RNA 表观遗传。
- 互补价值:**结构化 RNA 折叠-功能权威**——正是 ViennaRNA 式折叠 oracle 设计闭环最直接需要的"结构化 RNA / 热力学"领域,提供"序列如何折叠成功能结构"的真值。全目录中对折叠 oracle 命中度最高。
- 项目阶段:P1 / oracle 真值 / 结构 / 湿实验
- 主页:https://ibs.fudan.edu.cn/9f/e5/c54533a761829/page.htm · 邮箱:`drchen@fudan.edu.cn`(置信度**高**,官方 IBS 页)

**甘建华(Jianhua Gan)· 生命科学学院 / 遗传工程国家重点实验室**
- 背景 / 实验室类型:研究员、博导;**湿实验** X 射线晶体学家。北医本科(1997)、中科院上海有机所化学博士(2002),NIH-NCI 与纪念斯隆凯特琳博后,2012 回复旦。解析核酶/催化核酸结构、DNA 复制修复酶与病原蛋白;共同解出首个 SAM 依赖甲基化核酶晶体结构。
- 互补价值:**原子分辨率核酶/核酸实验晶体结构**——为计算折叠 oracle 提供硬结构真值来锚定/验证,正是 ML 侧最缺的湿实验-结构资产。
- 项目阶段:P1 / 结构
- 主页:https://life.fudan.edu.cn/b2/f5/c28175a307957/page.htm · 邮箱:`ganjhh@fudan.edu.cn`(置信度**高**,官方生科院页)

**麻锦彪(Jinbiao Ma)· 生命科学学院 / 生物化学与分子生物学系(遗传工程国重室副主任)**
- 背景 / 实验室类型:教授、博导(1973 生)。复旦化学本科(1997)、有机所博士(2002),纪念斯隆凯特琳博后、UAB 助理教授,2011 回复旦。**湿实验** RNA-蛋白结构生物学(X 射线 + 冷冻电镜):m6A writer(METTL3/METTL14)与 reader(YTHDF2)、hnRNP A2/B1-RNA 识别、Dicer-2-Loqs-PD dsRNA 加工、C3PO、small-RNA 3' 修剪。
- 互补价值:**RNA 及 RNA 修饰/沉默机器的原子分辨率结构**——深度序列-结构与修饰生化,把 ML 设计模型锚定到真实 RNA-蛋白识别。
- 项目阶段:P1 / 结构
- 主页:https://life.fudan.edu.cn/66/0f/c31281a353807/page.htm · 邮箱:`majb@fudan.edu.cn`(置信度**中-高**;来自其论文/ResearchGate,符合复旦标准命名,未在单一官方页当场核实——outreach 前建议经主页确认)

**林金钟(Jinzhong Lin)· 生命科学学院 / 遗传工程国重室;复旦 mRNA 药物研发中心创始主任;张江 mRNA 国际创新中心主任(兼中山医院研究员)**
- 背景 / 实验室类型:研究员、博导(1979 生)。中国农大本科(2001)、中科院生物物理所博士(2008),耶鲁 Thomas Steitz(诺奖,核糖体)博后(2011-16),2017 回复旦。**湿实验**核糖体结构生物学(高分辨冷冻电镜/X 射线/多维 NMR)、真核 5'UTR-翻译起始因子互作、核糖体靶向抗生素/抗性;2018 建成复旦 mRNA 平台(加帽/修饰/递送/制剂),联合领衔中国首个国产 mRNA 新冠疫苗;解出完整核糖体回收复合物。
- 互补价值:**mRNA 结构/翻译权威 + 端到端 mRNA 治疗与临床/产业管线**——把 RNA 结构一直接到真实 mRNA 治疗设计与制造,是最理想的**上海湿实验/治疗闭环**入口,可对设计出的构建体做实验真值。
- 项目阶段:P1 / 治疗 / P2 benchmark 验证
- 主页:https://life.fudan.edu.cn/b3/74/c28175a308084/page.htm(另 https://siidb.fudan.edu.cn/ljz/list.htm) · 邮箱:`linjinzhong@fudan.edu.cn`(置信度**中**;聚合列表见到,未在单页当场核实)

**胡璐璐(Lulu Hu)· 生物医学研究院 IBS**
- 背景 / 实验室类型:准聘副教授、青年 PI,国家级青年人才(2020)。**湿实验 + 方法学**表观转录组:超低起始量单碱基分辨 RNA 修饰(m6A)检测(与芝加哥 He Chuan 合作 m6A-SAC-seq,Nat Biotechnol 2024)、超低起始表观组检测(Uli-epic,~1 ng cfDNA,Genome Biology)、细胞表面糖基化 ncRNA(glycoRNA)与免疫。
- 互补价值:**定量单碱基 RNA 修饰测量技术**——产出修饰感知模型/verifier 训练与验证所需的实验表观转录组标签/数据集;可作**修饰感知 oracle 扩展**。
- 项目阶段:湿实验 / P1(修饰感知 oracle 扩展)
- 主页:https://ibs.fudan.edu.cn/e3/cd/c54534a254925/page.htm · 邮箱:`luluhu@fudan.edu.cn`(置信度**存在分歧**:一次检索标高置信、另一次未在页面见到——建议 outreach 前经 IBS 主页确认)

**陆伟(Wei Lu)· 药学院;智能化递药教育部重点实验室副主任**
- 背景 / 实验室类型:**湿实验**药剂/递送科学家。非病毒 & LNP 核酸/mRNA 疫苗递送系统、诊疗一体化纳米药、多模态分子影像探针、肿瘤放/化/免疫纳米药。Nat Biotechnol、Nat Commun。
- 互补价值:**LNP / 非病毒递送**——把计算设计出的 RNA 变成可递送、可测的治疗实体(制剂 + 体外/体内)。注意你的红线把"成功"定在计算 oracle 上,故这是**下游转化伙伴**,非 oracle。
- 项目阶段:湿实验 / 治疗递送
- 主页:https://spfdu.fudan.edu.cn/cc/e4/c41556a511204/page.htm · 邮箱:`wlu@fudan.edu.cn`(置信度**中**,电话 021-51980185)

**徐彦辉(Yanhui Xu)· 生物医学研究院 IBS & 附属肿瘤医院**
- 背景 / 实验室类型:研究员、博导。**湿实验**转录与表观调控结构生物学;RNA Pol II / Pol III 转录起始超级复合物、延伸复合物、转录终止、转录泡冷冻电镜;重建 16 个连续从头转录起始状态。
- 互补价值:世界级转录机器结构生物学(Pol II/III、RNA 合成)。**邻近而非核心**:讲"RNA 如何被合成"而非折叠/设计——很强的结构生物学盟友,但偏离折叠 oracle 中心一步。
- 项目阶段:结构
- 主页:http://xulab.fudan.edu.cn · 邮箱:`xuyh@fudan.edu.cn`(置信度**高**)

**应天雷(Tianlei Ying)· 上海医学院 / 基础医学院;上海市重大传染病与生物安全研究院 SIIDB;合成免疫学工程中心主任**
- 背景 / 实验室类型:**湿实验**治疗性抗体/免疫工程。全人源治疗 mAb 与纳米抗体(抗 SARS-CoV-2、抗 H7N9);近年 mRNA-LNP 编码核酸抗体(抗 HBsAg mRNA-LNP,mBio,与林金钟合作)。100+ 论文,25 专利,多项转化。
- 互补价值:**mRNA/RNA 治疗转化专家(mRNA 编码抗体)**,有临床/产业管线且上海本地——利于下游 P2 验证与产业对接;偏治疗应用而非折叠 oracle 核心。
- 项目阶段:湿实验 / 治疗转化
- 主页:https://siidb.fudan.edu.cn/ytl/list.htm · 邮箱:未在官方页见到(置信度**低**,经主页/研究院联系)

**鞠佃文(Dianwen Ju)· 药学院 生物制药学系(系主任)**
- 背景 / 实验室类型:教授、博导。**湿实验**生物制药,聚焦抗体与 mRNA 免疫治疗药物研发;张江 mRNA 国际创新中心成员。
- 互补价值:**mRNA 免疫治疗开发 + 转化/临床前管线**——设计 harness 产出 RNA 构建体的下游治疗应用伙伴(P3/湿实验转化)。
- 项目阶段:治疗
- 主页 / 邮箱:未找到官方页,仅经张江 mRNA 中心团队页确认姓名/单位(来源 shmitc.org.cn/research/team)——**未核实,不臆造**

**邱敏(Min Qiu)· 人类表型组研究院**
- 背景 / 实验室类型:青年研究员。**湿实验** mRNA 纳米药:靶向 LNP mRNA 递送系统;张江 mRNA 中心描述为 mRNA 纳米药递送专家。
- 互补价值:**LNP / mRNA 递送制剂**——把设计序列变成可测治疗体的物理递送层,下游互补能力,与 LLM/agent 赛道不重叠。
- 项目阶段:治疗
- 主页 / 邮箱:未找到官方页,仅经张江 mRNA 中心团队页确认(来源 shmitc.org.cn/research/team)——**未核实,不臆造**

**沈宏杰(Hongjie Shen)· 生物医学研究院 IBS**
- 背景 / 实验室类型:研究员。**湿实验** RNA 与 RNA 结合蛋白在染色质调控中的作用、表观修饰筛选/鉴定;METTL3/m6A 调控新生 RNA 转录(Molecular Cell,与蓝斐合作);曾与 Yang Shi(牛津)共同通讯。
- 互补价值:**RNA 结合蛋白与染色质/表观修饰机制湿实验**;邻近 RNA 结构/功能但以染色质调控为中心,是**次级/支撑**性 RNA 领域联系人,而非核心折叠/结构盟友。
- 项目阶段:湿实验 / RNA 修饰生物学
- 主页:https://ibs.fudan.edu.cn/55/e5/c21164a415205/page.htm · 邮箱:`hongjieshen@fudan.edu.cn`(置信度**高**,官方 IBS 页)

### 二档 · hybrid(干湿/计算结合)

**杨力(Li Yang)· 生物医学研究院 IBS / 医学遗传研究院 / 附属儿科医院**
- 背景 / 实验室类型:研究员、博导;**hybrid(计算偏重)** RNA 生物学家。外显子来源环状 RNA(circRNA)的生物合成/加工/功能;建立并维护 CIRCpedia(>2.6M circRNA / 20 物种);全基因组碱基编辑分析管线;陈玲玲(中科院上海)长期合作者。Google Scholar 引用 >16,000–23,000。
- 互补价值:**circRNA/ncRNA 领域深度 + 大规模 RNA-seq 管线与精选数据库(CIRCpedia)**。计算侧与你部分重叠,但领域锚在 circRNA,供数据/benchmark 资产(利于 P2/P3),非纯 ML 同行。
- 项目阶段:P2/P3(数据与 benchmark)/ ncRNA 数据
- 主页:https://imgg.fudan.edu.cn/kydw/jzPI/yl.htm · 邮箱:未在官方页见到(置信度**低**,经实验室/研究院页联系)

### 三档 · 算法工具

- 复旦本次未识别出独立的"纯算法/工具型"RNA 折叠热力学理论 PI(区别于结构生物学家与 ML 同行)。如需 ViennaRNA 式二级结构预测理论家,复旦范围内**结果偏弱**,可能需其他机构。

### 四档 · 同赛道 ML 镜像(peer/baseline)— 互补性低

**孙思琦(Siqi Sun)· 智能复杂体系基础理论与关键技术实验室 / 智能复杂体系研究院(教育部前沿);兼上海人工智能实验室**
- 背景 / 实验室类型:**ML/深度学习 + LLM**(AI4Science)。语言模型 DL 做 RNA 3D 结构预测(Nature Methods 2024)、RNA 二级结构预测(PriFold,AAAI 2025)、CRAFTS(~9 亿序列对比 RNA 结构筛选)、IntelliFold 生物分子结构基础模型、OriGene 智能体科学发现系统。复旦数学本科,TTIC CS 博士(Jinbo Xu),前微软研究院。
- 判定:**与你同赛道**(ML/AI-for-RNA:结构预测/设计 + 智能体科学),是 peer/baseline 而非所需领域专家。但仍有用:其 RNA 结构模型可作**额外折叠 oracle/baseline**,且上海本地(你此前想确认的复旦 AI4Science 组)。其 CRAFTS 共一 吴浩(Wu Hao)、许晟(Xu Sheng)为博士生,非 PI。
- 项目阶段:P1 oracle 模型 / P3 RL(ML 侧)
- 主页:https://intersun.github.io/ · 邮箱:`siqisun@fudan.edu.cn`(置信度**高**,自建主页)

**RhoFold+ 工作(复旦智能复杂体系研究院参与)**
- RNA 语言模型 3D 结构预测(Nature Methods 2024;与港中文、哈佛、上海 AI Lab、Zelixir 合作)。属 **AI/ML-for-RNA** ——与你同赛道(peer/baseline)。**未能定位到单一明确署名的复旦 PI**,故列此处而非作为个人;若需归属请再核实。

---

## 上海交通大学

### 一档 · 互补型领域专家(domain_expert)— 最该接触

**雷鸣(Ming Lei)· 上海精准医学研究院(常务副院长)/ 医学院附属第九人民医院**
- 背景 / 实验室类型:哈佛生物物理博士,前密歇根大学教授。**湿实验**大分子组装体结构生物学(X 射线/NMR/冷冻电镜),聚焦基因组稳定性。解出端粒酶 RNA–TERT 逆转录酶复合物晶体结构;端粒/端粒酶 RNP、H/ACA RNP 深度工作。60+ CNS 论文,院士级权威。
- 互补价值:**顶级实验 RNP / RNA-蛋白结构权威**——提供 ML 折叠 oracle 自身永远无法生成的物理结构真值。
- 项目阶段:结构 / P1 oracle 验证
- 主页:http://www.shipm.cn/…/jyjz_eg_lmz_sysjj/List/index.htm · 邮箱:`leim@shsmu.edu.cn`(置信度**中**,来自 shsmu 英文页检索片段,建议页面确认)

**吴健(Jian Wu)· 上海精准医学研究院 / 第九人民医院**
- 背景 / 实验室类型:教授 / co-PI。**湿实验**明确做"长非编码 RNA-蛋白复合物的结构与功能"(结构生物学 + 生化 + 细胞 + 遗传);把 lncRNA-蛋白组装联系到干细胞维持与致癌。
- 互补价值:**实验解析 lncRNA-蛋白复合物结构**——正是 LLM/agent 设计者所缺的 RNA 结构真值;RNP 折叠的天然 verifier/合作者。
- 项目阶段:结构
- 主页:https://www.shsmu.edu.cn/english/info/1304/2349.htm · 邮箱:`wujian@shsmu.edu.cn`(置信度**高**,官方英文页)

**于翔(Xiang Yu)· 生命科学技术学院(tenure-track 副教授)**
- 背景 / 实验室类型:宾大博士(2014-2021);**hybrid(干湿结合)** RNA 生物学。RNA 修饰(NAD+ 加帽、ac4C、m5C)与 RNA 降解,分子生物学 + 生信 + 纳米孔直接 RNA 测序。
- 互补价值:**表观转录组 + 纳米孔直接 RNA 测序**——供纯折叠 oracle 忽略的真实 RNA 化学/修饰/降解数据与检测能力。
- 项目阶段:湿实验 / 修饰(P1 邻近)
- 主页:http://life.sjtu.edu.cn/teacher/Lab/yuxiang · 邮箱:`yuxiang2021@sjtu.edu.cn`(置信度**高**,官方教师页)
- ⚠️ 重名提醒:此 于翔(交大生科、宾大博士、RNA 修饰)与知名神经科学家 于翔(中科院/复旦)**并非同一人**,勿混。

**张良(Liang Zhang)· 化学化工学院 化学生物学系 / 化学生物学与分子医学研究所(2024 起教授)**
- 背景 / 实验室类型:**hybrid** 核酸与蛋白修饰酶系统化学生物学。三方向:DNA/RNA 表观修饰化学;蛋白修饰化学;肿瘤代谢调控。做核酸修饰高通量检测、鉴定修饰/去修饰酶、质谱代谢流分析。
- 互补价值:**检测与定量 RNA 修饰及其酶的化学生物学工具箱**——ML 无法产出的真值化学与 assay。
- 项目阶段:湿实验 / 修饰
- 主页:https://chembioins.sjtu.edu.cn/info/1028/1554.htm · 邮箱:`liangzhang2014@sjtu.edu.cn`(置信度**中**,系页)

**徐颖洁(Yingjie Xu)· 基础医学院 生化与分子细胞生物学系;上海市肿瘤微环境与炎症重点实验室**
- 背景 / 实验室类型:哈佛医学院博后;研究员、博导。**湿实验**癌症转移/耐药机制 + mRNA 治疗。蓝鹊生物(RNACure,2019 创)联创兼 CSO;与复旦林金钟合作领衔交大侧新冠 mRNA 疫苗突破;多项 mRNA 应用专利。
- 互补价值:**端到端 mRNA 药设计/优化→合成→递送→临床,背后有真实 mRNA 公司**——ML/agent 实验室无从触及的治疗性 RNA 转化层。
- 项目阶段:治疗
- 主页:https://www.shsmu.edu.cn/cbms/info/1104/3217.htm · 邮箱:`xuyingjie@shsmu.edu.cn`(置信度**中**,官方页片段)

**蔡宇伽(Yuga Cai)· 系统生物医学研究院(VLP-mRNA 工作时);本导基因 BDgene 创始人 — 当前单位需核实,可能已变动**
- 背景 / 实验室类型:**湿实验**。研制中国首个全自主基因治疗载体:VLP-mRNA 递送,把 CRISPR/Cas9 以 mRNA 携带做瞬时、低脱靶体内编辑(用 mRNA 茎环 / 噬菌体衣壳蛋白识别)。Nat Biomed Eng 2021,黄斑变性模型体内编辑。
- 互补价值:**mRNA / CRISPR-mRNA 递送与基因治疗转化**——设计-only ML harness 永远见不到的物理 RNA 载荷与递送现实。
- 项目阶段:治疗 / 湿实验
- 主页 / 邮箱:**未找到官方页,无邮箱(不臆造)**;⚠️ VLP-mRNA(2021)发表时在交大系统生物医学研究院,创办 BDgene 后**可能已迁**,outreach 前务必核实当前单位。

**张川(Chuan Zhang)· 药学院**
- 背景 / 实验室类型:普渡博士(2011),西北大学 Chad Mirkin 博后(2011-14)。**湿实验**核酸纳米凝胶/纳米结构,共递送 siRNA、miRNA、Cas9-sgRNA、mRNA,含协同癌症光免疫治疗。
- 互补价值:**把设计出的功能 RNA 变成可递送核酸纳米结构**——RNA 载荷的物理实现与递送,与 in-silico 设计互补。
- 项目阶段:治疗 / 湿实验
- 主页 / 邮箱:**未找到官方页,无邮箱(不臆造)**

**石毅(Yi Shi)· Bio-X 研究院**
- 背景 / 实验室类型:**湿实验** circRNA 生物学——IRES/核糖体招募驱动的 circRNA 编码蛋白(帽非依赖翻译)。2026 Dev Cell:circRNA 编码 RIPK1-98 驱动肺腺癌进展与耐药(与 UTSW、交大胸科医院共同通讯)。
- 互补价值:**ncRNA/circRNA 功能生物学与 IRES 驱动翻译**——为设计出的 RNA 元件提供功能读出与疾病相关性。
- 项目阶段:湿实验 / ncRNA
- 主页 / 邮箱:**未找到官方页,无邮箱(不臆造)**

**樊春海(Chunhai Fan)· 化学化工学院 / 张江高等研究院 / 变革性分子前沿科学中心;中科院院士**
- 背景 / 实验室类型:**hybrid**。框架核酸、DNA 纳米技术、DNA 计算/数据存储、生物传感与成像;2018 中国 DNA 纳米首篇 Nature 突破。合著机器学习驱动核酸分子工程综述(与李明强)。
- 互补价值:核酸设计/纳米技术与 DNA 计算权威,**已跨接 ML 与核酸工程**——但**以 DNA 为中心**,RNA 折叠互补性**部分**。
- 项目阶段:结构 / 设计(核酸纳米技术)
- 主页:https://dnastorage.sjtu.edu.cn/ · 邮箱:**未在官方页见到(不臆造)**

### 二档 · hybrid

- 交大侧 hybrid(干湿结合)人才已按其 domain_expert 属性归入一档(于翔、张良、樊春海),此处不重复;另有 胥川(Bio-X,circRNA 系统/进化基因组学,偏计算)、俞章盛(Bio-X/医学院,空间转录组/生物统计 HERGAST)属计算偏向、与 RNA 设计切线相关,优先级低。

### 三档 · 算法工具

- 交大本次**未发现**纯 RNA 热力学 / ViennaRNA 式二级结构预测理论家;最接近的"折叠权威"是实验结构生物学家(雷鸣、吴健)。如专门需要计算折叠/热力学理论家,交大范围内**结果偏弱**,建议另找机构。

### 四档 · 同赛道 ML 镜像(peer/baseline)— 互补性低

**郑双佳(Shuangjia Zheng)· 溥渊未来技术学院 GIFT + 人工智能学院(GENTEL 实验室,tenure-track 助理教授)**
- 背景 / 实验室类型:**ML** 治疗性生物分子发现/设计(生物-化学-医学)。直接做生成式 RNA 设计:RiboFlow(flow matching 条件从头 RNA co-design)、基于结构的 RNA 设计(潜在扩散逐步优化)、BEACON(RNA 任务 benchmark)/ RNA 语言模型。GitHub: GENTEL-lab。
- 判定:**与你同赛道**(生成式 ML for RNA design),互补性低,但是**关键 peer/baseline/benchmark 来源**(RiboFlow、BEACON)——要知道、引用或超越的对象,而非领域互补方。
- 项目阶段:P2 baseline / P3
- 主页:https://zhenglab.sjtu.edu.cn/ · 邮箱:**未在官方页见到(不臆造)**

**杨洋(Yang Yang)& 沈红斌(Hongbin Shen)· 电院/计算机(DRAG)**
- 按你此前 brief 已知的 ML 同行,**未重新调研**;确认为**同赛道 ml_agent_peer**。

---

## 🎯 上海本地首选(互补型)Top outreach — 复旦优先

只选 domain_expert / hybrid;每人一句"能给你 LLM/agent RNA 工作补上的本地领域资产 + 上海地利为何有用"。

**复旦(优先)**

1. **陈东戎(Chen Dongrong,IBS)** — 给你 riboswitch/ribozyme 的**折叠-功能真值**,几乎正对 ViennaRNA oracle + Eterna 设计闭环的核心;上海本地可当场验证"计算折出的结构化 RNA 是否真折叠、真有功能"。邮箱高置信 `drchen@fudan.edu.cn`,**建议第一封信**。
2. **林金钟(Lin Jinzhong,生科院 / 复旦 mRNA 中心 / 张江 mRNA 创新中心)** — 给你**mRNA 结构-翻译 + 完整加帽/修饰/递送/临床管线**,是把设计序列推进到可测 mRNA 治疗的最佳上海湿实验/产业闭环入口;张江本地便于产业对接与制造。
3. **甘建华(Gan Jianhua,生科院)** — 给你**原子分辨率核酶/催化核酸晶体结构**,为折叠 oracle 提供硬结构锚点;上海本地便于就设计构建体做晶体学验证。邮箱高置信 `ganjhh@fudan.edu.cn`。
4. **麻锦彪(Ma Jinbiao,生科院)** — 给你**RNA 及 m6A 修饰/沉默机器结构**(序列-结构-修饰生化),把 ML 模型锚到真实 RNA-蛋白识别;本地结构平台成熟。
5. **胡璐璐(Hu Lulu,IBS)** — 给你**单碱基 RNA 修饰定量数据/标签**,可做修饰感知 oracle/verifier 扩展;本地方法学实验室便于共建数据集(邮箱置信有分歧,先核实)。
6. **陆伟(Lu Wei,药学院)** — 给你**LNP/非病毒递送**这一"设计→可测治疗体"的下游转化层;本地便于制剂-体外闭环(注意红线:递送属转化伙伴,非 oracle)。

**上海交大**

7. **雷鸣(Ming Lei,精准医学研究院)** — 给你**顶级 RNP/端粒酶 RNA-蛋白实验结构真值**,oracle 无法自产;本地结构平台强,适合 P1 oracle 验证。
8. **吴健(Wu Jian,精准医学研究院)** — 给你**lncRNA-蛋白复合物实验结构**,RNP 折叠的天然 verifier;邮箱高置信 `wujian@shsmu.edu.cn`,本地同院可对接。
9. **于翔(Xiang Yu,生科院)** — 给你**表观转录组 + 纳米孔直接 RNA 测序**的真实 RNA 化学/修饰/降解数据;干湿结合、上海本地便于数据共建;邮箱高置信 `yuxiang2021@sjtu.edu.cn`。
10. **徐颖洁(Xu Yingjie,基础医学院 / 蓝鹊生物)** — 给你**端到端 mRNA 药转化 + 一家真实 mRNA 公司**;与林金钟已有合作史,上海产业闭环极强。

---

## ⚠️ 提醒

**同赛道 ML 镜像(与你重叠,当 baseline/benchmark,而非领域合作方):**
- **孙思琦(Siqi Sun,复旦智能复杂体系研究院 / 上海 AI Lab)** — RNA 3D/二级结构预测 + agentic science(RhoFold 系、PriFold、CRAFTS、OriGene)。是你想确认的复旦 AI4Science 组,但**与你同赛道**:其 RNA 结构模型可当**额外 oracle/baseline**,不是湿实验领域专家。
- **RhoFold+ 工作(复旦)** — ML-for-RNA,同赛道;**无单一明确复旦 PI 归属**,勿当个人合作方。
- **郑双佳(Shuangjia Zheng,交大 GENTEL/GIFT)** — 生成式 RNA 设计(**RiboFlow、BEACON**),这是交大最强的新 ML-for-RNA-design 同行,和你的工作直接重叠——**要对标/超越的 baseline 与 benchmark 来源**,不是领域互补。
- **杨洋 / 沈红斌(交大 DRAG)** — 已知 ML 同行,同赛道 peer。

**联系方式诚实标注(未核实 / 无邮箱,绝不臆造):**
- 复旦:**鞠佃文、邱敏**仅经张江 mRNA 中心团队页确认姓名/单位,**无官方页、无邮箱**;**应天雷、杨力**主页有、邮箱未见(低置信,经研究院/实验室页联系)。
- **胡璐璐邮箱两次检索结论不一致**(一次标 `luluhu@fudan.edu.cn` 高、一次未见)——outreach 前务必经 IBS 主页确认。
- **麻锦彪 `majb@fudan.edu.cn`** 来自论文/ResearchGate、符合复旦命名规律,但**未在单一官方页当场核实**(中-高)。
- **林金钟 `linjinzhong@fudan.edu.cn`、陆伟 `wlu@fudan.edu.cn`**:聚合列表见到,**未逐页核实**(中)。
- 交大:**蔡宇伽、张川、石毅、樊春海、郑双佳无官方邮箱(不臆造)**;**蔡宇伽当前单位可能已从交大迁至 BDgene,务必核实**。**雷鸣 `leim@shsmu.edu.cn`、张良 `liangzhang2014@sjtu.edu.cn`、徐颖洁 `xuyingjie@shsmu.edu.cn`** 为中置信,建议页面确认。**吴健、于翔**为官方页高置信。

**结果强弱诚实评估:**
- 复旦的**结构/折叠真值**(陈东戎、甘建华、麻锦彪)与**mRNA 治疗闭环**(林金钟)命中度最高,是本目录最强互补资产。
- 两校都**未找到纯 RNA 热力学 / ViennaRNA 式二级结构预测理论家**——最接近"折叠权威"的都是**实验结构生物学家**;若你要计算折叠/热力学理论家,上海(复旦+交大)范围内**结果偏弱**,建议另寻机构。
- 越往下(治疗递送/circRNA/染色质)越偏离折叠 oracle 中心,属邻近或下游互补,非核心。

**目录外但值得知道(本次按 Fudan/SJTU scope 排除):** 陈玲玲(Chen Lingling,中科院分子细胞科学卓越创新中心,上海)— 领军 ncRNA/circRNA 湿实验权威、复旦杨力常合作者;詹叶超(Zhan Xiechao,西湖大学)— 人源剪接体冷冻电镜。二者都是长三角/上海 RNA 领域强互补专家,若日后放宽出复旦/交大可优先考虑。