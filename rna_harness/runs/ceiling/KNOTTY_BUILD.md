# 第五裁判 Knotty:构建方法(可复现)

> 为什么要第五个:ShapeKnots(RNAstructure)与 RNAPKplex(ViennaRNA)虽是两个软件包,但**都是"先找螺旋再拼假结"的启发式,都用 Turner 系参数**。
> Knotty 在**两个轴上同时不同**:精确 sparsified DP(不做启发式拼接)+ HotKnots DP09 参数。
> 所以 ShapeKnots 与 Knotty 的一致,既不能归因于共享搜索启发式,也不能归因于共享能量参数。

```bash
mkdir -p /data/wenhui/pk_judges && cd /data/wenhui/pk_judges
git clone --depth 1 https://github.com/HosnaJabbari/Knotty.git
cd Knotty
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build -j8
```

## 两处必须知道的坑

**1. 需要给 `src/h_struct.h` 补一行 `#include <algorithm>`。**
`std::sort` 在旧 libstdc++ 下靠传递包含进来,GCC 13 下不再传递。**只加 include,不动任何逻辑。**

```
src/h_struct.h:118  error: 'sort' is not a member of 'std'
```

**2. 调用时必须清掉 `CONDA_PREFIX`,并把 cwd 设在 Knotty 根目录。**
`src/param_path.cpp` 优先找 `$CONDA_PREFIX/share/simfold/params/`,回退路径是**相对 cwd** 的 `simfold/params/`。conda 环境一激活,它就去读一个不存在的参数文件。

```python
env = dict(os.environ); env.pop("CONDA_PREFIX", None)
subprocess.run([KNOTTY, seq], cwd=KNOTTY_HOME, env=env, ...)
```

> 这个坑**会大声失败**(`Cannot open file .../multirnafold.conf`),不会静默给错答案,所以不危险 —— 但必须处理。

## 输出格式

```
Seq: GGCGCGGCACCGUCCGCGGAACAAACGG
RES: ..[[[[[..((((]]]]]......))))  -6.96
```

## wrapper 自检(`knotty_judge.py::self_test`,判任何序列之前必须通过)

本项目有过**四次** wrapper 缺陷,每次都产出一个自信的错数字而不是报错(`ct2dot` 三行拼接、RNAduplex 按最后开括号配对配出物理上不可能的平行双链、bifold 多结构块当成一块、ProbKnot 漏 `--sequence`)。所以:

| 用例 | 期望 | 它抓的是什么 |
|---|---|---|
| 已知 H-type 假结 | 有交叉配对 | 工具/解析根本没在识别假结 |
| 工具自带示例 | 嵌套,无交叉 | 把嵌套误判成假结 |
| poly-A | 零配对 | **负对照必须【能够】通过** |
| 三个用例的配对数 | ≥2 种不同值 | 输出对输入不敏感(常量) |

配对集另外过 `instrument.check_pair_set`:必须是**不重用位置、在范围内、i<j 的匹配**。

---

# 第六裁判 IPknot:构建方法

> 第三个**算法类**:配对概率 + 整数规划,而且概率来自 **LinearPartition-C(学习模型)**,连热力学打分基础都不共享。

```bash
/home/ubuntu/miniconda3/bin/conda create -y -n ipk -c conda-forge -c bioconda viennarna glpk pkg-config cxx-compiler
cd /data/wenhui/pk_judges
git clone --recursive --depth 1 https://github.com/satoken/ipknot.git && cd ipknot
E=/home/ubuntu/miniconda3/envs/ipk
PKG_CONFIG_PATH=$E/lib/pkgconfig cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$E
cmake --build build -j16
```

一次编成,无需打补丁。调用时需 `LD_LIBRARY_PATH=$E/lib`,输入是 **FASTA 文件**(不是 stdin 序列)。

**全部跑默认参数,并在脚本里声明。** IPknot 有 `-t` 阈值与 `-e` 模型可调;**一边看着它和我们设计的同意率一边调,就是本项目已经犯过的那个错**(曾让被测模型自己挑对齐方式)。

## 自检的一处修正,而且它本身是个发现

`knotty_judge.py` 最初的自检里有一条"**已知假结必须判出交叉配对**"。用在 IPknot 上会失败 —— 但那是**工具灵敏度**,不是 **wrapper 正确性**,`instrument_audit.py` 早已立下这个区分(9/9 外部工具全清,失败全是分析者的)。更糟的是那条用例的期望标签是从 **Knotty 自己的输出**抄来的,等于拿 Knotty 检验 Knotty。

**现在的自检只断言 wrapper 级性质**(合法匹配、长度正确、poly-A 零配对、动态范围),**工具灵敏度改为量出来的 S_ext**,不设任何发明出来的阈值。
