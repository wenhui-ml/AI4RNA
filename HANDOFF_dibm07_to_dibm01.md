# 迁移交接:dibm07 → dibm01(2026-08-18)

dibm07 明天 10:00 从 slurm 下线,本地盘 `/data` 随节点消失(存储 deadline 本周五 10:00)。
本文件记录**数据落在哪、环境要怎么重建、以及这次对话停在哪一步**。

---

## 1. 数据落点

| dibm07 上的路径 | dibm01 上的路径 | 说明 |
|---|---|---|
| `/data/wenhui/openclaude_rna` | `/data/wenhui/openclaude_rna` | **主仓库**,同名不冲突 |
| `/data/wenhui/DesiRNA` | 同名 | SOTA 求解器,`DESIRNA_CMD` 指向它 |
| `/data/wenhui/openknot_data` | 同名 | OpenKnotBench v4.5.1 CSV |
| `/data/wenhui/rnet_weights` | 同名 | RibonanzaNet-SS 权重 |
| `/data/wenhui/rnet_env` | 同名 | RibonanzaNet 的 venv(**不是 conda**) |
| `pk_judges` `openscience` `openclaude` `RibonanzaNet` `UFold` `AI4Catalysis` `RNAInvBench` `inferenv` `wh-llm.tar.gz` `ec2_run` `ec23` | 同名 | 无冲突 |
| `InceptBench_Prompt` `openclaude_sat` `tmp` `plan_before_you_patch` `ms-swift` `hf_cache` | **`/data/wenhui/_from_dibm07/<同名>`** | dibm01 上已有同名项目,**隔离存放,绝不覆盖** |
| `/home/ubuntu/.claude`(整棵) | `/data/wenhui/_dibm07_claude_state/dot_claude/` | 含本对话 transcript 与 32 个记忆文件 |

**注意 `hf_cache` / `ms-swift` 是可重新下载的,放在最后同步;若来不及,损失可接受。**

---

## 2. 环境重建(在 dibm01 上)

```bash
# 求解器
export DESIRNA_CMD="/home/ubuntu/miniconda3/envs/rna/bin/python /data/wenhui/DesiRNA/DesiRNA.py"

# 两个 venv 的分工(踩过的坑,别再踩)
#   rna 环境   有 ViennaRNA + 氧化物二进制,没有 torch  -> 跑求解器和面板
#   rnet_env   有 torch + RibonanzaNet,没有 RNA 模块    -> 只跑打分
#   所有两阶段脚本都是按这个分工写的:xxx.py 用 rna,xxx_score.py 用 rnet_env
/home/ubuntu/miniconda3/envs/rna/bin/python     # 注意:conda 里【没有】名为 rnet 的环境
/data/wenhui/rnet_env/bin/python                # RibonanzaNet 是普通 venv

# scipy 只在 rna / base 里有,rnet_env 里没有(label_calibration.py 需要 scipy)
```

**核数上限 160**(硬约束)。DesiRNA 每个 job fork 10 个副本,所以 `--workers` 最多 14–16。
长任务用 tmux,**不要用 nohup**;**不要在启动脚本的同一条命令里用 `pkill -f`**(会杀掉自己,本项目踩了 4 次)。

---

## 3. 代码状态

分支 `rna-ceiling`,**全部已 commit 并 push 到 GitHub**(`TitanResearchLabs/openclaude_rna`),
所以即使 dibm07 的盘丢了,代码和数据产物都在远端。最后几个提交:

```
6c21cb65  weekly report: objective 4 takes one row each way, and a number this project must stop quoting
e9df320e  the certificate finally has an error rate against physics, and the ruler ... beaten by an off-the-shelf model
529c8c9a  weekly report renamed to today, and the week's largest result
6abe8f49  the objective-function knob is a null end to end, and the five worst targets are a selection defect
```

`rna_harness/runs/ceiling/pool_dump/`(57 个池、8,730 条候选)也在仓库里,所有重放实验都读它。

---

## 4. 这次对话停在哪一步

### 已完成并已提交

- **`screen_first` 选择规则**:同一次运行、同一批候选,交付中位 **0.826 → 0.955**,
  L1 对整个已发表领域 **−0.075 → ±0.000**,逐方法 3赢5平4输 → **6赢3平3输**。
  200 次切分留出检验通过(+0.1264,200/200)。**已验证,尚未上线。**
- **证书档位对物理测量的错误率**(④ 的核心成果):
  `not_found` 51.7% / `knot_screened` 34.1% / `knot_confirmed` 20.8% / 不用 harness 29.2%,
  单调,靶标聚类 p = 4.7e-6。
- **一个必须停止引用的数字**:「桌面上还有 +9.5 分」是 100% 胜者诅咒(置换零 +9.32 对实测 +9.37)。

### 待办,按优先级

1. **修标签一致性** —— 36 个靶标仍挂 `knot_confirmed`,但序列已不是确认档挑的(`apply_screen_first.py` 只换序列没换 verdict)。
2. **把 `screen_first` 落进 `certify.py`**,并把"顺序来自求解器"显式化,而不是靠数组下标的巧合。
3. **读 DesiRNA 真实的目标函数值**(现在拿数组下标当代理;真值写在 `*_results.csv` 和 FASTA 头里)。

### 悬而未决的方向问题(对话结束时正在讨论)

④ 的正向侧卡在一个结构性问题上,**下一次对话应该从这里接**:

```
造判据这套方法要扩到第三个领域,需要该领域「判据缺失 AND 有大规模物理测量」
但这两个条件反相关:有大规模测量的地方,就已经有人训出模型了,判据就不缺失
今天的 ❌ 正是一个实例 —— 假结领域两条都满足,而 RibonanzaNet 把我们手工造的尺子全面压制

待验证的主张:「造判据」的有效范围被【测量的可得性】和【模型的存在性】从两头夹住
证伪方式:找到一个「有大规模测量、但那批测量还没被训成可用预测器」的判据缺失领域
若那一格是空的 —— 这本身是 ④ 的边界结论,值得写下来,并停止在这条路上投入
```

**已排除的候选:** Eterna100(DesiRNA 99/100,无残差)、OpenKnot 上打赢 gRNAde(判据饱和,且
gRNAde 已发表同款「RNet 预测反应性 → in-silico OpenKnot Score → 排序过滤」流程)、
OpenVaccine(有 6,043 条单碱基降解测量,但已有一批已发表的监督模型,判据不缺失)。

---

## 5. 恢复对话

dibm01 上已有 `/home/ubuntu/.claude/projects/-data-wenhui-openclaude-rna/`,**内容与本机不同**,
所以本机的状态同步到了 `/data/wenhui/_dibm07_claude_state/dot_claude/`,没有覆盖。恢复时:

```bash
# 记忆文件(32 个)—— 逐个比对后再合并,不要整目录覆盖
ls /data/wenhui/_dibm07_claude_state/dot_claude/projects/-data-wenhui-openclaude-rna/memory/

# 本对话完整 transcript(72 MB)
/data/wenhui/_dibm07_claude_state/dot_claude/projects/-data-wenhui-openclaude-rna/a26f6ec5-7bf6-4442-b1f1-d17e22916349.jsonl
```

**记忆里有一条已知过期,恢复后要改:** `project-rna-ceiling-branch.md` 写着「Eterna100 是 64/100」,
那是旧管线(`_archive_runs/eterna_paired/memon/`)的陈数;现在的事实是 **DesiRNA 解出 99/100,
剩下那道已被证明不可设计**。
