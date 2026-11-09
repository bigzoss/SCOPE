# SCOPE-GRPO

State-conditioned Context Optimization with Provenance and Evidence.

这是基于[完整设计](docs/design.md)的独立科研实现。主路径为真实ShopSimulator＋本地HF SFT epoch3模型，使用公开证据、版本化状态图谱、LLM管理器及分角色GRPO。代码不修改MA-GRPO、原始数据或共享模型权重。

当前版本为0.2.0。逐模块核验、已修复缺陷、执行证据与尚未实现的研究能力见[模块核验报告](docs/module_audit.md)。原始验证文档记录的是0.1版本，不能替代本轮检查。

## 实现范围

| 能力 | 实现 |
|---|---|
| 原始公开证据 | 完整原文、内容哈希、精确字符引用；评分对象与策略输入分开 |
| 状态图谱 | SQLite节点/关系、版本事务、冲突、幂等提交、退役、视图失效与恢复 |
| 状态支持检查 | 字段规则、作用域、实体、版本；可选项/尝试动作不升级为选中确认 |
| 上下文管理 | BM25、两跳有界关系扩展、精确片段回查、读写事实、过滤、来源映射摘要 |
| LLM管理器 | 现有HF底座＋manager LoRA；工具JSON的生成token可训练 |
| 购物执行器 | actor LoRA、多轮历史、公开状态、无进展提醒、完整动作预算 |
| 训练 | 管理器工具SFT；全局、完整窗口局部、分阶段条件局部GRPO；执行器或交替更新 |
| 局部续跑 | 官方模拟器的私有原生状态快照、公开结果核验、相同剩余预算 |
| 恢复 | 双角色/参考适配器、优化器、数据随机状态；在完整update边界恢复 |
| 评估 | 参数冻结、任务范围检查、序列空事实库、结果全集与SHA核验、配对簇bootstrap |

首版采用可核验的**抽取式结构化摘要**，不把自由改写当作已证实事实。没有部署额外语义裁判或向量模型。另含ALFWorld public-v1 HTTP适配器及官方manifest划分检查；真实迁移尚未执行，其公开接口没有私有快照，代码明确拒绝在该接口上运行局部GRPO。自动经验晋升也不属于首版运行链路，经验库须事先审核后冻结。

## 目录

```text
scope/
  contracts.py       # Scope / Sample / Budget / Boundary
  atlas.py           # EvidenceStore + StateAtlasStore + reducer transactions
  validation.py      # evidence support and summary checks
  retrieval.py       # lexical retrieval and bounded relation traversal
  tools.py           # structured management operations
  experience.py      # immutable audited training-experience bank
  policy.py          # HF generation, replay log-prob, separate adapters and losses
  environments.py    # public HTTP client; explicit unit-test fixture
  alfworld.py        # optional target HTTP and official split adapter
  service.py         # isolated official simulator, opaque native snapshots
  rollout.py         # public observations -> management -> actor -> environment
  credit.py          # global/local advantages and diagnostic interaction
  training.py        # progressive / frozen-role / alternating updates
  warmup.py          # public-prefix SFT data and resumable manager warmup
  checkpoint.py      # atomic full training checkpoints
  evaluation.py      # frozen evaluation and paired cluster bootstrap
  launcher.py        # owned-service lifecycle
  cli.py             # actual entrypoint and immutable execution receipts
configs/shopsim.json
scripts/{make_configs,validate_native,validate_hf,validate_cli}.py
tests/
docs/design.md
```

## 环境和数据

在服务器项目目录运行：

```bash
cd /home/hyx/lm_workspace/SCOPE
PY=/home/hyx/lm_workspace/longagent/.venv-verl/bin/python3
$PY -m scope.cli preflight
CUDA_VISIBLE_DEVICES='' $PY -m pytest -q
```

复用已安装的torch、transformers和PEFT，无须升级全局环境。默认配置使用物理GPU2。模拟器单独使用现有`.venv-shopsim`和Java/Lucene，`CUDA_VISIBLE_DEVICES`为空，绑定`127.0.0.1:5112`。

原始GRPO池3000任务在代码中按隔离组固定划出2700内部训练、300内部验证；原开发300和历史测试900另行保留。切分器检查task ID、isolation_group、已记录SFT暴露。任务目录中的ASIN、规格答案等仅参与数据审计，模型目标取自环境公开reset响应。

`dev`曾参与协议调试，900历史集合及原任务池也曾被旧模型预测，不作为全新盲测。默认不启动任何最终测试。正式测试前还需按设计完成暴露审计。

## 管理器预热

先收集**训练池**公开轨迹。以下示例20任务只是数据收集规模，预热规模需按设计确定：

```bash
$PY -m scope.cli collect --split rl_train --limit 20 \
  --manage-service --output runs/public_train
$PY -m scope.cli warmup-build --split rl_train \
  --public-file runs/public_train/public.jsonl \
  --examples runs/public_train/manager_examples.jsonl
$PY -m scope.cli warmup --examples runs/public_train/manager_examples.jsonl \
  --epochs 1 --batch-size 1 --output runs/manager_warmup
```

样例仅使用每一时刻已经公开的前缀。生成引用、检索、摘要、FINISH等规则示范，并实际经过工具校验。此数据不是当前策略的GRPO样本。预热报告工具学习与参数变化，不报告为购物成功率。

SFT每100次update、最终update或`--stop-after`边界保存checkpoint。使用相同示例、epoch、batch size与输出目录恢复：

```bash
$PY -m scope.cli warmup --examples runs/public_train/manager_examples.jsonl \
  --epochs 1 --batch-size 1 --output runs/manager_warmup \
  --resume runs/manager_warmup/checkpoints/step_00100
```

实际checkpoint名称见`checkpoints/latest.json`，不要假定预热恰好100步。

## B4 / B5 / B6训练

从实际预热checkpoint生成三个对照配置：

```bash
$PY scripts/make_configs.py --warmstart /absolute/path/to/warmup/checkpoints/step_NNNNN
$PY -m scope.cli train --config configs/experiments/global.json \
  --manage-service --output runs/b4_global
$PY -m scope.cli train --config configs/experiments/window.json \
  --manage-service --output runs/b5_window
$PY -m scope.cli train --config configs/experiments/conditional.json \
  --manage-service --output runs/b6_conditional
```

以上是待执行命令，不表示正式48步实验已经完成。默认48个update、K4、序列长度1/2/4，每seed448条全局查询。B5/B6在12个预定update增加局部候选，每次4条当前任务续跑，计划共48条额外查询。任务之间事实库独立，序列长度表示一组独立查询的回报聚合，不表示跨任务记忆已经贯通。局部估计明确限定`probe_query_horizon=1`；不支持的跨任务范围会在配置预检中被拒绝。

恢复训练：

```bash
$PY -m scope.cli train --config configs/experiments/conditional.json \
  --manage-service --output runs/b6_conditional \
  --resume runs/b6_conditional/checkpoints/step_00016
```

`training.role=manager`固定执行器；`actor`固定管理器；`alternating`按`alternating_window`切换角色。`runtime.semantic_validation=false`是去掉语义支持检查的B2消融，范围/引用存在性检查仍保留；正常方案保持true。`manager_mode=rule`为不调用LLM管理器的FINISH规则基线；`disable_experience=true`禁用长期经验检索。

checkpoint只在完成的update边界提交。中途进程失败，从最后完成的update重新采样，不宣称能跨进程恢复尚未完成的原生服务handle。旧轨迹、失败日志和部分数据仍留在各run中，未完成组不填成模型失败分。

## 奖励与梯度

主要任务回报为`(strict_success + 0.1 * loose_reward) / 1.1`；扣除有上限的非法操作和实际token成本。默认不使用额外进展势函数。

局部组从相同私有环境快照、公开图谱、工作视图、目标、策略版本和剩余预算续跑。B5比较整个管理窗口；B6比较一次状态整理或一次上下文管理，分别归一化优势，训练被比较的动作token。检索结果、环境页面、校验器回复属于观察，生成损失掩码为0。

候选回报完全相同，任务优势为0。此时可能仍有KL正则梯度，必须与任务学习区分。参数变化不构成管理有效或任务提升证明。

日志分别记录任务回报方差及扣成本后的回报方差；`non_task_only_advantage=true`表示区分只来自非法操作或成本。尤其全体任务失败时，微小token差也可能被GRPO标准化放大，不能当作严格成功的学习证据。

共享底座的两个可训练适配器与两个固定参考适配器独立保存。角色切换不复用KV cache。默认确定性CUDA设置及关闭bf16缩减精度归约已进入配置；实测恢复与连续更新的位级一致性见验证文件。

## 冻结评估及角色交叉

```bash
$PY -m scope.cli evaluate --split dev --checkpoint runs/b6_conditional/checkpoints/step_00048 \
  --manage-service --output runs/dev_b6
$PY -m scope.cli compare --left runs/dev_b5 --right runs/dev_b6 --output runs/paired_b6_vs_b5
```

通过`--actor-checkpoint`和`--manager-checkpoint`分别选择角色，完成四格交叉；省略角色覆盖时使用`--checkpoint`中的两个角色。加载时核验checkpoint SHA和冻结底座身份。

每个评估序列以独立命名空间开始，任务事实不跨任务共享，训练经验库只读。长期在此表示超出当前上下文仍可回查的持久证据/事实；不同购物任务不被伪造成同一用户的稳定偏好。摘要与检索可改变工作视图，但不能改写原始公开证据。

评估单位先以`sequences/sequence_NNNNN.json`原子保存，重启跳过完整单位。汇总检查ID全集、重复、参数与经验库不变及结果SHA。`--limit`明确标记为子集。`--split test`会切换独立test manifest，但仅在明确需要该历史测试时使用。

## 运行产物与清理

每个run保存`receipt.json`、`status.json`、分支轨迹、训练指标、完整checkpoint及数据/源码/模型SHA。`--manage-service`仅清理本进程启动的5112服务，不复用或停止5084。服务端保留快照私有内容，客户端只接收不透明handle及指纹；评分只能进入Reward/metrics通道。

同一输出目录有进程文件锁，未指定resume不能覆盖已有训练结果。receipt变化会拒绝恢复；更改方法或配置时使用新run，避免静默混合实验身份。

## ALFWorld接口

```bash
$PY scripts/make_alfworld_config.py --manifest /absolute/path/to/alfworld_manifest.json
$PY -m scope.cli preflight --config configs/alfworld.json
$PY -m scope.cli evaluate --config configs/alfworld.json --split test \
  --checkpoint /absolute/path/to/source/checkpoint --output runs/alfworld_transfer
```

使用项目已有的外部ALFWorld public-v1服务，SCOPE不启动、重启或清理该共享服务。适配器仅向模型提供公开任务和观察，另行保存二元任务分数。manifest按官方train/valid_seen/valid_unseen区分，并检查精确game/task重叠；官方seen不等于模型训练暴露已知。已核验现有manifest接口可被解析，但没有执行目标任务，不报告泛化成功率。

### 跨 benchmark 的公开交互接口

环境暴露 `action_adapter` 和 `step_command(command)`。`ActionAdapter` 定义 `parse`、`normalize`、`bind` 和 `observe`。`observe` 接收公开观察、上次投影、已发送的动作和公开按钮，可返回 JSON 状态字段。只根据公开反馈确认状态变化，避免将动作尝试或未公开的目标写进投影。可用 `progress_key` 排除时间戳等易变字段。`EpisodeRunner` 复用动作分派、停滞检测和轨迹记录。新环境还需实现 `reset`、`close` 与评分接口。如需通过现有 CLI 选择该环境，再注册其配置、环境工厂和任务划分。旧的 `step(response)` 接口可以继续使用 `legacy`。

`runtime.action_mode` 可选 `legacy`、`normalize`、`grounded`，分别使用原解析、增加规范化、增加公开证据绑定。`runtime.loop_policy` 可选 `legacy` 或 `state`。两项默认 `legacy`，便于保留已有实验配置。使用 `state` 时，同一公开状态内的动作交替或状态往返循环会生成重选提示和证据引用。`grounded` 模式下，ALFWorld 缺编号或候选歧义的动作不会发送给环境，ShopSimulator 只在适配器掌握当前公开按钮时核验点击。

评估的 `results.jsonl` 包含原始命令、规范化命令、发送状态、绑定诊断、前后证据 ID、公开状态变化和终止原因。`public_evidence` 保存可供定位的公开观察与动作回执。`summary.json` 分开统计格式错误、绑定失败、环境拒绝和合法无进展。对照三种动作模式时，在三个独立输出目录用同一开发任务、checkpoint、生成预算和循环策略运行评估，再用 `scope.cli compare --allow-interaction-contrast --left ... --right ...` 显式允许动作或循环模式差异。其他协议字段仍需一致。

## 验证

```bash
PYTHONPATH=. $PY scripts/validate_native.py
PYTHONPATH=. $PY scripts/validate_hf.py
PYTHONPATH=. $PY scripts/validate_cli.py
```

这些是有独立输出目录的工程验证，包含明确的短生成预算，不能代替正式训练或benchmark结果。`validate_native`检查真实模拟器恢复，`validate_hf`检查真实8B模型的角色梯度和下一步恢复，`validate_cli`检查真实入口的两次update、新进程恢复与冻结评估。已有验证输出不会静默覆盖，可通过`SCOPE_VALIDATION_OUTPUT`指定新目录。当前执行结果以[模块核验报告](docs/module_audit.md)及服务器`runs/module_audit/`为准；`docs/implementation_validation.md`保留历史结果。
