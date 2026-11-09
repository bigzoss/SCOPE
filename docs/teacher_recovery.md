# SCOPE：强教师辅助的执行恢复与记忆训练

更新日期：2026-11-09。本文描述本轮已经接入的代码，以及尚需真实训练验证的环节。

## 实现范围与实验边界

服务器主项目已更新；历史 `runs/` 中的源代码快照、checkpoint、结果不改写。本轮进行 CPU 回归测试、微型模型梯度与恢复测试、历史轨迹只读诊断；没有启动新的 GPU 训练，没有调用付费教师 API。接口可用不等于教师示范有效，示范有效也不等于学生独立成功率提高。

旧恢复试验为两个环境各 6 个开发任务，三个条件共 36 条轨迹。它们已用于诊断，应作为回归集；新的训练示范必须使用 `rl_train`，效果验证使用新的开发任务。最终测试期间不调用教师、不更新参数、不写入跨任务测试经验。

## 模块变更

| 模块 | 行为 |
|---|---|
| `scope/interaction.py` | 不再在格式错误或绑定失败时清空循环窗口；记录 `invalid_no_progress` 和 `repeated_rejection`。诊断协议升级为 `public-interaction-v2`。 |
| `scope/state_graph.py`、`scope/alfworld.py` | 解析“On the desk 1, you see …”列表，将物品位置绑定到原文完整区间。看到物品不等于持有。增加基于公开持有和位置的放置、处理物品、使用台灯候选。规范化 `check inventory`，保留不能确定等价的命令错误。 |
| `scope/recovery.py` | 区分新观察、重复观察、没有收到观察。页面变化不再被解释为任务进展；`task_progress_verified` 保持空值。对同状态无新增信息的动作累计记录并降低候选优先级。 |
| `scope/rollout.py`、`scope/config.py` | 可选精简执行器视图：保留目标、核心状态、待确认项和证据引用，移除重复目标与候选库的冗余元数据，保留最近两轮历史。原文仍在证据库；该开关改变信息协议，必须单独对照。 |
| `scope/teacher.py` | 教师可选择菜单，也可提出菜单外操作。统一在图谱副本、相同预算上校验，不修改原前缀。提供相关历史与覆盖率；历史不完整时不接受“证据完全不存在”的确定标签。教师诊断标记为未独立证实。 |
| `scope/teacher_api.py` | 增加 USTC JSON 教师客户端，默认模型 `qwen3.8-flash-next`。HTTPS 主机白名单、禁用重定向、有限重试、请求预算、模型与终止原因检查、内容哈希缓存、输入输出 token 统计。日志与缓存不含密钥。 |
| `scope/warmup.py` | 自由生成管理器可学习菜单外的合法工具调用。候选选择模式无法表示的新操作明确计入 `online_menu_gap`，不会静默冒充教师标签。 |
| `scope/teacher_execution.py` | 强教师使用学生实际公开决策视图，提出带可见证据 ID 的合法动作。默认只在错误、循环或冲突后介入，也支持全程教师诊断。语法合法和引用存在不被宣称为任务效用。 |
| `scope/cli.py`、`scope/rollout.py` | 训练集 `collect` 可进行真实教师辅助续跑，保存教师介入、环境反馈和终局结果。只从成功任务中提取被环境接受的教师动作供执行器 SFT。失败轨迹保留用于诊断，不伪造成功标签。 |
| `scope/warmup.py`、`scope/policy.py` | 支持 `--warmup-role actor`，冻结管理器并使用独立适配器和 checkpoint；管理器预热保持原接口。教师/SFT样本带来源标记，禁止进入 on-policy GRPO。 |
| `scope/probe_selection.py`、`scope/training.py` | 教师可离线推荐配对续跑前缀。训练时核验清单 SHA、任务分区及公开前缀签名，匹配后才捕获边界；候选仍由学生采样，收益仍由真实续跑确定。无匹配时跳过，不偷偷替换成其他节点。 |
| `scope/credit.py` | 新增全分支失败、正负增益候选和任务成功数诊断；拒绝教师来源候选。全失败不产生严格任务优势；成功且保留必要证据的压缩才允许成本辅助信用。 |

## 两种管理器接口

原 `*_recovery.json` 配置继续使用候选选择。新增 `*_teacher_recovery.json` 使用自由 JSON 操作并启用精简视图，GPU 设置为 1。两个模式的输入和输出分布不同，必须分别预热、分别评价，不能直接把候选选择 checkpoint 当成已学会自由工具调用的模型。

选择模式下，教师发现菜单缺项是候选生成器的缺陷证据；自由生成模式下，经过同一工具校验的新增操作可直接成为 SFT 标签。程序仍负责来源、实体、版本与作用域校验，LLM不能通过提高置信度越过校验。

ALFWorld 的 HTTP 接口仍没有原生快照，新增配置保留全局训练模式，不把命令重放冒充同状态原生配对。ShopSimulator 新配置采用配对信用、确定性续跑和两次配对重复。固定更新数只是配置模板，本轮未执行这些更新。

## 使用顺序

以下命令从项目根目录执行，`python` 指已经安装项目依赖的解释器。`$USTC_CONFIG` 指向本机/服务器已有的 USTC 配置文件，字段为 `base_url`、`api_key` 和可选 `model`；不要把配置提交到仓库。独立实验目录不得与旧结果复用。

### 1. 在训练任务中采集学生失败后的教师纠正

```bash
python -m scope.cli collect --config configs/alfworld_teacher_recovery.json \
  --split rl_train --limit 2 --output runs/teacher_recovery_collect \
  --teacher-api-config "$USTC_CONFIG" --teacher-max-calls 64 \
  --teacher-intervention recovery
```

使用已有学生 checkpoint 时另外传入 `--actor-checkpoint` 和 `--manager-checkpoint`。服务需提前就绪；不改变原有服务归属规则。`--teacher-intervention all` 是全程强教师执行器对照，不是学生独立结果。

输出：`public.jsonl`、`teacher_rollouts.jsonl`、`summary.json`，有成功示范时生成 `actor_examples.jsonl`。API请求预算包含重试；耗尽或接口失败作为基础设施错误报告，不计入任务失败率。保存的终局指标不传入教师提示。

### 2. 基于真实公开轨迹构建管理器示范及续跑选点

```bash
python -m scope.cli warmup-build --config configs/alfworld_teacher_recovery.json \
  --split rl_train --public-file runs/teacher_recovery_collect/public.jsonl \
  --examples runs/teacher_recovery_manager/examples.jsonl \
  --output runs/teacher_recovery_manager \
  --teacher-api-config "$USTC_CONFIG" --teacher-max-calls 64 \
  --probe-output runs/teacher_recovery_manager/probes.jsonl
```

教师给出 `probe_worthy=true` 只表示值得测试，不是正奖励。`probes.jsonl` 的公开状态签名用于选择机会，真正的信用估计仍使用完整同前缀快照与相同续跑策略。教师生成过程不读取未来任务评分。

### 3. 分别预热管理器和执行器

```bash
python -m scope.cli warmup --config configs/alfworld_teacher_recovery.json \
  --examples runs/teacher_recovery_manager/examples.jsonl \
  --output runs/teacher_recovery_manager_sft --warmup-role manager

python -m scope.cli warmup --config configs/alfworld_teacher_recovery.json \
  --examples runs/teacher_recovery_collect/actor_examples.jsonl \
  --output runs/teacher_recovery_actor_sft --warmup-role actor
```

执行器示范文件不存在或没有成功轨迹时，应先诊断教师和环境接口，不能伪造训练数据。预热验证需要报告合法率、绑定正确率、待确认状态准确性，以及学生独立真实任务结果。

### 4. 在原生可恢复环境中使用局部 GRPO

使用 `shopsim_teacher_recovery.json`，载入对应预热后的管理器和执行器。可在 `training` 添加：

```json
"teacher_probe_hints": {
  "path": "runs/shop_teacher_manager/probes.jsonl",
  "sha256": "实际文件的SHA256"
}
```

任务必须来自当前训练分区。离线选点不调用教师，不强制执行教师候选；训练日志保存 `probe_selection`，并报告 `all_branches_failed`、正负效用候选数、严格成功差和实际续跑成本。先固定执行器训练管理器，再启用已有交替训练；每次比较期间策略版本固定。

## 验证与限制

- 全量 CPU 测试覆盖已有模块以及新增循环、证据位置、教师越权引用、缓存、训练分区、预热隔离与恢复等行为。
- 历史轨迹审计入口：`python -m scripts.audit_recovery_traces --run runs/recovery_dev_20261109.a4DCDE --output ccfa-workfiles/engineering/teacher_recovery/trace_audit.json`。它只重算诊断，不重新执行环境，也不修改旧结果。
- 教师诊断缺口是否正确、子目标是否有效、检索是否真正改变决策，需要独立标注与真实对照验证；当前没有把这些假设包装成任务奖励。
- 教师纠正示范与完整教师轨迹应分别报告调用成本。成功轨迹中的动作也不能逐个被断言为因果必要，局部续跑才检验记忆操作的增量效用。
- 对照顺序：同执行器下的原管理器、教师预热管理器、轨迹级 GRPO、配对信用 GRPO；执行器适配单独控制。教师辅助在线结果与学生独立结果分开。
- 最终需要测试关闭教师后的小模型表现。本次代码与测试结果不构成成功率提升证据。
