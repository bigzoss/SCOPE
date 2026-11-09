# SCOPE：证据回查、决策恢复与统一记忆接口

本次实现针对 `hotpot_transfer_20261109.nf63uA` 暴露的三个问题：未入图证据无法通过槽位检索找回，循环检测缺少可执行的恢复建议，以及源域训练与目标环境的管理接口不一致。原实验快照、checkpoint 和结果不修改。本文件描述代码行为，不宣称成功率已经提高。

## 1. 运行开关与兼容性

新增 `runtime.decision_recovery`，默认关闭。新配方为 `configs/alfworld_recovery.json` 和 `configs/shopsim_recovery.json`，均启用 state_graph、memory_choice、shared_memory、decision_evidence、grounded action 和 state loop。配置校验要求这些前置条件成立。配方沿用各环境已有服务和数据路径，使用 Qwen3-8B；运行前仍需选择对应 checkpoint 与独立开发划分。

开关进入评估 contract，不允许把新旧协议缓存混合续跑。旧实验的独立 source 快照继续工作。ALFWorld 命令规范化及公开位置提取属于适配器修复，未来调用当前代码的实验应保留新的源码哈希。

## 2. 原始证据回查

`Retriever.search_raw` 只搜索当前 atlas 内的公开 observation，不要求先存在 Claim。排除需求、动作回执、终局记录与观察中附带的任务描述；检查实体编号边界和版本范围。返回长度有界的原文片段、原始字符位置和来源，不返回“状态已确认”的结论。

`RETRIEVE(source=raw, entity, predicate, query)` 必须对应当前决策目标。若存在待确认动作，只允许查该动作之后的观察；商品候选切换后尊重 candidate_since。相同公开状态、目标和原文的重复回查拒绝为 `NO_NEW_DECISION_EVIDENCE`。公开状态改变后，旧 raw 片段从当前执行视图移除。

回查后仍需单独执行 `PROPOSE_STATE_UPDATE`，通过原有语义与版本校验器。菜单能从刚取回的历史观察生成更新候选。看到 apple 1 不等于持有 apple 1；原文提及价格也不补造币种。

日志区分：`extractor_gap`（适配器可识别原文事实但图谱尚无可用确认）、`unverified_raw_evidence`（只找到相关原文）、`validation_rejected`、候选容量不足和预算不足。检索没匹配到原文不能证明环境中不存在事实。

## 3. 跨环境决策恢复

`ActionAdapter.decision_targets` 提供待查槽位；`recovery_candidates` 提供基于公开观察的行动建议。共享 `scope/recovery.py` 校验绑定、附上公开来源，并将结果送入管理器与执行器视图。建议不会自动执行，成功与否仍由环境决定。

ALFWorld 提供已观察位置的探索、公开关闭容器的检查，以及当前位置已有公开位置证据的目标物品拿取候选。保存已访问位置、历史已知实体、持有物和物体位置。`open receptacle fridge 1` 可规范化为 `open fridge 1`；缺编号和多匹配仍要求模型选择。

ShopSimulator 从当前公开 clickables 提供重新检查规格或其他候选的建议，不把 Buy Now 自动当作恢复动作。两种适配器共享恢复记录和诊断接口；领域动作语法保留在适配器中。

同一公开状态下同一命令两次失败或无进展，且存在有来源的替代候选时，再次提出该命令会被标记 `no_progress_blocked`，不发送给环境。此次生成仍计 token 和回合预算。状态变化后允许合法重复。这个标记独立于格式错误、绑定失败和环境无效动作统计。

恢复候选不是完整规划器：尚未实现所有任务类型的完整子目标规划，也不保证小模型会选择更好的动作。没有替代候选时不会无限拒绝全部行动。

## 4. 教师预热与 GRPO 接口

在线 rollout 与离线 warmup 共用恢复计划。教师候选现在包含原始证据回查；教师只能选择通过工具检查的操作，并引用学生可见的信息。预热标签不再一见 pending 就优先 FINISH，而是在有可用回查时提供回查示范。

HotpotQA 的教师数据生成脚本 `experiments/memory_curriculum/hotpot_teacher_rewards.py` 和 GRPO 脚本 `hotpot_grpo.py` 均新增 `--shared-interface`；GRPO 还要求 `--bridge-candidates`。它们通过相同的 choice prompt 构造器呈现状态写入、原文获取、摘要、过滤和 FINISH，并将接口选项写入 receipt。旧教师缓存不能混入新接口。

原实验仍是固定阶段的 HotpotQA 课程。统一操作词汇和提示格式减少接口偏移，但不等于已经实现 QA 到动态操作任务的完全同分布训练。写入和压缩阶段不提前泄露后来的问题；答案和 supporting facts 仅进入评分与诊断。

新接口下，压缩可选择保留各片段的开头或结尾完整句子，模型选择具体内容；每条摘要仍引用原文。它是有界抽取式压缩候选，不是任意自由生成的摘要器。成本比较计入压缩操作本身以及后续管理器、执行器的 token。只有两条分支均答对且支持证据完整时才给成本信用；无效比较记录具体原因。

原有同边界、同续跑种子的局部信用路径保留。没有新增“检索一次就加分”的奖励。ALFWorld 当前 HTTP 适配器仍无原生快照，不能把未核验的命令重放标成原生快照局部信用实验。

## 5. 验证及证据边界

回归测试覆盖：无 Claim 的原文回查到校验入图；看到物体不确认持有；旧确认不能满足新动作的 pending；apple 1 与 apple 10；重复回查；检索预算和关闭检索消融；循环拦截与状态改变后的恢复；教师选择 raw 回查；完整 rollout 的候选呈现与日志；源域接口统一后的同前缀对照；压缩操作成本。

只读轨迹检查入口：

```bash
python -m experiments.memory_curriculum.audit_recovery \
  --results runs/hotpot_transfer_20261109.nf63uA/alfworld/initial/results.jsonl \
  --output ccfa-workfiles/engineering/decision_recovery_20261109/public_prefix_audit.json
```

它逐步重建每个决策点当时可见的公开前缀，不执行模型或环境。对已有 134 个失败任务、3,216 轮记录，所有轮次可构造有来源的候选；97 个任务的 2,130 个循环轮次可提供替代建议。沿原失败轨迹有 2,713 次命令会触发重复拦截。这些数字说明恢复入口覆盖情况，不是新策略实际行动数、成功率或因果收益。

测试及部署哈希位于同目录 `tests.log`、`deployment.json`。原始副本为 `before.tar.gz` 和 `before_sha.json`。测试不调用付费教师 API，不占用 GPU，也不修改正在运行的实验。

## 6. 下一步实测

先在独立开发任务上固定执行器和生成预算，比较：旧协议、新恢复协议、关闭检索的新协议。报告严格成功、合法动作、循环、恢复候选选择及公开状态进展、管理器跳过原因、总 token。另设完整公开历史／训练集操作示范的执行器对照，判断执行器基础能力是否仍构成瓶颈。

本次尚未启动新的教师采样、SFT、GRPO 或 GPU 成功率评估。旧的 134 个 valid_unseen 任务已用于诊断，后续针对其修改的结果应按开发性证据报告。运行新训练时使用新目录和新 receipt；先通过合法与有效操作验证，再扩充训练规模。
