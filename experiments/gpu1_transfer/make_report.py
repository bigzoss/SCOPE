"""Render the completed pilot from measured JSON; no training or selection."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args();root=Path(args.root)
    summary=load(root/'summary.json');analysis=load(root/'analysis.json');verify=load(root/'verification.json');protocol=load(root/'protocol.json')
    if not summary['complete'] or not verify['passed']:raise RuntimeError('Results are not verified')
    names={'source_initial':'源域：共同工具预热M0','source_trained':'源域：M0＋12步GRPO',
        'target_initial':'目标：共同工具预热M0','target_transfer':'目标：源域12步GRPO后冻结',
        'target_adapted':'目标：源域12步＋目标8步GRPO',
        'target_only_adapted':'目标：M0直接做目标8步GRPO'}
    conditions=summary['conditions'];elapsed=verify['pipeline_elapsed_wall_seconds'];h=int(elapsed//3600);m=int(elapsed%3600//60);s=int(elapsed%60)
    finished=verify['date_local']
    lines=['# GPU1：ShopSimulator训练→ALFWorld冻结迁移→少量适配对照','',
        f'核验完成：{finished}。实验目录：`{root}`。','',
        '**本轮全流程完成，工程核验通过；没有观察到成功率提升、检索/压缩行为改善或token节省。**',
        '这是当前SCOPE基础实现的单seed、固定24目标任务试验，不是官方完整134任务评估，也不构成完整长程记忆泛化证明。','',
        '## 1. 实际执行范围','',
        '- 仅使用物理GPU1，固定seed/split_seed=233。共享SFT底座未改写；所有条件使用同一冻结的ShopSimulator SFT epoch3执行器，仅更新管理器LoRA。',
        '- 从6个源域训练任务收集公开轨迹，生成335条工具校验通过的规则示例；按操作均衡选128条预热，实际覆盖5个任务，完成128次SFT更新。这个共同初始化记为M0。',
        '- 源域GRPO：固定12次更新，K4，48条全局查询＋16条原生快照局部续跑。源域前后分别评估同一24个internal_dev任务。',
        '- 两条目标训练分支：各8次更新、32条轨迹，实际各使用8个不同train任务，任务顺序一致。一个从源域GRPO初始化，一个从M0初始化。',
        '- 目标冻结评估：四条件各24任务，每个官方任务类型4个，来自valid_unseen；总计96条目标评估记录。目标适配采用全局GRPO，public-v1接口没有原生快照。',
        '- 源域最多12轮，目标最多30轮；actor最多256新token、manager最多192新token；同环境条件使用相同预算。任务事实每次重建，经验库为空且只读。',
        '- 固定最终checkpoint，不根据目标评估结果选模、换提示或改预算。',
        '- 回报为`(strict_success + 0.1 * loose_reward)/1.1`；本轮成本及非法操作惩罚系数均为0，仅监测其指标，避免成本微差单独产生组内优势。','',
        '“直接目标域GRPO”仍共享源域SFT底座和工具预热，仅没有本轮12步源域GRPO。两条分支匹配目标训练预算，累计源域＋目标训练算力不同。','',
        '## 2. 冻结评估结果','',
        '| 条件 | 严格成功 | 成功率 | 平均宽松奖励 | 平均严格奖励 | 平均总token |',
        '|---|---:|---:|---:|---:|---:|']
    for name,res in conditions.items():
        lines.append(f"| {names[name]} | {res['success_count']}/{res['n']} | {res['strict_success_rate']:.2%} | {res['mean_loose_reward']:.6f} | {res['mean_strict_reward']:.6f} | {res['mean_total_tokens']:.2f} |")
    lines+=['','源域前后均为17/24（70.83%），平均宽松/严格奖励和token相同。四种目标管理器均为0/24，平均87763.17 token；非法动作率均为49.25%，生成长度终止率均为0。','',
        '这里的严格成功是所有终局要求均满足的二元判定；源域平均严格奖励仍可能包含部分匹配分值，不能与成功率互换。目标环境的宽松/严格奖励在本接口中均为二元成功信号。','',
        '| 配对问题 | 成功率差值（后者－前者） | 经验簇bootstrap 95%区间 |',
        '|---|---:|---|']
    pair_names={'source_grpo_effect':'源域GRPO效应','zero_shot_manager_transfer':'源域训练后的零样本迁移效应',
        'target_adaptation_effect':'源域初始化后的目标适配效应','source_initialization_effect':'相同目标预算下的源域初始化效应'}
    for name,value in summary['paired_comparisons'].items():
        lines.append(f"| {pair_names[name]} | {value['difference']:+.2%} | [{value['ci95'][0]:+.2%}, {value['ci95'][1]:+.2%}] |")
    lines+=['','以上比较每组n=24，bootstrap重复20000次。配对结果完全一致导致经验差值区间退化为[0,0]，不能据此证明总体真实效应严格为0。',
        '单条件描述性Wilson近似区间：17/24约为[50.83%,85.09%]，0/24约为[0%,13.80%]；这些区间不替代分层抽样不确定性和训练seed差异。','',
        '## 3. 管理器究竟做了什么','',
        '| 条件 | 实际采样的管理器调用 | 操作 | 检索次数 | 摘要次数 |',
        '|---|---:|---|---:|---:|']
    for name,value in analysis['conditions'].items():
        lines.append(f"| {names[name]} | {value['sampled_manager_calls']} | {json.dumps(value['operations'],ensure_ascii=False)} | {value['costs']['retrieval_calls']} | {value['costs']['summary_calls']} |")
    lines+=['','**六组冻结评估中，管理器均只选择FINISH。工具接受率100%只表示这些FINISH通过了格式与版本检查，不表示有效管理了记忆。** 工作视图中的需求/尝试动作由运行器自动维护，不能算作管理器学到的检索、压缩或状态更新。','',
        '源域训练采样出现过2次成功RETRIEVE，其余主要为FINISH或被拒绝操作；冻结后的策略没有表现出实际检索/摘要行为。',
        '两条目标训练分支各32条轨迹全部失败，所有8个组的任务回报均为0，任务优势也均为0。参数确实有变化，但该阶段优化仅剩KL正则项，不能称为任务技能学习。','',
        '## 4. 信用与预热诊断','',
        '- 源域12次更新中，8个全局组复合任务回报零方差，9个组严格成功零方差；1个组的优势仅来自宽松奖励差异。',
        '- 4个局部组的任务回报全部零方差，均未产生局部任务优势。本轮不能证明条件局部信用机制带来了收益。',
        '- 源域全局更新4、8、11存在回报差异，却没有有效非FINISH管理操作。其梯度没有隔离执行器/环境随机性，不能直接归因于记忆效用。',
        '- 128条预热输入全部没有已提交Claim；53个不同输入中有48个对应多种操作标签。真实运行包含受保护需求和尝试动作Claim，存在输入分布差异；是否造成FINISH行为，需要独立对照确认。',
        '- 目标任务记录中出现`go to receptacle`、`go to desklamp.`等未正确绑定观察对象名称/编号的命令。当前固定执行器/提示接口的能力不足，可能掩盖记忆管理的潜在收益。该现象不等于所有跨benchmark迁移都必须重新训练。','',
        '不能把这些负结果解释为长短期记忆机制普遍无效：本轮没有建立有效管理行为，目标执行器也没有完成目标任务。它说明当前初始化、任务和信用设置尚不足以验证原研究假设。','',
        '## 5. 成本与资源','',
        '| 阶段 | 更新数 | 真实轨迹/续跑数 | 不同训练任务 | 日志计费token |',
        '|---|---:|---:|---:|---:|']
    for name,title in [('source_train','源域GRPO'),('target_adapt','源域初始化＋目标GRPO'),('target_only_adapt','M0直接目标GRPO')]:
        v=analysis['training'][name];lines.append(f"| {title} | {v['updates']} | {v['trajectories']} | {v['unique_tasks']} | {v['charged_tokens']:,} |")
    logged=verify['logged_grpo_tokens']+verify['logged_evaluation_tokens']
    manager_ratio=(analysis['conditions']['target_initial']['costs']['manager_input']+analysis['conditions']['target_initial']['costs']['manager_output'])/analysis['conditions']['target_initial']['costs']['charged_tokens']
    lines += ['',f"服务器流程墙钟时间：{h}小时{m}分钟{s}秒。已记录GRPO采样token为{verify['logged_grpo_tokens']:,}，冻结评估token为{verify['logged_evaluation_tokens']:,}，合计{logged:,}。",
        '源域收集没有保存采样token，SFT教师强制重算也未计入上述生成采样总量，因此这不是整个准备与训练过程的完整token成本。墙钟时间包含CPU与服务阶段，不作为独占GPU计费时长。',
        f'目标评估中管理器输入/输出占总token的{manager_ratio:.2%}，却没有执行实际检索/摘要。本轮没有测到上下文管理的成本收益。','',
        '## 6. 完整性与数据范围','',
        '- 六组结果各24条，ID唯一、与固定任务清单完全一致；逐任务结果SHA均通过。四个目标条件使用同一24个任务。',
        '- 源域训练12步、两条目标训练各8步，提交步号连续且无重复；目标适配任务顺序一致。',
        '- 四个训练阶段checkpoint的actor张量SHA完全一致；评估前后LoRA、冻结底座版本与经验库不变。',
        '- 所有执行receipt均为物理GPU1，同一底座文件SHA、执行源码SHA及checkpoint原始receipt校验通过。',
        '- 目标train/holdout/test按game/task分离；未知基础模型预训练暴露不由这些检查证明。原valid_seen140任务本轮未查询。',
        '- 24任务为原134任务的按类型子集。原始子run summary中的subset=false仅表示没有再次对当前manifest使用--limit，不表示完成了全部134任务。',
        '- 任务事实不跨任务共享，当前经验库为空；没有验证跨会话长期经验形成或完整ALFWorld类型化状态图谱。',
        '- 自己的GPU任务已结束，5131/5132专用服务已清理；原5084健康检查为HTTP200。','',
        '## 7. 下一轮应先解决的问题','',
        '1. 用实际已提交状态和工作视图构造有状态预热，按缺失信息与操作效用组织标签；先检查必要操作覆盖率与条件正确性，不能只看格式接受率。',
        '2. 在ALFWorld训练任务上检查执行器的公开命令语法与对象绑定能力，先建立可用的固定执行器；再用同一个执行器比较管理器。当前结果只覆盖管理器适配。',
        '3. 使用确实需要历史证据的任务与受控上下文预算，建立管理操作的有效回报差异；固定执行器权重之外，还要控制其续跑随机性，减少全局信用污染。',
        '本轮不自动扩大到134任务或继续训练。后续协议若基于这24任务的诊断修改，应将它们标注为已用开发证据，另保留目标评估任务。','',
        '## 8. 证据与复核','',
        '`protocol.json`、`summary.json`、`analysis.json`、`verification.json`为本报告依据；原始各条件results、receipt、日志和checkpoint保留在实验目录。',
        f"最终执行器适配器SHA：`{verify['actor_hash']}`。",
        f"协议SHA：`{sha(root/'protocol.json')}`。",
        f"原始汇总SHA：`{sha(root/'summary.json')}`。"]
    report='\n'.join(lines)+'\n'
    if '\ufffd' in report:raise RuntimeError('Invalid text encoding')
    (root/'report.md').write_text(report,encoding='utf-8')
    bundle={'summary':summary,'analysis':analysis,'verification':verify,'protocol':protocol,
            'report_sha256':sha(root/'report.md'),'generator_sha256':sha(__file__)}
    (root/'results_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report_sha256':bundle['report_sha256'],'bundle_sha256':sha(root/'results_bundle.json'),'wall_seconds':elapsed}))


if __name__=='__main__':main()
