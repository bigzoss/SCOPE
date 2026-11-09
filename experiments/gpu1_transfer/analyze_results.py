"""Descriptive analysis of the fixed experiment; never selects or modifies a policy."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def load(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).open() if line.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(success,n,z=1.959963984540054):
    p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    summary=load(root/'summary.json')
    if not summary['complete'] or load(root/'status.json')['phase']!='complete':raise RuntimeError('Study is not complete')
    protocol=load(root/'protocol.json');conditions={};training={}
    for name in summary['conditions']:
        observed=rows(root/name/'results.jsonl');result=load(root/name/'summary.json')
        if sha(root/name/'results.jsonl')!=result['results_sha256']:raise RuntimeError('Changed result file')
        operations=collections.Counter();rejections=collections.Counter();skip_reasons=collections.Counter();sampled=accepted=skipped=0
        for row in observed:
            for event in row['manager_operations']:
                response=event['result']
                if not event.get('sampled'):
                    skipped+=1;skip_reasons[response.get('reason','unspecified')]+=1;continue
                sampled+=1;accepted+=int(bool(response.get('accepted')))
                operations[response.get('operation','unknown')]+=1
                if not response.get('accepted'):rejections[response.get('reason','unknown')]+=1
        conditions[name]={'n':len(observed),'success_count':result['success_count'],
            'strict_success_rate':result['strict_success_rate'],'wilson95':wilson(result['success_count'],len(observed)),
            'sampled_manager_calls':sampled,'accepted_manager_calls':accepted,'manager_acceptance_rate':accepted/max(1,sampled),
            'skipped_manager_calls':skipped,'skip_reasons':dict(skip_reasons),'operations':dict(operations),'rejection_reasons':dict(rejections),
            'costs':{key:sum(row['costs'].get(key,0) for row in observed) for key in
                ['charged_tokens','manager_input','manager_output','actor_input','actor_output','retrieval_calls','summary_calls','tool_wall_seconds']}}
    for name in ('source_train','target_adapt','target_only_adapt'):
        outcomes=rows(root/name/'trajectories.jsonl');updates=rows(root/name/'train_metrics.jsonl')
        strict_groups=collections.defaultdict(list);active_operations=collections.defaultdict(int)
        for row in outcomes:
            key=(row['identity']['step'],row['identity']['kind'])
            strict_groups[key].append(row['metrics']['success'])
            active_operations[key]+=sum(event['phase']!='execute' and event.get('sampled',False)
                and event.get('result',{}).get('accepted',False) and event['result'].get('operation')!='FINISH' for event in row['events'])
        partial_only=0
        for update in updates:
            values=strict_groups[(update['step'],'global')]
            if values and max(values)==min(values) and any(a!=0 for a in update['global']['advantages']):partial_only+=1
        training[name]={'updates':len(updates),'trajectories':len(outcomes),
            'unique_tasks':len({row['task_id'] for row in outcomes}),
            'sampled_successes':sum(int(row['metrics']['success']) for row in outcomes),
            'charged_tokens':sum(row['costs']['charged_tokens'] for row in outcomes),
            'role_tokens':{role:sum(len(sample['prompt_ids'])+len(sample['completion_ids']) for row in outcomes
                                  for sample in row['samples'] if sample['role']==role) for role in ('manager','actor')},
            'global_task_zero_variance_updates':sum(bool(u['global']['task_zero_variance']) for u in updates),
            'global_strict_zero_variance_updates':sum(max(strict_groups[(u['step'],'global')])==min(strict_groups[(u['step'],'global')]) for u in updates),
            'global_loose_only_advantage_updates':partial_only,
            'varying_global_updates_without_active_management':[u['step'] for u in updates
                if not u['global']['task_zero_variance'] and active_operations[(u['step'],'global')]==0],
            'local_task_zero_variance_updates':sum(bool(u.get('local',{}).get('task_zero_variance')) for u in updates if u.get('local')),
            'updates_with_parameter_change':sum(bool(u['parameters_changed']) for u in updates),
            'task_only_reward_verified':not any(u['global']['non_task_only_advantage'] for u in updates),
            'reported_training_wall_seconds':sum(u['elapsed_wall_seconds'] for u in updates)}
        if sum(training[name]['role_tokens'].values())!=training[name]['charged_tokens']:
            raise RuntimeError('Continuation token cost contains duplicated prefixes')
    # Compare the actor weights across all stage checkpoints, independently of train logs.
    import torch
    hashes={}
    for name in ('warmup','source_train','target_adapt','target_only_adapt'):
        checkpoint=root/name/'checkpoints'/load(root/name/'checkpoints/latest.json')['path']
        data=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        h=hashlib.sha256()
        for key,value in sorted(data['adapters']['actor'].items()):
            h.update(key.encode());h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        hashes[name]=h.hexdigest()
    if len(set(hashes.values()))!=1:raise RuntimeError('Frozen actor differs across conditions')
    examples=rows(root/'manager_examples.jsonl');claim_counts=[];prompt_labels=collections.defaultdict(set)
    for row in examples:
        claim_counts.append(len(json.loads(row['messages'][-1]['content'])['claims']))
        prompt_labels[json.dumps(row['messages'],sort_keys=True)].add(json.loads(row['completion'])['operation'])
    warmup_diagnostic={'examples':len(examples),'unique_source_tasks':len({row['task_id'] for row in examples}),
        'examples_with_committed_claims':sum(n>0 for n in claim_counts),'max_committed_claims':max(claim_counts),
        'unique_prompts':len(prompt_labels),'prompts_with_multiple_operation_labels':sum(len(v)>1 for v in prompt_labels.values()),
        'interpretation':'Warmup has empty Claim inputs and alternative operations for identical prefixes; online rollout has protected requirement and attempted-action Claims. A possible distribution mismatch, not a causally established explanation.'}
    analysis={'conditions':conditions,'training':training,'actor_hashes':hashes,'same_actor_across_all_stages':True,
        'warmup_diagnostic':warmup_diagnostic,'original_benchmark_subset':True,'original_target_test_tasks':134,
        'protocol_sha256':sha(root/'protocol.json'),'raw_summary_sha256':sha(root/'summary.json'),
        'source_collection_token_cost':None,
        'cost_limit':'Source collection stores public evidence without sampled-token costs; it is not included in the logged token totals.',
        'comparison_limit':'Target-only and source-initialized adaptation match target updates, not cumulative source+target training compute.',
        'uncertainty_limit':'One training seed; 24 fixed equal-type target tasks. Degenerate paired bootstrap does not establish zero population effect.'}
    (root/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2))
    lines=['','补充核验：四个阶段checkpoint的actor张量哈希完全一致。目标适配预算相同，但源域初始化分支有额外源域训练成本。','',
        '| 条件 | 成功率Wilson 95%区间 | 管理器工具接受率 | 管理器输入/输出token合计 | 执行器输入/输出token合计 |',
        '|---|---|---:|---:|---:|']
    for name,result in conditions.items():
        cost=result['costs'];lo,hi=result['wilson95']
        lines.append(f"| {name} | [{lo:.2%}, {hi:.2%}] | {result['manager_acceptance_rate']:.2%} | {cost['manager_input']+cost['manager_output']} | {cost['actor_input']+cost['actor_output']} |")
    lines+=['','| 训练阶段 | 更新数 | 实际轨迹 | 独立任务 | 计费token | 全局任务回报零方差组 |','|---|---:|---:|---:|---:|---:|']
    for name,result in training.items():lines.append(f"| {name} | {result['updates']} | {result['trajectories']} | {result['unique_tasks']} | {result['charged_tokens']} | {result['global_task_zero_variance_updates']} |")
    lines+=['','上述token总量包含日志中的GRPO真实采样与局部续跑，但不包含未记录采样token的源域公开轨迹收集，也不把SFT教师强制重算当作生成采样。',
        '复合任务回报包含0.1倍宽松奖励；严格结果相同时，部分组的优势仍可能仅来自宽松奖励差异，analysis.json单独记录这些组。',
        '另行标记存在回报方差但没有有效非FINISH管理操作的全局组；其梯度没有隔离执行器/环境随机性，不能自动解释为记忆效用。',
        '工具接受率表示通过当前校验器，不等于完整语义准确率或真实购物/家务成功率。',
        '若配对样本全失败，bootstrap差值区间可能退化为[0,0]，只描述这组样本；单条件Wilson区间和样本限制必须同时保留。',
        '原始summary中的subset=false表示未对当前24任务manifest再次使用--limit，不表示完成了原始134任务评估。',
        f"预热诊断：{len(examples)}条示例实际覆盖{warmup_diagnostic['unique_source_tasks']}个源任务，包含已提交Claim的示例为{warmup_diagnostic['examples_with_committed_claims']}条；{warmup_diagnostic['unique_prompts']}个独立输入中，{warmup_diagnostic['prompts_with_multiple_operation_labels']}个对应多种操作标签。实际rollout包含需求/尝试动作Claim，存在输入分布差异；其对FINISH行为的因果作用仍需后续对照。"]
    (root/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':True,'same_actor':True,'conditions':{k:v['success_count'] for k,v in conditions.items()}}))


if __name__=='__main__':main()
