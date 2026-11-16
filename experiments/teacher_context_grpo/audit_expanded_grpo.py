"""Verify independent manager branches and separate SFT gains from GRPO gains."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scope.io import atomic_json, file_sha
from evidence_data import format_signals


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def audit(root,source,parent):
    torch.set_num_threads(4)
    receipt,summary=read(root/'receipt.json'),read(root/'summary.json')
    assert summary['complete'] and read(root/'status.json')['phase']=='complete'
    assert not (root/'error.json').exists() and read(root/'exit.json')['exit_code']==0
    assert file_sha(root/'receipt.json')==summary['receipt_sha256']
    assert file_sha(root/'expanded_grpo.py')==receipt['source_sha256']
    assert file_sha(source/'summary.json')==receipt['source_summary_sha256']
    assert file_sha(source/'receipt.json')==receipt['source_receipt_sha256']
    n=receipt['questions_per_arm'];train=receipt['train_ids'];dev=receipt['dev_ids']
    assert len(train)==len(set(train))==n==256
    assert len(dev)==len(set(dev))==96
    assert not set(train).intersection(dev)
    assert not set(train).intersection(read(source/'receipt.json')['train_ids'])
    initial_path=parent/'initial/state.pt'
    actor_path=Path(read(source/'summary.json')['training']['answer_only']['checkpoint'])/'state.pt'
    assert file_sha(initial_path)==receipt['initial_checkpoint_sha256']
    assert file_sha(actor_path)==receipt['teacher_actor_checkpoint_sha256']
    initial_adapters=torch.load(initial_path,map_location='cpu',weights_only=False)['adapters']
    teacher_adapters=torch.load(actor_path,map_location='cpu',weights_only=False)['adapters']
    arms={}; distributions={};parameter_checks={}
    for arm in receipt['arms']:
        rows=[read(p) for p in sorted((root/arm/'steps').glob('*.json'))]
        assert [r['task_id'] for r in rows]==train
        assert [r['question_index'] for r in rows]==list(range(1,n+1))
        counts=Counter();distribution=Counter();choices=Counter();expanded_format=0
        for r in rows:
            assert r['version_after']-r['version_before'] in (0,1)
            assert r['seed_step']==receipt['seed_positions'][r['task_id']]
            assert r['version_after']-r['version_before']==int(r['grpo']['optimizer_step'])
            local=r.get('local')
            if local:
                winners=[c for c in local['candidates'] if c['score']['em']>local['control']['score']['em']]
                expanded_format+=bool(winners and format_signals(local['control']['answer'],winners[0]['answer'])['format_compatible'])
                returns=local['metrics']['strict_returns']
                distribution['all_correct' if all(x==1 for x in returns) else
                             'all_wrong' if all(x==0 for x in returns) else 'mixed']+=1
                choices[str(local['metrics']['unique_choices'])]+=1
                counts['local_groups']+=1
                counts['local_all_failed']+=local['metrics']['all_failed']
                counts['local_nonzero_task_advantage']+=not local['metrics']['task_zero_variance']
                counts['local_strict_win_groups']+=local['has_strict_win_vs_noop']
                counts['local_format_compatible_win_groups']+=local['format_compatible_win']
                counts['local_support_coverage_win_groups']+=local['support_coverage_win']
                for c in local['candidates']:
                    s=c['sample']
                    assert s['origin']=='policy' and s['version']==r['version_before']
                    assert s['boundary']==local['boundary']
                    assert c['seed']==local['control']['seed']==local['continuation_seeds'][0]
                    assert s['old_logprobs'] and len(s['old_logprobs'])==len(s['completion_ids'])
                if local['metrics']['task_zero_variance']:
                    assert not r['grpo']['optimizer_step']
            elif 'trajectory' in r:
                counts['trajectory_groups']+=1
                counts['trajectory_all_failed']+=all(x['score']['em']==0 for x in r['trajectory'])
            counts['local_optimizer_steps' if r['question_index']%2 else 'trajectory_optimizer_steps']+=r['grpo']['optimizer_step']
        for k,v in counts.items():assert v==summary['counts'][arm][k],(arm,k,v)
        assert read(root/arm/'resume_validation.json')['passed']
        assert read(root/arm/'checkpoints/latest.json')['step']==n
        checkpoint=root/arm/'checkpoints'/f'step_{n:05d}'
        assert file_sha(checkpoint/'state.pt')==read(checkpoint/'metadata.json')['state_sha256']
        adapters=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)['adapters']
        maximum=0.;changed=0
        for name,tensor in adapters['manager'].items():
            old=initial_adapters['manager'][name]
            changed+=int(torch.count_nonzero(tensor!=old))
            maximum=max(maximum,float((tensor.float()-old.float()).abs().max()))
        assert changed>0
        for role in ('manager_ref','actor','actor_ref'):
            expected=teacher_adapters[role] if arm=='teacher_actor' and role!='manager_ref' else initial_adapters[role]
            assert all(torch.equal(tensor,expected[name]) for name,tensor in adapters[role].items())
        parameter_checks[arm]={'manager_changed_elements':changed,'max_abs_manager_delta':maximum,
                              'actor_and_reference_adapters_unchanged':True}
        path=root/arm/'dev.jsonl';assert file_sha(path)==summary['results_sha256'][arm]
        records=jsonl(path);assert [x['task_id'] for x in records]==dev
        assert len({x['parameters'] for x in records})==1
        assert sum(x['em'] for x in records)==summary['dev'][arm]['em_count']
        arms[arm]=records
        distributions[arm]={'local_reward_groups':dict(distribution),'unique_choice_histogram':dict(choices),
                            'extended_format_compatible_win_groups':expanded_format}
    before={}
    for arm,old in (('initial_actor','initial'),('teacher_actor','answer_only')):
        old_records=jsonl(source/old/'results.jsonl')
        assert [r['task_id'] for r in old_records]==dev
        before[arm]=[r['cells']['native']['em'] for r in old_records]
    differences=np.asarray([(t['em']-tb)-(i['em']-ib) for t,tb,i,ib in zip(
        arms['teacher_actor'],before['teacher_actor'],arms['initial_actor'],before['initial_actor'],strict=True)])
    rng=np.random.default_rng(233)
    boot=rng.choice(differences,size=(5000,len(differences)),replace=True).mean(axis=1)
    did={'n':len(differences),'teacher_grpo_gain_minus_control_grpo_gain':float(differences.mean()),
         'bootstrap_95':[float(x) for x in np.quantile(boot,[.025,.975])]}
    result={'passed':True,'audit_source_sha256':file_sha(Path(__file__)),
            'analysis_helper_sha256':file_sha(Path(__file__).with_name('evidence_data.py')),'questions_per_arm':n,'dev_n':len(dev),'source_train_disjoint_from_sft':True,
            'on_policy_local_samples_verified':True,'same_local_continuation_seeds':True,
            'restoration_passed':True,'difference_in_differences':did,'local_distributions':distributions,
            'checkpoint_parameter_checks':parameter_checks,
            'summary_sha256':file_sha(root/'summary.json'),
            'interpretation':'Exploratory follow-up chosen after development results; final teacher/control gap includes SFT effects. '
                             'Within-arm pre/post changes and difference in differences isolate measured GRPO increments.'}
    atomic_json(root/'audit.json',result)
    names={'initial_actor':'原执行器＋管理器 GRPO','teacher_actor':'教师预热执行器＋管理器 GRPO'}
    lines=['# 教师执行器预热后的 256 题管理器 GRPO 对照','',
           '日期：2026-11-12。两组训练、恢复验证和冻结开发评估完成，数据与采样核验通过。','',
           '这是在执行器开发结果之后新建的探索性对照。额外证据选择监督没有超过短答案监督，原定联合组件晋级门槛仍未通过。'
           '为完成后续管理器训练验证，使用训练 token 更少、开发 EM 相同的短答案预热执行器，与原执行器进行同预算对照。'
           '这一选择使用了开发反馈；下面的 96 题结果不是独立最终测试。','',
           '## 分开报告 SFT 与 GRPO 的收益','',
           '| 执行器 | GRPO 前 | 256 题 GRPO 后 | GRPO 净变化 |','| --- | ---: | ---: | ---: |']
    for arm in receipt['arms']:
        m=summary['dev'][arm];old=sum(before[arm]);delta=(m['em_count']-old)/len(dev)
        lines.append(f"| {names[arm]} | {old}/96（{old/96*100:.2f}%） | {m['em_count']}/96（{m['em_count']/96*100:.2f}%） | {delta*100:+.2f} 个百分点 |")
    lines += ['', '配对 bootstrap 95% CI（百分点）：','']
    for key,pair in summary['paired'].items():
        lines.append(f"- {key}：{pair['delta_em']*100:+.2f}，[{pair['bootstrap_95'][0]*100:.2f}, {pair['bootstrap_95'][1]*100:.2f}]。")
    lines += ['', f"差上差，即“教师执行器分支的 GRPO 增量 − 原执行器分支的 GRPO 增量”："
              f"{did['teacher_grpo_gain_minus_control_grpo_gain']*100:+.2f} 个百分点，95% CI "
              f"[{did['bootstrap_95'][0]*100:.2f}, {did['bootstrap_95'][1]*100:.2f}]。",'',
              '最终两组之间的差距包含之前的执行器 SFT 收益，不能全部归因于过程信用或 GRPO。','',
              '## 局部信用是否形成可学习信号','',
              '| 分支 | 局部组 | 全错组 | 全对组 | 混合回报组 | 局部实际更新 | 整轨迹实际更新 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in receipt['arms']:
        c=summary['counts'][arm];d=distributions[arm]['local_reward_groups']
        lines.append(f"| {names[arm]} | {c['local_groups']} | {d.get('all_wrong',0)} | {d.get('all_correct',0)} | {d.get('mixed',0)} | "
                     f"{c['local_optimizer_steps']} | {c['trajectory_optimizer_steps']} |")
    lines += ['', '候选相对不操作对照有收益，不等于候选之间存在差异。全对组和全错组都可能得到零标准化优势。'
              '因此必须把教师预热降低全错率、组内差异和实际梯度更新分开解释。','']
    for arm in receipt['arms']:
        c=summary['counts'][arm]
        lines.append(f"- {names[arm]}：相对不操作有严格收益的局部组 {c['local_strict_win_groups']} 个；"
                     f"原始词序列规则标记格式相关候选 {c['local_format_compatible_win_groups']} 个；补充标点/空格和 yes/no 解释规则后为 "
                     f"{distributions[arm]['extended_format_compatible_win_groups']} 个。标注支持句覆盖增加 {c['local_support_coverage_win_groups']} 个。"
                     f"每组不同候选数的分布：{json.dumps(distributions[arm]['unique_choice_histogram'])}。")
    lines += ['', '格式标记只用于分析，不改写严格 EM 奖励；支持句覆盖是离线诊断，不进入管理器输入。'
              '两个分支各自的管理器会随训练演化，因此训练期间的局部组统计是各自策略下的观测，不能当成完全相同状态上的纯执行器因果效应。','',
              '## 实施与边界','',
              '- 两组管理器从同一原始 checkpoint 和 Adam 状态开始；教师分支只替换 actor/actor_ref。',
              '- 每组 256 道新 HotpotQA Training 题，与教师示范的 128 题无 ID 重叠；交替局部与整轨迹训练，K=4，每题最多一次更新。',
              '- 执行器与底座冻结。局部候选来自当前学生策略，保留旧 log-prob、策略版本、同一前缀与配对续跑种子；教师强制动作不混入损失。',
              '- 严格 EM 主目标，token 成本系数为零；无在线教师 API 调用。教师作用来自前面的源域执行器监督。',
              '- 每 64 题及最后保存 checkpoint；第 1 题完成实际参数和优化器恢复验证。',
              '- 无目标 benchmark 更新，没有改变或替换原 90,447 题主训练。',
              f"- 本阶段墙钟 {summary['elapsed_seconds']/60:.2f} 分钟，共享 GPU2；这不是独占 GPU 计时。",'',
              '来源与结果均为一个固定种子的开发实验。ID 与规范化问题隔离不等于 Wikipedia 文档或事实完全不重叠；'
              '最终泛化仍需冻结方案后在独立测试集检查。','',
              f'实验路径：{root}', '',
              '结果：summary.json、audit.json、receipt.json；每组 steps 保存逐题信用记录，dev.jsonl 保存逐题结果，checkpoints/step_00256 为最终模型。', '']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.source,a.parent)
