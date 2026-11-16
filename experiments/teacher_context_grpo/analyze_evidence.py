"""Separate fact-access and answer-format-compatible changes after fixed evaluation."""
import argparse
from pathlib import Path
import json
from collections import Counter

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scope.hotpot import normalize, split_row
from scope.io import atomic_json, file_sha
from evidence_data import format_signals


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def contains(answer, target):
    return format_signals(answer,target)['format_compatible']


def main(root, base, parent):
    receipt=read(root/'receipt.json')
    ids=set(receipt['dev_ids'])
    gold={r['id']:split_row(r)[1] for r in pq.read_table(base/'dataset/validation-00000-of-00001.parquet').to_pylist()
          if r['id'] in ids}
    results={a:jsonl(root/a/'results.jsonl') for a in ('initial','answer_only','evidence_and_answer')}
    records={}
    for arm,rows in results.items():
        native_gains=[];native_losses=[];fact_access=[];format_access=[]
        evidence_use_gains=[]
        for before,after in zip(results['initial'],rows,strict=True):
            task=after['task_id'];target=gold[task]['answer']
            old=before['cells']['native'];new=after['cells']['native']
            if new['em']>old['em']:
                native_gains.append({'task_id':task,'before':old['answer'],'after':new['answer'],
                                     'format_compatible':contains(old['answer'],target),
                                     'initial_category':after['initial_category']})
            if new['em']<old['em']:
                native_losses.append({'task_id':task,'before':old['answer'],'after':new['answer']})
            augmented=after['cells']['support_augmented'];distractor=after['cells']['distractor_augmented']
            if (after['diagnostic_meta']['missing_support_sentences'] and after['diagnostic_meta']['token_matched_20pct']
                and augmented['available'] and distractor['available'] and augmented['em']==1 and new['em']==0
                and distractor['em']==0):
                detail={'task_id':task,'native':new['answer'],'augmented':augmented['answer'],
                        'distractor':distractor['answer']}
                (format_access if contains(new['answer'],target) else fact_access).append(detail)
            previous=before['cells']['support_augmented']
            if previous['available'] and augmented['available'] and augmented['em']>previous['em']:
                evidence_use_gains.append({'task_id':task,'before':previous['answer'],'after':augmented['answer'],
                                           'format_compatible':contains(previous['answer'],target)})
        records[arm]={'native_gains':native_gains,'native_losses':native_losses,
                      'fact_access_candidates_excluding_target_in_old_answer':fact_access,
                      'format_compatible_access_candidates':format_access,
                      'gains_using_same_augmented_evidence':evidence_use_gains}
    tokenizer=AutoTokenizer.from_pretrained(read(parent/'receipt.json')['model'],local_files_only=True)
    costs={}
    for arm in ('answer_only','evidence_and_answer'):
        examples=jsonl(root/arm/'examples.jsonl')
        inp=sum(len(tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)) for r in examples)
        out=sum(len(tokenizer.encode(r['completion'],add_special_tokens=False))+1 for r in examples)
        costs[arm]={'epochs':2,'input_tokens_over_training':2*inp,'label_tokens_including_eos_over_training':2*out,
                    'normal_labels_exceed_24_tokens':sum(len(tokenizer.encode(r['completion'],add_special_tokens=False))+1>24
                        for r in examples if r['supervision']=='short_answer')}
    data={'arms':records,'training_token_costs':costs,
          'analysis_source_sha256':{p.name:file_sha(p) for p in (Path(__file__),Path(__file__).with_name('evidence_data.py'))},
          'format_analysis_revision':'posthoc inclusion of punctuation/spacing and binary-answer explanations; raw training categories and rewards unchanged',
          'interpretation':'Target substring, punctuation/spacing, and binary prefix matches flag format-compatible changes only; they do not certify semantic equivalence. '
                           'Fact access uses oracle-support diagnostics, not learned retrieval success.'}
    atomic_json(root/'evidence_analysis.json',data)
    summary=read(root/'summary.json')
    audit=read(root/'audit.json')
    assert audit['passed']
    names={'initial':'初始化','answer_only':'教师短答案监督','evidence_and_answer':'教师证据选择＋短答案监督'}
    lines=['# SCOPE 扩大教师证据使用训练：结果与诊断','',
           '日期：2026-11-12。两组固定训练与冻结开发评估已完成，独立核验通过。','',
           '本轮从 128 道新的 HotpotQA Training 问题生成教师示范，使用与上一轮 48 道诊断题无重叠的 96 道开发题。'
           '教师模型为 qwen3.8-flash-next。普通评估无教师调用，管理器参数全程冻结。','',
           '## 主结果：正常记忆视图','',
           '| 配置 | 严格 EM | F1 |','| --- | ---: | ---: |']
    for arm in names:
        m=summary['results'][arm]['native']
        lines.append(f"| {names[arm]} | {m['em_count']}/{m['n']}（{m['em']*100:.2f}%） | {m['f1']*100:.2f}% |")
    lines += ['', '配对差值使用 5,000 次 task bootstrap；以下区间以百分点计。','']
    for key,p in summary['paired'].items():
        lines.append(f"- {key}：{p['delta']*100:+.2f}，95% CI [{p['bootstrap_95'][0]*100:.2f}, {p['bootstrap_95'][1]*100:.2f}]；"
                     f"由错到对 {p['wins']} 题，由对到错 {p['losses']} 题。")
    lines += ['', '## 训练数据与执行','',
              f"128 份教师响应中，最终接受 {summary['data']['accepted_tasks']} 道题。接受样本按初始学生表现分类：", '']
    lines += [f'- {k}：{v} 题。' for k,v in summary['data']['categories'].items()]
    lines += ['', '拒收原因：'+json.dumps(dict(Counter(x['reason'] for x in summary['data']['rejected'])),ensure_ascii=False)+'。', '',
              '引用未覆盖全部标注支持句，不等于教师答案必然错误；这是本轮对完整证据示范的保守筛选条件。'
              'format_candidate 也只是答案格式相关候选，不能仅凭包含标准答案就断言语义正确。','',
              '两组从相同第 20,600 题 checkpoint 初始化，执行器 LoRA 学习率 1e-5、2 epochs、batch size 2，管理器冻结。'
              '短答案组每题重复两条短答案，联合组每题一条短答案加一条带句子 ID 的证据选择示范，匹配参数更新次数。'
              '两组都在第 1 步实际重载参数与优化器后继续；固定最终 checkpoint，无开发集选模。','',
              '| 配置 | 更新次数 | 训练输入 token（含重复轮次） | 监督输出 token（含 EOS） |',
              '| --- | ---: | ---: | ---: |']
    for arm,c in costs.items():
        lines.append(f"| {names[arm]} | {summary['training'][arm]['updates']} | {c['input_tokens_over_training']:,} | {c['label_tokens_including_eos_over_training']:,} |")
    lines += ['', '更新次数相同不代表训练 token 或计算量相同。上述数量为 tokenizer 实测的训练数据暴露，不是 API 计费。','',
              '## 事实获取与格式修正分开检查','',
              '普通答题始终使用相同提示、4,096 输入 token 上限与 24 输出 token 上限。'
              '下表另外列出用标注定位缺失支持句的 oracle 诊断，以及添加相近长度非支持文档句子的干扰对照。'
              'oracle 诊断不是正常检索系统成绩，不能混入主结果。','',
              '| 配置 | 补齐支持句 EM | 无关片段 EM | 排除原答案含目标后的事实获取候选 | 格式相关获取候选 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for arm in names:
        m=summary['results'][arm];a=m['support_augmented'];d=m['distractor_augmented'];r=records[arm]
        lines.append(f"| {names[arm]} | {a['em_count']}/{a['n']} | {d['em_count']}/{d['n']} | "
                     f"{len(r['fact_access_candidates_excluding_target_in_old_answer'])} | {len(r['format_compatible_access_candidates'])} |")
    lines += ['', '受控候选要求补齐后答对、原视图和无关片段对照均答错、新增 token 差不超过 20%。'
              '再排除原答案已含目标词序列、标点空格等价和 yes/no 前缀相同的情况，以降低格式变化混入事实收益的风险。'
              '删除新增支持句后视图与原视图完全一致，复用原输出，不虚增独立评估次数。'
              '这些条件仍不构成语义因果的充分证明。标点和 yes/no 规则是在逐题复核后补充的分析规则；原始数据分类、训练奖励和模型结果均未改变。'
              '完整逐题结果保存在 evidence_analysis.json。','']
    for arm in ('answer_only','evidence_and_answer'):
        r=records[arm]
        f=sum(x['format_compatible'] for x in r['native_gains'])
        lines.append(f"- {names[arm]}：正常视图新增答对 {len(r['native_gains'])} 题，其中 {f} 题与格式修正相容；"
                     f"同一补齐证据视图下新增答对 {len(r['gains_using_same_augmented_evidence'])} 题。")
    lines += ['', '## 是否扩大管理器 GRPO','',
              f"预设门槛通过：{summary['manager_grpo_promotion_gate_passed']}；本阶段自动晋级管理器 GRPO：{summary['manager_grpo_started']}。",'',
              '门槛要求联合训练的正常视图 EM 相对初始化和短答案组的配对区间下界均大于零，并至少有三个受控事实使用候选。'
              '没有通过时，停留在执行器开发验证阶段，不把新增参数更新或训练内成功当作泛化提升。','',
              '## 成本与保存位置','',
              f"批量续跑墙钟 {summary['elapsed_seconds']/60:.2f} 分钟；使用共享 GPU2，allocator 配额 26 GiB，不能等同于独占 GPU 用时。",'',
              'API 先前单题阶段与批量续跑分开记账，另有诊断请求；缓存命中不是免费生成新数据的证明。'
              '详细请求用量见 teacher_status.json、execution_receipt.json 与 api_diagnostic.json。'
              '最初单题等待进程在训练前主动停止，学生基线与有效教师响应复用；调用方式调整未改变训练/开发任务身份。','',
              '训练和开发按 ID 与规范化问题隔离；该 96 题仍只作为本轮开发证据，不是最终测试。'
              '本轮没有目标 benchmark 强化学习，也未替换原 90,447 题后台训练配置。','',
              f'服务器结果目录：{root}', '',
              '代码：/home/hyx/lm_workspace/SCOPE/experiments/teacher_context_grpo。', '',
              '关键产物：summary.json、receipt.json、audit.json、evidence_analysis.json；每组 sft/checkpoints 下保留最终模型和恢复点。', '']
    if (root/'grpo256/receipt.json').exists():
        lines += ['## 后续独立探索', '', '为落实后续管理器训练验证，另行登记了 grpo256 对照，使用开发 EM 相同且训练 token 更少的短答案监督执行器，与原执行器比较。原联合证据组件晋级门槛仍未通过；该后续实验属于使用开发反馈选择配置后的探索，不能把它描述为自动晋级成功。结果与分阶段增量见 grpo256/report.md。', '']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    compact={a:{k:len(v) for k,v in row.items()} for a,row in records.items()}
    print(json.dumps({'counts':compact,'training_token_costs':costs}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','base','parent'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.root,a.base,a.parent)
