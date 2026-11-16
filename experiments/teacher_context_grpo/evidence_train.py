"""Expanded, source-only teacher evidence distillation with frozen paired diagnosis."""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import random
import threading
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch

from evidence_data import (build_examples, category, diagnostic_views, evidence_messages,
                           parse_proposal)
from experiments.memory_curriculum import full_hotpot_grpo as full
from experiments.memory_curriculum.hotpot_teacher_rewards import TEACHER_MODEL
from scope.checkpoint import load_checkpoint
from scope.contracts import ContractError
from scope.hotpot import answer_from_view, answer_messages, rollout, score, split_row
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from scope.warmup import warmup
from train_speed_policy import TrainingSpeedPolicy


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def status(root, phase, **values):
    atomic_json(root/'status.json', {'phase': phase, **values})


def collect_baseline(policy, tasks, root, split):
    result = []
    before = policy.fingerprint()
    for i, (public, gold) in enumerate(tasks, 1):
        path = root/'baseline'/split/f'{public["id"]}.json'
        if path.exists():
            row = read(path)
        else:
            with policy.task_cache():
                trace, _ = rollout(policy, public, 'validation', full.rng_seed('validation', public['id']),
                                   greedy=True, **full.OPTIONS)
            row = {'task_id': public['id'], 'answer': trace['answer'], 'answer_view': trace['answer_view'],
                   'score': score(trace, gold), 'finish_reason': trace['answer_sample']['termination'],
                   'choices': [x['operation'] for x in trace['actions']],
                   'category': category(trace['answer'], trace['answer_view'], gold),
                   'public_sha256': digest(public), 'initial_parameters': before}
            atomic_json(path, row)
        if row['public_sha256'] != digest(public) or row['initial_parameters'] != before:
            raise ContractError('Baseline evidence does not match current initialization')
        result.append(row)
        status(root, 'baseline_'+split, done=i, n=len(tasks))
    if policy.fingerprint() != before:
        raise ContractError('Baseline evaluation changed parameters')
    return result


def evaluate_views(policy, tasks, fixed, baseline, root, arm):
    before, rows = policy.fingerprint(), []
    for i, ((public, gold), base, bundle) in enumerate(zip(tasks, baseline, fixed, strict=True), 1):
        assert base['task_id'] == public['id'] == bundle['task_id']
        path = root/arm/'diagnostics'/f'{public["id"]}.json'
        if path.exists():
            record = read(path)
            if record['parameters'] != before:
                raise ContractError('Cached evaluation used different parameters')
        else:
            cells, cache = {}, {}
            for condition, view in bundle['views'].items():
                key = digest(view)
                if key not in cache:
                    if len(policy.encode(answer_messages(public['question'], view))) > 4096:
                        cache[key] = {'available': False, 'reason': 'diagnostic_context_exceeds_budget'}
                    else:
                        answer, sample = answer_from_view(policy, public['question'], view, digest([public['id'], key]))
                        cache[key] = {'available': True, 'answer': answer,
                                      **score({'answer': answer, 'answer_view': view}, gold),
                                      'input_tokens': len(sample.prompt_ids),
                                      'output_tokens': len(sample.completion_ids), 'termination': sample.termination}
                cells[condition] = {**cache[key], 'view_sha256': key}
            # Removing only the added support returns exactly the native public view.
            cells['support_masked'] = {**cells['native'], 'reused_identical_native_prompt': True}
            record = {'task_id': public['id'], 'parameters': before, 'cells': cells,
                      'initial_category': base['category'], 'diagnostic_meta': bundle['meta']}
            atomic_json(path, record)
        rows.append(record)
        status(root, 'evaluation', arm=arm, done=i, n=len(tasks))
    if policy.fingerprint() != before:
        raise ContractError('Evaluation changed parameters')
    atomic_jsonl(root/arm/'results.jsonl', rows)
    return rows


def paired(left, right, condition='native', subset=None):
    assert [x['task_id'] for x in left] == [x['task_id'] for x in right]
    diffs = [y['cells'][condition]['em']-x['cells'][condition]['em']
             for x,y in zip(left,right,strict=True)
             if (subset is None or x['task_id'] in subset) and x['cells'][condition]['available']
             and y['cells'][condition]['available']]
    if not diffs:
        return {'n': 0, 'delta': None, 'bootstrap_95': None}
    rng = np.random.default_rng(233)
    draws = rng.choice(diffs, size=(5000,len(diffs)), replace=True).mean(axis=1)
    return {'n': len(diffs), 'delta': float(np.mean(diffs)),
            'wins': sum(x>0 for x in diffs), 'losses': sum(x<0 for x in diffs),
            'bootstrap_95': [float(x) for x in np.quantile(draws,[.025,.975])]}


def metrics(rows):
    out = {}
    for condition in ('native','support_augmented','distractor_augmented','support_masked'):
        cells=[r['cells'][condition] for r in rows if r['cells'][condition]['available']]
        out[condition]={'n':len(cells),'em_count':sum(x['em'] for x in cells),
                        'em':float(np.mean([x['em'] for x in cells])) if cells else None,
                        'f1':float(np.mean([x['f1'] for x in cells])) if cells else None,
                        'input_tokens':sum(x['input_tokens'] for x in cells),
                        'output_tokens':sum(x['output_tokens'] for x in cells)}
    out['by_initial_category']={c:{'n':sum(r['initial_category']==c for r in rows),
                                  'native_em_count':sum(r['cells']['native']['em'] for r in rows if r['initial_category']==c)}
                                for c in sorted({r['initial_category'] for r in rows})}
    out['controlled_fact_candidates'] = [r['task_id'] for r in rows
        if r['initial_category']=='support_missing' and r['diagnostic_meta']['token_matched_20pct']
        and all(r['cells'][c]['available'] for c in ('native','support_augmented','distractor_augmented'))
        and r['cells']['support_augmented']['em']==1 and r['cells']['native']['em']==0
        and r['cells']['distractor_augmented']['em']==0]
    return out


def main(args):
    root, parent, base = args.output, args.parent, args.base
    started = time.monotonic()
    original, parent_receipt = read(base/'train/receipt.json'), read(parent/'receipt.json')
    initial = parent/'initial'
    if file_sha(initial/'state.pt') != parent_receipt['base_checkpoint_sha256']:
        raise ContractError('Initial checkpoint changed')
    if file_sha(base/'train/receipt.json') != parent_receipt['base_receipt_sha256']:
        raise ContractError('Base training receipt changed')
    rows, old_dev, manifest = full.prepare(base/'dataset')
    train_keys = {full.question_key(row['question']) for row in rows}
    all_dev = pq.read_table(base/'dataset/validation-00000-of-00001.parquet').to_pylist()
    dev_keys = {full.question_key(row['question']) for row in all_dev}
    excluded_ids = {p['id'] for p,_ in old_dev}
    fresh_dev = [r for r in all_dev if r['id'] not in excluded_ids and full.question_key(r['question']) not in train_keys]
    fresh_dev.sort(key=lambda r:digest([233,'teacher-evidence-fresh-dev',r['id']]))
    dev = [split_row(r) for r in fresh_dev[:args.dev_n]]
    if len(dev)!=args.dev_n:
        raise ContractError('Insufficient isolated development tasks')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    random.seed(233); np.random.seed(233); torch.manual_seed(233)
    free,total = torch.cuda.mem_get_info()
    if free < 28*2**30:
        raise ContractError('Selected GPU has less than 28 GiB free; leave other jobs intact')
    torch.cuda.set_per_process_memory_fraction(min(1.,26*2**30/total))
    TrainingSpeedPolicy.fast_update = True
    policy = TrainingSpeedPolicy(parent_receipt['model'])
    load_checkpoint(initial,policy,receipt_hash=digest(original))
    initial_manager = policy.fingerprint('manager')
    tasks, over_budget = [], []
    for row in rows[parent_receipt['base_checkpoint_step']+len(parent_receipt['train_ids']):]:
        if row['id'] in parent_receipt['train_ids'] or full.question_key(row['question']) in dev_keys:
            continue
        public,gold=split_row(row)
        if len(policy.encode(evidence_messages(public)))>4096:
            over_budget.append(public['id']); continue
        tasks.append((public,gold))
        if len(tasks)==args.train_n:
            break
    if len(tasks)!=args.train_n:
        raise ContractError('Insufficient source tasks')
    receipt={'initial_checkpoint_sha256':file_sha(initial/'state.pt'),
             'base_receipt_sha256':file_sha(base/'train/receipt.json'),
             'parent_receipt_sha256':file_sha(parent/'receipt.json'),
             'train_ids':[p['id'] for p,_ in tasks],'dev_ids':[p['id'] for p,_ in dev],
             'excluded_old_dev_ids':sorted(excluded_ids),'skipped_source_over_budget':over_budget,
             'teacher':TEACHER_MODEL,'teacher_max_tokens_per_question':192,'teacher_batch_size':4,'normal_answer_budget':24,
             'training':{'epochs':2,'batch_size':2,'lr':1e-5,'max_input':4096,'max_output':192,
                         'arms':['answer_only','evidence_and_answer'],
                         'answer_only_duplicates_short_samples_to_match_optimizer_steps':True,
                         'manager_frozen':True,'checkpoint_selection':'fixed final','minimum_accepted':32},
             'gpu':args.gpu,'source_sha256':{p.name:file_sha(p) for p in (Path(__file__),Path(__file__).with_name('evidence_data.py'),Path(__file__).with_name('evidence_teacher.py'))},
             'diagnostic_protocol':'gold-support augmentation and public unrelated-sentence control; never benchmark score',
             'promotion_gate':'paired native EM lower95 > 0 vs initialization and answer-only, >=3 controlled fact candidates',
             'known_limitations':['fresh relative to prior 48 diagnostic tasks; development evidence only',
                                  'same optimizer steps do not equal completion-token or compute budgets']}
    if (root/'receipt.json').exists() and read(root/'receipt.json')!=receipt:
        raise ContractError('Resume receipt differs')
    atomic_json(root/'receipt.json',receipt)
    assert not set(receipt['train_ids']).intersection(receipt['dev_ids'])
    identity=digest(receipt)
    stop=threading.Event()
    from evidence_teacher import collect
    thread=threading.Thread(target=collect,args=(tasks,root,args.config,stop),daemon=True)
    thread.start()
    try:
        train_baseline=collect_baseline(policy,tasks,root,'train')
        dev_baseline=collect_baseline(policy,dev,root,'dev')
        fixed=[]
        for (public,gold),row in zip(dev,dev_baseline,strict=True):
            views,meta=diagnostic_views(public,gold,row['answer_view'],policy.count)
            fixed.append({'task_id':public['id'],'views':views,'meta':meta})
        atomic_jsonl(root/'fixed_dev_views.jsonl',fixed)
        initial_result=evaluate_views(policy,dev,fixed,dev_baseline,root,'initial')
        status(root,'waiting_teacher')
        while thread.is_alive():
            thread.join(1.)
        if (root/'teacher_error.json').exists():
            raise ContractError('Teacher collection failed; see teacher_error.json')
        examples, labels, rejected=[],Counter(),[]
        for (public,gold),trace in zip(tasks,train_baseline,strict=True):
            response=read(root/'teacher'/f'{public["id"]}.json')['response']
            if response is None:
                rejected.append({'task_id':public['id'],'reason':'teacher_unavailable'});continue
            pair,reason=build_examples(public,gold,trace,response['proposal'])
            if pair and (policy.count(pair[0]['completion'])+1>24 or
                         any(len(policy.encode(x['messages']))>4096 or policy.count(x['completion'])+1>192 for x in pair)):
                pair,reason=[],'student_training_budget_exceeded'
            if not pair:
                rejected.append({'task_id':public['id'],'reason':reason});continue
            examples.append(pair);labels[pair[0]['diagnostic_category']]+=1
        data={'accepted_tasks':len(examples),'categories':dict(labels),'rejected':rejected,
              'teacher_collection':read(root/'teacher_status.json')}
        atomic_json(root/'data_validation.json',data)
        if len(examples)<32:
            raise ContractError('Fewer than 32 validated examples; do not expand training')
        results={'initial':initial_result}; training={}
        for arm in ('answer_only','evidence_and_answer'):
            load_checkpoint(initial,policy,receipt_hash=digest(original))
            optimizer=torch.optim.AdamW(policy.parameters('actor'),lr=1e-5,weight_decay=0.)
            arm_data=[item for short,linked in examples for item in (short,short if arm=='answer_only' else linked)]
            path=root/arm/'examples.jsonl';atomic_jsonl(path,arm_data)
            status(root,'actor_training',arm=arm,examples=len(arm_data))
            kwargs=dict(epochs=2,batch_size=2,allowed_tasks=receipt['train_ids'],role='actor',
                        graph_token_budget=(4096,192),checkpoint_every=100)
            opt={'actor':optimizer}
            run=root/arm/'sft'
            latest=run/'checkpoints/latest.json'
            resume=run/'checkpoints'/read(latest)['path'] if latest.exists() else None
            if resume is None:
                first=warmup(policy,path,opt,run,identity,stop_after=1,**kwargs)
                first_fingerprint=policy.fingerprint()
                saved=load_checkpoint(Path(first['checkpoint']),policy,opt,identity)
                assert saved['step']==1 and policy.fingerprint()==first_fingerprint
                atomic_json(root/arm/'resume_validation.json',{'passed':True,'step':1,'optimizer_states':len(optimizer.state)})
                resume=Path(first['checkpoint'])
            summary=warmup(policy,path,opt,run,identity,resume=resume,**kwargs)
            if not summary['complete'] or summary['filtered_over_budget'] or policy.fingerprint('manager')!=initial_manager:
                raise ContractError('Actor training incomplete or manager changed')
            training[arm]=summary
            results[arm]=evaluate_views(policy,dev,fixed,dev_baseline,root,arm)
        comparison={'joint_vs_initial':paired(results['initial'],results['evidence_and_answer']),
                    'joint_vs_answer_only':paired(results['answer_only'],results['evidence_and_answer']),
                    'answer_only_vs_initial':paired(results['initial'],results['answer_only'])}
        stats={k:metrics(v) for k,v in results.items()}
        gate=all(comparison[k]['bootstrap_95'][0]>0 for k in ('joint_vs_initial','joint_vs_answer_only'))
        gate=gate and len(stats['evidence_and_answer']['controlled_fact_candidates'])>=3
        summary={'complete':True,'data':data,'training':training,'results':stats,'paired':comparison,
                 'manager_grpo_promotion_gate_passed':bool(gate),'manager_grpo_started':False,
                 'elapsed_seconds':time.monotonic()-started,
                 'results_sha256':{arm:file_sha(root/arm/'results.jsonl') for arm in results},
                 'receipt_sha256':file_sha(root/'receipt.json'),
                 'interpretation':'source-only evidence-use development validation; oracle diagnostics separate from native EM'}
        atomic_json(root/'summary.json',summary)
        status(root,'complete',summary_sha256=file_sha(root/'summary.json'))
    finally:
        stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('base','parent','output','config'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--train-n',type=int,default=128)
    parser.add_argument('--dev-n',type=int,default=96)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main(args)
        except Exception:
            atomic_json(args.output/'error.json',{'traceback':traceback.format_exc()})
            raise
