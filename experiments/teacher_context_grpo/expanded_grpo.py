"""Exploratory 256-question GRPO contrast after verified teacher actor distillation."""
import argparse
from collections import Counter
import fcntl
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.memory_curriculum import full_hotpot_grpo as full
from experiments.memory_curriculum.hotpot_grpo import step_data, rng_seed
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.contracts import ContractError
from scope.hotpot import normalize, rollout, score, split_row
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from train_speed_policy import TrainingSpeedPolicy


def read(path):
    return json.loads(path.read_text())


def contained(answer, target):
    target=normalize(target)
    return target not in {'yes','no','noanswer',''} and f' {target} ' in f' {normalize(answer)} '


def compact_local(group, gold):
    control=group['controls'][0]
    control_row={'answer':control['trace']['answer'],'score':control['score'],
                 'answer_view':control['trace']['answer_view'],'seed':control['seed']}
    candidates=[]
    for item in group['candidates']:
        run=item['runs'][0]
        candidates.append({'sample':item['sample'],'operation':item['operation'],
                           'seed':run['seed'],'answer':run['trace']['answer'],'score':run['score'],
                           'answer_view':run['trace']['answer_view']})
    winners=[r for r in candidates if r['score']['em']>control['score']['em']]
    return {'boundary':group['boundary'],'stage':group['stage'],'metrics':group['metrics'],
            'continuation_seeds':group['continuation_seeds'],'control':control_row,'candidates':candidates,
            'has_strict_win_vs_noop':bool(winners),
            'format_compatible_win':bool(winners and contained(control_row['answer'],gold['answer'])),
            'support_coverage_win':any(r['score']['support_visible_recall']>control['score']['support_visible_recall'] for r in winners),
            'sampling_tokens':group['actual_sampling_tokens']}


def evaluate(policy,tasks,output,arm):
    fingerprint=policy.fingerprint();rows=[]
    for i,(public,gold) in enumerate(tasks,1):
        path=output/arm/'dev'/f'{public["id"]}.json'
        if path.exists():
            row=read(path)
            if row['parameters']!=fingerprint:raise ContractError('Stale frozen evaluation')
        else:
            with policy.task_cache():
                trace,_=rollout(policy,public,'validation',rng_seed('validation',public['id']),greedy=True,**full.OPTIONS)
            row={'task_id':public['id'],'answer':trace['answer'],'answer_view':trace['answer_view'],
                 **score(trace,gold),'parameters':fingerprint,'choices':[a['operation'] for a in trace['actions']]}
            atomic_json(path,row)
        rows.append(row)
        atomic_json(output/'status.json',{'phase':'evaluation','arm':arm,'done':i,'n':len(tasks)})
    if policy.fingerprint()!=fingerprint:raise ContractError('Evaluation changed parameters')
    atomic_jsonl(output/arm/'dev.jsonl',rows)
    return rows


def main(a):
    started=time.monotonic();out=a.output
    source=read(a.source/'receipt.json');sft_summary=read(a.source/'summary.json')
    assert read(a.source/'audit.json')['passed'] and sft_summary['complete']
    if file_sha(a.source/'summary.json')!=read(a.source/'audit.json')['summary_sha256']:
        raise ContractError('Verified actor results changed')
    initial=a.parent/'initial';original=read(a.base/'train/receipt.json')
    prior=read(a.parent/'receipt.json')
    if file_sha(initial/'state.pt')!=prior['base_checkpoint_sha256']:
        raise ContractError('Initial checkpoint changed')
    actor_checkpoint=Path(sft_summary['training']['answer_only']['checkpoint'])
    rows,_,_=full.prepare(a.base/'dataset')
    indices={r['id']:i+1 for i,r in enumerate(rows)}
    offset=max(indices[x] for x in source['train_ids'])
    validation=pq.read_table(a.base/'dataset/validation-00000-of-00001.parquet').to_pylist()
    val_keys={full.question_key(r['question']) for r in validation}
    dev_map={r['id']:split_row(r) for r in validation}
    dev=[dev_map[k] for k in source['dev_ids']]
    selected=[r for r in rows[offset:] if full.question_key(r['question']) not in val_keys][:a.n]
    tasks=[split_row(r) for r in selected]
    assert len(tasks)==a.n and not {p['id'] for p,_ in tasks}.intersection(source['train_ids'])
    receipt={'source_receipt_sha256':file_sha(a.source/'receipt.json'),
             'source_summary_sha256':file_sha(a.source/'summary.json'),
             'initial_checkpoint_sha256':file_sha(initial/'state.pt'),
             'teacher_actor_checkpoint_sha256':file_sha(actor_checkpoint/'state.pt'),
             'base_receipt_sha256':file_sha(a.base/'train/receipt.json'),
             'train_ids':[p['id'] for p,_ in tasks],'dev_ids':source['dev_ids'],
             'seed_positions':{p['id']:indices[p['id']] for p,_ in tasks},
             'arms':['initial_actor','teacher_actor'],'questions_per_arm':a.n,
             'local_groups_per_arm':a.n//2,'candidate_count':4,'continuation':'greedy',
             'objective':'strict EM; cost coefficient 0; auxiliary format tags never alter rewards',
             'lr':1e-5,'kl_beta':.01,'checkpoint_interval':64,'actor_frozen':True,'teacher_online_calls':0,
             'teacher_role':'source-only actor SFT before this on-policy manager experiment',
             'comparison':'exploratory follow-up after development evidence; not promotion of evidence-selection auxiliary',
             'selection':'short-answer actor uses fewer training tokens and ties joint actor EM; decision uses development results',
             'prior_joint_promotion_gate_passed':sft_summary['manager_grpo_promotion_gate_passed'],
             'checkpoint_numbering':'processed questions in this independent branch; original global run unaffected',
             'source_sha256':file_sha(Path(__file__)),
             'local_builder_sha256':file_sha(a.parent/'run_validation.py'),'gpu':2}
    if (out/'receipt.json').exists() and read(out/'receipt.json')!=receipt:raise ContractError('Resume receipt changed')
    atomic_json(out/'receipt.json',receipt);identity=digest(receipt)
    sys.path.insert(0,str(a.parent));previous=importlib.import_module('run_validation')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    free,total=torch.cuda.mem_get_info()
    if free<28*2**30:raise ContractError('Less than 28 GiB free on selected GPU')
    torch.cuda.set_per_process_memory_fraction(min(1.,26*2**30/total))
    TrainingSpeedPolicy.fast_update=True
    policy=TrainingSpeedPolicy(prior['model']);all_counts={};results={}
    for arm in receipt['arms']:
        optimizer=torch.optim.AdamW(policy.parameters('manager'),lr=1e-5,weight_decay=0.)
        load_checkpoint(initial,policy,{'manager':optimizer},digest(original))
        if arm=='teacher_actor':
            load_checkpoint(actor_checkpoint,policy,receipt_hash=digest(source),roles=['actor'],restore_random=False)
        actor=policy.fingerprint('actor');backbone=policy.frozen_base_signature();counts=Counter();done=0
        latest=out/arm/'checkpoints/latest.json'
        if latest.exists():
            checkpoint=latest.parent/read(latest)['path']
            saved=load_checkpoint(checkpoint,policy,{'manager':optimizer},identity)
            done=saved['step'];counts.update(saved['extra']['counts'])
            if policy.fingerprint('actor')!=actor:raise ContractError('Resume actor differs')
        for i,(public,gold) in enumerate(tasks,1):
            if i<=done:continue
            step=indices[public['id']];torch.manual_seed(rng_seed('update',step));begin=time.monotonic()
            before=policy.versions['manager'];detail={'question_index':i,'task_id':public['id'],'seed_step':step,'version_before':before}
            local=i%2==1
            with policy.task_cache():
                if local:
                    weighted,anchor,group=previous.local_batch(policy,public,gold,step)
                    counts['local_groups']+=bool(group)
                    if group:
                        detail['local']=compact_local(group,gold)
                        counts['local_all_failed']+=group['metrics']['all_failed']
                        counts['local_nonzero_task_advantage']+=not group['metrics']['task_zero_variance']
                        counts['local_strict_win_groups']+=detail['local']['has_strict_win_vs_noop']
                        counts['local_format_compatible_win_groups']+=detail['local']['format_compatible_win']
                        counts['local_support_coverage_win_groups']+=detail['local']['support_coverage_win']
                    else:counts['no_opportunity']+=1
                else:
                    weighted,record=step_data(policy,(public,gold),step,'global',full.WARMUP,full.OPTIONS,0.,
                                              local_candidates=4,trajectory_objective='strict')
                    detail['trajectory']=[{'answer':r['trace']['answer'],'score':r['score']} for r in record['primary']]
                    detail['sampling_tokens']=record['sampling_tokens']
                    counts['trajectory_groups']+=1
                    counts['trajectory_all_failed']+=all(r['score']['em']==0 for r in record['primary'])
            if any(s.origin!='policy' or s.version!=before for s,_ in weighted):
                raise ContractError('Off-policy sample in GRPO loss')
            detail['grpo']=policy.grpo_update(weighted,'manager',optimizer,beta=.01)
            counts['local_optimizer_steps' if local else 'trajectory_optimizer_steps']+=detail['grpo']['optimizer_step']
            if policy.fingerprint('actor')!=actor or policy.frozen_base_signature()!=backbone:
                raise ContractError('Frozen actor or backbone changed')
            detail.update(version_after=policy.versions['manager'],seconds=time.monotonic()-begin)
            assert detail['version_after']-before in (0,1)
            atomic_json(out/arm/'steps'/f'{i:05d}.json',detail)
            if i==1 or i%64==0 or i==len(tasks):
                saved=save_checkpoint(out/arm/'checkpoints',i,policy,{'manager':optimizer},identity,{'counts':dict(counts)})
                if i==1:
                    fingerprint=policy.fingerprint()
                    restored=load_checkpoint(saved,policy,{'manager':optimizer},identity)
                    assert restored['step']==1 and policy.fingerprint()==fingerprint
                    atomic_json(out/arm/'resume_validation.json',{'passed':True,'step':1,'optimizer_states':len(optimizer.state)})
            atomic_json(out/'status.json',{'phase':'training','arm':arm,'done':i,'n':len(tasks),'counts':dict(counts)})
            print(arm,i,len(tasks),dict(counts),flush=True)
        all_counts[arm]=dict(counts)
        results[arm]=evaluate(policy,dev,out,arm)
    before={}
    for arm,old in (('initial_actor','initial'),('teacher_actor','answer_only')):
        saved=[json.loads(line) for line in (a.source/old/'results.jsonl').read_text().splitlines()]
        before[arm]=[{'task_id':r['task_id'],**r['cells']['native']} for r in saved]
    summary={'complete':True,'counts':all_counts,
             'dev':{arm:{'n':len(items),'em_count':sum(x['em'] for x in items),'f1':float(np.mean([x['f1'] for x in items]))}
                    for arm,items in results.items()},
             'paired':{'teacher_vs_initial_after_grpo':previous.paired(results['initial_actor'],results['teacher_actor']),
                       'initial_actor_grpo_change':previous.paired(before['initial_actor'],results['initial_actor']),
                       'teacher_actor_grpo_change':previous.paired(before['teacher_actor'],results['teacher_actor'])},
             'elapsed_seconds':time.monotonic()-started,'receipt_sha256':file_sha(out/'receipt.json'),
             'results_sha256':{arm:file_sha(out/arm/'dev.jsonl') for arm in results},
             'interpretation':'exploratory source development follow-up; teacher SFT and manager GRPO contributions separated'}
    atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('source','base','parent','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--n',type=int,default=256)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(args)
        except Exception:
            atomic_json(args.output/'error.json',{'traceback':traceback.format_exc()});raise
