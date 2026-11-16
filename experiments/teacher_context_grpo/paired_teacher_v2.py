"""Matched new-teacher vs actor-warmup comparison, with frozen development probes."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scope.contracts import ContractError
from scope.hotpot import split_row
from scope.io import atomic_json, atomic_jsonl, digest, file_sha


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]


def write_rows(path,rows):pq.write_table(pa.Table.from_pylist(rows),path)


def prepare(root,old,n):
    if (root/'receipt.json').exists():raise ContractError('Comparison already prepared')
    old=old.resolve();root=root.resolve();original=read(old/'receipt.json')
    if read(old/'summary.json').get('completed')!=3000:raise ContractError('Expected completed 3000-question source')
    train=pq.read_table(old/original['train']['file']).to_pylist()
    if [r['id'] for r in train]!=original['train']['ids']:raise ContractError('Source order mismatch')
    chosen=train[3000:3000+n];ids={r['id'] for r in chosen}
    ordered=chosen+[r for r in train if r['id'] not in ids]
    if len(chosen)!=n or len(ids)!=n:raise ContractError('Insufficient unique new tasks')
    dev=set(original['dev']['ids']);candidates=[];seen=set()
    # Actual failed student views, ranked by a fixed task hash, not by teacher outcome.
    for path in sorted((old/'updates').glob('step_*.json.gz')):
        record=json.loads(gzip.decompress(path.read_bytes()))
        for primary in record['primary']:
            if primary['score']['em']==0 and primary['score']['support_visible_recall']==1:
                trace=primary['trace'];task=trace['task_id']
                if task in seen:break
                if task in dev or task in ids:raise ContractError('Warmup/train/dev overlap')
                candidates.append({'task_id':task,'trace':trace,'source_update':str(path),'source_sha256':file_sha(path)})
                seen.add(task);break
    candidates.sort(key=lambda r:digest([233,'actor-warmup-v2',r['task_id']]))
    candidates=candidates[:64]
    if len(candidates)!=64:raise ContractError('Need 64 real failed source prefixes')
    root.mkdir(parents=True,exist_ok=True)
    source=root/'source';shutil.copytree(old/'source',source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    project=Path.cwd()
    for folder in ['scope','experiments/memory_curriculum','experiments/teacher_context_grpo']:
        for p in (project/folder).glob('*.py'):
            dest=source/p.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    hashes={str(p.relative_to(source)):file_sha(p) for p in source.rglob('*.py')}
    (root/'data').mkdir();write_rows(root/'data/train.parquet',ordered)
    shutil.copy2(old/original['dev']['file'],root/'data/dev.parquet')
    byid={r['id']:r for r in train}
    for c in candidates:c['row']=byid[c['task_id']]
    with gzip.open(root/'data/warmup_prefixes.json.gz','wt') as f:json.dump(candidates,f)
    manifest={'schema':'teacher-v2-paired-1','seed':233,'questions_per_arm':n,'dev_n':512,
        'source_run':str(old),'source_receipt_sha256':file_sha(old/'receipt.json'),
        'base_checkpoint':str(old/'checkpoints/step_03000'),'base_checkpoint_sha256':file_sha(old/'checkpoints/step_03000/state.pt'),
        'train_ids':[r['id'] for r in chosen],'dev_ids':original['dev']['ids'],
        'warmup_ids':[r['task_id'] for r in candidates],'warmup_prefix_sha256':file_sha(root/'data/warmup_prefixes.json.gz'),
        'source_sha256':hashes,'model':original['model'],'gpu':2,
        'warmup':{'candidate_prefixes':64,'content_only':True,'epochs':2,'lr':1e-5,'batch_size':1,
                  'max_teacher_http_requests':160,'minimum_examples':16,'actor_max_tokens':24},
        'arms':['new_teacher','new_teacher_actor_warmup'],'max_teacher_http_requests_per_arm':200,
        'frozen_dev_after_warmup':True,'selection':'fixed final checkpoint; no dev selection',
        'limits':'Single-seed development comparison; warmup is an additional data/compute intervention, not compute matched'}
    inference_equal=all(file_sha(source/name)==sha for name,sha in original['source_sha256'].items()
        if name.startswith('scope/') and name not in {'scope/warmup.py','scope/teacher_rate_limit.py'})
    manifest['reuse_initial_dev_if_inference_sources_match']=inference_equal
    atomic_json(root/'receipt.json',manifest)
    for arm in manifest['arms']:
        p=root/arm;p.mkdir();(p/'source').symlink_to(source,target_is_directory=True);(p/'data').symlink_to(root/'data',target_is_directory=True)
        receipt={**original,'experiment':arm,'warmstart':manifest['base_checkpoint'],
            'warmstart_sha256':manifest['base_checkpoint_sha256'],'source_sha256':hashes,
            'teacher_protocol':'evidence_v2','teacher_selection':'failed_public_gap',
            'teacher_schedule':'Every local group is eligible for outcome/public-gap gating',
            'max_teacher_http_requests':200,'fixed_questions':n,'comparison_receipt_sha256':file_sha(root/'receipt.json'),
            'selection':'original manifest positions 3001 onward, fixed before outcomes',
            'initial_evaluation_prefix_reused_from':None,
            'train':{**original['train'],'ids':[r['id'] for r in ordered],'sha256':file_sha(root/'data/train.parquet')},
            'dev':{**original['dev'],'sha256':file_sha(root/'data/dev.parquet')}}
        receipt.pop('repair_of_failed_run',None)
        if arm=='new_teacher' and inference_equal:
            # The same frozen manager/actor, evaluator and task order; execute() rechecks parameter hashes.
            shutil.copy2(old/'dev_final.jsonl',p/'dev_initial.jsonl')
            receipt['initial_evaluation_prefix_reused_from']=str(old/'dev_final.jsonl')
        atomic_json(p/'receipt.json',receipt)
    atomic_json(root/'status.json',{'phase':'prepared','questions_per_arm':n})
    print(str(root))


def warmup(root,config):
    import torch
    from scope.teacher_rate_limit import RateLimitedTeacher
    from scope.hotpot_evidence_use import propose, verify_student_and_build
    from scope.checkpoint import load_checkpoint
    from scope.policy import HFPolicy
    from scope.warmup import warmup as fit
    manifest=read(root/'receipt.json');p=root/'actor_warmup';p.mkdir(exist_ok=True)
    if (p/'summary.json').exists() and read(p/'summary.json').get('complete'):return
    if file_sha(root/'data/warmup_prefixes.json.gz')!=manifest['warmup_prefix_sha256']:raise ContractError('Warmup data changed')
    with gzip.open(root/'data/warmup_prefixes.json.gz','rt') as f:candidates=json.load(f)
    audits=lines(p/'teacher_audits.jsonl') if (p/'teacher_audits.jsonl').exists() else []
    teacher=RateLimitedTeacher(config,p/'teacher_cache',max_requests=160,model='qwen3.8-flash-next',allow_unusable=True,
        on_wait=lambda event:atomic_json(p/'status.json',{'phase':'teacher_cooldown',**event}))
    if (p/'teacher_usage.json').exists():teacher.counts.update(read(p/'teacher_usage.json'))
    try:
        for row in candidates[len(audits):]:
            public,gold=split_row(row['row'])
            audit,proposal=propose(public,gold,row['trace'],teacher)
            audits.append(audit);atomic_jsonl(p/'teacher_audits.jsonl',audits)
            atomic_json(p/'teacher_usage.json',dict(teacher.counts))
            atomic_json(p/'status.json',{'phase':'teacher_demonstrations','completed':len(audits),'expected':len(candidates)})
    finally:atomic_json(p/'teacher_usage.json',dict(teacher.counts));teacher.close()
    torch.set_num_threads(4);random.seed(233);np.random.seed(233);torch.manual_seed(233)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(min(1.,26*2**30/torch.cuda.get_device_properties(0).total_memory))
    policy=HFPolicy(manifest['model']);load_checkpoint(Path(manifest['base_checkpoint']),policy,restore_random=False)
    manager=policy.fingerprint('manager');actor=policy.fingerprint('actor');examples=[];checks=[]
    for row,audit in zip(candidates,audits,strict=True):
        if audit['task_id']!=row['task_id']:raise ContractError('Teacher audit order mismatch')
        if audit['reason']!='teacher_label_validated_pending_reader_replay':continue
        public,gold=split_row(row['row'])
        checked,example=verify_student_and_build(policy,public,gold,row['trace'],audit['proposal'],audit)
        checks.append(checked)
        if (example and example['diagnostic_category']=='support_present_content_error'
                and len(policy.tokenizer.encode(example['completion'],add_special_tokens=False))+1<=24):
            examples.append(example)
        atomic_jsonl(p/'reader_checks.jsonl',checks);atomic_jsonl(p/'content_examples.jsonl',examples)
        atomic_json(p/'status.json',{'phase':'frozen_reader_verification','completed':len(checks),'content_examples':len(examples)})
    if len(examples)<manifest['warmup']['minimum_examples']:
        raise ContractError(f'Only {len(examples)} content examples passed; need 16. Do not silently shrink the warmup study.')
    optimizer={'actor':torch.optim.AdamW(policy.parameters('actor'),lr=1e-5,weight_decay=0.)}
    identity=digest([manifest,file_sha(p/'content_examples.jsonl'),'actor-only-warmup'])
    latest=p/'sft/checkpoints/latest.json'
    resume=p/'sft/checkpoints'/read(latest)['path'] if latest.exists() else None
    result=fit(policy,p/'content_examples.jsonl',optimizer,p/'sft',identity,epochs=2,batch_size=1,
        allowed_tasks=manifest['warmup_ids'],graph_token_budget=(4096,24),checkpoint_every=20,role='actor',resume=resume)
    if manager!=policy.fingerprint('manager'):raise ContractError('Warmup changed manager')
    result.update(manager_sha256=manager,actor_before=actor,actor_after=policy.fingerprint('actor'),
        teacher_usage=read(p/'teacher_usage.json'),teacher_prefixes=len(candidates),
        reader_checked=len(checks),format_examples_excluded=sum(x.get('category')=='format_compatible_requires_review' for x in checks),
        content_examples_sha256=file_sha(p/'content_examples.jsonl'))
    atomic_json(p/'summary.json',result)
    target=root/'new_teacher_actor_warmup/receipt.json';r=read(target)
    r.update(warmstart=result['checkpoint'],warmstart_sha256=file_sha(Path(result['checkpoint'])/'state.pt'))
    atomic_json(target,r)


def contrast(a,b,key='em'):
    if [r['task_id'] for r in a]!=[r['task_id'] for r in b]:raise ContractError('Paired evaluation IDs differ')
    d=np.array([y[key]-x[key] for x,y in zip(a,b,strict=True)])
    boot=np.random.default_rng(233).choice(d,(5000,len(d)),replace=True).mean(1)
    return {'delta':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'wins':int((d>0).sum()),'losses':int((d<0).sum())}


def summarize(root):
    manifest=read(root/'receipt.json');B=root/'new_teacher';C=root/'new_teacher_actor_warmup'
    for p in [B,C]:
        summary=read(p/'summary.json')
        if not summary['complete'] or summary['completed']!=manifest['questions_per_arm']:raise ContractError('Incomplete arm')
    cells={'initial':lines(B/'dev_initial.jsonl'),'new_teacher':lines(B/'dev_final.jsonl'),
           'warmup_only':lines(C/'dev_initial.jsonl'),'warmup_plus_new_teacher':lines(C/'dev_final.jsonl')}
    for rows in cells.values():
        if [r['task_id'] for r in rows]!=manifest['dev_ids'] or len(rows)!=len({r['task_id'] for r in rows}):raise ContractError('Dev ID mismatch')
    pairs={'new_flow_training':('initial','new_teacher'),'warmup_alone':('initial','warmup_only'),
           'training_after_warmup':('warmup_only','warmup_plus_new_teacher'),
           'added_warmup_at_final':('new_teacher','warmup_plus_new_teacher')}
    result={'complete':True,'scope':'single-seed 512-task development comparison',
        'cells':{k:{'n':len(v),'em_count':sum(r['em'] for r in v),'em':float(np.mean([r['em'] for r in v])),
                    'f1':float(np.mean([r['f1'] for r in v]))} for k,v in cells.items()},
        'contrasts':{k:{metric:contrast(cells[a],cells[b],metric) for metric in ['em','f1']} for k,(a,b) in pairs.items()},
        'warmup':read(root/'actor_warmup/summary.json'),'arms':{p.name:read(p/'summary.json') for p in [B,C]},
        'limitation':'No matched old-teacher arm; B before/after does not isolate new protocol vs old protocol.'}
    atomic_json(root/'summary.json',result)
    report=['# SCOPE 新教师流程与执行器预热配对结果','',
        '| 条件 | 严格成功 | EM | F1 |','|---|---:|---:|---:|']
    for name,r in result['cells'].items():report.append(f"| {name} | {r['em_count']}/{r['n']} | {r['em']:.2%} | {r['f1']:.4f} |")
    report += ['',json.dumps(result['contrasts'],ensure_ascii=False,indent=2),'',
               '单 seed 开发证据；无匹配旧教师流程组，不能将 B 组前后差值解释为新协议相对旧协议的净收益。']
    (root/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')


def supervise(root,config):
    import fcntl
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    r=read(root/'receipt.json');source=root/'source'
    for name,sha in r['source_sha256'].items():
        if file_sha(source/name)!=sha:raise ContractError('Comparison source changed: '+name)
    if file_sha(Path(r['base_checkpoint'])/'state.pt')!=r['base_checkpoint_sha256']:raise ContractError('Base checkpoint changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(source),CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    tasks=[('actor_warmup',[sys.executable,'-m','experiments.teacher_context_grpo.paired_teacher_v2','warmup','--root',str(root),'--teacher-config',str(config)])]
    for arm in r['arms']:
        tasks.append((arm,[sys.executable,'-u','-m','experiments.teacher_context_grpo.repair_10k','--root',str(root/arm),'--teacher-config',str(config),'--target',str(r['questions_per_arm'])]))
    for name,cmd in tasks:
        folder=root/name;folder.mkdir(exist_ok=True)
        if (folder/'summary.json').exists() and read(folder/'summary.json').get('complete'):continue
        if name!='actor_warmup' and (folder/'checkpoints/latest.json').exists():cmd.append('--resume')
        atomic_json(root/'status.json',{'phase':name,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        for attempt in range(3):
            with (folder/'worker.log').open('a') as log:
                worker=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(root/'processes.json',{'stage':name,'pid':worker.pid,'command':cmd,'attempt':attempt+1})
                code=worker.wait()
            if code==0:break
            error_path=root/'warmup_error.json' if name=='actor_warmup' else folder/'worker_error.json'
            error=read(error_path) if error_path.exists() else {}
            transient=error.get('type')=='InfrastructureError' and any(x in error.get('error','') for x in ['429','transport','502','503','504'])
            if not transient or attempt==2:raise RuntimeError(f'{name} exited {code}; resumable artifacts retained')
            shutil.copy2(error_path,folder/f'transient_failure_{attempt+1}.json')
            latest=folder/'checkpoints/latest.json'
            if name!='actor_warmup' and latest.exists():
                step=int(read(latest)['path'].split('_')[-1]);archive=folder/f'uncommitted_retry_{attempt+1}';archive.mkdir(exist_ok=True)
                for path in (folder/'updates').glob('step_*.json.gz'):
                    if int(path.name.split('_')[1].split('.')[0])>step:shutil.move(str(path),archive/path.name)
                if '--resume' not in cmd:cmd.append('--resume')
            atomic_json(root/'status.json',{'phase':'transient_retry','stage':name,'attempt':attempt+1,'wait_seconds':60})
            time.sleep(60)
    summarize(root)
    atomic_json(root/'cleanup.json',{'own_processes_ended':True,'gpu':2,'external_services_touched':False})
    atomic_json(root/'status.json',{'phase':'complete'})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','warmup','supervise','summarize'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source-run',type=Path)
    p.add_argument('--questions',type=int,default=1000);p.add_argument('--teacher-config',type=Path)
    args=p.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.source_run,args.questions)
        elif args.action=='warmup':warmup(root,args.teacher_config)
        elif args.action=='supervise':supervise(root,args.teacher_config)
        else:summarize(root)
    except Exception as error:
        atomic_json(root/('supervisor_error.json' if args.action=='supervise' else args.action+'_error.json'),
                    {'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        atomic_json(root/'status.json',{'phase':'needs_attention','stage':args.action,'error':str(error)})
        raise


if __name__=='__main__':main()
