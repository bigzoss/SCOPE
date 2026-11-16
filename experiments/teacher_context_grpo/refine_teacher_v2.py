"""Audit completed source-training prefixes and collect bounded actor demonstrations.

Default is offline audit only. --live-teacher validates the compact manager
protocol on archived prefixes (no policy loss) and proposes source-only actor
labels. --verify-reader runs three frozen views, never training any adapter.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq

from scope.contracts import ContractError, TeacherBudgetExhausted
from scope.hotpot import split_row
from scope.hotpot_teacher_repair import repair_packet, check_proposal
from scope.hotpot_teacher_v2 import compact_packet, messages, validate_proposal, teacher_gate
from scope.hotpot_evidence_use import propose, verify_student_and_build
from scope.io import atomic_json, atomic_jsonl, digest, file_sha


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--live-teacher',action='store_true')
    parser.add_argument('--verify-reader',action='store_true')
    parser.add_argument('--teacher-config',type=Path)
    parser.add_argument('--teacher-cache',type=Path,help='Reuse verified request/response cache without new paid calls')
    parser.add_argument('--max-requests',type=int,default=12)
    parser.add_argument('--protocol-cases',type=int,default=4)
    parser.add_argument('--actor-cases',type=int,default=4)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    if args.verify_reader and not args.live_teacher:parser.error('Reader verification requires teacher proposals')
    if args.live_teacher and not args.teacher_config:parser.error('Teacher config required')
    if min(args.protocol_cases,args.actor_cases)<0:parser.error('Negative case limit')
    root=args.source_run;out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():raise ContractError('Use a fresh diagnostic output directory')
    started=time.time();receipt=read(root/'receipt.json');summary=read(root/'summary.json')
    if not summary.get('complete'):raise ContractError('Audit requires a completed source experiment')
    source_sha=file_sha(root/'receipt.json');trainfile=root/receipt['train']['file']
    if file_sha(trainfile)!=receipt['train']['sha256']:raise ContractError('Source dataset hash mismatch')
    rows=pq.read_table(trainfile).to_pylist();tasks={r['id']:split_row(r) for r in rows}
    trainids=set(receipt['train']['ids']);devids=set(receipt['dev']['ids'])
    if trainids & devids:raise ContractError('Source/dev identity overlap')
    gate_counts=Counter();old_counts=Counter();triage=[];protocol_cases=[];actor_cases=[];allfailed=[]
    for path in sorted((root/'updates').glob('step_*.json.gz')):
        record=json.loads(gzip.decompress(path.read_bytes()))
        if record['task_id'] not in trainids or record['task_id'] in devids:
            raise ContractError('Nontraining record entered teacher data')
        repair=record.get('teacher_repair')
        if not repair:continue
        public,gold=tasks[record['task_id']];anchor=record['primary'][0]['trace']
        gate=teacher_gate(public,gold,anchor,record['local_group']);gate_counts[gate['reason']]+=1
        old_counts[repair['reason']]+=1
        triage.append({'step':record['step'],'task_id':public['id'],'old_reason':repair['reason'],'gate':gate})
        # Frozen-prefix diagnostic: old local group only. Do not invent new-group outcomes.
        if gate['eligible'] and len(protocol_cases)<args.protocol_cases:
            protocol_cases.append((public,anchor,record['local_group']['stage_index']))
        branches=repair.get('branches',{})
        if branches and all(r['score']['em']==0 for runs in branches.values() for r in runs):
            native=branches['student'][0];full=native['score']['support_visible_recall']==1
            allfailed.append({'step':record['step'],'task_id':public['id'],
                'route':'actor_evidence_use' if full else 'memory_evidence_gap',
                'source_update':str(path),'source_update_sha256':file_sha(path),
                'native_view_sha256':digest(native['trace']['answer_view'])})
            if full and len(actor_cases)<args.actor_cases:actor_cases.append((public,gold,native['trace']))
    atomic_jsonl(out/'prefix_triage.jsonl',triage);atomic_jsonl(out/'all_failed_routes.jsonl',allfailed)
    manifest={'schema':'scope-teacher-refinement-v2','source_run':str(root),'source_receipt_sha256':source_sha,
        'source_summary_sha256':file_sha(root/'summary.json'),'source_train_sha256':file_sha(trainfile),
        'mode':'bounded_teacher_and_frozen_reader' if args.live_teacher else 'offline_audit',
        'protocol_task_ids':[p['id'] for p,_,_ in protocol_cases],
        'actor_task_ids':[p['id'] for p,_,_ in actor_cases],
        'source_only':True,'no_parameter_updates':True,'gpu':args.gpu if args.verify_reader else None,
        'max_teacher_http_requests':args.max_requests,'teacher_model':'qwen3.8-flash-next',
        'code_sha256':{str(p):file_sha(p) for p in [Path(__file__),Path('scope/hotpot_teacher_v2.py'),Path('scope/hotpot_evidence_use.py')]}}
    atomic_json(out/'receipt.json',manifest)
    project=Path.cwd().resolve()
    for name,sha in manifest['code_sha256'].items():
        source=Path(name).resolve();dest=out/'source'/source.relative_to(project)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if file_sha(dest)!=sha:raise ContractError('Diagnostic source snapshot mismatch')
    result={'complete':False,'old_interventions':len(triage),'old_reasons':dict(old_counts),
        'frozen_old_prefix_gate_counts':dict(gate_counts),'all_failed':len(allfailed),
        'all_failed_routes':dict(Counter(r['route'] for r in allfailed)),
        'gate_limitation':'Retrospective on old decision points, not new training outcomes',
        'protocol_checks':[],'actor_checks':[]}
    atomic_json(out/'status.json',{'phase':'audited','interventions':len(triage)})
    proposals=[]
    if args.live_teacher:
        from scope.teacher_rate_limit import RateLimitedTeacher
        teacher=RateLimitedTeacher(args.teacher_config,args.teacher_cache or out/'teacher_cache',max_requests=args.max_requests,
            model='qwen3.8-flash-next',allow_unusable=True,
            on_wait=lambda r:atomic_json(out/'status.json',{'phase':'teacher_cooldown',**r}))
        try:
            for public,anchor,stage in protocol_cases:
                original=repair_packet(public,anchor,stage);packet,sources=compact_packet(original)
                conversation=messages(packet);item={'task_id':public['id'],'mode':'archived_prefix_schema_only',
                    'accepted_for_training':False,'attempts':[],'valid':False}
                for attempt in range(2):
                    try:raw=teacher(conversation)
                    except TeacherBudgetExhausted:
                        item['reason']='request_budget_exhausted';break
                    try:
                        normalized=validate_proposal(raw,packet,sources);check_proposal(original,normalized)
                    except ContractError as error:
                        item['attempts'].append({'response':raw,'error':str(error)})
                        conversation += [{'role':'assistant','content':str(raw)[:4096]},
                            {'role':'user','content':json.dumps({'validation_error':str(error),'instruction':'Correct once using the same schema.'})}]
                        continue
                    item.update(valid=True,proposal=normalized,corrected_after_feedback=attempt==1)
                    item['attempts'].append({'response':raw,'error':None});break
                result['protocol_checks'].append(item);atomic_json(out/'summary.json',result)
            for public,gold,trace in actor_cases:
                item,proposal=propose(public,gold,trace,teacher)
                result['actor_checks'].append(item)
                if proposal:proposals.append((public,gold,trace,proposal,item))
                atomic_json(out/'summary.json',result)
        finally:
            result['teacher_usage']=dict(teacher.counts);teacher.close()
            atomic_json(out/'summary.json',result)
    examples=[]
    if args.verify_reader and proposals:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        import torch
        from scope.policy import HFPolicy
        from scope.checkpoint import load_checkpoint
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(min(1.,26*1024**3/torch.cuda.get_device_properties(0).total_memory),0)
        atomic_json(out/'status.json',{'phase':'loading_frozen_reader'})
        policy=HFPolicy(receipt['model'],gradient_checkpointing=False)
        checkpoint=root/'checkpoints/step_03000'
        load_checkpoint(checkpoint,policy,receipt_hash=digest(receipt),restore_random=False)
        result['reader_checkpoint_sha256']=file_sha(checkpoint/'state.pt')
        result['reader_actor_fingerprint']=policy.fingerprint('actor')
        for public,gold,trace,proposal,item in proposals:
            checked,example=verify_student_and_build(policy,public,gold,trace,proposal,item)
            index=next(i for i,r in enumerate(result['actor_checks']) if r['task_id']==public['id'])
            # Exactly the existing actor output budget; no overlong supervision silently admitted.
            if example and len(policy.tokenizer.encode(example['completion'],add_special_tokens=False))+1>24:
                checked.update(accepted=False,reason='teacher_answer_exceeds_actor_budget');example=None
            result['actor_checks'][index]=checked
            if example:examples.append(example)
            atomic_jsonl(out/'actor_examples.jsonl',examples);atomic_json(out/'summary.json',result)
        del policy;torch.cuda.empty_cache()
    if file_sha(root/'receipt.json')!=source_sha:raise ContractError('Source receipt changed')
    atomic_jsonl(out/'actor_examples.jsonl',examples)
    result.update(complete=True,actor_examples=len(examples),elapsed_seconds=time.time()-started,
                  actor_examples_sha256=file_sha(out/'actor_examples.jsonl'),no_parameter_updates=True)
    atomic_json(out/'summary.json',result);atomic_json(out/'status.json',{'phase':'complete'})
    print(json.dumps({k:result[k] for k in ['complete','old_interventions','all_failed_routes','actor_examples','elapsed_seconds']}))


if __name__=='__main__':main()
