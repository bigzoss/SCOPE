"""Fixed, single-seed source-to-target study. No outcome-dependent tuning or selection."""
from __future__ import annotations
import argparse
import collections
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

PROJECT = Path('/home/hyx/lm_workspace/SCOPE')
PYTHON = '/home/hyx/lm_workspace/longagent/.venv-verl/bin/python3'
ALF_PYTHON = '/home/hyx/lm_workspace/longagent/.venv-alfworld/bin/python'
ALF_BRIDGE = Path('/home/hyx/lm_workspace/longagent/outputs/code_revisions/teica-v11-optimized-20261025-v1/scripts/serve_alfworld.py')
SEED = 233
ALF_PORT = 5131
SHOP_PORT = 5132


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def read(path):return json.loads(Path(path).read_text())


def jsonl(path):return [json.loads(line) for line in Path(path).open() if line.strip()]


def stable(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def prepare(root):
    code=root/'code'
    if not code.exists():
        code.mkdir()
        shutil.copytree(PROJECT/'scope',code/'scope',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(PROJECT/'pyproject.toml',code/'pyproject.toml')
    source={str(p.relative_to(code)):sha(p) for p in sorted((code/'scope').glob('*.py'))}
    source['pyproject.toml']=sha(code/'pyproject.toml')
    identity=root/'source_identity.json'
    if identity.exists() and read(identity)!=source:raise RuntimeError('Immutable source snapshot changed')
    write(identity,source)
    inputs=root/'input_configs.json'
    if not inputs.exists():write(inputs,{'source':read(PROJECT/'configs/shopsim.json'),'target':read(PROJECT/'configs/alfworld_http.json')})
    source_cfg=read(inputs)['source']
    target_cfg=read(inputs)['target']
    original=Path(target_cfg['environment']['manifest'])
    original_data=read(original)
    by_type=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in original_data['tasks']:by_type[row['split']][row['task_type']].append(row)
    types=sorted(by_type['valid_unseen'])
    if len(types)!=6:raise RuntimeError('Expected six official ALFWorld task types')
    selected=[];train_ids=[];test_ids=[]
    for typ in types:
        for split,n in [('train',8),('valid_unseen',4)]:
            rows=sorted(by_type[split][typ],key=lambda row:stable([SEED,split,row['task_id']]))[:n]
            if len(rows)!=n:raise RuntimeError('Insufficient tasks in '+split+'/'+typ)
            selected.extend(rows)
            (train_ids if split=='train' else test_ids).extend(row['task_id'] for row in rows)
    # Keep the original valid_seen only as recorded development inventory; never query it.
    selected.extend(row for row in original_data['tasks'] if row['split']=='valid_seen')
    manifest={**original_data,'tasks':selected,'counts':dict(collections.Counter(r['split'] for r in selected)),
        'selection':{'seed':SEED,'training_pool_per_type':8,'evaluation_per_type':4,
                     'original_manifest_sha256':sha(original),'outcome_independent':True}}
    manifest_path=root/'alfworld_manifest.json'
    if manifest_path.exists() and read(manifest_path)!=manifest:raise RuntimeError('Manifest changed')
    write(manifest_path,manifest)
    code_path=str(code)
    if code_path not in sys.path:sys.path.insert(0,code_path)
    from scope.data import SplitRegistry
    from scope.alfworld import ALFWorldRegistry
    source_registry=SplitRegistry(source_cfg['data_root'],SEED)
    target_registry=ALFWorldRegistry(manifest_path,SEED)
    for row in selected:
        if sha(row['gamefile'])!=row['game_sha256']:raise RuntimeError('ALFWorld game file changed')
    source_cfg['gpu']=target_cfg['gpu']=1
    source_cfg['seed']=target_cfg['seed']=SEED
    source_cfg['split_seed']=target_cfg['split_seed']=SEED
    source_cfg['environment']['base_url']=f'http://127.0.0.1:{SHOP_PORT}'
    target_cfg['environment']['base_url']=f'http://127.0.0.1:{ALF_PORT}'
    target_cfg['environment']['manifest']=str(manifest_path)
    for cfg in (source_cfg,target_cfg):
        # Pure task advantages avoid amplifying tiny cost differences in all-failed groups.
        cfg['runtime'].update(invalid_weight=0.,cost_weight=0.)
        cfg['training'].update(role='manager',group_size=4,local_group_size=4,lr=1e-5)
        cfg['evaluation'].update(sequence_length=1,seed=SEED)
    source_cfg['training'].update(updates=12,curriculum=[{'through':12,'length':1}],probe_updates=[3,6,9,12])
    target_cfg['training'].update(method='global',updates=8,curriculum=[{'through':8,'length':1}],probe_updates=[])
    collect_cfg=json.loads(json.dumps(source_cfg));collect_cfg['runtime']['manager_mode']='rule'
    cfg_dir=root/'configs'
    for name,cfg in [('source',source_cfg),('target',target_cfg),('collect',collect_cfg)]:write(cfg_dir/(name+'.json'),cfg)
    protocol={'study':'GPU1 single-seed manager-transfer pilot','seed':SEED,'physical_gpu':1,
        'source_updates':12,'source_group_size':4,'source_local_updates':[3,6,9,12],
        'source_global_queries':48,'source_extra_local_continuations':16,
        'target_updates_per_branch':8,'target_group_size':4,'target_queries_per_branch':32,
        'target_training_pool':target_registry.ids('rl_train'),'target_internal_holdout':target_registry.ids('internal_dev'),
        'target_test_tasks':sorted(test_ids),'target_test_per_type':4,'target_test_type_count':6,
        'target_eval_tasks':24,'source_diagnostic_tasks':source_registry.ids('internal_dev')[:24],
        'warmup_collect_tasks':source_registry.ids('rl_train')[:6],'warmup_max_examples':128,
        'actor':'same frozen ShopSimulator SFT epoch3 actor in every condition; target competence is measured, not assumed',
        'initial_manager':'common source-only checked tool SFT, no GRPO',
        'conditions':['warmup_only','source_grpo','source_grpo_target_adapt','warmup_target_only_grpo'],
        'source_actor_turns':12,'target_actor_turns':30,'actor_max_new_tokens':256,'manager_max_new_tokens':192,
        'reward':'(strict_success + 0.1*loose_reward)/1.1; cost and invalid penalties=0; costs monitored',
        'target_adaptation_method':'global GRPO; public-v1 has no native snapshots',
        'checkpoint_selection':'fixed final update; no target outcome-based tuning',
        'evaluation':'greedy, frozen weights and experience; fresh task facts; all target conditions same IDs/budgets',
        'scope':'24-task stratified pilot, not official full134, not cross-task memory training or full long-horizon generalization proof',
        'source_test_exposure':'source internal holdout used for diagnostics; original dev300/test900 not used',
        'external_model_pretraining_contamination':'not certified',
        'original_target_manifest_sha256':sha(original),'manifest_sha256':sha(manifest_path),
        'source_snapshot_sha256':source,'alf_bridge_sha256':sha(ALF_BRIDGE)}
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:raise RuntimeError('Fixed protocol changed')
    write(root/'protocol.json',protocol)
    return code,source_cfg,target_cfg


def assert_snapshot(root):
    for name,value in read(root/'source_identity.json').items():
        if sha(root/'code'/name)!=value:raise RuntimeError('Source changed during experiment: '+name)


@contextmanager
def alf_service(root):
    for port in (ALF_PORT,SHOP_PORT):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Dedicated port occupied: '+str(port))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(root/'code')}
    bootstrap='from scope.lifecycle import install_parent_guard;import sys,runpy;install_parent_guard(int(sys.argv[1]));sys.argv=sys.argv[2:];runpy.run_path(sys.argv[0],run_name="__main__")'
    args=[ALF_PYTHON,'-c',bootstrap,str(os.getpid()),str(ALF_BRIDGE),'--manifest',str(root/'alfworld_manifest.json'),
          '--port',str(ALF_PORT),'--pool-size','1']
    log=(root/'alf_service.log').open('a')
    process=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(root/'alf_service_process.json',{'pid':process.pid,'port':ALF_PORT,'command':args})
    try:
        for _ in range(90):
            if process.poll() is not None:raise RuntimeError('ALF service initialization failed; inspect alf_service.log')
            try:
                data=json.load(urllib.request.urlopen(f'http://127.0.0.1:{ALF_PORT}/healthz',timeout=2))
                if data.get('ready'):
                    if data.get('manifest_sha256')!=sha(root/'alfworld_manifest.json'):raise RuntimeError('ALF service identity mismatch')
                    break
            except (OSError,ValueError):pass
            time.sleep(1)
        else:raise RuntimeError('ALF service startup timed out')
        # Only a training task is used to check the newly connected public environment path.
        from scope.alfworld import ALFWorldEnvironment
        task=read(root/'protocol.json')['target_training_pool'][0]
        env_client=ALFWorldEnvironment(f'http://127.0.0.1:{ALF_PORT}',[task],root/'alfworld_manifest.json')
        try:
            env_client.health();obs=env_client.reset(task);env_client.step('Action: look')
            write(root/'alf_public_preflight.json',{'passed':True,'task_split':'rl_train','task_id':task,
                'public_observation_nonempty':bool(obs),'no_expert_or_hidden_infos_requested':True})
        finally:env_client.close()
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        log.close()
        write(root/'alf_cleanup.json',{'own_processes_ended':process.poll() is not None,'pid':process.pid,'port':ALF_PORT,
              'shared_5084_and_5088_untouched':True})


def stage(root,name,args,config,output,completion='summary.json'):
    assert_snapshot(root)
    output=Path(output)
    done=output/completion
    if done.exists() and read(done).get('complete'):
        return read(done)
    command=[PYTHON,'-m','scope.cli',*args,'--config',str(config),'--output',str(output)]
    if args[0] in ('train','warmup') and (output/'checkpoints/latest.json').exists():
        command+=['--resume',str(output/'checkpoints'/read(output/'checkpoints/latest.json')['path'])]
    write(root/'status.json',{'phase':name,'output':str(output),'command':command,'started_at':time.time()})
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','PYTHONPATH':str(root/'code'),'TOKENIZERS_PARALLELISM':'false'}
    with (root/(name+'.log')).open('a') as log:
        process=subprocess.Popen(command,cwd=root/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(root/'current_process.json',{'pid':process.pid,'stage':name,'command':command,'physical_gpu':1})
        try:
            if process.wait()!=0:raise RuntimeError(name+' failed; inspect '+str(root/(name+'.log')))
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            raise
    assert_snapshot(root)
    if not done.exists() or not read(done).get('complete'):raise RuntimeError(name+' exited without a complete summary')
    return read(done)


def checkpoint(output):
    output=Path(output);latest=read(output/'checkpoints/latest.json')
    return output/'checkpoints'/latest['path']


def summarize(root):
    from scope.evaluation import paired_comparison
    source_names=['source_initial','source_trained']
    target_names=['target_initial','target_transfer','target_adapted','target_only_adapted']
    conditions={name:read(root/name/'summary.json') for name in source_names+target_names}
    expected=sorted(read(root/'protocol.json')['target_test_tasks'])
    type_for={r['task_id']:r['task_type'] for r in read(root/'alfworld_manifest.json')['tasks']}
    for name,result in conditions.items():
        rows=jsonl(root/name/'results.jsonl')
        if result['n']!=24 or not result['parameters_unchanged_during_eval'] or not result['memory_unchanged'] or not result['frozen_base_parameters_unchanged']:
            raise RuntimeError('Evaluation invariants failed: '+name)
        if sha(root/name/'results.jsonl')!=result['results_sha256']:raise RuntimeError('Result SHA mismatch')
        ids=[r['task_id'] for r in rows]
        if len(set(ids))!=24:raise RuntimeError('Duplicate evaluation IDs')
        if name in target_names and sorted(ids)!=expected:raise RuntimeError('Target task receipt mismatch')
        if name in target_names:
            groups=collections.defaultdict(list)
            for row in rows:groups[type_for[row['task_id']]].append(row)
            result['per_task_type']={kind:{'n':len(group),'success_count':sum(int(r['metrics']['success']) for r in group)} for kind,group in groups.items()}
        result['mean_manager_tokens']=sum(r['costs']['manager_input']+r['costs']['manager_output'] for r in rows)/24
        result['mean_actor_tokens']=sum(r['costs']['actor_input']+r['costs']['actor_output'] for r in rows)/24
    pairs={key:paired_comparison(root/a,root/b) for key,a,b in [
        ('source_grpo_effect','source_initial','source_trained'),('zero_shot_manager_transfer','target_initial','target_transfer'),
        ('target_adaptation_effect','target_transfer','target_adapted'),('source_initialization_effect','target_only_adapted','target_adapted')]}
    training={name:read(root/name/'train_summary.json') for name in ('source_train','target_adapt','target_only_adapt')}
    for name,result in training.items():
        expected_updates=12 if name=='source_train' else 8
        if result['completed_updates']!=expected_updates or not result['complete'] or not result['frozen_role_unchanged']:
            raise RuntimeError('Training incomplete: '+name)
        metrics=jsonl(root/name/'train_metrics.jsonl')
        result['non_task_only_advantage_groups']=sum(bool(r['global']['non_task_only_advantage']) for r in metrics)
        result['global_task_zero_variance_groups']=sum(bool(r['global']['task_zero_variance']) for r in metrics)
        result['wall_seconds']=sum(r['elapsed_wall_seconds'] for r in metrics)
        if result['non_task_only_advantage_groups']:raise RuntimeError('Pure-task reward unexpectedly produced cost-only advantages')
    def tasks(name):
        import torch
        return [torch.load(root/name/'checkpoints'/f'step_{i:05d}'/'state.pt',map_location='cpu',weights_only=False)['extra']['tasks'] for i in range(1,9)]
    first,second=tasks('target_adapt'),tasks('target_only_adapt')
    if first!=second:raise RuntimeError('Adaptation controls used different training task schedules')
    training['matched_target_task_schedule']=first
    summary={'complete':True,'study':'GPU1 single-seed 24-task transfer pilot','physical_gpu':1,'conditions':conditions,
        'paired_comparisons':pairs,'training':training,'protocol_sha256':sha(root/'protocol.json'),
        'target_test_tasks_unchanged':True,'target_checkpoint_selection':'fixed final update, no target tuning',
        'scientific_scope':'limited pilot; no multi-seed or full benchmark claims','created_at':time.time()}
    write(root/'summary.json',summary)
    lines=['# SCOPE GPU1迁移与少量适配对照','',
        '单seed、固定24个ALFWorld unseen任务，每类4个；固定执行器。所有管理器共享源域工具SFT预热。结果不代表官方全部134任务或多seed结论。','',
        '| 条件 | 严格成功 | 成功率 | 平均总token | 非法动作率/轮 | 长度终止率/轮 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,result in conditions.items():
        lines.append(f"| {name} | {result['success_count']}/{result['n']} | {result['strict_success_rate']:.2%} | {result['mean_total_tokens']:.0f} | {result['illegal_action_rate_per_turn']:.2%} | {result['length_termination_rate_per_turn']:.2%} |")
    lines+=['','配对差值为右侧条件减左侧条件，置信区间来自任务单位簇bootstrap，不能替代训练seed不确定性。','',
        '| 比较 | 成功率差值 | 95%CI |','|---|---:|---|']
    for name,result in pairs.items():lines.append(f"| {name} | {result['difference']:+.2%} | [{result['ci95'][0]:+.2%}, {result['ci95'][1]:+.2%}] |")
    lines+=['','训练使用纯任务回报，成本和非法操作仅监测；ALFWorld适配采用全局GRPO，其public-v1接口没有原生快照。',
        '源域使用条件局部GRPO；各任务事实独立，不声称跨任务长期记忆已经实现。',
        '执行器仍为源域SFT epoch3模型，未另外训练ALFWorld执行器；若目标成功率处于地板，不能据此断言记忆管理没有潜在效用。',
        '模型、代码、数据、训练任务和每条件结果SHA保留于protocol、receipt和原始run。训练阶段的额外续跑成本见各train_metrics。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.supervisor.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    code,source,target=prepare(root)
    if args.prepare_only:
        print(json.dumps({'prepared':True,'root':str(root),'target_tasks':24}));return
    if (root/'summary.json').exists() and read(root/'summary.json').get('complete'):raise RuntimeError('Study already completed')
    def cancelled(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,cancelled)
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    start=time.time()
    try:
        with alf_service(root):
            stage(root,'collect_source',['collect','--split','rl_train','--limit','6','--manage-service'],root/'configs/collect.json',root/'public_source')
            from scope.warmup import build_examples
            all_examples=root/'all_manager_examples.jsonl';examples=root/'manager_examples.jsonl'
            if not examples.exists():
                result=build_examples(root/'public_source/public.jsonl',all_examples,read(root/'protocol.json')['warmup_collect_tasks'])
                buckets=collections.defaultdict(list)
                for row in jsonl(all_examples):buckets[json.loads(row['completion'])['operation']].append(row)
                picked=[]
                while len(picked)<128 and any(buckets.values()):
                    for op in sorted(buckets):
                        if buckets[op] and len(picked)<128:picked.append(buckets[op].pop(0))
                temporary=examples.with_name(examples.name+'.tmp')
                with temporary.open('w') as stream:
                    for row in picked:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    stream.flush();os.fsync(stream.fileno())
                write(root/'warmup_data.json',{'builder':result,'used_examples':len(picked),
                    'operation_counts':dict(collections.Counter(json.loads(r['completion'])['operation'] for r in picked)),
                    'examples_sha256':sha(temporary),'source_only':True})
                os.replace(temporary,examples)
            if sha(examples)!=read(root/'warmup_data.json')['examples_sha256']:raise RuntimeError('Warmup corpus changed')
            stage(root,'manager_warmup',['warmup','--examples',str(examples),'--epochs','1','--batch-size','1'],root/'configs/source.json',root/'warmup')
            initial=checkpoint(root/'warmup')
            source['warmstart']=str(initial);write(root/'configs/source_train.json',source)
            stage(root,'source_train',['train','--manage-service'],root/'configs/source_train.json',root/'source_train','train_summary.json')
            trained=checkpoint(root/'source_train')
            for name,cp in [('source_initial',initial),('source_trained',trained)]:
                stage(root,name,['evaluate','--checkpoint',str(cp),'--split','internal_dev','--limit','24','--manage-service'],root/'configs/source.json',root/name)
            for name,cp in [('target_initial',initial),('target_transfer',trained)]:
                stage(root,name,['evaluate','--checkpoint',str(cp),'--split','test'],root/'configs/target.json',root/name)
            for train_name,eval_name,cp in [('target_adapt','target_adapted',trained),('target_only_adapt','target_only_adapted',initial)]:
                cfg=json.loads(json.dumps(target));cfg['warmstart']=str(cp);cfg_path=root/'configs'/(train_name+'.json');write(cfg_path,cfg)
                stage(root,train_name,['train'],cfg_path,root/train_name,'train_summary.json')
                stage(root,eval_name,['evaluate','--checkpoint',str(checkpoint(root/train_name)),'--split','test'],root/'configs/target.json',root/eval_name)
            summarize(root)
        write(root/'cleanup.json',{'own_processes_ended':read(root/'alf_cleanup.json')['own_processes_ended'],
            'physical_gpu':1,'shared_services_untouched':True,'elapsed_wall_seconds':time.time()-start})
        write(root/'status.json',{'phase':'complete','elapsed_wall_seconds':time.time()-start})
    except BaseException as exc:
        write(root/'supervisor_error.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
        write(root/'status.json',{'phase':'needs_attention','error':str(exc)})
        raise


if __name__=='__main__':main()
