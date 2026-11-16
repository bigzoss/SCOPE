"""Persistently supervise the existing 3000-question run, not another GPU job.

Restart only after BOTH recorded processes have exited and a saved error proves
a transient teacher transport failure. Never kill live workers, retry contract
errors, reset API accounting, or change the training/evaluation protocol.
"""
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback


TRANSIENT = (
    'Teacher API HTTP 429', 'Teacher API HTTP 500', 'Teacher API HTTP 502',
    'Teacher API HTTP 503', 'Teacher API HTTP 504', 'Teacher API transport failed',
    'Teacher HTTP 429 persisted beyond bounded cooldown; checkpoint remains recoverable',
)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def owned_alive(pid, root, execution):
    try:
        proc=Path(f'/proc/{pid}')
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        args=(proc/'cmdline').read_bytes().decode().split('\0')
        return state!='Z' and str(execution/'runner.py') in args and str(root) in args
    except (FileNotFoundError,ProcessLookupError):
        return False


def eligible_error(error):
    return error.get('type')=='InfrastructureError' and error.get('error') in TRANSIENT


def verify_execution(root, execution):
    receipt=read(execution/'execution_receipt.json')
    if receipt['target']!=3000 or sha(root/'receipt.json')!=receipt['original_receipt_sha256']:
        raise RuntimeError('Run identity changed; automatic recovery refused')
    for name,expected in receipt['execution_sha256'].items():
        if sha(execution/name)!=expected:raise RuntimeError('Execution source changed; automatic recovery refused')


def gpu_snapshot():
    try:
        result=subprocess.run(['nvidia-smi','--id=2',
            '--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=10)
        if result.returncode:
            return {'available':False,'reason':'nvidia-smi failed'}
        fields=result.stdout.strip().split(',')
        return {'available':True,'index':int(fields[0]),'memory_used_mib':int(fields[1]),
                'memory_free_mib':int(fields[2]),'utilization_percent':int(fields[3])}
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
        return {'available':False,'reason':'GPU telemetry unavailable'}


def hourly_audit_if_due(root, execution, monitoring, snapshot, config, now=None):
    """A persisted hourly timer inside the running server-side supervisor."""
    interval=config.get('hourly_audit_seconds',0)
    if not interval:return False
    now=time.time() if now is None else now
    schedule=monitoring/'hourly_schedule.json'
    previous=read(schedule) if schedule.exists() else None
    if previous is not None and now<previous['next_check_epoch']:return False
    verify_execution(root,execution)
    latest=read(root/'checkpoints/latest.json')
    path=root/'checkpoints'/latest['path']
    if path.name!=f"step_{latest['step']:05d}" or path.resolve().parent!=(root/'checkpoints').resolve():
        raise RuntimeError('Hourly checkpoint path identity failed')
    metadata=read(path/'metadata.json')
    if metadata['step']!=latest['step'] or sha(path/'state.pt')!=metadata['state_sha256']:
        raise RuntimeError('Hourly checkpoint checksum failed')
    state=read(root/'status.json')
    report={'checked_at_epoch':now,'checked_at_beijing':datetime.fromtimestamp(now,timezone(timedelta(hours=8))).isoformat(),
            'state':state,'supervision':snapshot,'gpu2':gpu_snapshot(),'checkpoint_sha256_valid':True,
            'checkpoint_step':latest['step'],'execution_source_hashes_valid':True,
            'teacher_api_usage':read(root/'teacher_api_usage.json')}
    if previous is not None:
        report['seconds_since_previous_check']=now-previous['last_check_epoch']
    reports=monitoring/'hourly_reports';reports.mkdir(exist_ok=True)
    stamp=datetime.fromtimestamp(now,timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    write(reports/(stamp+'.json'),report)
    # Retain the original timer cadence after delays/restarts; do not fabricate
    # observations for missed hours.
    next_due=(previous['next_check_epoch'] if previous is not None else now)+interval
    while next_due<=now:next_due+=interval
    write(schedule,{'last_check_epoch':now,'next_check_epoch':next_due,'interval_seconds':interval,
                   'next_check_beijing':datetime.fromtimestamp(next_due,timezone(timedelta(hours=8))).isoformat(),
                   'latest_report':str(reports/(stamp+'.json'))})
    return True


def audit_complete(root, execution):
    summary=read(root/'summary.json');receipt=read(root/'receipt.json')
    if not summary.get('complete') or summary.get('completed')!=3000 or summary.get('expected')!=3000:
        raise RuntimeError('Incomplete 3000-question summary')
    if not summary.get('training_ids_complete_unique') or not summary.get('actor_unchanged'):
        raise RuntimeError('Training identity/frozen actor audit failed')
    for label,key in [('initial','initial'),('final','final')]:
        metrics=summary[key];path=root/f'dev_{label}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if (metrics['n']!=512 or not metrics.get('parameters_unchanged')
                or [row['task_id'] for row in rows]!=receipt['dev']['ids']
                or len({row['task_id'] for row in rows})!=512 or sha(path)!=metrics['sha256']):
            raise RuntimeError('Frozen evaluation coverage/hash audit failed')
    if not read(root/'resume_validation.json').get('passed'):
        raise RuntimeError('Missing successful resume validation')
    if not read(execution/'cleanup.json').get('own_worker_ended'):
        raise RuntimeError('Worker cleanup not confirmed')
    return {'passed':True,'training_questions':3000,'dev_questions':512,
            'initial_em_count':summary['initial']['em_count'],'final_em_count':summary['final']['em_count'],
            'summary_sha256':sha(root/'summary.json')}


def prepare_recovery(root, execution, monitoring, attempt):
    """Archive an ended attempt before replaying the unsaved suffix."""
    verify_execution(root,execution)
    processes=read(root/'process.json')
    if any(owned_alive(processes[k],root,execution) for k in ('worker_pid','supervisor_pid')):
        raise RuntimeError('A run process is still alive; duplicate recovery refused')
    error=read(root/'worker_error.json')
    if not eligible_error(error):raise RuntimeError('Failure is not eligible for automatic recovery')
    latest=read(root/'checkpoints/latest.json');step=latest['step']
    if type(step) is not int or not 0<=step<=3000:
        raise RuntimeError('Checkpoint lies outside the requested run')
    checkpoint=root/'checkpoints'/f'step_{step:05d}'
    if latest['path']!=checkpoint.name:
        raise RuntimeError('Checkpoint path does not match saved step')
    metadata=read(checkpoint/'metadata.json')
    if metadata['step']!=step or sha(checkpoint/'state.pt')!=metadata['state_sha256']:
        raise RuntimeError('Checkpoint checksum failed')
    history=monitoring/'recoveries'/f'attempt_{attempt:02d}'
    history.mkdir(parents=True,exist_ok=False)
    (history/'uncheckpointed_updates').mkdir()
    archived=[]
    for path in sorted((root/'updates').glob('step_*.json.gz')):
        number=int(path.name.split('_')[1].split('.')[0])
        if number>step:
            archived.append({'name':path.name,'sha256':sha(path)})
            path.rename(history/'uncheckpointed_updates'/path.name)
    for path in [root/name for name in ['status.json','process.json','supervisor_process.json',
                                      'worker_error.json','resume_validation.json']]+[
                 execution/name for name in ['worker.log','supervisor.log','supervisor_error.json','teacher_transport_status.json']]:
        if path.exists():
            prefix='execution_' if path.parent==execution else 'root_'
            path.rename(history/(prefix+path.name))
    event={'attempt':attempt,'resumed_from_step':step,'checkpoint_sha256':metadata['state_sha256'],
           'reason':error['error'],'archived_uncheckpointed_updates':archived,
           'preserved_teacher_api_usage':read(root/'teacher_api_usage.json'),'time':time.time()}
    write(history/'recovery_receipt.json',event)
    write(root/'status.json',{'phase':'resuming','completed':step,'expected':3000,'automatic_recovery':attempt})
    return event


def monitor(root, execution, monitoring):
    with (monitoring/'watchdog.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_execution(root,execution)
        config=read(monitoring/'config.json')
        restarts=len(list((monitoring/'recoveries').glob('attempt_*'))) if (monitoring/'recoveries').exists() else 0
        waiting=None
        while True:
            processes=read(root/'process.json')
            live={k:owned_alive(processes[k],root,execution) for k in ['worker_pid','supervisor_pid']}
            state=read(root/'status.json')
            if state.get('phase')=='complete' and not any(live.values()) and (execution/'cleanup.json').exists():
                verify_execution(root,execution)
                audit=audit_complete(root,execution)
                write(monitoring/'completion_audit.json',audit)
                write(monitoring/'status.json',{'phase':'complete','checked_at':time.time(),'automatic_restarts':restarts,**audit})
                return
            activity=max((p.stat().st_mtime for p in [root/'status.json',execution/'worker.log',
                execution/'teacher_transport_status.json',root/'teacher_api_usage.json'] if p.exists()),default=0)
            snapshot={'phase':state.get('phase'),'completed':state.get('completed'),
                'expected':state.get('expected'),'durable_step':read(root/'checkpoints/latest.json')['step'],
                'checked_at':time.time(),'process_alive':live,'automatic_restarts':restarts,
                'seconds_since_activity':time.time()-activity}
            hourly_audit_if_due(root,execution,monitoring,snapshot,config)
            if any(live.values()):
                waiting=None
                if time.time()-activity>config['stall_seconds']:
                    write(monitoring/'attention.json',{'reason':'No new progress or log activity; live processes retained for diagnosis',**snapshot})
                    snapshot['phase']='needs_attention_live_stall'
                write(monitoring/'status.json',snapshot)
                time.sleep(config['poll_seconds']);continue
            error=read(root/'worker_error.json') if (root/'worker_error.json').exists() else {}
            if not eligible_error(error) or restarts>=config['max_automatic_restarts']:
                reason='Non-transient/unclassified failure' if not eligible_error(error) else 'Automatic restart budget exhausted'
                write(monitoring/'attention.json',{'reason':reason,'actual_error':error,**snapshot})
                write(monitoring/'status.json',{**snapshot,'phase':'needs_attention','reason':reason})
                return
            if waiting is None:
                waiting=time.time()+config['restart_cooldowns_seconds'][restarts]
            write(monitoring/'status.json',{**snapshot,'phase':'recovery_cooldown','retry_at':waiting,'reason':error['error']})
            if time.time()<waiting:
                time.sleep(min(config['poll_seconds'],waiting-time.time()));continue
            # A failed/replaced launch must be reviewed rather than retried in a
            # tight loop. Persist preparation before starting exactly one worker.
            restarts+=1
            prepare_recovery(root,execution,monitoring,restarts)
            with (execution/'supervisor.log').open('a') as log:
                child=subprocess.Popen(['python3',str(execution/'runner.py'),'--root',str(root)],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'supervisor_process.json',{'pid':child.pid,'execution':str(execution),'automatic_recovery':restarts})
            deadline=time.monotonic()+60
            while not (root/'process.json').exists():
                if child.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Recovery launcher failed to publish a worker; no duplicate launch attempted')
                time.sleep(.5)
            waiting=None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    execution=root/'resume_3000_429_20261112';monitoring=execution/'monitoring'
    try:monitor(root,execution,monitoring)
    except Exception as error:
        write(monitoring/'attention.json',{'error':str(error),'traceback':traceback.format_exc(),'time':time.time()})
        write(monitoring/'status.json',{'phase':'needs_attention','reason':str(error),'checked_at':time.time()})
        raise


if __name__=='__main__':main()
