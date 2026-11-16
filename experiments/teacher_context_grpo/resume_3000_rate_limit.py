"""Audited continuation from an existing checkpoint with a bounded teacher cooldown."""
import argparse
import fcntl
import functools
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    root=args.root.resolve();execution=Path(__file__).resolve().parent
    sys.path.insert(0,str(root/'source'))
    from scope.io import file_sha
    receipt=json.loads((execution/'execution_receipt.json').read_text())
    if file_sha(root/'receipt.json')!=receipt['original_receipt_sha256']:
        raise RuntimeError('Original receipt changed')
    for name,digest in receipt['execution_sha256'].items():
        if file_sha(execution/name)!=digest:raise RuntimeError('Execution source changed')
    if args.worker:
        retry=load_module('scope.teacher_rate_limit',execution/'teacher_rate_limit.py')
        driver=load_module('scope_resumed_driver',execution/'driver.py')
        def on_wait(event):
            write(execution/'teacher_transport_status.json',event)
            print(json.dumps(event),flush=True)
        driver.APITeacher=functools.partial(retry.RateLimitedTeacher,on_wait=on_wait)
        sys.argv=[str(execution/'driver.py'),'--root',str(root),'--teacher-config',
                  '/home/hyx/.config/evomem/ustc_20261018.json','--resume','--target','3000']
        driver.main()
        return
    with (execution/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(root/'source'),
                     TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            command=['/home/hyx/lm_workspace/longagent/.venv-verl/bin/python','-u',str(Path(__file__).resolve()),
                     '--root',str(root),'--worker']
            with (execution/'worker.log').open('a') as log:
                p=subprocess.Popen(command,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
                write(root/'process.json',{'supervisor_pid':os.getpid(),'worker_pid':p.pid,'gpu':2,
                      'phase':'resume_500_to_3000_then_frozen_dev','command':command,'execution':str(execution)})
                code=p.wait()
            if code:raise RuntimeError(f'Resumed worker exited {code}; inspect worker.log')
            summary=json.loads((root/'summary.json').read_text())
            if not summary.get('complete') or summary['completed']!=3000:
                raise RuntimeError('Resumed worker did not complete requested 3000 questions')
            summary.update(execution_receipt_sha256=file_sha(execution/'execution_receipt.json'),
                           original_planned_questions=10000,teacher_used_during_frozen_eval=False)
            write(root/'summary.json',summary);write(execution/'summary.json',summary)
            write(execution/'cleanup.json',{'own_worker_ended':True,'other_processes_untouched':True})
        except Exception as error:
            write(execution/'supervisor_error.json',{'error':str(error),'traceback':traceback.format_exc()})
            write(root/'status.json',{'phase':'needs_attention','expected':3000,'error':str(error),'execution':str(execution)})
            raise


if __name__=='__main__':main()
