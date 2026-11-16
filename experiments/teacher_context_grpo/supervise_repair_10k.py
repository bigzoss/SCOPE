"""Own one fixed experiment; an intentional step-8 exit tests real resumption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def write(path,value):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PYTHONPATH=str(root/'source'),
                     TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                     CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            python='/home/hyx/lm_workspace/longagent/.venv-verl/bin/python'
            command=[python,'-u',str(root/'source/experiments/teacher_context_grpo/repair_10k.py'),
                     '--root',str(root),'--teacher-config','/home/hyx/.config/evomem/ustc_20261018.json']
            # No retry or automatic restart on actual errors. The second phase
            # is solely the predeclared completed-update resume validation.
            for phase,flags in [('initial_and_8',['--stop-after','8']),('resume_to_10000',['--resume'])]:
                with (root/(phase+'.log')).open('a') as log:
                    process=subprocess.Popen(command+flags,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
                    write(root/'process.json',{'supervisor_pid':os.getpid(),'worker_pid':process.pid,
                          'phase':phase,'gpu':2,'started_at':time.time(),'command':command+flags})
                    code=process.wait()
                if code:raise RuntimeError(f'{phase} worker exited with code {code}; see {phase}.log')
                if phase=='initial_and_8':
                    latest=json.loads((root/'checkpoints/latest.json').read_text())
                    if latest['step']!=8:raise RuntimeError('Intentional step-8 boundary was not saved')
            summary=json.loads((root/'summary.json').read_text())
            if not summary.get('complete') or summary.get('completed')!=10000:
                raise RuntimeError('Worker exit without complete 10000-question summary')
            write(root/'cleanup.json',{'own_worker_ended':True,'other_processes_untouched':True})
        except Exception as error:
            write(root/'supervisor_error.json',{'error':str(error),'traceback':traceback.format_exc()})
            write(root/'status.json',{'phase':'needs_attention','error':str(error)})
            raise


if __name__=='__main__':main()
