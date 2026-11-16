"""Finish an already running experiment at a user-selected saved milestone.

Keep its pinned source/receipt intact. Stop only the recorded owning processes
after the requested checkpoint commits, then evaluate that checkpoint frozen.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process_identity(pid, root):
    proc = Path(f'/proc/{pid}')
    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    command = (proc/'cmdline').read_bytes().decode().replace('\0', ' ')
    if str(root) not in command or stat[0] == 'Z':
        raise RuntimeError('Process ownership/liveness check failed')
    return {'pid': pid, 'start_ticks': stat[19], 'command': command}


def still_owned(identity, root):
    try:
        current = process_identity(identity['pid'], root)
        return current == identity
    except (FileNotFoundError, ProcessLookupError, RuntimeError):
        return False


def frozen_evaluation(root, output, target):
    # The caller pins PYTHONPATH to the original immutable experiment source.
    import gzip
    import pyarrow.parquet as pq
    import torch
    from scope.checkpoint import load_checkpoint
    from scope.hotpot import split_row
    from scope.io import digest
    from train_speed_policy import TrainingSpeedPolicy
    from experiments.teacher_context_grpo.repair_10k import evaluate, paired, verify_inputs

    receipt = read(root/'receipt.json')
    verify_inputs(root, receipt)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 26*2**30/total))
    policy = TrainingSpeedPolicy(Path(receipt['model']))
    path = root/'checkpoints'/f'step_{target:05d}'
    saved = load_checkpoint(path, policy, receipt_hash=digest(receipt))
    if saved['step'] != target:
        raise RuntimeError('Wrong evaluation checkpoint')
    actor = policy.fingerprint('actor')
    base = policy.frozen_base_signature()
    initial = [json.loads(line) for line in (root/'dev_initial.jsonl').read_text().splitlines()]
    if len(initial) != 512 or any(row['actor_sha256'] != actor for row in initial):
        raise RuntimeError('Actor changed from initial frozen evaluation')
    tasks = [split_row(row) for row in pq.read_table(root/receipt['dev']['file']).to_pylist()]
    if [p['id'] for p, _ in tasks] != receipt['dev']['ids'] or [r['task_id'] for r in initial] != receipt['dev']['ids']:
        raise RuntimeError('Development task identity changed')
    ids = []
    for index in range(1, target+1):
        with gzip.open(root/'updates'/f'step_{index:05d}.json.gz', 'rt', encoding='utf-8') as stream:
            row = json.load(stream)
        if row['step'] != index or not row['actor_unchanged']:
            raise RuntimeError('Training journal boundary/freeze mismatch')
        ids.append(row['task_id'])
    if ids != receipt['train']['ids'][:target] or len(set(ids)) != target:
        raise RuntimeError('Incomplete or duplicated training prefix')
    if sha(root/'updates'/f'step_{target:05d}.json.gz') != saved['extra']['journal_sha256']:
        raise RuntimeError('Checkpoint/journal digest mismatch')
    (output/'dev_initial.jsonl').write_bytes((root/'dev_initial.jsonl').read_bytes())
    final = evaluate(output, policy, tasks, 'final', receipt['rollout_options'], actor)
    if base != policy.frozen_base_signature():
        raise RuntimeError('Frozen evaluation changed backbone')
    summary = {'complete': True, 'completed': target, 'expected': target,
        'original_planned_questions': 10000, 'training_ids_complete_unique': True,
        'initial': read(root/'initial_summary.json'), 'final': final, 'paired': paired(output),
        'counts': saved['extra']['counts'], 'train_seconds': saved['extra']['train_seconds'],
        'teacher_api': read(root/'teacher_api_usage.json'), 'teacher_used_during_eval': False,
        'actor_unchanged': policy.fingerprint('actor') == actor,
        'checkpoint': str(path), 'checkpoint_sha256': sha(path/'state.pt'),
        'receipt_sha256': sha(root/'receipt.json'), 'execution_receipt_sha256': sha(output/'execution_receipt.json'),
        'scope': 'user-requested 3000-question interim source development evaluation; single seed'}
    write(output/'summary.json', summary)
    write(output/'status.json', {'phase': 'complete', 'completed': target, 'expected': target})


def supervise(root, output, target):
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = read(output/'execution_receipt.json')
        if (config['target'] != target or config['original_receipt_sha256'] != sha(root/'receipt.json')
                or config['controller_sha256'] != sha(Path(__file__))):
            raise RuntimeError('Execution amendment identity changed')
        worker, supervisor = config['worker'], config['supervisor']
        last_state = None
        while True:
            latest = read(root/'checkpoints/latest.json')
            if latest['step'] >= target:
                if not (root/'checkpoints'/f'step_{target:05d}'/'metadata.json').exists():
                    raise RuntimeError('Requested milestone checkpoint is missing')
                break
            state = read(root/'status.json')
            if state.get('phase') == 'needs_attention' or not still_owned(worker, root) or not still_owned(supervisor, root):
                raise RuntimeError('Original training stopped before requested milestone; inspect original error logs')
            if state != last_state:
                write(output/'status.json', {**state, 'expected': target, 'original_planned_questions': 10000})
                last_state = state
            time.sleep(.1)
        # Suspend the worker first to avoid running beyond the saved boundary.
        # The immutable milestone, not any later in-memory state, is evaluated.
        if not still_owned(worker, root) or not still_owned(supervisor, root):
            raise RuntimeError('Process identity changed before milestone handoff')
        os.kill(worker['pid'], signal.SIGSTOP)
        os.kill(supervisor['pid'], signal.SIGTERM)
        os.kill(worker['pid'], signal.SIGTERM)
        os.kill(worker['pid'], signal.SIGCONT)
        deadline = time.monotonic()+60
        while still_owned(worker, root) or still_owned(supervisor, root):
            if time.monotonic()>deadline:
                raise RuntimeError('Recorded training processes did not exit after milestone handoff')
            time.sleep(.25)
        write(output/'training_handoff.json', {'checkpoint_step': target, 'own_training_processes_ended': True,
                                              'original_worker': worker, 'original_supervisor': supervisor})
        write(root/'status.json', {'phase': 'training_stopped_for_requested_evaluation', 'completed': target,
                                  'expected': target, 'authoritative_evaluation_status': str(output/'status.json')})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', PYTHONPATH=str(root/'source'),
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        command = ['/home/hyx/lm_workspace/longagent/.venv-verl/bin/python', '-u', str(Path(__file__).resolve()),
                   '--root', str(root), '--target', str(target), '--evaluate']
        with (output/'evaluation.log').open('a') as log:
            process = subprocess.Popen(command, cwd=root/'source', env=env, stdout=log, stderr=subprocess.STDOUT)
            write(output/'evaluation_process.json', {'pid': process.pid, 'gpu': 2, 'command': command})
            code = process.wait()
        if code or not read(output/'summary.json').get('complete'):
            raise RuntimeError(f'Milestone evaluation failed, exit code {code}')
        write(output/'cleanup.json', {'own_training_processes_ended': True, 'own_eval_process_ended': True,
                                     'other_processes_untouched': True})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--target',type=int,default=3000);parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();root=args.root.resolve();target=args.target
    if target<=0 or target>=10000 or target%1000:
        raise ValueError('Requested target must be an existing retained milestone below 10000')
    output=root/f'eval_after_{target}'
    try:
        if args.evaluate:frozen_evaluation(root,output,target)
        else:supervise(root,output,target)
    except Exception as error:
        write(output/('evaluation_error.json' if args.evaluate else 'controller_error.json'),
              {'error':str(error),'traceback':traceback.format_exc()})
        write(output/'status.json',{'phase':'needs_attention','error':str(error),'expected':target})
        raise


if __name__=='__main__':main()
