"""Replay ALFWorld prefixes to measure action effects of short public quotations."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import traceback

import torch

from action_memory_dev import AuditedRunner, DevPolicy, bootstrap
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.alfworld import ALFWorldEnvironment
from scope.checkpoint import load_checkpoint
from scope.config import evaluation_runtime
from scope.contracts import ContractError
from scope.decision_evidence import review
from scope.interaction import execute_action
from scope.io import append_jsonl, atomic_json, digest, file_sha
from scope.policy import HFPolicy
from scope.rollout import EpisodeRunner
from scope.tools import context_view


class Capture(Exception):
    def __init__(self, data):
        self.data = data


class RecordingEnvironment(ALFWorldEnvironment):
    def reset(self, task):
        observation = super().reset(task)
        self.initial_hash = digest(observation)
        self.commands = []
        return observation

    def step_command(self, command):
        observation, done, public = super().step_command(command)
        self.commands.append({'command': command, 'observation_hash': digest(observation), 'done': done})
        return observation, done, public


def exact_quote(atlas, working, goal):
    for slot in review(atlas, working, goal):
        if slot['status'] != 'supported':
            continue
        for ref in slot['evidence_refs']:
            record = atlas.evidence(ref['id'])
            if (record['kind'] == 'observation' and not record.get('terminal')
                    and record['id'] != working.get('observation_id')
                    and slot['min_revision'] <= record['revision'] <= atlas.revision
                    and 0 < ref['end'] - ref['start'] <= 240):
                return {'id': record['id'], 'entity': slot['entity'], 'predicate': slot['predicate'],
                        'span_start': ref['start'], 'text': record['text'][ref['start']:ref['end']],
                        'source_revision': record['revision'], 'support_status': 'source_quote'}
    return None


class ProbeRunner(AuditedRunner):
    def _actor_messages(self, atlas, working, goal, observation, history):
        factual, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        quote = exact_quote(atlas, working, goal)
        view = context_view(atlas, working, goal)
        if not quote or not self.action_menu or not (view['selected_evidence'] or view['fetched_evidence']):
            return factual, view_hash, view_id
        if quote['id'] in (working.get('summary') or {}).get('source_hashes', {}):
            return factual, view_hash, view_id
        inputs = {'existing_retrieval': factual}
        shown = {'existing_retrieval': view}
        suffix = '\nChoose one public action label only.\n' + '\n'.join(
            f"{x['label']}: {x['command']}" for x in self.action_menu)
        for name in ('no_retrieved_excerpt', 'short_slot_quote'):
            modified = copy.deepcopy(working)
            modified.update(selected_ids=[], selected_spans={}, fetched_span=quote if name == 'short_slot_quote' else None)
            clone = type(atlas).restore(atlas.export())
            try:
                messages, _, _ = EpisodeRunner._actor_messages(self, clone, modified, goal, observation, history)
                messages[-1]['content'] += suffix
                inputs[name] = messages
                shown[name] = context_view(clone, modified, goal)
            finally:
                clone.close()
        limit = self.cfg.get('actor_max_input', 6144)
        if any(len(self.policy.encode(msg)) > limit for msg in inputs.values()):
            return factual, view_hash, view_id
        labels = [x['label'] for x in self.action_menu]
        choices = {x['label']: x['command'] for x in self.action_menu}
        cells = {}
        for name, messages in inputs.items():
            action = self.policy.actor_choice(messages, digest([atlas.scope.key, atlas.revision, name]), labels, limit)
            cells[name] = {'command': choices[action.text], 'actor_tokens': len(action.prompt_ids) + 1,
                           'quote': shown[name]['fetched_evidence'],
                           'selected_ids': [r['id'] for r in shown[name]['selected_evidence']]}
        raise Capture({'task_id': atlas.scope.task, 'revision': atlas.revision, 'goal': goal,
                       'initial_observation_sha256': self.env.initial_hash, 'replay': list(self.env.commands),
                       'last_observation_sha256': digest(observation), 'public_state': copy.deepcopy(working['public_state']),
                       'prefix_action_count': len(history), 'menu': self.action_menu, 'cells': cells})


def evaluate_command(env, capture, command):
    observation = env.reset(capture['task_id'])
    if digest(observation) != capture['initial_observation_sha256']:
        raise ContractError('ALFWorld reset changed the replay prefix')
    for item in capture['replay']:
        observation, done, _ = env.step_command(item['command'])
        if digest(observation) != item['observation_hash'] or done != item['done'] or done:
            raise ContractError('ALFWorld command replay diverged')
    if digest(observation) != capture['last_observation_sha256']:
        raise ContractError('ALFWorld replay did not reach the recorded decision boundary')
    result, done, _, status = execute_action(env, env.action_adapter, 'Action: ' + command,
                                             observation, capture['public_state'], 'grounded')
    legal = status['action_status'] == 'accepted'
    target = env.action_adapter.confirmation_target(command, 'task') if legal else None
    confirmed = bool(target and any((fact.entity, fact.predicate, fact.value) == target and fact.status == 'supported'
                                    for fact in env.action_adapter.public_facts({'text': result})))
    return {'legal': legal, 'action_status': status['action_status'], 'sent': status['sent'], 'terminal': done,
            'command_effect_confirmed': confirmed, 'terminal_success': bool(env.metrics['success']),
            'observation_after_sha256': digest(result)}


def run(root, source):
    root.mkdir(parents=True, exist_ok=True)
    while not (source / 'summary.json').exists() or not shared.read(source / 'summary.json')['complete']:
        status = shared.read(source / 'status.json')
        if status['phase'] == 'needs_attention':
            raise ContractError('Action interface comparison needs repair')
        pid = shared.read(source / 'process.json')['pid']
        cmdline = Path('/proc') / str(pid) / 'cmdline'
        if not cmdline.exists() or b'action_memory_dev.py' not in cmdline.read_bytes():
            if not (source / 'summary.json').exists() or not shared.read(source / 'summary.json')['complete']:
                raise ContractError('Action interface comparison exited before producing all results')
        atomic_json(root / 'status.json', {'phase': 'waiting_for_action_study'})
        time.sleep(20)
    manifest = shared.read(source / 'receipt.json')
    arm = source / 'alfworld/public_candidates_recovery'
    records = [json.loads(x) for x in (arm / 'results.jsonl').read_text().splitlines()]
    selected = [r for r in records if any(e.get('retrieved_evidence_in_view') for e in r['history'])]
    selected = sorted(selected, key=lambda r: digest([233, 'probe', r['task_id']]))[:12]
    atomic_json(root / 'receipt.json', {'source_receipt': file_sha(source / 'receipt.json'), 'target_updates': 0,
        'checkpoint': manifest['checkpoint'], 'task_ids': [r['task_id'] for r in selected], 'protocol': 'same reconstructed prefix, one executed action per condition'})
    if not selected:
        atomic_json(root / 'summary.json', {'complete': True, 'paired_prefixes': 0, 'eligible_tasks': 0})
        atomic_json(root / 'status.json', {'phase': 'complete'})
        return
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(min(1., 26 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    policy = DevPolicy(HFPolicy(manifest['model'], gradient_checkpointing=False))
    checkpoint = Path(manifest['checkpoint']['path'])
    if file_sha(checkpoint / 'state.pt') != manifest['checkpoint']['sha256']:
        raise ContractError('Evidence probe checkpoint changed')
    load_checkpoint(checkpoint, policy.policy, restore_random=False)
    before = policy.fingerprint()
    alf = source / 'alfworld_dev_manifest.json'
    endpoint = f'http://127.0.0.1:{shared.port()}'
    config = {'environment': {'kind': 'alfworld_http', 'manifest': str(alf), 'base_url': endpoint},
              'runtime': shared.read(arm / 'receipt.json')['runtime'], 'evaluation': {'seed': 233, 'sequence_length': 1}}
    env = RecordingEnvironment(endpoint, [r['task_id'] for r in selected], alf)
    shared.BRIDGE = source / 'serve_alfworld.py'
    outcomes = []
    with shared.alf_service(config, root):
        env.health()
        for index, record in enumerate(selected):
            task = record['task_id']
            atomic_json(root / 'status.json', {'phase': 'replaying', 'completed': index, 'expected': len(selected)})
            runner = ProbeRunner(policy, env, evaluation_runtime(config), 'dev', domain='alfworld', arm='public_candidates_recovery')
            try:
                runner.run(task, f"eval:{record['sequence_id']}:{task}", greedy=True)
            except Capture as captured:
                item = captured.data
                executed = {}
                for cell in item['cells'].values():
                    command = cell['command']
                    if command not in executed:
                        try:
                            executed[command] = evaluate_command(env, item, command)
                        finally:
                            env.close()
                    cell.update(executed[command])
            else:
                item = {'task_id': task, 'reason': 'no_exact_supported_historical_quote_with_live_retrieval'}
            append_jsonl(root / 'prefixes.jsonl', item)
            outcomes.append(item)
    if policy.fingerprint() != before or not shared.read(root / 'alf_cleanup.json')['own_processes_ended']:
        raise ContractError('Evidence probe changed policy or left an owned service running')
    paired = [r['cells'] for r in outcomes if 'cells' in r]
    summary = {'complete': True, 'paired_prefixes': len(paired), 'eligible_tasks': len(selected), 'comparisons': {}}
    for name in ('existing_retrieval', 'short_slot_quote'):
        if paired:
            base = [r['no_retrieved_excerpt'] for r in paired]
            alt = [r[name] for r in paired]
            summary['comparisons'][name] = {'changed_commands': sum(a['command'] != b['command'] for a, b in zip(base, alt)),
                **{key: bootstrap([x[key] for x in base], [x[key] for x in alt])
                   for key in ('legal', 'command_effect_confirmed', 'terminal_success', 'actor_tokens')}}
    atomic_json(root / 'summary.json', summary)
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.source_run.resolve())
        except Exception as exc:
            atomic_json(args.root / 'error.json', {'error': str(exc), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
