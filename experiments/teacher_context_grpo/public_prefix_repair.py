"""Collect offline teacher demonstrations with verified same-prefix student continuations."""
import argparse
import copy
import fcntl
from pathlib import Path
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import DevPolicy
from target_action_adaptation import AdaptationRunner, GroundedScienceWorld, GuardedScienceWorld, ThermalScienceWorld
from scope.checkpoint import load_checkpoint, save_checkpoint
from scope.config import DEFAULT_RUNTIME
from scope.contracts import ContractError, Sample
from scope.io import atomic_json, atomic_jsonl, digest, file_sha
from scope.policy import HFPolicy
from scope.public_action_repair import teacher_messages, validate_proposal, verified_gain
from scope.rollout import EpisodeRunner
from scope.teacher_execution import TeacherActionRejected
from scope.teacher_rate_limit import RateLimitedTeacher
from experiments.memory_curriculum.shared_memory_eval import read


class RepairRunner(AdaptationRunner):
    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        turn = len(self.turn_audits) - 1
        if self.repair is None:
            if len(self.action_menu) >= 2:
                records = [row for row in atlas.all_evidence()
                           if row['kind'] in {'requirement', 'observation'} and not row.get('terminal')]
                facts = [{k: item[k] for k in ('entity', 'predicate', 'value', 'support_status') if k in item}
                         for item in atlas.nodes('Claim')[-16:]]
                packet = {'goal': goal, 'menu': [{k: x[k] for k in ('label', 'command', 'reason')}
                                                for x in self.action_menu],
                          'records': [{'id': row['id'], 'kind': row['kind'], 'revision': row['revision'],
                                       'text': row['text'][:1000]} for row in records[-6:]],
                          'current_observation_id': working['observation_id'], 'public_claims': facts}
                if self.cfg.get('teacher_protocol') == 2:
                    packet['protocol'] = 2
                    packet['decision_slots'] = [list(x) for x in working.get('decision_targets', [])]
                if self.cfg.get('late_checkpoint_only'):
                    pipeline = self.turn_audits[-1]['menu_pipeline']
                    packet['public_experiment_stage'] = {k: pipeline[k] for k in
                        ('focus_stage', 'focus_target', 'offered_focus')}
                    if 'thermal_evidence' in pipeline:
                        trial = pipeline['thermal_evidence']
                        packet['public_experiment_stage']['thermal_evidence'] = trial
                        ids = {r['id'] for r in packet['records']}
                        relevant = {eid for row in trial['samples'] for eid in row['evidence_ids']}
                        packet['records'] += [{k: row[k] if k != 'text' else row[k][:1000]
                                              for k in ('id', 'kind', 'revision', 'text')}
                                              for row in records if row['id'] in relevant - ids]
                self.packets[turn] = {'packet': packet, 'prompt_hash': digest(messages)}
            return messages, view_hash, view_id
        if turn != self.repair['turn']:
            return messages, view_hash, view_id
        if digest(messages) != self.repair['prompt_hash']:
            raise ContractError('Replayed actor prompt did not reach the same public boundary')
        mode = self.repair['mode']
        if mode in {'keep', 'short'}:
            stripped = dict(working, selected_ids=[], selected_spans={}, fetched_span=None)
            clone = type(atlas).restore(atlas.export())
            try:
                messages, view_hash, clone_view_id = EpisodeRunner._actor_messages(
                    self, clone, stripped, goal, observation, history)
                clone_view = clone.view(clone_view_id)
            finally:
                clone.close()
            sources = list(clone_view['selected_ids'])
            if mode == 'short':
                ref = self.repair['quote']
                messages[-1]['content'] += f"\nHistorical public observation {ref['id']}: {ref['quote']}"
                sources = sorted(set(sources + [ref['id']]))
            messages[-1]['content'] += '\nChoose one public action label only.\n' + '\n'.join(
                f"{x['label']}: {x['command']}" for x in self.action_menu)
            if len(self.policy.encode(messages)) > self.cfg.get('actor_max_input', 6144):
                raise ContractError('Paired intervention exceeds the original actor input budget')
            view_id = atlas.save_view(sources, clone_view['summary'])['id']
            if isinstance(self.env, ThermalScienceWorld):
                messages, view_hash, view_id = self._measurements(atlas, messages, view_hash, view_id)
        self.repair['messages'] = messages
        return messages, digest([view_hash, self.action_menu]), view_id

    def execute(self, task_id, repair=None):
        self.repair, self.packets = repair, {}
        choose = self.policy.actor_choice

        def actor_choice(messages, boundary, labels, limit):
            turn = len(self.turn_audits) - 1
            if repair and repair['mode'] == 'teacher' and turn == repair['turn']:
                label = repair['label']
                if label not in labels:
                    raise ContractError('Teacher choice left the replayed public menu')
                token_ids = self.policy.choice_ids(labels)
                return Sample('actor', 'execute', self.policy.encode(messages), [token_ids[labels.index(label)]],
                              [], label, self.policy.versions['actor'], boundary, 'choice',
                              origin='teacher', choice_token_ids=token_ids)
            return choose(messages, boundary, labels, limit)

        with patch.object(self.policy, 'actor_choice', actor_choice):
            result = self.run(task_id, 'teacher-source:' + task_id, greedy=True)
        steps = [e for e in result.events if e['phase'] == 'execute']
        if repair is not None and ('messages' not in repair or repair['turn'] >= len(steps)):
            raise ContractError('Replay did not reach the intervention')
        return result, steps


def select_prefix(outcome, steps, packets):
    if outcome.metrics['success']:
        return None
    for index in sorted(packets, key=lambda i: (
        0 if steps[i]['progress'] == 'terminal' else 1 if steps[i]['action_status'] == 'environment_invalid'
        else 2 if steps[i].get('loop') else 3, i)):
        if steps[index]['progress'] in {'terminal', 'invalid_no_progress', 'legal_no_progress'} or steps[index].get('loop'):
            return index
    return None


def select_late_prefix(outcome, steps, packets):
    """Use native progress only to select an offline training prefix."""
    if outcome.metrics['success']:
        return None
    ready = [i for i in packets if i + 3 < len(steps)
             and steps[i]['public_action_audit']['native_progress_before'] > 0
             and not steps[i]['native_score_improved']]
    if not ready:
        return None
    return max(ready, key=lambda i: (
        steps[i]['public_action_audit']['native_progress_before'],
        steps[i]['action_status'] == 'environment_invalid' or bool(steps[i].get('loop')),
        steps[i]['progress'] in {'invalid_no_progress', 'legal_no_progress'}, -i))


def prefix_identity(steps, turn):
    return [(e['normalized_command'], e['sent'], e['observation_after_id']) for e in steps[:turn]]


def cell(outcome, steps, index):
    selected = steps[index]
    return {'command': selected['normalized_command'], 'action_status': selected['action_status'],
            'failed_on_intervention': selected['progress'] == 'terminal' and not outcome.metrics['success'],
            'metrics': outcome.metrics, 'actor_turns': len(steps), 'tokens': outcome.costs['charged_tokens']}


def repair_task(runner, teacher, task_id):
    factual, steps = runner.execute(task_id)
    late = runner.cfg.get('late_checkpoint_only')
    turn = (select_late_prefix if late else select_prefix)(factual, steps, runner.packets)
    if turn is None:
        return {'task_id': task_id, 'route': 'no_late_failed_prefix' if late else 'no_failed_prefix',
                'baseline_metrics': factual.metrics}
    data = runner.packets[turn]
    packet = data['packet']
    before = prefix_identity(steps, turn)
    control = cell(factual, steps, turn)

    def continuation(mode, **kwargs):
        intervention = {'turn': turn, 'prompt_hash': data['prompt_hash'], 'mode': mode, **kwargs}
        outcome, replay = runner.execute(task_id, intervention)
        if prefix_identity(replay, turn) != before:
            raise ContractError('Public actions or observations diverged before the teacher boundary')
        return cell(outcome, replay, turn), intervention['messages']

    row = {'task_id': task_id, 'turn': turn, 'packet': packet, 'existing_retrieval': control,
           'native_progress_before': steps[turn]['public_action_audit']['native_progress_before']}
    messages = teacher_messages(packet)
    if late:
        messages[0]['content'] += (' Use the current experiment stage to finish the next unmet requirement. '
                                    'If the stage requires an action absent from the offered menu, choose the best '
                                    'available preparatory action or null.')
    try:
        proposal = validate_proposal(teacher(messages), packet)
    except TeacherActionRejected as error:
        if packet.get('protocol') != 2:
            return {**row, 'route': 'teacher_rejected', 'reason': str(error)}
        row['first_rejection'] = str(error)
        messages[0]['content'] += '\nThe previous response was rejected: ' + str(error) + '. Repair the same JSON against the given menu and sources.'
        try:
            proposal = validate_proposal(teacher(messages), packet)
        except TeacherActionRejected as retry_error:
            return {**row, 'route': 'teacher_rejected', 'reason': str(retry_error)}
    row['teacher_proposal'] = proposal
    kept, keep_prompt = continuation('keep')
    row['keep'] = kept
    historical = next((ref for ref in proposal['evidence']
                       if ref['id'] != packet['current_observation_id']
                       and next(x for x in packet['records'] if x['id'] == ref['id'])['kind'] == 'observation'
                       and not any(ref['quote'] in turn_message['content'] for turn_message in keep_prompt)), None)
    if historical:
        short, _ = continuation('short', quote=historical)
        row['short_evidence'] = short
        if short['command'] != kept['command'] and verified_gain(kept, short):
            return {**row, 'route': 'manager_evidence', 'verified_quote': historical}
    if control['command'] != kept['command'] and verified_gain(kept, control):
        return {**row, 'route': 'manager_evidence', 'verified_existing_retrieval': True}
    label = proposal['action_label']
    if label is None or proposal['command'] == control['command']:
        return {**row, 'route': 'no_distinct_teacher_action'}
    corrected, prompts = continuation('teacher', label=label)
    row['teacher_continuation'] = corrected
    if not verified_gain(control, corrected):
        return {**row, 'route': 'teacher_utility_unverified'}
    return {**row, 'route': 'actor_demonstration',
            'demonstration': {'task_id': task_id, 'turn': turn, 'messages': prompts,
                              'completion': label, 'menu_labels': [act['label'] for act in packet['menu']],
                              'command': proposal['command'],
                              'evidence': proposal['evidence'], 'expected_subgoal': proposal.get('expected_subgoal'),
                              'observed_gain': {'strict': corrected['metrics']['success'] - control['metrics']['success'],
                                                'native': corrected['metrics']['r_loose'] - control['metrics']['r_loose']},
                              'positive_outcome_verified': True}}


def verify_sft_update(model, demos, root):
    """One isolated actor step on environment-verified source training labels."""
    if not demos:
        return {'applied': False, 'verified_examples': 0}
    output = root / 'actor_sft_smoke'
    status = output / 'validation.json'
    if status.exists():
        prior = read(status)
        if prior['examples_sha256'] != digest(demos) or file_sha(
                output / 'step_00001/state.pt') != prior['checkpoint_sha256']:
            raise ContractError('Actor verification checkpoint identity changed')
        return prior
    manager = model.fingerprint('manager')
    actor = model.fingerprint('actor')
    samples = []
    for demo in demos:
        labels = demo['menu_labels']
        ids = model.choice_ids(labels)
        samples.append(Sample('actor', 'execute', model.encode(demo['messages']),
                              [ids[labels.index(demo['completion'])]], [], demo['completion'],
                              model.versions['actor'], digest(demo['messages']),
                              origin='teacher', choice_token_ids=ids))
    optimizer = torch.optim.AdamW(model.parameters('actor'), lr=1e-5, weight_decay=0.)
    result = model.sft_update(samples, optimizer, role='actor')
    if result['grad_norm'] <= 0 or model.fingerprint('actor') == actor or model.fingerprint('manager') != manager:
        raise ContractError('Evidence-use update did not change only the actor adapter')
    checkpoint = save_checkpoint(output, 1, model, {'actor': optimizer}, digest(demos))
    validation = {'applied': True, 'verified_examples': len(demos), 'examples_sha256': digest(demos),
                  'checkpoint_sha256': file_sha(checkpoint / 'state.pt'), **result}
    atomic_json(status, validation)
    return validation


def run(root, teacher_config, max_prefixes):
    manifest = read(root / 'receipt.json')
    for file, sha in manifest['source_hashes'].items():
        if file_sha(root / 'source' / file) != sha:
            raise ContractError('Public action repair snapshot changed')
    if file_sha(Path(manifest['checkpoint']['path']) / 'state.pt') != manifest['checkpoint']['sha256']:
        raise ContractError('Source checkpoint changed')
    if manifest['split'] != 'native_train' or {t['id'] for t in manifest['tasks']} & set(manifest['excluded_ids']):
        raise ContractError('Teacher demonstrated on a held-out task')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., manifest.get('gpu_memory_gib', 26) * 2**30 / total))
    model = HFPolicy(manifest['model'], gradient_checkpointing=False)
    load_checkpoint(Path(manifest['checkpoint']['path']), model, restore_random=False)
    before = model.fingerprint()
    teacher = RateLimitedTeacher(teacher_config, root / 'teacher_cache', max_requests=max_prefixes * 4,
                                 model='qwen3.8-flash-next', allow_unusable=True)
    grounded = manifest.get('teacher_protocol') == 2
    env_type = GroundedScienceWorld if grounded else GuardedScienceWorld
    if grounded and manifest.get('thermal_evidence'):
        env_type = ThermalScienceWorld
    env = env_type(manifest['tasks'])
    policy = DevPolicy(model)
    runtime = {**DEFAULT_RUNTIME, 'state_graph': True, 'memory_choice': True, 'shared_memory': True,
               'decision_evidence': True, 'action_mode': 'normalize', 'loop_policy': 'state',
               'max_environment_steps': 30, 'max_turns': 60, 'total_tokens': 850000}
    if grounded:
        runtime['teacher_protocol'] = 2
    if manifest.get('late_checkpoint_only'):
        runtime['late_checkpoint_only'] = True
    runner = RepairRunner(policy, env, runtime, 'train', domain='scienceworld', arm='public_candidates')
    rows = []
    try:
        for task in manifest['tasks']:
            if sum('packet' in row for row in rows) >= max_prefixes:
                break
            target = root / 'cases' / (digest(task['id']) + '.json')
            if target.exists():
                row = read(target)
                if row['task_id'] != task['id']:
                    raise ContractError('Teacher repair resume task differs')
            else:
                row = repair_task(runner, teacher, task['id'])
                atomic_json(target, row)
            rows.append(row)
            atomic_json(root / 'status.json', {'phase': 'evaluating', 'completed_tasks': len(rows),
                                               'prefixes': sum('packet' in v for v in rows)})
        if model.fingerprint() != before:
            raise ContractError('Frozen policy changed in offline teacher verification')
        demos = [row['demonstration'] for row in rows if 'demonstration' in row]
        atomic_jsonl(root / 'actor_sft.jsonl', demos)
        if grounded:
            atomic_jsonl(root / 'manager_positive_evidence.jsonl',
                         [row for row in rows if row['route'] == 'manager_evidence'])
        atomic_jsonl(root / 'verified_prefixes.jsonl', rows)
        sft = (verify_sft_update(model, demos, root) if not manifest.get('collect_only')
               else {'applied': False, 'verified_examples': len(demos), 'collect_only': True})
        summary = {'complete': True, 'actual_tasks': len(rows), 'actual_prefixes': sum('packet' in v for v in rows),
                   'routes': {route: sum(row['route'] == route for row in rows) for route in sorted({r['route'] for r in rows})},
                   'verified_actor_demonstrations': len(demos), 'teacher_usage': dict(teacher.counts),
                   'teacher': teacher.identity, 'parameters_unchanged_during_verification': True,
                   'actor_sft_smoke': sft}
        atomic_json(root / 'summary.json', summary)
        atomic_json(root / 'status.json', {'phase': 'complete', **summary})
    finally:
        teacher.close()
        env.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--teacher-config', type=Path, required=True)
    parser.add_argument('--max-prefixes', type=int, default=4)
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root, args.teacher_config, args.max_prefixes)
        except Exception as error:
            atomic_json(args.root / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
