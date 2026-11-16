"""Freeze the actor and manager while adapting four public action interfaces."""
import argparse
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
import fcntl
import json
from pathlib import Path
import re
import traceback
from unittest.mock import patch

import torch

from action_memory_dev import AuditedRunner, DevBabyAI, DevPolicy, analyze, bootstrap
from experiments.memory_curriculum import native_target_eval as native
from experiments.memory_curriculum import shared_memory_eval as shared
from scope.alfworld import ALFWorldEnvironment
from scope.checkpoint import load_checkpoint
from scope.config import DEFAULT_RUNTIME
from scope.contracts import ContractError
from scope.evaluation import evaluate
from scope.io import atomic_json, digest, file_sha
from scope.policy import HFPolicy
from scope.scienceworld_actions import (ScienceWorldActions, ThermalScienceWorldActions, focus_guard,
                                        public_focus_stage, science_command_guard, thermal_preparation_priority,
                                        thermal_trial)


DOMAINS = ('alfworld', 'scienceworld', 'pddl', 'babyai')
ARMS = ('current_interface', 'adapted_interface')
GUARDED_SCIENCE_ARM = 'guarded_interface'
GROUNDED_SCIENCE_ARM = 'grounded_experiment'
THERMAL_SCIENCE_ARM = 'public_thermal_evidence'
SEED = 233


def words(text):
    return set(re.findall(r'[a-z]+\d*', text.lower())) - {'the', 'to', 'a', 'of', 'in', 'on', 'with', 'and', 'is'}


class RankedActions:
    """Rank current public legal commands without accessing a solution or hidden goal."""

    def public_action_candidates(self, atlas, working, goal, observation, history):
        seen = Counter(x['command'] for x in history[-8:] if x.get('command'))
        stagnant = Counter(x['command'] for x in history[-5:] if x.get('command') and x.get('observation') == observation)
        goal_words = words(goal)
        visible_words = words(self.public_description())
        choices = sorted({self.action_adapter.normalize(x) for x in self.legal_commands() if x.strip()})
        def rank(command):
            tokens = words(command)
            score = 5 * len(tokens & goal_words) + min(3, len(tokens & visible_words)) - 4 * seen[command]
            score -= 8 * stagnant[command]
            return -score, digest([SEED, command])
        ranked = sorted(choices, key=rank)
        # Retain at least one navigation/inspection action when the list is large.
        selected = ranked[:8]
        if len(ranked) > 8:
            exploration = next((x for x in ranked if x.startswith(('go to ', 'open ', 'move ')) and stagnant[x] < 2), None)
            if exploration and exploration not in selected:
                selected[-1] = exploration
        return [{'command': cmd, 'reason': 'public_legal_actions',
                 'evidence_ids': [working['observation_id']]} for cmd in selected]


class DevScienceWorld(native.ScienceWorld):
    def __init__(self, specs):
        from scienceworld import ScienceWorldEnv
        self.env = ScienceWorldEnv(envStepLimit=native.STEP_CAP)
        self.tasks = {r['id']: r for r in specs}


class AdaptedScienceWorld(RankedActions, DevScienceWorld):
    actor_system = ('Complete the experiment. Use the current observation and the exact available commands. '
                    'Use observations to find missing objects and advance the goal. Return one action label.')

    def legal_commands(self):
        # This is the same list exposed by the original check-valid-actions command.
        return self.env.get_valid_action_object_combinations()

    def public_description(self):
        return self.env.look() + '\n' + self.env.inventory()


class GuardedScienceWorld(AdaptedScienceWorld):
    actor_system = (AdaptedScienceWorld.actor_system + ' Focus can end the task immediately. '
                    'Inspect and prepare required objects when a focus target is pending.')

    def reset(self, task_id):
        self.current_goal, self.public_history, self.exact_commands = '', [], {}
        return super().reset(task_id)

    def public_action_candidates(self, atlas, working, goal, observation, history):
        self.current_goal, self.public_history = goal, list(history)
        return super().public_action_candidates(atlas, working, goal, observation, history)

    def public_command_guard(self, command):
        return focus_guard(command, self.current_goal, self.public_history)

    def legal_commands(self):
        by_normalized = {}
        for raw in self.env.get_valid_action_object_combinations():
            command = self.action_adapter.normalize(raw)
            choices = by_normalized.setdefault(command, set())
            choices.add(raw)
        self.exact_commands = {key: next(iter(values)) for key, values in by_normalized.items() if len(values) == 1}
        return ['look around', *(cmd for cmd in self.exact_commands
                                if cmd != 'look around' and not self.public_command_guard(cmd))]

    def step_command(self, command):
        return super().step_command(self.exact_commands.get(command, command))


class GroundedScienceWorld(GuardedScienceWorld):
    """Offer the public experiment's next plausible action with an auditable menu."""

    action_adapter = ScienceWorldActions()
    thermal = False

    def reset(self, task_id):
        self.public_state, self.last_public_observation, self.menu_audit = {}, '', {}
        return super().reset(task_id)

    def public_command_guard(self, command):
        return science_command_guard(command, self.current_goal, self.public_state,
                                     self.public_history, self.last_public_observation, thermal=self.thermal)

    def public_action_candidates(self, atlas, working, goal, observation, history):
        self.current_goal, self.public_history = goal, list(history)
        self.public_state, self.last_public_observation = working['public_state'], observation
        raw = self.env.get_valid_action_object_combinations()
        by_normalized = {}
        for text in raw:
            cmd = self.action_adapter.normalize(text)
            if cmd:
                by_normalized.setdefault(cmd, set()).add(text)
        by_normalized.setdefault('look around', {'look around'})
        self.exact_commands = {}
        removed = Counter()
        raw_focus, allowed_focus = [], []
        for cmd, variants in sorted(by_normalized.items()):
            is_focus = cmd.startswith('focus on ')
            if is_focus:
                raw_focus.append(cmd)
            reason = 'ambiguous_normalization' if len(variants) != 1 else self.public_command_guard(cmd)
            if reason:
                removed[reason] += 1
                continue
            self.exact_commands[cmd] = next(iter(variants))
            if is_focus:
                allowed_focus.append(cmd)
        focus, status = public_focus_stage(goal, history, self.public_state, thermal=self.thermal)
        trial = thermal_trial(goal, history) if self.thermal else None
        if trial:
            evidence = defaultdict(deque)
            for record in atlas.all_evidence():
                if record['kind'] == 'observation' and not record.get('terminal'):
                    evidence[record.get('action_command'), record['text']].append(record['id'])
            by_turn = {}
            for index, past in enumerate(history):
                queue = evidence[past.get('command'), past['observation']]
                if queue:
                    by_turn[index] = queue.popleft()
            for sample in trial['samples']:
                turns = (sample['turn'], sample['phase_source_turn'])
                sample['evidence_ids'] = sorted({by_turn[i] for i in turns if i in by_turn})
        desired = words(focus or goal)
        goal_words = words(goal)
        recent = Counter(row['command'] for row in history[-8:] if row.get('command'))
        visited = {row['command'] for row in history if (row.get('command') or '').startswith('go to ')}
        preparing = bool(re.search(r'\b(?:temperature|melting point|boil|melt|freeze)\b', goal, re.I))
        electrical = bool(re.search(r'\b(?:circuit|conductive|renewable|electric)\b', goal, re.I))
        investigate = lambda c: c.startswith(('look', 'examine ', 'read ', 'inventory'))
        navigate = lambda c: c.startswith(('go to ', 'open '))

        def rank(cmd):
            tokens = words(cmd)
            score = 6 * len(tokens & desired) + 2 * len(tokens & goal_words) - 9 * recent[cmd]
            if cmd.startswith('go to ') and cmd not in visited:
                score += 4
            if cmd.startswith('connect ') and not electrical:
                score -= 12
            if preparing and cmd.startswith(('use thermometer', 'turn on stove', 'turn on blast furnace')):
                score += 12
            if cmd in allowed_focus:
                score += 40
            score += thermal_preparation_priority(cmd, trial, history)
            return -score, digest([SEED, cmd])

        ranked = sorted(self.exact_commands, key=rank)
        categories = [lambda c: c in allowed_focus, navigate, investigate]
        if trial and not allowed_focus:
            categories.insert(1, lambda c: thermal_preparation_priority(c, trial, history) > 0)
        reserved = [next((cmd for cmd in ranked if category(cmd)), None) for category in categories]
        selected = list(dict.fromkeys([cmd for cmd in [*reserved, *ranked] if cmd]))[:8]
        target_refs = {ref['id'] for claim in atlas.nodes('Claim')
                       if (claim['entity'], claim['predicate']) in working.get('decision_targets', [])
                       and claim['support_status'] == 'supported' and not claim.get('conflicted')
                       for ref in claim['evidence_refs']}
        if trial and trial['interval_c']:
            target_refs.update(eid for sample in trial['samples'] for eid in sample['evidence_ids'])
        self.menu_audit = {'native_unique_commands': len(by_normalized), 'eligible_commands': len(ranked),
                           'removed_by_reason': dict(removed), 'focus_stage': status, 'focus_target': focus,
                           'native_focus_count': len(raw_focus), 'eligible_focus': allowed_focus,
                           'offered_focus': [cmd for cmd in selected if cmd in allowed_focus],
                           'eligible_navigation': sum(navigate(cmd) for cmd in ranked),
                           'offered_commands': selected}
        if trial:
            self.menu_audit['thermal_evidence'] = trial
        return [{'command': cmd, 'reason': 'public_experiment_stage',
                 'evidence_ids': sorted({working['observation_id']} | (target_refs if cmd in allowed_focus else set()))}
                for cmd in selected]


class ThermalScienceWorld(GroundedScienceWorld):
    action_adapter = ThermalScienceWorldActions()
    thermal = True


class DevPDDL(native.PDDL):
    def __init__(self, root, specs):
        super().__init__(root)
        self.tasks = {r['id']: r for r in specs}

    def render(self, literal):
        args = [var.name for var in literal.variables]
        name = literal.predicate.name
        template = self.formats.get(name)
        if template:
            try:
                return template.format(*args)
            except IndexError:
                # A predicate with the same name may have another arity in a new domain.
                pass
        return ' '.join([name, *args])


class AdaptedPDDL(RankedActions, DevPDDL):
    actor_system = ('Satisfy every public goal condition. Choose one exact legal action label. '
                    'Use the current satisfied and pending goal conditions to plan the next action.')

    def legal_commands(self):
        return [self.action_adapter.normalize(self.render(x)) for x in self.env.action_space.all_ground_literals(self.state)]

    def public_description(self):
        return self.render_state()

    def goal_checklist(self):
        state = set(self.state.literals)
        return {'satisfied': sorted(self.render(x) for x in self.goal_literals & state),
                'pending': sorted(self.render(x) for x in self.goal_literals - state)}

    def render_state(self):
        return super().render_state() + '\nPublic goal checklist: ' + json.dumps(self.goal_checklist(), ensure_ascii=False)


class AdaptedBabyAI(DevBabyAI):
    actor_system = ('Complete the public BabyAI mission. The observations are relative to where you face. '
                    'Turn toward the target, then move into free space. Pickup, drop and toggle act on the tile directly ahead. '
                    'Distinguish currently visible objects from previously visited views. Choose one action label.')

    def describe(self, observation):
        self.visible_observation = observation
        return super().describe(observation)

    def public_action_candidates(self, atlas, working, goal, observation, history):
        view = self.visible_observation['image']
        center = view.shape[0] // 2
        front_kind, _, front_state = map(int, view[center, -2])
        carried_kind = int(view[center, -1, 0])
        front, held = self.objects[front_kind], self.objects[carried_kind]
        empty = held in {'empty', 'floor', 'unseen'}
        names = ['left', 'right']
        if front in {'empty', 'floor', 'goal', 'unseen'} or (front == 'door' and front_state == 0):
            names.append('forward')
        if empty and front in {'key', 'ball', 'box'}:
            names.append('pickup')
        if not empty and front in {'empty', 'floor', 'unseen'}:
            names.append('drop')
        if front in {'door', 'box', 'unseen'}:
            names.append('toggle')
        recent = history[-3:]
        repeated = {x['command'] for x in recent if x.get('command') and
                    sum(z.get('command') == x['command'] and z['observation'] == observation for z in recent) >= 2}
        choices = [x for x in names if x not in repeated] or names
        return [{'command': name, 'reason': 'public_egocentric_preconditions',
                 'evidence_ids': [working['observation_id']]} for name in choices]


class AdaptationRunner(AuditedRunner):
    def _actor_messages(self, atlas, working, goal, observation, history):
        messages, view_hash, view_id = super()._actor_messages(atlas, working, goal, observation, history)
        self.turn_audits[-1]['native_progress_before'] = float(self.env.metrics['r_loose'])
        if isinstance(self.env, GroundedScienceWorld):
            self.turn_audits[-1]['menu_pipeline'] = self.env.menu_audit
            return self._measurements(atlas, messages, view_hash, view_id)
        return messages, view_hash, view_id

    def _measurements(self, atlas, messages, view_hash, view_id):
        trial = self.env.menu_audit.get('thermal_evidence')
        if not trial:
            return messages, view_hash, view_id
        info = {'subject': trial['subject'], 'samples': [{k: row[k] for k in ('phase', 'temperature_c', 'evidence_ids')}
                for row in trial['samples']], 'interval_c': trial['interval_c'], 'focus_stage': self.env.menu_audit['focus_stage']}
        prior = messages[-1]['content']
        messages[-1]['content'] += '\nPublic experiment measurements: ' + json.dumps(info, ensure_ascii=False)
        if len(self.policy.encode(messages)) > self.cfg.get('actor_max_input', 6144):
            messages[-1]['content'] = prior
            return messages, view_hash, view_id
        view = atlas.view(view_id)
        refs = {eid for sample in trial['samples'] for eid in sample['evidence_ids']}
        saved = atlas.save_view(sorted(set(view['selected_ids']) | refs), view['summary'])
        return messages, digest(messages), saved['id']

    def run(self, *args, **kwargs):
        outcome = super().run(*args, **kwargs)
        steps = [event for event in outcome.events if event['phase'] == 'execute']
        for index, step in enumerate(steps):
            following = steps[index + 1]['public_action_audit']['native_progress_before'] if index + 1 < len(steps) else outcome.metrics['r_loose']
            step['native_score_improved'] = following > step['public_action_audit']['native_progress_before']
            confirmation = step.get('public_confirmation') or {}
            step['new_public_action_confirmation'] = bool(step.get('action_receipt_id') and
                confirmation.get('status') == 'confirmed' and confirmation.get('attempt_id') == step['action_receipt_id'])
        return outcome


def create_env(root, domain, arm, specs, config):
    adapted = arm == 'adapted_interface'
    if domain == 'alfworld':
        endpoint = f'http://127.0.0.1:{shared.port()}'
        path = root / 'alfworld_manifest.json'
        config['environment'].update(base_url=endpoint, manifest=str(path))
        env = ALFWorldEnvironment(endpoint, [r['id'] for r in specs], path)
        return env, shared.alf_service
    if domain == 'scienceworld':
        if arm == THERMAL_SCIENCE_ARM:
            return ThermalScienceWorld(specs), None
        if arm == GROUNDED_SCIENCE_ARM:
            return GroundedScienceWorld(specs), None
        if arm == GUARDED_SCIENCE_ARM:
            return GuardedScienceWorld(specs), None
        return (AdaptedScienceWorld if adapted else DevScienceWorld)(specs), None
    if domain == 'pddl':
        return (AdaptedPDDL if adapted else DevPDDL)(root, specs), None
    return (AdaptedBabyAI if adapted else DevBabyAI)(root, specs), None


def run(root, domains):
    manifest = shared.read(root / 'receipt.json')
    for file, sha in manifest['source_hashes'].items():
        if file_sha(root / 'source' / file) != sha:
            raise ContractError('Adaptation code snapshot changed')
    for file, sha in manifest['assets'].items():
        if file_sha(file) != sha:
            raise ContractError('Target dataset asset changed')
    checkpoint = Path(manifest['checkpoint']['path'])
    if file_sha(checkpoint / 'state.pt') != manifest['checkpoint']['sha256']:
        raise ContractError('Adaptation checkpoint changed')
    shared.BRIDGE = root / 'serve_alfworld.py'
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    memory_gib = manifest.get('memory_gib', 26)
    torch.cuda.set_per_process_memory_fraction(min(1., memory_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inner = HFPolicy(manifest['model'], gradient_checkpointing=False)
    load_checkpoint(checkpoint, inner, restore_random=False)
    policy = DevPolicy(inner)
    weights = policy.fingerprint()
    report = {}
    for domain in domains:
        specs = manifest['tasks'][domain]
        ids = [r['id'] for r in specs]
        registry = shared.FixedTasks(ids)
        cells, rows = {}, {}
        arms = manifest.get('arms_by_domain', {}).get(domain, ARMS)
        for arm in arms:
            output = root / domain / arm
            output.mkdir(parents=True, exist_ok=True)
            cap = 50 if domain == 'alfworld' else 30
            runtime = {**DEFAULT_RUNTIME, 'state_graph': True, 'memory_choice': True, 'shared_memory': True,
                       'decision_evidence': True, 'action_mode': 'grounded' if domain == 'alfworld' else 'normalize',
                       'loop_policy': 'state', 'max_environment_steps': cap, 'max_turns': 2 * cap,
                       'total_tokens': 850000, 'decision_recovery': False}
            config = {'environment': {'kind': 'alfworld_http' if domain == 'alfworld' else domain},
                      'runtime': runtime, 'evaluation': {'seed': SEED, 'sequence_length': 1}}
            receipt = {'domain': domain, 'arm': arm, 'task_ids': ids, 'runtime': runtime, 'seed': SEED,
                       'weights': weights, 'manifest_sha256': file_sha(root / 'receipt.json'),
                       'target_weight_updates': 0, 'teacher_calls': 0}
            path = output / 'receipt.json'
            if path.exists() and shared.read(path) != receipt:
                raise ContractError('Adaptation resumption identity differs')
            atomic_json(path, receipt)
            atomic_json(root / 'status.json', {'phase': 'evaluating', 'domain': domain, 'arm': arm, 'expected': len(ids)})
            env, service = create_env(root, domain, arm, specs, config)
            try:
                with (service(config, output) if service else nullcontext()):
                    try:
                        if service:
                            env.health()
                        mode = 'current_interface' if arm == 'current_interface' else 'public_candidates'
                        with patch('scope.evaluation.EpisodeRunner', lambda *a, **kw: AdaptationRunner(*a, domain=domain, arm=mode, **kw)):
                            summary = evaluate(policy, env, registry, config, output,
                                               'test' if manifest.get('split') == 'test' else 'dev')
                    finally:
                        env.close()
            finally:
                if isinstance(env, native.ScienceWorld):
                    env.shutdown()
            if policy.fingerprint() != weights or not summary['complete'] or summary['n'] != len(ids):
                raise ContractError('Target parameters or task count changed')
            records, diagnostic = analyze(output)
            if {r['task_id'] for r in records} != set(ids) or len(records) != len(ids):
                raise ContractError('Target result IDs differ from the manifest')
            if service and not shared.read(output / 'alf_cleanup.json')['own_processes_ended']:
                raise ContractError('Owned ALFWorld service did not exit')
            rows[arm] = {r['task_id']: r for r in records}
            steps = [e for r in records for e in r['history'] if e['phase'] == 'execute']
            diagnostic['actions'].update(native_score_improvements=sum(e['native_score_improved'] for e in steps),
                newly_confirmed_action_effects=sum(e['new_public_action_confirmation'] for e in steps),
                turns_using_public_action_menu=sum(bool(e.get('public_action_audit', {}).get('selected_label')) for e in steps),
                public_state_loops=sum(bool(e.get('loop')) for e in steps),
                no_progress=sum(e.get('progress') in {'legal_no_progress', 'invalid_no_progress'} for e in steps))
            if arm in {GROUNDED_SCIENCE_ARM, THERMAL_SCIENCE_ARM}:
                pipeline = [e['public_action_audit']['menu_pipeline'] for e in steps]
                diagnostic['action_menu_pipeline'] = {
                    'native_focus_turns': sum(item['native_focus_count'] > 0 for item in pipeline),
                    'eligible_focus_turns': sum(bool(item['eligible_focus']) for item in pipeline),
                    'offered_focus_turns': sum(bool(item['offered_focus']) for item in pipeline),
                    'sent_focus_turns': sum(bool(e['sent']) and (e.get('normalized_command') or '').startswith('focus on ')
                                            for e in steps),
                    'removed_by_reason': dict(sum((Counter(item['removed_by_reason']) for item in pipeline), Counter()))}
            failed_early = [r for r in records if r['termination_reason'] == 'environment_done'
                            and not r['metrics']['success']]
            diagnostic['actions']['terminal_failures'] = len(failed_early)
            diagnostic['actions']['terminal_focus_failures'] = sum(
                str(next(e for e in reversed(r['history']) if e['phase'] == 'execute').get('command') or '')
                .startswith('focus on ') for r in failed_early)
            cells[arm] = {**diagnostic, **{k: summary[k] for k in ('n', 'success_count', 'strict_success_rate',
                'mean_loose_reward', 'illegal_action_rate_per_turn', 'mean_total_tokens', 'results_sha256')}}
        baseline, alternative = rows[arms[0]], rows[arms[-1]]
        paired = {metric: bootstrap([baseline[k]['metrics'][metric] for k in ids],
                                    [alternative[k]['metrics'][metric] for k in ids])
                  for metric in ('success', 'r_loose')}
        additional = {}
        if GUARDED_SCIENCE_ARM in rows and 'adapted_interface' in rows:
            old_menu = rows['adapted_interface']
            additional['guarded_vs_previous_menu'] = {
                metric: bootstrap([old_menu[k]['metrics'][metric] for k in ids],
                                  [alternative[k]['metrics'][metric] for k in ids])
                for metric in ('success', 'r_loose')}
        report[domain] = {'cells': cells, 'paired': paired, 'additional_contrasts': additional}
        atomic_json(root / 'summary.json', {'complete': len(report) == len(domains), 'domains': report,
            'target_weight_updates': 0, 'teacher_calls': 0, 'parameters_unchanged': True})
    atomic_json(root / 'status.json', {'phase': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    args = parser.parse_args()
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.root.resolve(), args.domains)
        except Exception as error:
            atomic_json(args.root / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
            atomic_json(args.root / 'status.json', {'phase': 'needs_attention'})
            raise
