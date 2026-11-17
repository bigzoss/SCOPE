"""Prepare generated PDDL instances and official ScienceWorld dev ablations."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path('/home/hyx/lm_workspace/SCOPE')
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/teacher_context_grpo'),
               str(ROOT / 'runs/hotpot_four_bench_20261110.3j9gw8h4/native_deps')]
os.environ['PATH'] = '/home/hyx/lm_workspace/longagent/.java/bin:' + os.environ['PATH']
from prerequisite_dev_eval import prepare
from scope.io import atomic_json, file_sha


def problem(name, domain, objects, initial, goals):
    return (f'(define (problem {name}) (:domain {domain}) (:objects {objects}) '
            f'(:init {" ".join(initial)}) (:goal (and {" ".join(goals)})))')


def main(output):
    prepare(ROOT, output, 'closure')
    receipt = json.loads((output / 'receipt.json').read_text())
    folder = output / 'development_pddl/pddl'
    native = ROOT / 'baseline/AgentBoard/agentboard/environment/pddl_env/pddlgym/pddl'
    specs = receipt['tasks']['pddl']
    for domain in ('blockworld', 'barman', 'tyreworld'):
        (folder / domain).mkdir()
        shutil.copyfile(native / f'{domain}.pddl', folder / f'{domain}.pddl')
        for index in range(2):
            if domain == 'blockworld':
                blocks = [f'b{i}' for i in range(1, 5 + index)]
                objects = ' '.join(blocks)
                initial = ['(arm-empty)'] + [f'(on-table {b}) (clear {b})' for b in blocks]
                goals = [f'(on {a} {b})' for a, b in zip(blocks[:-1], blocks[1:])]
                domain_name = 'blocksworld-4ops'
            elif domain == 'barman':
                objects = ('shaker1 - shaker left right - hand shot1 shot2 - shot '
                           'ingredient1 ingredient2 ingredient3 ingredient4 - ingredient '
                           'cocktail1 - cocktail dispenser1 dispenser2 dispenser3 dispenser4 - dispenser l0 l1 l2 - level')
                initial = [f'(ontable {x}) (clean {x}) (empty {x})' for x in ('shaker1', 'shot1', 'shot2')]
                initial += [f'(dispenses dispenser{i} ingredient{i})' for i in range(1, 5)]
                initial += ['(handempty left) (handempty right)', '(shaker-empty-level shaker1 l0)',
                            '(shaker-level shaker1 l0)', '(next l0 l1) (next l1 l2)',
                            f'(cocktail-part1 cocktail1 ingredient{1+index})', '(cocktail-part2 cocktail1 ingredient4)']
                goals = ['(contains shot1 cocktail1)', f'(contains shot2 ingredient{3-index})']
                domain_name = 'barman'
            else:
                n = index + 2
                objects = ('wrench jack pump - tool boot - container '
                           + ' '.join(f'h{i}' for i in range(n)) + ' - hub '
                           + ' '.join(f'n{i}' for i in range(n)) + ' - nut '
                           + ' '.join(f'r{i} w{i}' for i in range(n)) + ' - wheel')
                initial = ['(in jack boot) (in pump boot) (in wrench boot)', '(unlocked boot) (isopen boot)']
                goals = []
                for i in range(n):
                    inflation = 'inflated' if i == 0 else 'not-inflated'
                    initial += [f'(intact r{i}) (in r{i} boot) ({inflation} r{i})',
                                f'(on w{i} h{i}) (on-ground h{i}) (tight n{i} h{i}) (fastened h{i})']
                    goals += [f'(on r{i} h{i}) (inflated r{i}) (in w{i} boot)']
                domain_name = 'tyreworld'
            task_id = f'generated-{domain}-{index}'
            path = folder / domain / f'dev{index}.pddl'
            path.write_text(problem(task_id, domain_name, objects, initial, goals))
            specs.append({'id': task_id, 'domain': domain, 'index': index, 'goal': ' '.join(goals)})
    receipt.update(profile='components', scope_note='Generated PDDL engineering tasks across four domains; '
        'official ScienceWorld development variants. No target model updates.',
        arms_by_domain={'pddl': ['grounded_dependencies_guard', 'readable_state_only', 'whole_goal_only', 'transport_only'],
                        'scienceworld': ['public_budget_aliases', 'continuous_experiment']})
    receipt['assets'].update({str(p): file_sha(p) for p in folder.rglob('*.pddl')})
    atomic_json(output / 'receipt.json', receipt)
    atomic_json(output / 'status.json', {'phase': 'prepared', 'rollouts': 40})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
