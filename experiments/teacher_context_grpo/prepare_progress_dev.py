"""Paired thermal development evaluation, without target model updates."""
import json
import os
from pathlib import Path
import sys

PROJECT = Path('/home/hyx/lm_workspace/SCOPE')
sys.path[:0] = [str(PROJECT), str(PROJECT / 'experiments/teacher_context_grpo'),
    str(PROJECT / 'runs/hotpot_four_bench_20261110.3j9gw8h4/native_deps')]
os.environ['PATH'] = '/home/hyx/lm_workspace/longagent/.java/bin:' + os.environ['PATH']
from prerequisite_dev_eval import prepare
from scope.io import atomic_json
from scienceworld import ScienceWorldEnv


def main(root):
    prepare(PROJECT, root, 'closure')
    receipt = json.loads((root / 'receipt.json').read_text())
    env = ScienceWorldEnv(envStepLimit=30)
    tasks = []
    try:
        for name in ('use-thermometer', 'measure-melting-point-unknown-substance'):
            env.load(name, 0)
            variations = sorted(env.get_variations_dev())[:3]
            assert len(variations) == 3 and not set(variations) & set(env.get_variations_test())
            tasks.extend({'id': f'{name}:{v}', 'task': name, 'variation': v} for v in variations)
    finally:
        env.close()
    receipt.update(profile='components', tasks={'scienceworld': tasks},
        arms_by_domain={'scienceworld': ['public_budget_aliases', 'progress_experiment']},
        scope_note='Six official development variants: two previously diagnosed, four additional. '
        'Frozen weights/budget; no teachers, target updates or claims of untouched final testing.')
    atomic_json(root / 'receipt.json', receipt)
    atomic_json(root / 'status.json', {'phase': 'prepared', 'rollouts': 12})


if __name__ == '__main__':
    main(Path(sys.argv[1]))
