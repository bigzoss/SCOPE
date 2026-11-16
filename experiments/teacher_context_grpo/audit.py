"""Audit completed actor-distillation/manager-GRPO pilot without loading models."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, parent):
    receipt = read(root/'receipt.json')
    summary = read(root/'summary.json')
    assert summary['complete'] and read(root/'status.json')['phase'] == 'complete'
    assert read(root/'exit.json')['exit_code'] == 0
    assert not (root/'error.json').exists()
    assert read(root/'actor_resume_validation.json')['passed']
    assert read(root/'actor_sft/summary.json')['manager_unchanged']
    assert sha(parent/'receipt.json') == receipt['parent_receipt_sha256']
    for name, expected in receipt['source_sha256'].items():
        assert sha(root/name) == expected, name
    train, dev = receipt['train_ids'], receipt['dev_ids']
    assert len(train) == len(set(train)) == 64
    assert len(dev) == len(set(dev)) == 48
    assert not set(train).intersection(dev)
    auxiliary = []
    files = {}
    for arm in ('actor_student', 'actor_teacher'):
        records = [read(p) for p in sorted((root/arm/'steps').glob('*.json'))]
        assert [r['task_id'] for r in records] == train
        for r in records:
            assert 0 <= r['version_after']-r['version_before'] <= 1
            if r.get('auxiliary'):
                a,t = r['auxiliary'],r['teacher']
                assert r['local']['metrics']['all_failed'] and t['verified']
                assert a['origin'] == a['sample']['origin'] == 'validated_auxiliary'
                assert not a['sample']['old_logprobs']
                assert a['new_logprob'] > a['old_logprob']
                assert t['boundary'] == r['local']['boundary'] == a['boundary']
                cells = t['evidence_intervention']['cells']
                visible = cells['a_visible']['score']['em']-cells['b_visible']['score']['em']
                masked = cells['a_masked']['score']['em']-cells['b_masked']['score']['em']
                assert visible > 0 and masked <= 0
                auxiliary.append({'step':r['step'],'task_id':r['task_id'],
                                  'strict_visible_delta':visible,'strict_masked_delta':masked,
                                  'old_logprob':a['old_logprob'],'new_logprob':a['new_logprob'],
                                  'cells':{k:{'answer':v['answer'],'em':v['score']['em'],
                                             'support_visible_recall':v['score']['support_visible_recall']}
                                           for k,v in cells.items()}})
    mapping = {'initial':parent/'initial/dev.jsonl',
               'manager_student':parent/'student/dev.jsonl','manager_teacher':parent/'teacher/dev.jsonl'}
    for arm in ('actor_only','actor_student','actor_teacher'):
        mapping[arm]=root/arm/'dev.jsonl'
    for arm,path in mapping.items():
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert [r['task_id'] for r in rows] == dev
        assert sum(r['em'] for r in rows) == summary['dev'][arm]['em_count']
        files[str(path)]=sha(path)
    result={'passed':True,'train_tasks_per_arm':64,'development_tasks_per_arm':48,
            'train_dev_disjoint':True,'actor_recovery_passed':True,
            'auxiliary_events':auxiliary,'result_sha256':files,
            'summary_sha256':sha(root/'summary.json'),
            'limitations':['48 development questions were previously used for diagnostics.',
                           'Evidence masking alone cannot separate factual retrieval from answer-format effects.',
                           'Eight teacher SFT examples are selected source-training successes; no generalization proof.']}
    (root/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True)
    a=p.parse_args()
    audit(a.root,a.parent)
