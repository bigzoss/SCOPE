"""Recompute completed evidence-training coverage, filtering, metrics and provenance."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scope.io import atomic_json, digest, file_sha


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(root):
    receipt, summary = read(root/'receipt.json'), read(root/'summary.json')
    assert summary['complete'] and read(root/'status.json')['phase']=='complete'
    assert read(root/'exit.json')['exit_code']==0
    assert not (root/'error.json').exists() and not (root/'teacher_error.json').exists()
    assert file_sha(root/'receipt.json')==summary['receipt_sha256']
    for name,sha in receipt['source_sha256'].items():
        assert file_sha(root/name)==sha
    train,dev=receipt['train_ids'],receipt['dev_ids']
    assert len(train)==len(set(train))==128
    assert len(dev)==len(set(dev))==96
    assert not set(train).intersection(dev)
    assert not set(dev).intersection(receipt['excluded_old_dev_ids'])
    execution=read(root/'execution_receipt.json')
    previous=read(Path(execution['previous_run'])/'receipt.json')
    assert previous['train_ids']==train and previous['dev_ids']==dev
    for name,sha in execution['reused_files_sha256'].items():
        assert file_sha(root/name)==sha,name
    ids={p.stem for p in (root/'teacher').glob('*.json')}
    assert ids==set(train)
    for p in (root/'teacher_requests').glob('*.json'):
        request=read(p)
        assert request['model']==receipt['teacher']
        payload=json.loads(request['messages'][-1]['content'])
        assert set(payload)=={'requests'}
        for task in payload['requests']:
            assert set(task)=={'task_id','question','sentences'}
            assert task['task_id'] in train
            assert all(set(sentence)=={'title','sentence_index','text'} for sentence in task['sentences'].values())
    results={}
    for arm in ('initial','answer_only','evidence_and_answer'):
        path=root/arm/'results.jsonl'
        assert file_sha(path)==summary['results_sha256'][arm]
        records=jsonl(path)
        assert [r['task_id'] for r in records]==dev
        assert len({r['parameters'] for r in records})==1
        assert sum(r['cells']['native']['em'] for r in records)==summary['results'][arm]['native']['em_count']
        for r in records:
            assert r['cells']['native']['view_sha256']==r['cells']['support_masked']['view_sha256']
        results[arm]=records
    for left,right in zip(results['initial'],results['answer_only'],strict=True):
        assert {k:v['view_sha256'] for k,v in left['cells'].items()}=={k:v['view_sha256'] for k,v in right['cells'].items()}
    for left,right in zip(results['initial'],results['evidence_and_answer'],strict=True):
        assert {k:v['view_sha256'] for k,v in left['cells'].items()}=={k:v['view_sha256'] for k,v in right['cells'].items()}
        base=read(root/'baseline/dev'/f'{left["task_id"]}.json')
        assert left['cells']['native']['answer']==base['answer']
    training={}
    for arm in ('answer_only','evidence_and_answer'):
        examples=jsonl(root/arm/'examples.jsonl')
        counts=Counter(row['task_id'] for row in examples)
        assert set(counts).issubset(set(train)) and set(counts.values())=={2}
        assert len(counts)==summary['data']['accepted_tasks']
        assert read(root/arm/'resume_validation.json')['passed']
        trained=summary['training'][arm]
        assert trained['complete'] and trained['actor_changed'] and trained['manager_unchanged'] and not trained['filtered_over_budget']
        assert trained['updates']==len(examples)  # two epochs, batch size two
        ledger=jsonl(root/arm/'sft/warmup_metrics.jsonl')
        assert [r['update'] for r in ledger]==list(range(1,trained['updates']+1))
        training[arm]={'examples':len(examples),'updates':trained['updates'],
                       'supervision':dict(Counter(r['supervision'] for r in examples)),
                       'examples_sha256':file_sha(root/arm/'examples.jsonl')}
    answer_data=jsonl(root/'answer_only/examples.jsonl')
    joint_data=jsonl(root/'evidence_and_answer/examples.jsonl')
    for offset in range(0,len(answer_data),2):
        assert answer_data[offset]==answer_data[offset+1]==joint_data[offset]
        link=joint_data[offset+1]
        assert set(json.loads(link['messages'][-1]['content']))=={'question','sentences'}
    verified={'passed':True,'train':128,'fresh_dev':96,'train_dev_disjoint':True,
              'initial_native_matches_full_rollout':True,'same_views_all_arms':True,
              'training':training,'summary_sha256':file_sha(root/'summary.json'),
              'limitations':['Public source identity and offline support coverage are checked; semantic entailment is not formally proven.',
                             'Oracle support augmentation is separate from native benchmark performance.',
                             'Format candidate classification can include negated or ambiguous assertions.']}
    atomic_json(root/'audit.json',verified)
    print(json.dumps(verified,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    audit(parser.parse_args().root)
