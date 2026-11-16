"""Contract, leakage and gradient ownership checks for opt-in teacher routing."""
from copy import deepcopy
import json
import pytest
from scope.contracts import ContractError
from scope.hotpot import rollout
from scope.hotpot_teacher_repair import repair_packet, check_proposal
from scope.hotpot_teacher_v2 import (compact_packet, messages, validate_proposal,
    repair_prefix_v2, teacher_gate, public_gap_opportunities)
from scope.hotpot_evidence_use import packet_for, propose, verify_student_and_build
from test_hotpot_teacher_repair import failing_anchor, RepairReader
from test_hotpot_local_credit import task


def reply_for(packet, operation='STORE'):
    label,op=next((k,v) for k,v in packet['actions'].items() if v['operation']==operation)
    return {'decision':'select','choice':label,'evidence_ids':op['required_evidence_ids']}


def test_compact_sources_roundtrip_and_no_future_query():
    _,public,gold,trace=failing_anchor()
    original=repair_packet(public,trace,1);packet,sources=compact_packet(original)
    assert 'question' not in packet and 'supporting_facts' not in json.dumps(messages(packet))
    assert check_proposal(original,validate_proposal(reply_for(packet),packet,sources))['choice']
    assert public_gap_opportunities(public,trace)


@pytest.mark.parametrize('bad',[
    {'decision':'select','choice':'Z','evidence_ids':[]},
    {'decision':'select','choice':'A','evidence_ids':['E999']},
    {'decision':'abstain','choice':'A','evidence_ids':[]},
    {'decision':'abstain','choice':None,'evidence_ids':[],'diagnosis':'no_repair'},
    'not json'])
def test_malformed_proposals_rejected(bad):
    _,p,_,t=failing_anchor();packet,sources=compact_packet(repair_packet(p,t,1))
    with pytest.raises(ContractError):validate_proposal(bad,packet,sources)


def test_correction_once_and_teacher_never_enters_on_policy_loss():
    policy,p,g,t=failing_anchor();requests=[]
    def teacher(msg):
        requests.append(deepcopy(msg))
        if len(requests)==1:return '{"choice":"A","diagnosis":"evidence_use"}'
        return json.dumps(reply_for(json.loads(msg[1]['content'])))
    r,s=repair_prefix_v2(policy,p,g,t,1,teacher,rollout_options={'record_prompts':True})
    assert len(requests)==2 and 'validation_error' in requests[1][-1]['content']
    assert r['accepted'] and r['corrected_after_feedback']
    assert s.origin=='teacher' and s.old_logprobs==[]
    assert r['loss_owner']=='manager_sft_only'
    calls=[]
    r,s=repair_prefix_v2(policy,p,g,t,1,lambda msg:(calls.append(msg) or '{}'))
    assert len(calls)==2 and s is None and r['reason']=='invalid_proposal'


def test_wrong_policy_version_stops_before_paid_call():
    policy,p,g,t=failing_anchor();policy.versions['manager']+=1
    with pytest.raises(ContractError,match='version'):
        repair_prefix_v2(policy,p,g,t,1,lambda _:pytest.fail('Should not call teacher'))


def test_gate_skips_successful_anchor_and_already_useful_student_groups():
    _,p,g,t=failing_anchor();stage=public_gap_opportunities(p,t)[0]['stage_index']
    local={'stage_index':stage,'metrics':{'strict_returns':[0,0]},'controls':[{'score':{'em':0}}]}
    assert teacher_gate(p,g,t,local)['eligible']
    ok=deepcopy(t);ok['answer']=g['answer']
    assert teacher_gate(p,g,ok,local)['reason']=='anchor_already_correct'
    local['metrics']['strict_returns']=[0,1]
    assert teacher_gate(p,g,t,local)['reason']=='student_group_already_has_success'
    local['metrics']['strict_returns']=[0,0];local['controls'][0]['score']['em']=1
    assert teacher_gate(p,g,t,local)['reason']=='keep_control_already_succeeds'


class WrongReader(RepairReader):
    def sample(self,*args,**kwargs):
        sample=super().sample(*args,**kwargs);sample.text='London';return sample
    def fingerprint(self):return 'frozen-test-parameters'


def evidence_fixture():
    policy=WrongReader();public,gold=task()
    trace,_=rollout(policy,public,'rl_train',1,greedy=True,record_prompts=True)
    return policy,public,gold,trace


def test_evidence_use_verified_label_is_not_claimed_as_student_improvement():
    policy,p,g,t=evidence_fixture();calls=[]
    def teacher(msg):
        calls.append(msg);packet=json.loads(msg[1]['content'])
        return json.dumps({'answer':'Glasgow','evidence_ids':list(packet['evidence'])})
    r,proposal=propose(p,g,t,teacher)
    assert not r['accepted'] and proposal and 'supporting_facts' not in json.dumps(calls)
    r,example=verify_student_and_build(policy,p,g,t,proposal,r)
    assert r['accepted'] and r['parameters_unchanged'] and r['memory_grpo_reward'] is None
    assert not r['selection_rescues_student']
    assert example['completion']=='Glasgow' and example['phase']=='execute'
    assert not example['student_improvement_after_training_verified']
    assert len(r['reader_cells'])==3


def test_evidence_use_rejects_fabrication_validation_tasks_and_missing_evidence():
    _,p,g,t=evidence_fixture()
    forged=deepcopy(t);forged['answer_view'][0]['text']='fabricated'
    with pytest.raises(ContractError):packet_for(p,forged)
    wrongsplit=deepcopy(t);wrongsplit['split']='validation'
    with pytest.raises(ContractError):packet_for(p,wrongsplit)
    missing=deepcopy(t);missing['answer_view']=[]
    r,proposal=propose(p,g,missing,lambda _:pytest.fail('No evidence, no teacher call'))
    assert proposal is None and r['reason']=='missing_evidence_route_to_memory'


def test_evidence_teacher_bad_answer_or_incomplete_citations_not_admitted():
    _,p,g,t=evidence_fixture()
    for reply,reason in [({'answer':'Berlin','evidence_ids':[]},'teacher_answer_not_exact'),
                         ({'answer':'Glasgow','evidence_ids':[]},'teacher_citations_incomplete')]:
        r,proposal=propose(p,g,t,lambda _:json.dumps(reply))
        assert not r['accepted'] and proposal is None and r['reason']==reason


def test_training_entry_v2_success_never_calls_teacher():
    from experiments.memory_curriculum.hotpot_grpo import step_data
    class AlwaysCorrect(RepairReader):
        def sample(self,*a,**kw):
            s=super().sample(*a,**kw);s.text='Glasgow';return s
    p,g=task()
    _,r=step_data(AlwaysCorrect(),(p,g),1,'alternating',0,{},0.,teacher_protocol='evidence_v2',
                 teacher=lambda _:pytest.fail('Correct anchor cannot trigger teacher'))
    assert r['teacher_gate']['reason']=='anchor_already_correct' and 'teacher_repair' not in r
    assert 'evidence_use_queue' not in r


def test_audited_hotpot_supervision_updates_actor_only_and_rejects_tampering(tmp_path):
    import torch
    from types import SimpleNamespace
    from test_policy import tiny_policy
    from scope.warmup import warmup
    from scope.io import atomic_jsonl
    reader,p,g,t=evidence_fixture()
    def teacher(msg):
        packet=json.loads(msg[1]['content'])
        return json.dumps({'answer':'Glasgow','evidence_ids':list(packet['evidence'])})
    audit,proposal=propose(p,g,t,teacher)
    audit,example=verify_student_and_build(reader,p,g,t,proposal,audit)
    assert example['source']=='teacher_verified_hotpot_answer' and 'environment_accepted' not in example
    learner=tiny_policy();learner.encode=lambda _:[1,2,3]
    learner.tokenizer=SimpleNamespace(encode=lambda text,**kw:[4,5],eos_token_id=6)
    optimizer={'actor':torch.optim.AdamW(learner.parameters('actor'),lr=.001)}
    data=tmp_path/'examples.jsonl';atomic_jsonl(data,[example])
    manager=learner.fingerprint('manager')
    result=warmup(learner,data,optimizer,tmp_path/'sft','test',allowed_tasks=[p['id']],role='actor')
    assert result['actor_changed'] and learner.fingerprint('manager')==manager
    example['completion']='tampered';atomic_jsonl(data,[example])
    with pytest.raises(ContractError,match='audited teacher answer'):
        warmup(learner,data,optimizer,tmp_path/'bad','test',allowed_tasks=[p['id']],role='actor')
