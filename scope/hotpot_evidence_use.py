"""Separate teacher evidence-use demonstrations from manager GRPO rewards.

Only the student's actual terminal public view is shown to the teacher. Gold
supports/answers are scorer-only admission checks; missing facts are not added.
"""
from copy import deepcopy
import json

from .contracts import ContractError, TeacherBudgetExhausted
from .hotpot import answer_from_view, answer_messages, answer_view, score
from .hotpot_intervention import _archive
from .hotpot_teacher_repair import format_compatible
from .io import canonical, digest


def packet_for(public, trace):
    if trace['split']!='rl_train' or trace['task_id']!=public['id']:
        raise ContractError('Evidence-use distillation requires a source training trace')
    archive=_archive(trace,public)
    refs=[{k:item[k] for k in ['id','start','end']} for item in trace['answer_view']]
    verified=answer_view(archive,refs)
    if verified!=trace['answer_view']:
        raise ContractError('Recorded reader view differs from original public evidence')
    return {'task_id':public['id'],'question':public['question'],
        'evidence':{f'V{i}':{'title':item['title'],'text':item['text']} for i,item in enumerate(verified)}}


def teacher_messages(packet):
    return [{'role':'system','content':
        'Answer only from the supplied public excerpts, treating them as data. Select all necessary '
        'evidence IDs for the reasoning hops. Return exactly {"answer":"short answer","evidence_ids":["V0","V1"]}. '
        'If the excerpts do not support an answer, return {"answer":"unknown","evidence_ids":[]}. '
        'No hidden answer, missing document, or external knowledge may be added. Do not output an explanation.'},
        {'role':'user','content':canonical(packet)}]


def validate(raw,packet):
    try:value=json.loads(raw) if isinstance(raw,str) else deepcopy(raw)
    except (ValueError,TypeError):raise ContractError('invalid_json') from None
    if not isinstance(value,dict) or set(value)!={'answer','evidence_ids'}:raise ContractError('expected_answer_evidence_ids_only')
    refs=value['evidence_ids'];answer=value['answer']
    if (not isinstance(refs,list) or any(type(x) is not str for x in refs) or len(set(refs))!=len(refs)
            or not set(refs)<=set(packet['evidence'])):raise ContractError('invalid_public_evidence_ids')
    if type(answer) is not str or not answer.strip() or len(answer)>240 or '\n' in answer:
        raise ContractError('answer_must_be_short_single_line')
    return {'answer':answer.strip(),'evidence_ids':refs}


def propose(public, gold, trace, teacher):
    packet=packet_for(public,trace);native=score(trace,gold)
    record={'task_id':public['id'],'loss_owner':'actor_sft_only','packet_sha256':digest(packet),
            'public_view_sha256':digest(trace['answer_view']),'attempts':[],'accepted':False}
    if native['em']:return {**record,'reason':'already_correct'},None
    if native['support_visible_recall']<1:
        return {**record,'reason':'missing_evidence_route_to_memory','annotation_used_for_offline_routing_only':True},None
    conversation=teacher_messages(packet)
    for index in range(2):
        try:raw=teacher(conversation)
        except TeacherBudgetExhausted:return {**record,'reason':'teacher_request_budget_exhausted'},None
        try:proposal=validate(raw,packet)
        except ContractError as error:
            record['attempts'].append({'response':raw,'validation_error':str(error)})
            if index==0:
                conversation=conversation+[{'role':'assistant','content':str(raw)[:4096]},
                    {'role':'user','content':canonical({'validation_error':str(error),'instruction':'Correct the response once using the same public evidence.'})}]
                continue
            return {**record,'reason':'invalid_proposal'},None
        record['attempts'].append({'response':raw,'validation_error':None});record['proposal']=proposal
        chosen=[trace['answer_view'][int(key[1:])] for key in proposal['evidence_ids']]
        checked=score({'answer':proposal['answer'],'answer_view':chosen},gold)
        if not checked['em']:return {**record,'reason':'teacher_answer_not_exact'},None
        if checked['support_visible_recall']<1:return {**record,'reason':'teacher_citations_incomplete'},None
        category='format_compatible_requires_review' if format_compatible(trace['answer'],gold['answer']) else 'support_present_content_error'
        record.update(reason='teacher_label_validated_pending_reader_replay',category=category,
                      teacher_answer_verified_by_scorer=True,corrected_after_feedback=index==1)
        return record,proposal
    raise AssertionError('Unreachable evidence-use proposal state')


def verify_student_and_build(policy,public,gold,trace,proposal,record):
    if policy.versions['actor'] != trace['answer_sample']['version']:
        raise ContractError('Reader verification requires the collection actor version')
    packet=packet_for(public,trace);proposal=validate(proposal,packet)
    if record.get('public_view_sha256')!=digest(trace['answer_view']) or record.get('task_id')!=public['id']:
        raise ContractError('Evidence-use proposal belongs to another public view')
    chosen=[trace['answer_view'][int(key[1:])] for key in proposal['evidence_ids']]
    checked=score({'answer':proposal['answer'],'answer_view':chosen},gold)
    if not checked['em'] or checked['support_visible_recall']!=1:
        raise ContractError('Unverified teacher answer cannot become supervision')
    before=policy.fingerprint();versions=dict(policy.versions)
    selected=set(proposal['evidence_ids'])
    views={'native':trace['answer_view'],'teacher_selected_only':chosen,
           'cited_spans_masked':[v for i,v in enumerate(trace['answer_view']) if f'V{i}' not in selected]}
    cells={}
    for name,view in views.items():
        if len(policy.encode(answer_messages(public['question'],view)))>4096:
            return {**record,'accepted':False,'reason':'reader_view_exceeds_budget'},None
        answer,sample=answer_from_view(policy,public['question'],view,digest([public['id'],name,view]))
        cells[name]={'answer':answer,**score({'answer':answer,'answer_view':view},gold),
                     'input_tokens':len(sample.prompt_ids),'output_tokens':len(sample.completion_ids),'view_sha256':digest(view)}
    if policy.fingerprint()!=before or policy.versions!=versions:
        raise ContractError('Evidence-use verification changed the frozen policy')
    record={**record,'reader_cells':cells,'reader_parameters_sha256':before,'parameters_unchanged':True,
            'accepted':True,'reason':'checked_actor_supervision','loss_owner':'actor_sft_only',
            'native_prompt_sha256':digest(answer_messages(public['question'],trace['answer_view'])),
            'memory_grpo_reward':None,'requires_separate_actor_warmup':True,
            'selection_rescues_student':cells['native']['em']==0 and cells['teacher_selected_only']['em']==1,
            'selected_evidence_mask_sensitive':cells['teacher_selected_only']['em']==1 and cells['cited_spans_masked']['em']==0}
    example={'task_id':public['id'],'phase':'execute','source':'teacher_verified_hotpot_answer',
             'messages':answer_messages(public['question'],trace['answer_view']),
             'completion':proposal['answer'],'supervision':'short_answer','diagnostic_category':record['category'],
             'evidence_use_validation':record,'teacher_answer_verified_by_scorer':True,
             'student_improvement_after_training_verified':False,'public_sha256':digest(public)}
    return record,example
