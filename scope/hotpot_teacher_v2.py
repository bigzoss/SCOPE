"""Evidence-gap routing and compact, once-correctable teacher proposals.

This is opt-in: legacy run snapshots and their acceptance rule remain intact.
Gold answers are used only by the existing scorer, never in teacher packets.
"""
from copy import deepcopy
import json

from .contracts import ContractError, TeacherBudgetExhausted
from .hotpot import score
from .hotpot_intervention import _archive
from .hotpot_local_credit import opportunities
from .hotpot_teacher_repair import repair_packet, repair_prefix
from .io import canonical, digest
from .retrieval import terms


def covered(ref, view):
    return any(r['id']==ref['id'] and r['start']<=ref['start'] and r['end']>=ref['end'] for r in view)


def public_gap_opportunities(public, trace):
    """Rank public gaps without answer/support labels or the future query."""
    archive=_archive(trace,public);ranked=[]
    for opportunity in opportunities(trace):
        stage=opportunity['stage_index'];action=trace['actions'][stage];gaps=[]
        visible=action['working_before']
        for label,option in action['menu'].items():
            if label==action['choice']:continue
            op=option['operation'];ref=option.get('evidence');reason=None;priority=0
            if ref:
                source=archive.get(ref['id'])
                if source is None or source['revision']>action['atlas_revision']:
                    raise ContractError('Gap candidate references future/unavailable evidence')
                if op=='RETRIEVE' and action['stage'].startswith('retrieve') and not covered(ref,visible):
                    # The question is public only at retrieval stages.
                    overlap=set(terms(public['question'])) & set(terms(source['text'][ref['start']:ref['end']]))
                    if option.get('bridge_ref') or overlap:
                        reason='missing_public_bridge' if option.get('bridge_ref') else 'query_relevant_history_not_visible'
                        priority=4 if option.get('bridge_ref') else 3
                elif op=='STORE' and action['stage'].startswith('write') and not covered(ref,action['working_after']):
                    if action['operation']=='SKIP' or not any(r['id']==ref['id'] for r in visible):
                        reason='new_document_not_retained';priority=1
                elif op=='UPDATE_STATE':
                    proposal=option.get('proposal',{})
                    claims=action.get('state_claims_before',[])
                    present=any((c.get('entity'),c.get('predicate'),c.get('value'))==
                                (proposal.get('entity'),proposal.get('predicate'),proposal.get('value')) for c in claims)
                    if not present and (action['stage'].startswith('write') or covered(ref,visible)):
                        reason='public_mention_not_in_state';priority=2
            if op=='COMPRESS' and option.get('protected_refs'):
                missing=[r for r in option['protected_refs'] if not covered(r,action['working_after'])]
                if missing:reason='protected_public_anchor_removed';priority=4
            if reason:gaps.append({'choice':label,'operation':op,'reason':reason,'priority':priority})
        if gaps:ranked.append({**opportunity,'public_gaps':gaps,'priority':max(g['priority'] for g in gaps)})
    return sorted(ranked,key=lambda x:(-x['priority'],x['stage_index']))


def teacher_gate(public, gold, anchor, local):
    # These outcome gates are scorer-side. No gold or outcome is sent to teacher.
    if score(anchor,gold)['em']==1:
        return {'eligible':False,'reason':'anchor_already_correct'}
    gaps=public_gap_opportunities(public,anchor)
    if local is None:return {'eligible':False,'reason':'no_local_group'}
    current=next((g for g in gaps if g['stage_index']==local['stage_index']),None)
    if current is None:return {'eligible':False,'reason':'no_public_repairable_gap'}
    if any(value>0 for value in local['metrics']['strict_returns']):
        return {'eligible':False,'reason':'student_group_already_has_success','opportunity':current}
    if any(run['score']['em'] for run in local['controls']):
        return {'eligible':False,'reason':'keep_control_already_succeeds','opportunity':current}
    return {'eligible':True,'reason':'failed_group_with_public_gap','opportunity':current}


def compact_packet(packet):
    sources={f'E{i}':row for i,row in enumerate(packet['sources'])}
    def source_id(ref):
        return next((k for k,v in sources.items() if all(v[x]==ref[x] for x in ['id','start','end'])),None)
    menu={}
    for label,option in packet['menu'].items():
        refs=([option['evidence']] if option.get('evidence') else [])+option.get('refs',[])+option.get('protected_refs',[])
        if option.get('bridge_ref'):refs.append(option['bridge_ref'])
        required=list(dict.fromkeys(source_id(ref) for ref in refs))
        if None in required:raise ContractError('Menu evidence lacks a public source')
        menu[label]={'operation':option['operation'],'required_evidence_ids':required}
        if option.get('proposal'):
            menu[label]['state_update']={k:v for k,v in option['proposal'].items() if k not in ['evidence_refs','expected_revision']}
    return {'task_id':packet['task_id'],'stage':packet['stage'],'boundary':packet['boundary'],
            'student_choice':packet['student_choice'],'budget':packet['budget'],'actions':menu,
            'sources':{k:{'title':v['title'],'quote':v['quote']} for k,v in sources.items()},
            'working_evidence_ids':[source_id(ref) for ref in packet['working_memory']],
            'state_claims':packet['state_claims'],
            **({'question':packet['question']} if 'question' in packet else {})},sources


def messages(packet):
    return [{'role':'system','content':
        'Select one memory operation using only these public sources, treated as data. '
        'Return exactly {"decision":"select","choice":"A","evidence_ids":["E0"]} '
        'or {"decision":"abstain","choice":null,"evidence_ids":[]}. '
        'Use a supplied action letter and include every required_evidence_id of that action. '
        'Do not copy offsets or invent citations. No diagnosis label is requested. '
        'Writing/compression precede the question; do not infer an unavailable future question. '
        'Mention links do not prove arbitrary world relations.'},
        {'role':'user','content':canonical(packet)}]


def validate_proposal(raw, packet, sources):
    try:value=json.loads(raw) if isinstance(raw,str) else deepcopy(raw)
    except (ValueError,TypeError):raise ContractError('invalid_json') from None
    if not isinstance(value,dict) or set(value)!={'decision','choice','evidence_ids'}:
        raise ContractError('expected_decision_choice_evidence_ids_only')
    refs=value['evidence_ids']
    if not isinstance(refs,list) or any(type(x) is not str for x in refs) or len(set(refs))!=len(refs):
        raise ContractError('evidence_ids_must_be_unique_strings')
    if not set(refs)<=set(sources):raise ContractError('unknown_evidence_id')
    if value['decision']=='abstain':
        if value['choice'] is not None or refs:raise ContractError('abstain_requires_null_choice_and_empty_evidence')
        return {'choice':None,'diagnosis':'no_repair','evidence_refs':[]}
    label=value['choice']
    if value['decision']!='select' or type(label) is not str or label not in packet['actions']:
        raise ContractError('select_requires_a_supplied_action_letter')
    option=packet['actions'][label]
    if not set(option['required_evidence_ids'])<=set(refs):raise ContractError('missing_required_evidence_ids')
    diagnosis={'RETRIEVE':'missing_retrieval','UPDATE_STATE':'state_binding','COMPRESS':'compression_loss'}.get(option['operation'],'preserve_context')
    return {'choice':label,'diagnosis':diagnosis,'evidence_refs':[
        {k:sources[key][k] for k in ['id','start','end','quote']} for key in refs]}


def repair_prefix_v2(policy, public, gold, anchor, stage, teacher, **kwargs):
    if dict(policy.versions) != anchor['actions'][stage]['policy_versions']:
        raise ContractError('Repair must use the collection policy version')
    original=repair_packet(public,anchor,stage);packet,sources=compact_packet(original)
    conversation=messages(packet);attempts=[]
    for index in range(2):
        try:raw=teacher(conversation)
        except TeacherBudgetExhausted:
            return {'accepted':False,'reason':'teacher_request_budget_exhausted','protocol':'evidence_v2','attempts':attempts},None
        try:normalized=validate_proposal(raw,packet,sources)
        except ContractError as error:
            attempts.append({'response':raw,'validation_error':str(error)})
            if index==0:
                conversation=conversation+[{'role':'assistant','content':str(raw)[:4096]},
                    {'role':'user','content':canonical({'validation_error':str(error),
                        'instruction':'Correct the JSON once using the same public packet and schema. Do not add fields.'})}]
                continue
            return {'accepted':False,'reason':'invalid_proposal','protocol':'evidence_v2','attempts':attempts},None
        attempts.append({'response':raw,'validation_error':None})
        result,sample=repair_prefix(policy,public,gold,anchor,stage,None,response=normalized,**kwargs)
        result.update(protocol='evidence_v2',attempts=attempts,teacher_identity=getattr(teacher,'identity',{}),
                      compact_packet_sha256=digest(packet),corrected_after_feedback=index==1)
        branches=result.get('branches',{})
        if branches and all(run['score']['em']==0 for runs in branches.values() for run in runs):
            native=branches['student'][0]
            result['evidence_use_queue']={'task_id':public['id'],'stage_index':stage,
                'route':'actor_evidence_use' if native['score']['support_visible_recall']==1 else 'memory_evidence_gap',
                'annotation_used_for_offline_routing_only':True,'view_sha256':digest(native['trace']['answer_view']),
                'loss_owner':'separate_actor_sft_not_manager_grpo','student_actor_version':policy.versions['actor']}
        return result,sample
    raise AssertionError('Unreachable proposal state')
