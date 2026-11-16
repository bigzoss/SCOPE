"""Bounded batch transport for independently validated public-only teacher labels."""
from collections import Counter
import json
import re
import time
import traceback

import requests

from evidence_data import evidence_messages, parse_proposal
from experiments.memory_curriculum.hotpot_teacher_rewards import USTCTeacher, TEACHER_MODEL
from scope.contracts import ContractError
from scope.io import atomic_json, canonical, digest


def parse_batch(text, publics):
    match=re.fullmatch(r'(?:```(?:json)?\s*)?(\{.*\})(?:\s*```)?',text.strip(),re.S)
    result=json.loads(match[1] if match else text)
    if not isinstance(result,dict) or set(result)!={'results'} or not isinstance(result['results'],list):
        raise ValueError('Expected a batch results list')
    by_id={p['id']:p for p in publics}; output={}
    for row in result['results']:
        if not isinstance(row,dict) or set(row)!={'task_id','answer','evidence'}:
            raise ValueError('Invalid batch row')
        task_id=row['task_id']
        if type(task_id) is not str or task_id not in by_id or task_id in output:
            raise ValueError('Unknown or repeated batch task')
        output[task_id]=parse_proposal(json.dumps({k:row[k] for k in ('answer','evidence')}),by_id[task_id])
    if set(output)!=set(by_id):
        raise ValueError('Missing batch task')
    return output


def collect(tasks,root,config,stop):
    client=USTCTeacher(config,root/'teacher_cache');client.cache.mkdir(exist_ok=True)
    counts=Counter();pending=[];done=0
    def save(public,response):
        nonlocal done
        atomic_json(root/'teacher'/f'{public["id"]}.json',{'task_id':public['id'],'response':response})
        done+=1
        atomic_json(root/'teacher_status.json',{'phase':'collecting','done':done,'n':len(tasks),'api':dict(counts)})
    try:
        for public,_ in tasks:
            cache=client.cache/(digest([TEACHER_MODEL,evidence_messages(public),192])+'.json')
            if cache.exists():
                response=json.loads(cache.read_text())
                if response['model']!=TEACHER_MODEL or response['public_sha256']!=digest(public):
                    raise ContractError('Cached teacher identity changed')
                parse_proposal(json.dumps(response['proposal']),public)
                counts['cached_responses']+=1;save(public,response)
            else:
                pending.append((public,cache))
        for start in range(0,len(pending),4):
            group=pending[start:start+4];publics=[p for p,_ in group]
            if stop.is_set():return
            prompt={'requests':[{'task_id':p['id'],**json.loads(evidence_messages(p)[1]['content'])} for p in publics]}
            instruction=(evidence_messages(publics[0])[0]['content']+
                ' Process every request independently; sentence IDs belong only to that request. '
                'Do not transfer facts or citations between requests. Instead of a single result, return '
                '{"results":[{"task_id":"exact supplied ID","evidence":["D0:S1"],"answer":"short answer"}]}.' )
            messages=[{'role':'system','content':instruction},{'role':'user','content':canonical(prompt)}]
            request_id=digest([TEACHER_MODEL,messages,192*len(group)])
            proposals=None;usage={};finish=None
            for attempt in range(4):
                if stop.wait(max(0.,6.5-(time.monotonic()-client.last_request))):return
                client.last_request=time.monotonic();counts['requests']+=1
                try:
                    result=client.http.post(client.url,json={'model':TEACHER_MODEL,'messages':messages,
                        'temperature':0,'max_tokens':192*len(group),'chat_template_kwargs':{'enable_thinking':False}},
                        timeout=(12,60),allow_redirects=False)
                except requests.RequestException:
                    counts['transport_retries']+=1
                    if stop.wait(2**attempt):return
                    continue
                limits={k:v for k,v in result.headers.items() if 'ratelimit' in k.lower() or k.lower()=='retry-after'}
                atomic_json(root/'teacher_limits.json',{'status_code':result.status_code,'headers':limits})
                if result.status_code in {429,500,502,503,504}:
                    counts[f'http_{result.status_code}']+=1
                    if stop.wait(60 if result.status_code==429 else 2**attempt):return
                    continue
                if result.status_code!=200:
                    raise ContractError(f'Teacher returned HTTP {result.status_code}')
                body=result.json()
                if body.get('model')!=TEACHER_MODEL:raise ContractError('Teacher model identity changed')
                usage={k:int((body.get('usage') or {}).get(k,0) or 0) for k in ('prompt_tokens','completion_tokens')}
                counts.update(usage)
                try:
                    row=body['choices'][0];finish=row.get('finish_reason')
                    if finish=='length':raise ValueError('Batch output truncated')
                    proposals=parse_batch(row['message']['content'],publics)
                except (ValueError,KeyError,TypeError,IndexError):
                    counts['invalid_response_retries']+=1;continue
                atomic_json(root/'teacher_requests'/f'{request_id}.json',{'messages':messages,'model':TEACHER_MODEL,
                    'proposals':proposals,'usage':usage,'finish_reason':finish})
                break
            for public,cache in group:
                response=None
                if proposals is not None:
                    response={'model':TEACHER_MODEL,'proposal':proposals[public['id']],
                              'public_sha256':digest(public),'finish_reason':finish,'request_id':request_id,
                              'batch_task_ids':[p['id'] for p in publics],
                              'usage_scope':'see teacher_requests, counted once per batch'}
                    atomic_json(cache,response);counts['new_valid_responses']+=1
                else:counts['unavailable']+=1
                save(public,response)
        atomic_json(root/'teacher_status.json',{'phase':'complete','done':done,'n':len(tasks),'api':dict(counts)})
    except Exception:
        atomic_json(root/'teacher_error.json',{'traceback':traceback.format_exc(),'api':dict(counts)})
    finally:client.close()
