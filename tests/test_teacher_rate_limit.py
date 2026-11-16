from collections import Counter
from types import SimpleNamespace
import json

import pytest

from scope.contracts import InfrastructureError, TeacherBudgetExhausted
from scope.teacher_api import APITeacher
from scope.teacher_rate_limit import RateLimitedSession, retry_after_seconds
import scope.teacher_rate_limit as retry


class Response:
    def __init__(self, status, header=None):
        self.status_code=status;self.headers={'Retry-After':header} if header is not None else {};self.closed=False
    def close(self):self.closed=True
    def json(self):
        return {'model':'teacher','choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}],
                'usage':{'prompt_tokens':10,'completion_tokens':4}}


@pytest.fixture
def clock(monkeypatch):
    state={'now':0.,'sleeps':[]}
    def sleep(seconds):state['sleeps'].append(seconds);state['now']+=seconds
    monkeypatch.setattr(retry.time,'monotonic',lambda:state['now'])
    monkeypatch.setattr(retry.time,'sleep',sleep)
    return state


def test_429_retries_identical_request_counts_cost_and_preserves_cache(tmp_path,clock):
    teacher=APITeacher.__new__(APITeacher)
    teacher.model='teacher';teacher.url='https://unused.invalid';teacher.identity={'model':'teacher'}
    teacher.counts=Counter();teacher.limit=10;teacher.cache=tmp_path;teacher.allow_unusable=True
    responses=[Response(429,'90'),Response(429),Response(200)];calls=[]
    def post(*args,**kwargs):calls.append((args,kwargs));return responses[len(calls)-1]
    teacher.http=RateLimitedSession(SimpleNamespace(post=post),teacher)
    messages=[{'role':'user','content':'public prefix'}]
    assert json.loads(teacher(messages))=={'ok':True}
    assert len(calls)==3 and calls[0]==calls[1]==calls[2]
    assert clock['sleeps']==[90.,60.]
    assert teacher.counts['requests']==3 and teacher.counts['rate_limited_attempts']==2
    assert teacher.counts['prompt_tokens']==10
    teacher.limit=3
    assert json.loads(teacher(messages))=={'ok':True}
    assert len(calls)==3 and teacher.counts['cache_hits']==1


def test_persistent_rate_limit_is_infrastructure_error_with_bounded_wait(clock):
    teacher=SimpleNamespace(counts=Counter(requests=1),limit=100)
    calls=[]
    def post(*a,**kw):calls.append(1);return Response(429)
    session=RateLimitedSession(SimpleNamespace(post=post),teacher,attempts=2)
    with pytest.raises(InfrastructureError,match='429 persisted'):
        session.post('unused')
    assert len(calls)==2 and teacher.counts['requests']==2 and clock['sleeps']==[30.]


def test_retry_never_exceeds_total_request_budget(clock):
    teacher=SimpleNamespace(counts=Counter(requests=1),limit=1)
    session=RateLimitedSession(SimpleNamespace(post=lambda *a,**kw:Response(429)),teacher)
    with pytest.raises(TeacherBudgetExhausted):session.post('unused')
    assert teacher.counts['requests']==1 and clock['sleeps']==[]


def test_long_retry_after_is_not_ignored_or_shortened(clock):
    teacher=SimpleNamespace(counts=Counter(requests=1),limit=10)
    session=RateLimitedSession(SimpleNamespace(post=lambda *a,**kw:Response(429,'3600')),teacher)
    with pytest.raises(InfrastructureError):session.post('unused')
    assert clock['sleeps']==[]
    assert retry_after_seconds('Thu, 12 Nov 2026 00:01:00 GMT',1794441600)==60
