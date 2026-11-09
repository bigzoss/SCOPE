import copy
import pytest
from scope.contracts import Boundary,Budget,Scope,Sample,ContractError
from scope.credit import local_samples,standardize,interaction


def boundary():
    return Boundary(Scope("train","1","ep"),{}, {},Budget(),"goal","page",[],"context",{"manager":0,"actor":0},None,{},0,working={"selected_ids":[]})


def test_local_credit_target_and_zero_variance():
    b=boundary()
    samples=[Sample("manager","context",[1],[2],[0.],"{}",0,b.fingerprint()) for _ in range(2)]
    candidates=[{"boundary_hash":b.fingerprint(),"return":r,"selected":[s]} for s,r in zip(samples,[0.,1.])]
    weighted,info=local_samples(b,candidates)
    assert weighted[0][0].advantage<0<weighted[1][0].advantage
    assert standardize([1.,1.,1.])==[0.,0.,0.]
    samples[0].phase="state"
    with pytest.raises(ContractError,match="phase"):local_samples(b,candidates)


def test_boundary_includes_working_view_and_interaction_is_not_additive_credit():
    b=boundary();other=copy.deepcopy(b);other.working["selected_ids"]=["obs_other"]
    assert b.fingerprint()!=other.fingerprint()
    assert interaction(0.,0.,0.,1.)["interaction"]==1.

