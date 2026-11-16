import pytest
from scope.contracts import ContractError
from experiments.teacher_context_grpo.paired_teacher_v2 import contrast


def test_paired_effect_preserves_task_matching_and_direction():
    a=[{'task_id':str(i),'em':v} for i,v in enumerate([0,1,0,0])]
    b=[{'task_id':str(i),'em':v} for i,v in enumerate([1,1,0,1])]
    result=contrast(a,b)
    assert result['delta']==.5 and result['wins']==2 and result['losses']==0
    assert contrast(b,a)['delta']==-.5
    assert contrast(a,a)['ci95']==[0.,0.]
    with pytest.raises(ContractError):contrast(a,b[::-1])


def test_source_parquet_preserves_nested_hotpot_fields(tmp_path):
    import pyarrow.parquet as pq
    from experiments.teacher_context_grpo.paired_teacher_v2 import write_rows
    rows=[{'id':'a','question':'Which?','context':{'title':['X'],'sentences':[['public fact']]}}]
    target=tmp_path/'train.parquet';write_rows(target,rows)
    assert pq.read_table(target).to_pylist()==rows
