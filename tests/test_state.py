import copy
import pytest
from scope.atlas import StateAtlas
from scope.contracts import ContractError,Scope,Budget
from scope.tools import ToolExecutor
from scope.validation import Validator
from scope.retrieval import Retriever


def proposal(atlas,record,predicate,value,status="supported",entity="product:123"):
    return {"operation":"PROPOSE_STATE_UPDATE","expected_revision":atlas.revision,"entity":entity,
            "predicate":predicate,"value":value,"support_status":status,
            "evidence_refs":[{"id":record["id"],"start":0,"end":len(record["text"])}]}


def test_support_is_not_reference_existence_and_scope_is_hard():
    atlas=StateAtlas(Scope("train","1","episode1"))
    record=atlas.append_public("商品123，黑色可选，价格：358元。","observation","product:123",clickables=["黑色"])
    validator=Validator()
    assert validator.check(atlas,proposal(atlas,record,"price",358))["support_status"]=="supported"
    with pytest.raises(ContractError,match="DOES_NOT_SUPPORT"):
        validator.check(atlas,proposal(atlas,record,"selected_color","黑色"))
    assert validator.check(atlas,proposal(atlas,record,"selected_color","黑色","pending"))["support_status"]=="pending"
    other=StateAtlas(Scope("dev","1","episode2"))
    other.append_public("另一商品","observation","product:123")
    with pytest.raises(ContractError):validator.check(other,proposal(other,record,"price",358))
    with pytest.raises(ContractError):atlas.append_public("x",reward_detail={"r_type":1})
    atlas.close();other.close()


def test_transactions_conflicts_idempotence_and_restore():
    atlas=StateAtlas(Scope("train","1","ep"));v=Validator()
    source=atlas.append_public("价格：358元","observation","product:123")
    checked=v.check(atlas,proposal(atlas,source,"price",358))
    revision=atlas.revision
    atlas.commit(checked,revision,"p1")
    assert atlas.commit(checked,revision,"p1")["idempotent"]
    view=atlas.save_view([source["id"]])
    changed=atlas.append_public("价格：438元","observation","product:123")
    with pytest.raises(ContractError,match="STALE"):atlas.view(view["id"])
    checked2=v.check(atlas,proposal(atlas,changed,"price",438))
    atlas.commit(checked2,atlas.revision,"p2")
    assert not any(c.get("conflicted") for c in atlas.nodes("Claim"))
    assert atlas.db.execute("SELECT count(*) FROM edges WHERE relation='DIFFERS_FROM'").fetchone()[0]==1
    snapshot=atlas.export();clone=StateAtlas.restore(snapshot)
    assert clone.fingerprint()==atlas.fingerprint()
    with pytest.raises(ContractError,match="STALE"):clone.commit(checked2,0,"new")
    assert clone.fingerprint()==atlas.fingerprint()
    atlas.close();clone.close()


def test_extract_summary_preserves_source_and_stale_updates_do_not_commit():
    atlas=StateAtlas(Scope("train","1","ep"));tools=ToolExecutor()
    source=atlas.append_public("不需要红色，需要黑色","requirement")
    summary={"operation":"SUMMARIZE","expected_revision":atlas.revision,"items":[{
        "text":"需要红色","evidence_ref":{"id":source["id"],"start":0,"end":5}}]}
    with pytest.raises(ContractError,match="exact source"):tools.execute(atlas,{},summary,"context",Budget(),source["text"])
    update=proposal(atlas,source,"requirement",source["text"],entity="task")
    checked=tools.validator.check(atlas,update)
    atlas.append_public("新页面")
    before=atlas.fingerprint()
    with pytest.raises(ContractError):atlas.commit(checked,update["expected_revision"],"late")
    assert before==atlas.fingerprint()
    assert not Retriever().search(atlas,"黑色",entity="product:other")
    atlas.close()


def test_exact_fetch_beyond_prompt_prefix_and_view_use_provenance():
    atlas=StateAtlas(Scope("train","1","ep"));tools=ToolExecutor();budget=Budget()
    text="x"*900+"公开规格黑色"
    source=atlas.append_public(text)
    proposal={"operation":"FETCH_EVIDENCE","expected_revision":atlas.revision,"evidence_id":source["id"],"start":900,"end":len(text)}
    result=tools.execute(atlas,{},proposal,"context",budget,"goal")
    assert result["evidence"]["text"]=="公开规格黑色" and budget.retrievals==5
    view=atlas.save_view([source["id"]])
    event=atlas.record_use(view["id"],"Action: search[test]",{"actor":0})
    assert atlas.node(event["id"])["status"]=="proposed_action"
    assert atlas.db.execute("SELECT count(*) FROM edges WHERE relation='USED_IN'").fetchone()[0]==1
    with pytest.raises(ContractError):atlas.view(view["id"])
    atlas.close()


def test_semantic_ablation_does_not_disable_scope_checks():
    atlas=StateAtlas(Scope("train","1","ep"))
    record=atlas.append_public("黑色可选","observation","product:123")
    accepted=Validator(False).check(atlas,proposal(atlas,record,"selected_color","黑色"))
    assert accepted["validation_mode"]=="semantic_check_ablated"
    record["id"]="out_of_scope"
    with pytest.raises(ContractError):Validator(False).check(atlas,proposal(atlas,record,"selected_color","黑色"))
    atlas.close()


def test_price_units_types_and_simultaneous_ambiguity():
    atlas=StateAtlas(Scope("train","1","ep"));v=Validator()
    record=atlas.append_public("价格：358元；页面另一价格：438元","observation","product:123")
    first=proposal(atlas,record,"price",358)
    assert v.check(atlas,first)["unit"]=="CNY"
    bad={**first,"unit":"USD"}
    with pytest.raises(ContractError,match="UNIT"):v.check(atlas,bad)
    with pytest.raises(ContractError,match="boolean"):v.check(atlas,{**first,"value":True})
    atlas.commit(v.check(atlas,first),atlas.revision,"a")
    atlas.commit(v.check(atlas,proposal(atlas,record,"price",438)),atlas.revision,"b")
    assert all(c["conflicted"] for c in atlas.nodes("Claim"))
    atlas.close()

