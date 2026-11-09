"""Regression cases found during the module-by-module audit, not method-quality scores."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scope.atlas import StateAtlas
from scope.contracts import Scope,Budget,ContractError,Sample
from scope.tools import ToolExecutor,context_view
from scope.validation import Validator
from scope.warmup import build_examples
from scope.io import append_jsonl,atomic_json


def claim(atlas,record,predicate,value):
    return {"expected_revision":atlas.revision,"entity":record["entity"],"predicate":predicate,
            "value":value,"support_status":"supported","evidence_refs":[{"id":record["id"],"start":0,"end":len(record["text"])}]}


@pytest.mark.parametrize("text,predicate,value",[
    ("未确认已选颜色：黑色","selected_color","黑色"),
    ("价格高于358元","price",358),
    ("价格不是358元","price",358),
])
def test_negative_or_bound_evidence_cannot_confirm(text,predicate,value):
    atlas=StateAtlas(Scope("train","1","ep"));record=atlas.append_public(text,"observation","product:1")
    with pytest.raises(ContractError):Validator().check(atlas,claim(atlas,record,predicate,value))
    atlas.close()


def test_requirement_extraction_cannot_drop_negation():
    atlas=StateAtlas(Scope("train","1","ep"));record=atlas.append_public("不要红色，需要黑色","requirement")
    with pytest.raises(ContractError):Validator().check(atlas,claim(atlas,record,"requirement","红色"))
    atlas.close()


def test_fetch_reaches_actor_and_summary_survives_unrelated_audit_event():
    atlas=StateAtlas(Scope("train","1","ep"));tools=ToolExecutor();working={}
    text="x"*800+"需要保留的历史证据"
    record=atlas.append_public(text)
    tools.execute(atlas,working,{"operation":"FETCH_EVIDENCE","expected_revision":atlas.revision,
        "evidence_id":record["id"],"start":800,"end":len(text)},"context",Budget(),"goal")
    assert "需要保留的历史证据" in json.dumps(context_view(atlas,working,"goal"),ensure_ascii=False)
    tools.execute(atlas,working,{"operation":"SUMMARIZE","expected_revision":atlas.revision,
        "items":[{"text":text[800:],"evidence_ref":{"id":record["id"],"start":800,"end":len(text)}}]},"context",Budget(),"goal")
    atlas.append_public("下一轮的新页面")
    assert context_view(atlas,working,"goal")["summary"] is not None
    atlas.close()


def test_public_evidence_reserved_metadata_cannot_be_overridden():
    atlas=StateAtlas(Scope("dev","1","ep"))
    with pytest.raises(ContractError):atlas.append_public("text",scope={"split":"train"},revision=-1)
    atlas.close()


def test_budget_failure_is_transactional():
    budget=Budget(manager_tokens=1,total_tokens=10)
    before=copy.deepcopy(budget)
    with pytest.raises(ContractError):budget.charge("manager",2,2)
    assert budget==before


def test_terminal_evaluator_text_cannot_enter_sft_and_partial_corpus_is_not_published(tmp_path):
    source=tmp_path/"public.jsonl";target=tmp_path/"examples.jsonl"
    append_jsonl(source,{"task_id":"1","evidence":[{"kind":"requirement","text":"need black"},
        {"kind":"observation","text":"Target [SEP] asin [SEP] SECRET [SEP] Your score (min 0.0, max 1.0)"}]})
    with pytest.raises(ContractError):build_examples(source,target,["1"])
    assert not target.exists()


def test_independent_future_tasks_do_not_become_local_memory_credit():
    from scope.training import local_rollouts
    from scope.contracts import Boundary
    b=Boundary(Scope("train","1","ep"),{}, {},Budget(),"goal","obs",[],"context",{},None,{},0)
    calls=[]
    class Runner:
        cfg={"probe_query_horizon":4}
        def run(self,task,*args,**kwargs):
            calls.append(str(task))
            sample=Sample("manager","context",[1],[2],[0.],"{}",0,"boundary")
            return SimpleNamespace(samples=[sample],reward=0.,metrics={"success":0.,"r_loose":0.})
    local_rollouts(Runner(),b,["2","3"],2,"conditional")
    assert calls==["1","1"]


def test_sft_isolation_group_overlap_is_detected(tmp_path):
    from scope.data import SplitRegistry
    parts={"grpo/train.jsonl":[{"task_id":"1","isolation_group":"overlap","domain":"x"}],
           "grpo/dev.jsonl":[{"task_id":"2","isolation_group":"dev","domain":"x"}],
           "test/unseen.jsonl":[{"task_id":"3","isolation_group":"test","domain":"x"}],
           "sft/train_tasks.jsonl":[{"task_id":"4","isolation_group":"overlap"}],"sft/validation_tasks.jsonl":[]}
    for name,rows in parts.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ContractError):SplitRegistry(tmp_path)


def test_graph_expansion_reserves_capacity_and_respects_relation_filter():
    from scope.retrieval import Retriever
    atlas=StateAtlas(Scope("dev","1","ep"))
    linked=atlas.append_public("hidden historical option","observation","product:1")
    atlas.append_public("needle needle","observation","product:1")
    atlas.append_public("needle unrelated","observation","product:2")
    search=Retriever()
    assert linked["id"] not in [r["id"] for r in search.search(atlas,"needle",top_k=2)]
    expanded=search.search(atlas,"needle",top_k=2,max_hops=2)
    assert any(r["id"]==linked["id"] and len(r["retrieval_path"])>1 for r in expanded)
    assert linked["id"] not in [r["id"] for r in search.search(atlas,"needle",top_k=2,max_hops=2,relation_types=[])]
    atlas.close()


def test_terminal_projection_never_exposes_evaluator_fields():
    from scope.public_data import project_transition_text,TERMINAL_PUBLIC_TEXT
    assert project_transition_text("Your score (min 0.0) SECRET",True)==TERMINAL_PUBLIC_TEXT
    with pytest.raises(ContractError):project_transition_text("Reward Details: hidden",False)


def test_action_must_be_an_explicit_single_command():
    from scope.environments import parse_action
    assert parse_action("Thought: maybe search[unexecuted]") is None
    assert parse_action("Action: click[a]\nAction: click[b]") is None
    assert parse_action("Action: search[x] trailing garbage") is None
    assert parse_action("Thought: consider another item\nAction: search[black]")=="search[black]"
    assert parse_action("click[Buy Now]")=="click[Buy Now]"


def test_memory_update_supersedes_atomically_and_cannot_retire_entity():
    atlas=StateAtlas(Scope("train","1","ep"));tools=ToolExecutor();working={}
    original=atlas.append_public("价格：358元","observation","product:1")
    proposal={**claim(atlas,original,"price",358),"operation":"ADD_MEMORY","proposal_id":"first"}
    tools.execute(atlas,working,proposal,"context",Budget(),"goal")
    old=atlas.nodes("Claim")[0]
    newer=atlas.append_public("价格：438元","observation","product:1")
    update={**claim(atlas,newer,"price",438),"operation":"UPDATE_MEMORY","node_id":old["id"],"proposal_id":"second"}
    tools.execute(atlas,working,update,"context",Budget(),"goal")
    assert not atlas.node(old["id"])["active"]
    assert len(atlas.nodes("Claim"))==1
    assert atlas.db.execute("SELECT count(*) FROM edges WHERE relation='SUPERSEDES'").fetchone()[0]==1
    revision=atlas.revision
    assert tools.execute(atlas,working,update,"context",Budget(),"goal")["idempotent"]
    assert atlas.revision==revision
    with pytest.raises(ContractError):tools.execute(atlas,working,{"operation":"RETIRE_MEMORY","node_id":"product:1",
        "expected_revision":atlas.revision},"context",Budget(),"goal")
    atlas.close()


def test_config_validation_rejects_unsupported_modes():
    from scope.config import validate_config
    base=json.loads((Path(__file__).resolve().parents[1]/"configs/shopsim.json").read_text())
    assert validate_config(base)["split_seed"]==233
    for section,key,value in [("runtime","probe_query_horizon",2),("runtime","semantic_validation","false"),
            ("runtime","total_tokens",-1),("environment","kind","typo"),("training","method","ppo")]:
        bad=copy.deepcopy(base);bad[section][key]=value
        with pytest.raises(ContractError):validate_config(bad)
    bad=copy.deepcopy(base);bad["gpu"]="2,3"
    with pytest.raises(ContractError):validate_config(bad)


def test_evaluation_pairing_rejects_duplicates_and_different_budgets(tmp_path):
    from scope.evaluation import paired_comparison
    from scope.io import atomic_jsonl,file_sha
    def write(name,rows,budget):
        path=tmp_path/name
        atomic_jsonl(path/"results.jsonl",rows)
        atomic_json(path/"summary.json",{"complete":True,"n":len(rows),"expected":len(rows),
            "results_sha256":file_sha(path/"results.jsonl"),"evaluation_contract":{"budget":budget}})
        return path
    row={"task_id":"1","sequence_id":0,"metrics":{"success":1}}
    left=write("left",[row],256);duplicate=write("duplicate",[row,row],256)
    mismatch=write("mismatch",[row],128)
    with pytest.raises(ContractError,match="Duplicate"):paired_comparison(left,duplicate,repeats=10)
    with pytest.raises(ContractError,match="budgets differ"):paired_comparison(left,mismatch,repeats=10)


def test_cached_evaluation_checksum_detects_tampered_scores(tmp_path,monkeypatch):
    import scope.evaluation as ev
    from scope.io import digest
    class Policy:
        def fingerprint(self):return "frozen"
    class Registry:
        def evaluation_sequences(self,*args):return [["1"]]
    monkeypatch.setattr(ev,"EpisodeRunner",lambda *args:None)
    config={"runtime":{},"evaluation":{}}
    contract=ev.evaluation_contract(config,"dev",[["1"]])
    record={"task_id":"1","sequence_id":0,"metrics":{"success":0}}
    saved={"tasks":["1"],"parameter_hash":"frozen","contract_sha256":digest(contract),
        "records":[record],"records_sha256":digest([record])}
    saved["records"][0]["metrics"]["success"]=1
    atomic_json(tmp_path/"sequences/sequence_00000.json",saved)
    with pytest.raises(ContractError,match="checksum"):ev.evaluate(Policy(),None,Registry(),config,tmp_path)


def test_linux_service_parent_guard_ends_owned_child(tmp_path):
    import os,subprocess,sys,time
    if not sys.platform.startswith("linux"):pytest.skip("Linux server ownership mechanism")
    ready=tmp_path/"ready.json";release=tmp_path/"release"
    child="from scope.lifecycle import install_parent_guard;import os,sys,time,pathlib;install_parent_guard(int(sys.argv[1]));pathlib.Path(sys.argv[2]).write_text(str(os.getpid()));time.sleep(30)"
    parent="import subprocess,sys,os,time,pathlib;subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);\nwhile not pathlib.Path(sys.argv[3]).exists():time.sleep(.02)"
    controller=subprocess.Popen([sys.executable,"-c",parent,child,str(ready),str(release)])
    pid=None
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists();pid=int(ready.read_text());release.touch();controller.wait(timeout=3)
        while time.monotonic()<deadline:
            stat=Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2]=="Z":break
            time.sleep(.02)
        else:pytest.fail("Owned child survived controller exit")
    finally:
        if controller.poll() is None:controller.terminate();controller.wait(timeout=3)
        if pid and Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2]!="Z":os.kill(pid,15)
