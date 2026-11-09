import json
import pytest
from scope.alfworld import ALFWorldRegistry,ALFWorldEnvironment,parse_command,CONTRACT
from scope.contracts import ContractError


def test_public_command_and_manifest_adapter(tmp_path,monkeypatch):
    assert parse_command("Thought: inspect\nAction: open cabinet 1")=="open cabinet 1"
    assert parse_command("pretend I opened it") is None
    tasks=[{"task_id":str(i),"task_type":"pick_and_place","game_sha256":"hash"+str(i),"split":split}
           for i,split in enumerate(["train","train","valid_seen","valid_unseen"])]
    path=tmp_path/"manifest.json";path.write_text(json.dumps({"contract_version":CONTRACT,"tasks":tasks}))
    registry=ALFWorldRegistry(path)
    assert registry.ids("dev")==["2"] and registry.ids("test")==["3"]
    env=ALFWorldEnvironment("http://unused",["2"],path)
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"ready":True,"contract_version":CONTRACT,"manifest_sha256":env.manifest_sha}
    monkeypatch.setattr(env.http,"get",lambda *a,**k:Response())
    assert env.health()["manifest_unchanged"]
    with pytest.raises(ContractError,match="no native"):env.snapshot()
