import copy
import json
from scope.contracts import Sample
from scope.environments import DeterministicTestEnvironment
from scope.rollout import EpisodeRunner
from scope.io import canonical


class ScriptedPolicy:
    versions={"manager":0,"actor":0}
    def encode(self,messages):return [1]*max(1,len(canonical(messages))//10)
    def sample(self,messages,role,phase,boundary,max_input,max_new,greedy=False):
        if role=="manager":
            payload=json.loads(messages[-1]["content"])
            text=json.dumps({"operation":"FINISH","expected_revision":payload["revision"]})
        else:
            text="Action: click[Buy Now]" if "价格：358" in messages[-1]["content"] else "Action: search[黑色]"
        return Sample(role,phase,self.encode(messages),[2],[0.],text,0,boundary)


def test_real_pipeline_contract_with_explicit_fixture_and_native_boundary():
    env=DeterministicTestEnvironment()
    runner=EpisodeRunner(ScriptedPolicy(),env,{"max_turns":3})
    original=runner.run("1","one",greedy=True,capture_phase="context",capture_turn=1)
    assert original.metrics["success"]==1
    assert len(original.boundaries)==1
    repeated=runner.run("1","one",greedy=True,boundary=original.boundaries[0])
    assert repeated.metrics==original.metrics
    assert repeated.costs["actor_turns"]==1
    assert repeated.final_atlas==original.final_atlas
    assert "r_loose" not in canonical(original.final_atlas)
    assert "goal_options" not in canonical(original.final_atlas)

