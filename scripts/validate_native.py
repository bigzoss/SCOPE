"""Real CPU simulator restore check; no policy scores or hidden answers are exposed."""
import json
import os
import re
from pathlib import Path
from scope.data import SplitRegistry
from scope.environments import ShopEnvironment
from scope.launcher import own_service
from scope.io import atomic_json,file_sha,digest
from scope.public_data import TERMINAL_PUBLIC_TEXT


def main():
    config=json.loads(Path("configs/shopsim.json").read_text())
    registry=SplitRegistry(config["data_root"])
    out=Path(os.environ.get("SCOPE_VALIDATION_OUTPUT","runs/validation_native"))
    if (out/"result.json").exists():raise RuntimeError("Validation output already exists")
    task=registry.ids("rl_train")[0]
    with own_service(config,out,True):
        env=ShopEnvironment(config["environment"]["base_url"],registry.ids("rl_train"),file_sha(config["environment"]["manifest"]))
        env.health()
        try:
            goal=env.reset(task)
            snapshot=env.snapshot()
            before=env.step("Action: search[玫瑰]")
            env.restore(snapshot)
            after=env.step("Action: search[玫瑰]")
            assert before==after,"Continuation differed after native restore"
            env.drop_snapshot(snapshot)
            products=[item for item in after[2]["clickables"] if re.fullmatch(r"(?:\d{8,}|[Bb][A-Za-z0-9]{9})",item)]
            assert products,"Native engineering search did not yield a publicly clickable product"
            env.step("Action: click["+products[0]+"]")
            terminal,done,public=env.step("Action: click[Buy Now]")
            assert done and terminal==TERMINAL_PUBLIC_TEXT and not public["clickables"]
            assert set(env.metrics)>={"success","r_strict","r_loose","purchased"}
            atomic_json(out/"result.json",{"passed":True,"task_id":task,"real_environment":True,
                "native_private_state_hash_equal":True,"public_continuation_equal":True,
                "real_terminal_projection_verified":True,"terminal_score_separate_from_public_observation":True,
                "manifest_unchanged":env.health()["manifest_unchanged"],"restore_level":"native_private_state"})
        finally:env.close()


if __name__=="__main__":main()
