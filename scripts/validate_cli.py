"""Actual entrypoint: stop after update1, new-process resume to update2, frozen evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from scope.io import atomic_json


def main():
    root=Path(os.environ.get("SCOPE_VALIDATION_OUTPUT","runs/validation_cli"))
    if (root/"train").exists():raise RuntimeError("Validation output already exists; choose a fresh SCOPE_VALIDATION_OUTPUT")
    root.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path("configs/shopsim.json").read_text())
    config["runtime"].update(max_turns=2,actor_max_new=32,manager_max_new=24,probe_query_horizon=1)
    config["training"].update(updates=2,group_size=2,local_group_size=2,probe_updates=[1,2],probe_turn=0,
                              curriculum=[{"through":2,"length":1}])
    atomic_json(root/"config.json",config)
    base=[sys.executable,"-m","scope.cli"]
    def call(arguments,name):
        with (root/(name+".log")).open("w") as log:
            subprocess.run(base+arguments,stdout=log,stderr=subprocess.STDOUT,check=True)
    train=["train","--config",str(root/"config.json"),"--output",str(root/"train"),"--manage-service"]
    call(train+["--stop-after","1"],"first_process")
    assert json.loads((root/"train/status.json").read_text())["completed_updates"]==1
    call(train+["--resume",str(root/"train/checkpoints/step_00001")],"resumed_process")
    status=json.loads((root/"train/status.json").read_text())
    assert status["phase"]=="complete" and status["completed_updates"]==2
    call(["evaluate","--config",str(root/"config.json"),"--output",str(root/"eval"),"--manage-service",
          "--checkpoint",str(root/"train/checkpoints/step_00002"),"--split","rl_train","--limit","2"],"evaluation")
    summary=json.loads((root/"eval/summary.json").read_text())
    assert summary["complete"] and summary["n"]==2 and summary["parameters_unchanged_during_eval"] and summary["memory_unchanged"]
    atomic_json(root/"result.json",{"passed":True,"engineering_only":True,"train_updates":2,"new_process_resume":True,
              "evaluation_tasks":2,"split":"rl_train","partial_generation_budgets":True,
              "frozen_evaluation":True,"not_a_benchmark_success_estimate":True})


if __name__=="__main__":main()
