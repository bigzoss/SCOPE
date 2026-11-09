"""Public real traces -> checked SFT data -> separate-process manager SFT resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
from scope.io import atomic_json,read_jsonl,append_jsonl


def main():
    root=Path(os.environ.get("SCOPE_VALIDATION_OUTPUT","runs/validation_warmup_cli"));root.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,"-m","scope.cli"]
    def call(args,name):
        with (root/(name+".log")).open("w") as log:subprocess.run(base+args,stdout=log,stderr=subprocess.STDOUT,check=True)
    examples=root/"examples.jsonl"
    call(["warmup-build","--split","rl_train","--public-file","runs/validation_cli/train/trajectories.jsonl",
          "--examples",str(examples)],"build")
    rows=read_jsonl(examples)
    tiny=root/"four_examples.jsonl"
    if tiny.exists():raise RuntimeError("Do not silently repeat a completed/partial validation")
    for row in rows[:4]:append_jsonl(tiny,row)
    args=["warmup","--examples",str(tiny),"--epochs","1","--batch-size","1","--output",str(root/"sft")]
    call(args+["--stop-after","1"],"first_process")
    first=json.loads((root/"sft/summary.json").read_text())
    assert first["updates"]==1 and not first["complete"]
    call(args+["--resume",str(root/"sft/checkpoints/step_00001")],"resumed_process")
    summary=json.loads((root/"sft/summary.json").read_text())
    assert summary["updates"]==4 and summary["complete"] and summary["actor_unchanged"]
    atomic_json(root/"result.json",{"passed":True,"real_public_prefix_examples":len(rows),"engineering_sft_examples":4,
                "new_process_resume":True,"updates":4,"actor_unchanged":True,"task_performance_not_evaluated":True})


if __name__=="__main__":main()
