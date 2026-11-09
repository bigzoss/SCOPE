"""Read-only diagnostic replay; this is not a new environment evaluation."""
import argparse
import json
from pathlib import Path

from scope.alfworld import ALFWorldActions
from scope.interaction import record_progress
from scope.io import atomic_json, file_sha


def audit(root):
    report={"type":"read_only_historical_trace_audit", "new_environment_rollouts":0,
            "domains":{}, "input_sha256":{}}
    for domain in ("alfworld", "shopsimulator"):
        paths=sorted((root/domain/"recovery/sequences").glob("*.json"))
        old_loops=new_loops=invalid_cycles=count=0
        for path in paths:
            report["input_sha256"][str(path.relative_to(root))]=file_sha(path)
            for row in json.loads(path.read_text(encoding="utf-8"))["records"]:
                count+=1
                window=[]
                for event in row["history"]:
                    if event["phase"]!="execute":continue
                    old_loops+=bool(event.get("loop"))
                    _,cycle=record_progress(window,event["public_state_before"],event["public_state_after"],
                        event.get("normalized_command") or event.get("raw_command"),event["observation_before_id"],
                        not event["is_error"],event.get("progress")=="terminal")
                    new_loops+=bool(cycle)
                    invalid_cycles+=bool(cycle) and event["action_status"]!="accepted"
                if domain=="alfworld":
                    for obs in row["public_evidence"]:
                        if obs["kind"]=="observation" and obs["text"].startswith("You arrive at desk 1.") and "alarmclock 2" in obs["text"]:
                            adapter=ALFWorldActions()
                            state=adapter.observe(obs["text"],{},None,[])
                            report["alarmclock_location"]=state.get("object_locations",{}).get("alarmclock 2")
                            report["alarmclock_pickup_candidate"]=any(c["command"]=="take alarmclock 2 from desk 1"
                                for c in adapter.recovery_candidates(state,"look at alarmclock under the desklamp"))
        report["domains"][domain]={"tasks":count,"logged_old_loops":old_loops,
            "recomputed_loops":new_loops,"repeated_invalid_cycles":invalid_cycles}
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.run)
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!="input_sha256"},indent=2))
