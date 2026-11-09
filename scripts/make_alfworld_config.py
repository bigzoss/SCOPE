"""Frozen transfer configuration using the existing official public HTTP bridge."""
import argparse
import json
from pathlib import Path
from scope.io import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--url",default="http://127.0.0.1:5088")
    p.add_argument("--output",default="configs/alfworld.json")
    args=p.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    config=json.loads(Path("configs/shopsim.json").read_text())
    config["environment"]={"kind":"alfworld_http","base_url":args.url,"manifest":args.manifest}
    config["runtime"]["max_turns"]=30
    config["training"]["method"]="global"
    atomic_json(target,config)


if __name__=="__main__":main()
