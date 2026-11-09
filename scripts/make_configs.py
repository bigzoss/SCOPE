"""Write matched B4/B5/B6 configurations without changing the shared recipe."""
import argparse
import json
from pathlib import Path
from scope.io import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base",default="configs/shopsim.json")
    parser.add_argument("--directory",default="configs/experiments")
    parser.add_argument("--warmstart",required=True)
    parser.add_argument("--seed",type=int,default=233)
    args=parser.parse_args()
    if not (Path(args.warmstart)/"metadata.json").exists():parser.error("warmstart checkpoint does not exist")
    for method in ("global","window","conditional"):
        config=json.loads(Path(args.base).read_text())
        config["seed"]=args.seed;config["warmstart"]=str(Path(args.warmstart).resolve())
        config["training"]["method"]=method
        path=Path(args.directory)/(method+".json")
        if path.exists():raise FileExistsError("Refusing to overwrite "+str(path))
        atomic_json(path,config)


if __name__=="__main__":main()
