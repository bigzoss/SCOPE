"""Private CPU ShopSimulator service. No hidden state leaves the snapshot store."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import uuid
from .public_data import project_transition_text
from .lifecycle import install_parent_guard


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shop-root",required=True)
    parser.add_argument("--longagent-root",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--train-products",required=True)
    parser.add_argument("--eval-products",required=True)
    parser.add_argument("--port",type=int,default=5112)
    parser.add_argument("--seed",type=int,default=233)
    parser.add_argument("--max-snapshots",type=int,default=512)
    parser.add_argument("--owner-pid",type=int)
    args=parser.parse_args()
    install_parent_guard(args.owner_pid)
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    os.environ.setdefault("OPENAI_API_KEY","unused-local-bm25")
    os.environ["OPENAI_BASE_URL"]="http://127.0.0.1:1/v1"
    root=Path(args.shop_root).resolve()
    sys.path[:0]=[str(root),str(root/"shop_env"),args.longagent_root]
    os.chdir(root/"shop_env")
    import numpy as np
    from flask import Flask,request,jsonify
    from longagent.environments.official_data import load_products,build_manifest,file_digest
    import web_agent_site.envs.web_agent_text_env as text_env
    from web_agent_site.engine.engine import init_search_engine
    def chinese_search(num_products=None):
        searcher=init_search_engine(num_products);searcher.set_language("zh");return searcher
    text_env.init_search_engine=chinese_search
    random.seed(args.seed);np.random.seed(args.seed)
    env=text_env.WebAgentTextEnv(observation_mode="text",split="train")
    rows=build_manifest(env.server.goals,load_products(args.train_products),load_products(args.eval_products))
    available={str(row["task_id"]):row for row in rows}
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest["train_sha256"]!=file_digest(args.train_products) or manifest["eval_sha256"]!=file_digest(args.eval_products):
        raise RuntimeError("Partition catalog identity mismatch")
    allowed=set()
    for row in manifest["tasks"]:
        tid=str(row["task_id"])
        if tid in allowed or tid not in available or row["goal_signature"]!=available[tid]["goal_signature"]:
            raise RuntimeError("Partition task/goal identity mismatch")
        allowed.add(tid)
    manifest_sha=file_digest(args.manifest)
    app=Flask(__name__)
    lease=None;task=None;last_public="";snapshots={}
    static={"server","browser","text_to_clickable"}

    def check_lease(value):
        if not lease or value!=lease:raise ValueError("Unknown lease")

    def public_page():
        actions=env.get_available_actions()
        clicks=[c for c in actions["clickables"] if c!="search"]
        return env.observation+f"\n\n搜索功能是否可用: {actions['has_search_bar']}\n\n可点击的按钮: "+json.dumps(clicks,ensure_ascii=False),clicks

    def capture():
        # Copy every mutable environment attribute, browser data, sessions and RNG.
        # Product catalog/searcher are immutable shared inputs, not copied per branch.
        return {"env":copy.deepcopy({k:v for k,v in env.__dict__.items() if k not in static}),
                "browser":copy.deepcopy({k:v for k,v in env.browser.__dict__.items() if k!="server"}),
                "sessions":copy.deepcopy(env.server.user_sessions),
                "assigned_instruction_text":copy.deepcopy(env.server.assigned_instruction_text),
                "random":random.getstate(),"numpy":np.random.get_state(),"task":task,"public":last_public}

    def state_hash(state):
        # Canonicalize values rather than hashing pickle's incidental object aliasing.
        def normalize(value):
            if isinstance(value,dict):return {str(k):normalize(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
            if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
            if isinstance(value,set):return sorted(normalize(v) for v in value)
            if isinstance(value,np.ndarray):return {"dtype":str(value.dtype),"array":value.tolist()}
            if isinstance(value,np.generic):return value.item()
            return value
        return hashlib.sha256(json.dumps(normalize(state),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    @app.errorhandler(Exception)
    def error(exc):
        app.logger.exception("SCOPE simulator request failed")
        return jsonify(error=str(exc)),400

    @app.get("/healthz")
    def health():
        return {"ready":True,"manifest_sha256":manifest_sha,"manifest_unchanged":file_digest(args.manifest)==manifest_sha,
                "allowed_tasks":len(allowed),"available":int(lease is None),"snapshot_count":len(snapshots),
                "restore_level":"native_private_state","pid":os.getpid()}

    @app.post("/reset")
    def reset():
        nonlocal lease,task,last_public
        tid=str(request.json["task_id"])
        if tid not in allowed:raise ValueError("Task outside locked partition")
        if lease is not None:raise ValueError("SCOPE environment already leased")
        env.server.user_sessions.clear()
        env.reset(idx=int(tid))
        task=tid;lease=uuid.uuid4().hex;last_public=env.instruction_text
        return {"lease":lease,"observation":last_public}

    @app.post("/step")
    def step():
        nonlocal last_public
        check_lease(request.json["lease"])
        command=str(request.json["command"])
        actions=env.get_available_actions()
        name,_,value=command.partition("[")
        value=value.rstrip("]").lower()
        invalid=(name not in {"search","click"} or (name=="click" and value not in actions["clickables"]) or
                 (name=="search" and not actions["has_search_bar"]))
        _,status,_=env.step(command)
        done=bool(status["done"])
        if done:
            last_public=project_transition_text("",True);clicks=[]
        else:
            last_public,clicks=public_page()
            last_public=project_transition_text(last_public,False)
        metrics=None
        if done:
            detail=status.get("reward_detail") or {}
            if "reward" not in status or not {"r_type","r_att","r_option","r_price"}<=detail.keys():raise RuntimeError("Missing terminal evaluator components")
            factors=[float(detail.get(k,default)) for k,default in (("r_type",0),("r_att",0),("r_option",1),("r_price",0))]
            strict=float(np.prod(factors))
            metrics={"success":float(all(x==1 for x in factors)),"r_strict":strict,
                     "r_loose":float(status.get("reward",0)),"purchased":1.}
        return {"observation":last_public,"done":done,"is_error":invalid,"metrics":metrics,"clickables":clicks}

    @app.post("/snapshot")
    def snapshot():
        check_lease(request.json["lease"])
        if len(snapshots)>=args.max_snapshots:raise ValueError("Snapshot capacity exhausted; release old handles")
        state=capture();handle=uuid.uuid4().hex
        snapshots[handle]=state
        return {"handle":handle,"state_hash":state_hash(state)}

    @app.post("/restore")
    def restore():
        nonlocal lease,task,last_public
        if lease is not None:raise ValueError("Release current lease before restore")
        state=copy.deepcopy(snapshots[request.json["handle"]])
        for key in list(env.__dict__):
            if key not in static:del env.__dict__[key]
        env.__dict__.update(state["env"])
        env.browser.__dict__.update(state["browser"])
        env.server.user_sessions=state["sessions"]
        env.server.assigned_instruction_text=state["assigned_instruction_text"]
        random.setstate(state["random"]);np.random.set_state(state["numpy"])
        task=state["task"];last_public=state["public"];lease=uuid.uuid4().hex
        env.get_available_actions() # Derived BeautifulSoup clickables, not hidden policy evidence.
        restored=capture()
        if state_hash(restored)!=state_hash(state):raise RuntimeError("Snapshot restoration changed private state")
        return {"lease":lease,"observation":last_public,"state_hash":state_hash(restored)}

    @app.post("/release")
    def release():
        nonlocal lease
        check_lease(request.json["lease"]);lease=None
        return {"released":True}

    @app.post("/drop_snapshot")
    def drop_snapshot():
        snapshots.pop(request.json["handle"],None)
        return {"released":True}

    app.run(host="127.0.0.1",port=args.port,threaded=False)


if __name__=="__main__":main()
