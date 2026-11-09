"""Validate the effective recipe before allocating a model or starting a service."""
import copy
import math
from .contracts import ContractError
from .interaction import ACTION_MODES, LOOP_POLICIES

DEFAULT_RUNTIME={"max_turns":12,"actor_max_input":6144,"actor_max_new":256,"manager_max_input":2048,
    "manager_max_new":192,"manager_episode_tokens":1536,"total_tokens":100000,"retrievals":6,
    "probe_query_horizon":1,"sequence_memory_scope":"task","invalid_weight":.05,"cost_weight":.02,
    "action_mode":"legacy","loop_policy":"legacy","state_graph":False,"memory_choice":False,
    "compression_after":6,"compression_min_chars":1800,"shared_memory":False,"disabled_memory_tools":[],
    "decision_evidence":False,"decision_recovery":False,"compact_actor_view":False,"actor_history_turns":2}


def validate_config(source):
    cfg=copy.deepcopy(source)
    def integer(value,label,minimum=1):
        if type(value) is not int or value<minimum:raise ContractError(label+" must be an integer >= "+str(minimum))
    if cfg.get("environment",{}).get("kind","shopsim") not in {"shopsim","alfworld_http"}:raise ContractError("Unknown environment kind")
    for key in ("model","environment","runtime","training"):
        if not isinstance(cfg.get(key),dict):raise ContractError("Missing configuration section: "+key)
    if not cfg["model"].get("path") or not cfg["environment"].get("base_url") or not cfg["environment"].get("manifest"):
        raise ContractError("Model path, environment URL and manifest are required")
    integer(cfg["model"].get("rank",16),"model.rank");integer(cfg["model"].get("alpha",32),"model.alpha")
    if not str(cfg.get("gpu",2)).isdigit():raise ContractError("Exactly one physical GPU index is supported")
    runtime={**DEFAULT_RUNTIME,**cfg["runtime"]}
    known=set(DEFAULT_RUNTIME)|{"manager_mode","semantic_validation","disable_experience"}
    if set(runtime)-known:raise ContractError("Unknown runtime fields: "+str(sorted(set(runtime)-known)))
    for key in ("max_turns","actor_max_input","actor_max_new","manager_max_input","manager_max_new","total_tokens"):
        integer(runtime[key],"runtime."+key)
    for key in ("manager_episode_tokens","retrievals"):integer(runtime[key],"runtime."+key,0)
    if runtime["sequence_memory_scope"]!="task":raise ContractError("Sequence-scoped memory is not implemented")
    if runtime["probe_query_horizon"]!=1:raise ContractError("Task-scoped memory supports only current-query local credit (probe_query_horizon=1)")
    for key in ("invalid_weight","cost_weight"):
        if type(runtime[key]) not in (int,float) or not math.isfinite(runtime[key]) or runtime[key]<0:raise ContractError("Invalid reward coefficient "+key)
    for key in ("semantic_validation","disable_experience","state_graph","memory_choice","shared_memory","decision_evidence","decision_recovery","compact_actor_view"):
        if key in runtime and type(runtime[key]) is not bool:raise ContractError(key+" must be boolean")
    if runtime.get("manager_mode","learned") not in {"learned","rule"}:raise ContractError("Unknown manager mode")
    integer(runtime["actor_history_turns"], "runtime.actor_history_turns")
    if runtime["compact_actor_view"] and not runtime["decision_recovery"]:
        raise ContractError("Compact actor view requires decision recovery and its protected state")
    if runtime["action_mode"] not in ACTION_MODES:raise ContractError("Unknown runtime.action_mode")
    if runtime["memory_choice"] and not runtime["state_graph"]:raise ContractError("Memory choice requires state_graph")
    disabled = runtime["disabled_memory_tools"]
    if not isinstance(disabled, list) or any(x not in ("RETRIEVE","SUMMARIZE") for x in disabled) or len(disabled) != len(set(disabled)):
        raise ContractError("Unknown or duplicate disabled memory tools")
    if disabled and not runtime["memory_choice"]:
        raise ContractError("Tool ablations require memory_choice")
    if runtime["decision_evidence"] and not runtime["shared_memory"]:
        raise ContractError("Decision evidence requires shared_memory")
    if runtime["decision_recovery"] and (not runtime["decision_evidence"] or runtime["action_mode"] != "grounded" or runtime["loop_policy"] != "state"):
        raise ContractError("Decision recovery requires decision_evidence, grounded actions and state loops")
    if runtime["loop_policy"] not in LOOP_POLICIES:raise ContractError("Unknown runtime.loop_policy")
    integer(runtime["compression_after"],"runtime.compression_after",2)
    integer(runtime["compression_min_chars"],"runtime.compression_min_chars")
    train=cfg["training"]
    if train.get("method","conditional") not in {"global","window","conditional"}:raise ContractError("Unknown training method")
    credit = train.get("local_credit", "legacy")
    if credit not in {"legacy", "paired"}:raise ContractError("Unknown training.local_credit")
    if credit == "paired" and (train.get("method","conditional") != "conditional" or not runtime["state_graph"]):
        raise ContractError("Paired local credit requires conditional training and runtime.state_graph")
    hints = train.get("teacher_probe_hints")
    if hints is not None and (credit != "paired" or not isinstance(hints, dict) or
                             set(hints) != {"path", "sha256"} or not all(isinstance(v, str) and v for v in hints.values())):
        raise ContractError("Teacher probe hints require paired credit, path and sha256")
    integer(train.get("local_repeats", 1), "training.local_repeats")
    if train.get("local_continuation", "greedy") not in {"greedy", "paired_random"}:
        raise ContractError("Unknown training.local_continuation")
    cost = train.get("local_cost_weight", .05)
    if type(cost) not in (int, float) or not math.isfinite(cost) or not 0 <= cost <= 1:
        raise ContractError("training.local_cost_weight must be in [0, 1]")
    if train.get("role","manager") not in {"manager","actor","alternating"}:raise ContractError("Unknown training role")
    for key,default,minimum in (("updates",48,1),("group_size",4,2),("local_group_size",4,2),("alternating_window",8,1)):
        integer(train.get(key,default),"training."+key,minimum)
    for key,default in (("lr",1e-5),("kl_beta",.01),("local_weight",.25)):
        value=train.get(key,default)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or (key=="lr" and value==0):raise ContractError("Invalid "+key)
    if not 0<train.get("clip",.2)<1:raise ContractError("GRPO clip must be in (0,1)")
    curriculum=train.get("curriculum",[{"through":16,"length":1},{"through":32,"length":2},{"through":48,"length":4}])
    if not isinstance(curriculum,list) or not curriculum:raise ContractError("Empty curriculum")
    previous=0
    for part in curriculum:
        integer(part["through"],"curriculum.through");integer(part["length"],"curriculum.length")
        if part["through"]<=previous:raise ContractError("Curriculum boundaries must increase")
        previous=part["through"]
    if previous<train.get("updates",48):raise ContractError("Curriculum does not cover the configured updates")
    probes=train.get("probe_updates",[])
    opportunities=train.get("probe_opportunities",[])
    if not isinstance(opportunities,list) or any(value not in {"missing_evidence","public_confirmation_gap",
            "conflicting_evidence","context_threshold","decision_recovery","stalled_or_invalid_action"} for value in opportunities):
        raise ContractError("Unknown paired probe opportunity")
    if opportunities and (credit != "paired" or not runtime["decision_evidence"]):
        raise ContractError("Opportunity filters require paired credit and decision evidence")
    if len(probes)!=len(set(probes)):raise ContractError("Duplicate probe updates")
    for step in probes:integer(step,"probe update")
    integer(train.get("probe_turn",1),"probe_turn",0)
    if credit != "paired" and train.get("method","conditional")!="global" and train.get("probe_turn",1)>=runtime["max_turns"]:
        raise ContractError("Probe turn cannot be reached within the actor turn budget")
    integer(cfg.get("seed",233),"seed",0);integer(cfg.get("split_seed",233),"split_seed",0)
    evaluation=cfg.setdefault("evaluation",{})
    integer(evaluation.get("sequence_length",2),"evaluation.sequence_length")
    integer(evaluation.get("seed",233),"evaluation.seed",0)
    cfg["runtime"]=runtime
    cfg.setdefault("split_seed",233)
    return cfg
