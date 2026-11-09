"""Real environment rollout, public state pipeline and restorable management boundaries."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
import random
import uuid
import time
from .atlas import StateAtlas
from .contracts import Boundary, Budget, ContractError, Outcome, Scope
from .io import canonical, digest
from .interaction import ActionAdapter, execute_action, interaction_guidance, record_progress
from .tools import ToolExecutor, SCHEMA_HELP, context_view, parse_tool, public_snippet

ACTOR_SYSTEM = ("你是购物助手。依据用户需求和页面观察选择商品。每轮只执行一个操作，格式为 Thought: 简要理由\n"
                "Action: search[关键词] 或 click[可点击按钮]。只能点击页面提供的按钮，购买前选择所需规格。"
                "购买时点击 click[Buy Now]。不要编造页面内容。Thought保持简短，不超过32 tokens。"
                "尝试动作不等于已确认状态；结合多轮历史重新决策，不能机械重复操作。")


def capture_rng():
    import torch
    import numpy as np
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    import numpy as np
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if state["cuda"]:torch.cuda.set_rng_state_all(state["cuda"])


def manager_messages(atlas, observation, working, goal, phase, budget):
    evidence = [public_snippet(e,160) for e in atlas.all_evidence()[-2:]]
    for snippet in evidence:
        source=atlas.evidence(snippet["id"])["text"]
        if len(source)>160:snippet["tail"]={"span_start":max(0,len(source)-80),"text":source[-80:]}
    claims = [{k:c[k] for k in ("id","entity","predicate","value","support_status")}
              for c in atlas.nodes("Claim")]
    for item in claims:
        if item["predicate"]=="requirement":item.pop("value");item["goal_ref"]="goal"
        elif isinstance(item["value"],str) and len(item["value"])>96:
            value=item.pop("value");item.update(value_excerpt=value[:96],value_length=len(value))
    last=copy.deepcopy(working.get("last_tool_result"))
    if last and "retrieved" in last:
        last["candidate_ids"]=[x["id"] for x in last["retrieved"]]
        last["retrieved"]=[{**x,"text":x.get("text","")[:120]} for x in last["retrieved"][:4]]
    payload={"phase":phase,"goal":goal,"revision":atlas.revision,"claims":claims,
             "recent_evidence":evidence,"working_selection":working.get("selected_ids",[]),
             "last_tool_result":last,"fetched_span":working.get("fetched_span"),
             "remaining":asdict(budget)}
    if working.get("interaction_feedback"):
        payload["interaction_feedback"] = working["interaction_feedback"]
    return [{"role":"system","content":"You manage public evidence and memory, not shopping actions.\n"+SCHEMA_HELP},
            {"role":"user","content":canonical(payload)}]


class EpisodeRunner:
    def __init__(self,policy,env,config,split="train",experience=None):
        self.policy,self.env,self.cfg,self.split=policy,env,config,split
        self.tools=ToolExecutor(None if config.get("disable_experience") else experience,config.get("semantic_validation",True))
        self.experience=experience
        self.action_adapter = getattr(env, "action_adapter", None)
        if self.action_adapter is None and config.get("action_mode", "legacy") != "legacy":
            raise ContractError("Action normalization and binding require a benchmark action_adapter")
        self.public_projection = self.action_adapter or ActionAdapter()

    def _manager(self,atlas,observation,working,goal,budget,phase,greedy,boundary,costs,forced=None):
        self.last_tool_seconds=0.
        if budget.manager_tokens<=0 or budget.total_tokens<=16:
            return None,{"accepted":True,"finish":True,"reason":"manager_budget_exhausted"}
        if self.cfg.get("manager_mode")=="rule":
            return None,{"accepted":True,"finish":True,"reason":"rule_FINISH"}
        messages=manager_messages(atlas,observation,working,goal,phase,budget)
        max_input=self.cfg.get("manager_max_input",2048)
        if len(self.policy.encode(messages))>max_input:
            # Refuse to silently drop the goal or pending states. Hand back to the actor.
            return None,{"accepted":True,"finish":True,"reason":"manager_input_budget"}
        n_input=len(self.policy.encode(messages))
        room=min(self.cfg.get("manager_max_new",192),budget.manager_tokens,budget.total_tokens-n_input)
        if room<1:return None,{"accepted":True,"finish":True,"reason":"total_budget"}
        sample=forced or self.policy.sample(messages,"manager",phase,boundary,max_input,room,greedy)
        budget.charge("manager",len(sample.prompt_ids),len(sample.completion_ids))
        costs["manager_input"]+=len(sample.prompt_ids);costs["manager_output"]+=len(sample.completion_ids)
        try:
            tool_start=time.perf_counter()
            proposal=parse_tool(sample.text)
            result=self.tools.execute(atlas,working,proposal,phase,budget,goal)
            if proposal.get("operation")=="SUMMARIZE":working["compressed"]=True
        except (ContractError,ValueError,KeyError,TypeError) as exc:
            result={"accepted":False,"reason":str(exc),"operation":"rejected"}
        self.last_tool_seconds=time.perf_counter()-tool_start
        return sample,result

    def _actor_messages(self,atlas,working,goal,observation,history):
        view=context_view(atlas,working,goal)
        view["claims"]=[{k:v for k,v in c.items() if k in {"entity","predicate","value","support_status","conflicted","observed_at_revision","validity"}}
                        for c in view["claims"]]
        view["experience"]=working.get("experiences",[])
        history=list(history[-2:] if view["summary"] else history)
        limit=self.cfg.get("actor_max_input",6144)
        current=observation
        while True:
            messages=[{"role":"system","content":getattr(self.env,"actor_system",ACTOR_SYSTEM)},{"role":"user","content":goal}]
            for record in history:
                messages.extend([{"role":"assistant","content":record["response"]},
                                 {"role":"user","content":record["observation"]}])
            content="Public state and memory:\n"+canonical(view)+"\nCurrent observation:\n"+current
            if working.get("warning"):content+="\n"+working["warning"]
            if history:messages[-1]["content"]=content
            else:messages[-1]["content"]+="\n"+content
            if len(self.policy.encode(messages))<=limit:
                sources=list(working.get("selected_ids",[]))
                if working.get("fetched_span"):sources.append(working["fetched_span"]["id"])
                if view["summary"]:sources.extend(view["summary"]["source_hashes"])
                saved=atlas.save_view(sorted(set(sources))+[c["id"] for c in atlas.nodes("Claim")],view["summary"])
                return messages,digest(view),saved["id"]
            if history:history.pop(0)
            elif len(current)>600:
                half=max(150,len(current)//4)
                current=current[:half]+"\n[page middle omitted; complete source remains retrievable]\n"+current[-half:]
            else:raise ContractError("Cannot preserve goal and public state within actor input budget")

    def run(self,task_id,episode_id,greedy=False,capture_phase=None,capture_turn=1,boundary=None,forced=None):
        cfg=self.cfg
        action_mode = cfg.get("action_mode", "legacy")
        loop_policy = cfg.get("loop_policy", "legacy")
        samples=[];events=[];boundaries=[]
        if boundary:
            if boundary.role_versions!=self.policy.versions:
                raise ContractError("Continuation policy versions changed")
            observation=self.env.restore(boundary.environment)
            atlas=StateAtlas.restore(boundary.atlas)
            goal=boundary.goal;budget=copy.deepcopy(boundary.budget);history=copy.deepcopy(boundary.history)
            working=copy.deepcopy(boundary.working);costs=copy.deepcopy(boundary.costs);invalid=boundary.invalid
            phase_start=boundary.phase
            start_turn=cfg.get("max_turns",12)-budget.turns
            prefix_cost=sum(costs.values())
        else:
            observation=self.env.reset(task_id)
            goal=getattr(self.env,"task_goal",observation)
            atlas=StateAtlas(Scope(self.split,str(task_id),str(episode_id)))
            requirement=atlas.append_public(goal,"requirement")
            self.tools.execute(atlas,{}, {"operation":"PROPOSE_STATE_UPDATE","expected_revision":atlas.revision,
                    "entity":"task","predicate":"requirement","value":goal,"support_status":"supported",
                    "evidence_refs":[{"id":requirement["id"],"start":0,"end":len(goal)}]},"state",Budget(),goal)
            initial = atlas.append_public(observation, "observation") if observation != goal else requirement
            budget=Budget(cfg.get("max_turns",12),cfg.get("manager_episode_tokens",1536),
                          cfg.get("retrievals",6),cfg.get("total_tokens",100000))
            history=[];working={"selected_ids":[],"summary":None,"current_entity":"task","stagnant":{},
                                "interaction_window":[],"observation_id":initial["id"],
                                "public_state":self.public_projection.observe(observation, {}, None, [])}
            costs={"manager_input":0,"manager_output":0,"actor_input":0,"actor_output":0}
            invalid=0;phase_start="state";start_turn=0;prefix_cost=0
        done=False
        completed=False
        termination_reason = "turn_budget"
        try:
            for turn in range(start_turn,cfg.get("max_turns",12)):
                phases=["state","context"]
                if turn==start_turn and phase_start=="context":phases=["context"]
                for phase in phases:
                    if capture_phase and turn==capture_turn and not boundaries and phase==("state" if capture_phase=="window" else capture_phase):
                        snap=self.env.snapshot()
                        boundaries.append(Boundary(atlas.scope,atlas.export(),snap,copy.deepcopy(budget),goal,
                            observation,copy.deepcopy(history),capture_phase,dict(self.policy.versions),capture_rng(),
                            copy.deepcopy(costs),invalid,working=copy.deepcopy(working)))
                    if phase=="state" and phase_start=="window":phase_start="state"
                    local_forced=forced if forced is not None else None
                    if local_forced is not None:forced=None
                    sample,result=self._manager(atlas,observation,working,goal,budget,phase,greedy,
                                                 digest([atlas.fingerprint(),working,asdict(budget),phase]),costs,local_forced)
                    if sample:samples.append(sample)
                    if not result["accepted"]:invalid+=1
                    working["last_tool_result"]=result
                    events.append({"phase":phase,"result":result,"revision":atlas.revision,
                                   "sampled":sample is not None,"tool_wall_seconds":self.last_tool_seconds})
                phase_start="state"
                messages,view_hash,view_id=self._actor_messages(atlas,working,goal,observation,history)
                room=min(cfg.get("actor_max_new",256),budget.total_tokens-len(self.policy.encode(messages)))
                if room<1:
                    termination_reason = "total_token_budget"
                    break
                sample=self.policy.sample(messages,"actor","execute",digest([atlas.fingerprint(),view_hash]),
                                          cfg.get("actor_max_input",6144),room,greedy)
                budget.charge("actor",len(sample.prompt_ids),len(sample.completion_ids));budget.turns-=1
                costs["actor_input"]+=len(sample.prompt_ids);costs["actor_output"]+=len(sample.completion_ids)
                samples.append(sample)
                view = atlas.view(view_id)
                context_ids = {ref["id"] for nid in view["selected_ids"] if not nid.startswith("obs_")
                               for ref in atlas.node(nid).get("evidence_refs", [])}
                context_ids.update(nid for nid in view["selected_ids"] if nid.startswith("obs_"))
                atlas.record_use(view_id,sample.text,dict(self.policy.versions))
                before=observation
                before_id = working["observation_id"]
                state = working["public_state"]
                observation, done, public, action = execute_action(
                    self.env, self.action_adapter, sample.text, before, state, action_mode)
                if public["is_error"]:invalid+=1
                command=public.get("command")
                entity=working["current_entity"]
                receipt_id = None
                if command and not public["is_error"]:
                    entity = self.public_projection.evidence_entity(command, entity)
                    working["current_entity"]=entity
                    receipt=atlas.append_public(command,"action_receipt",entity,terminal=bool(done or budget.turns<=0))
                    receipt_id = receipt["id"]
                    attempted={"entity":entity,"predicate":"attempted_action","value":command,"support_status":"supported",
                        "evidence_refs":[{"id":receipt["id"],"start":0,"end":len(command)}],"expected_revision":atlas.revision}
                    checked=self.tools.validator.check(atlas,attempted)
                    atlas.commit(checked,atlas.revision,digest([receipt["id"],"attempted"]))
                if action["sent"] is not False or action_mode == "legacy":
                    record = atlas.append_public(observation, "observation", entity, clickables=public["clickables"],
                                                 terminal=bool(done or budget.turns<=0))
                    working["observation_id"] = record["id"]
                if not public["is_error"]:
                    working["public_state"] = self.public_projection.observe(observation, state, command, public["clickables"])
                next_state = working["public_state"]
                state_before = digest(self.public_projection.progress_key(state))
                state_after = digest(self.public_projection.progress_key(next_state))
                progress, cycle = record_progress(working["interaction_window"], state_before, state_after, command,
                                                  before_id, not public["is_error"], done)
                action.update(observation_before_id=before_id, observation_after_id=working["observation_id"],
                              action_receipt_id=receipt_id, public_state_before=state_before, public_state_after=state_after,
                              public_state_delta={k: {"before": state.get(k), "after": next_state.get(k)}
                                                  for k in sorted(state.keys() | next_state.keys())
                                                  if state.get(k) != next_state.get(k)},
                              progress=progress, loop=cycle)
                # Source-mapped historical quotations survive unrelated observations. The
                # next projection invalidates them if goal or protected state changed.
                if loop_policy == "legacy":
                    key=digest([before,command])
                    working["stagnant"][key]=working["stagnant"].get(key,0)+1 if before==observation else 0
                    working["warning"]=("同一页面重复相同动作且无观察变化，请结合历史重新决策。" if working["stagnant"][key]>=2 else "")
                else:
                    working["warning"] = interaction_guidance(action)
                if action_mode != "legacy" and action["sent"] is False:
                    working["warning"] = interaction_guidance(action)
                if loop_policy == "state" or action_mode != "legacy":
                    working["interaction_feedback"] = ({"message": working["warning"],
                        "evidence_ids": cycle["evidence_ids"] if cycle else [before_id]} if working["warning"] else None)
                history.append({"response":sample.text,"observation":observation,"command":command,"is_error":public["is_error"]})
                events.append({"phase":"execute","view_hash":view_hash,"view_id":view_id,"command":command,
                               "context_evidence_ids":sorted(context_ids),
                               "is_error":public["is_error"],"termination":sample.termination,"revision":atlas.revision,
                               **action})
                if done:
                    termination_reason = "environment_done"
                    break
            metrics=dict(self.env.metrics)
            tokens=sum(costs.values())-prefix_cost
            task_reward=(float(metrics["success"])+.1*float(metrics["r_loose"]))/1.1
            invalid_delta=invalid-(boundary.invalid if boundary else 0)
            reward=task_reward-cfg.get("invalid_weight",.05)*min(invalid_delta/4,1)-cfg.get("cost_weight",.02)*min(tokens/100000,2)
            outcome=Outcome(str(task_id),samples,metrics,float(reward),events,boundaries,
                            {**costs,"charged_tokens":tokens,"invalid":invalid_delta,"actor_turns":len([s for s in samples if s.role=="actor"])},
                            termination_reason=termination_reason)
            outcome.costs.update(tool_wall_seconds=sum(e.get("tool_wall_seconds",0) for e in events),
                retrieval_calls=sum(e.get("result",{}).get("operation") in {"RETRIEVE","FETCH_EVIDENCE"} for e in events),
                summary_calls=sum(e.get("result",{}).get("operation")=="SUMMARIZE" for e in events),
                skipped_manager_calls=sum(e["phase"]!="execute" and not e.get("sampled") for e in events))
            outcome.final_atlas=atlas.export()
            completed=True
            return outcome
        finally:
            try:
                if not completed:
                    for saved in boundaries:self.env.drop_snapshot(saved.environment)
            finally:
                atlas.close()
                self.env.close()
