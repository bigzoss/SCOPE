"""Real environment rollout, public state pipeline and restorable management boundaries."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
import random
import re
import uuid
import time
from .atlas import StateAtlas
from .candidate_actions import candidate_actions, choice_messages
from .decision_evidence import review
from .memory_gaps import confirmation_status, skipped_candidate_reason
from .recovery import prepare_recovery, record_recovery, information_progress
from .contracts import Boundary, Budget, ContractError, Outcome, Scope, Sample
from .io import canonical, digest
from .interaction import ActionAdapter, execute_action, interaction_guidance, record_progress
from .state_graph import apply_observation, compression_due, core_claims, management_reason, shopping_checklist
from .tools import ToolExecutor, SCHEMA_HELP, context_view, graph_tool_prompt, parse_tool, public_snippet

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


def seed_decision(pair_seed, turn, phase, attempt):
    import numpy as np
    import torch
    seed = int(digest([pair_seed, turn, phase, attempt])[:8], 16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def manager_messages(atlas, observation, working, goal, phase, budget):
    evidence = [public_snippet(e,160) for e in atlas.all_evidence()[-2:]]
    for snippet in evidence:
        source=atlas.evidence(snippet["id"])["text"]
        if len(source)>160:snippet["tail"]={"span_start":max(0,len(source)-80),"text":source[-80:]}
    graph = working.get("state_graph")
    claims = [{k:c[k] for k in ("id","entity","predicate","value","support_status")}
              for c in (core_claims(atlas, working) if graph else atlas.nodes("Claim"))]
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
    if working.get("confirmation"):
        payload["public_confirmation"] = working["confirmation"]
    if working.get("decision_evidence"):
        payload["decision_evidence"] = review(atlas,working,goal)
    if working.get("decision_recovery"):
        plan = working.get("recovery_plan", {})
        payload["recovery_plan"] = {k:plan.get(k) for k in ("reason","blocked_commands","unresolved_slots")}
        payload["recovery_plan"]["candidates"] = plan.get("candidates", [])[:3]
    schema = SCHEMA_HELP
    if graph:
        payload["management_reason"] = management_reason(atlas, working, goal, phase)
        payload["state_lookup_used"] = working.get("state_lookup_used", False)
        payload["constraint_checklist"] = shopping_checklist(atlas, goal, working.get("current_entity", "task"),
                                                             working.get("candidate_since", 0))
        schema = graph_tool_prompt(phase, atlas.revision, payload["state_lookup_used"])
        payload["domain_predicates"] = working.get("domain_predicates", [])
    return [{"role":"system","content":"You manage public evidence and memory, not shopping actions.\n"+schema},
            {"role":"user","content":canonical(payload)}]


class EpisodeRunner:
    def __init__(self,policy,env,config,split="train",experience=None,actor_teacher=None,probe_selector=None):
        self.policy,self.env,self.cfg,self.split=policy,env,config,split
        self.experience=experience
        if actor_teacher and split not in {"train", "rl_train"}:
            raise ContractError("Teacher demonstrations require a training partition")
        self.actor_teacher = actor_teacher
        self.probe_selector = probe_selector
        self.action_adapter = getattr(env, "action_adapter", None)
        if self.action_adapter is None and config.get("action_mode", "legacy") != "legacy":
            raise ContractError("Action normalization and binding require a benchmark action_adapter")
        self.public_projection = self.action_adapter or ActionAdapter()
        self.tools=ToolExecutor(None if config.get("disable_experience") else experience,
                                config.get("semantic_validation",True),
                                self.public_projection if config.get("state_graph") else None)

    def _manager_request(self, atlas, observation, working, goal, budget, phase, audit=None):
        if budget.manager_tokens<=0 or budget.total_tokens<=16:
            return None, 0, "manager_budget_exhausted", None
        if self.cfg.get("manager_mode")=="rule":
            return None, 0, "rule_FINISH", None
        if working.get("state_graph") and not management_reason(atlas, working, goal, phase) and not (
                phase == "state" and working.get("state_lookup_used")):
            return None, 0, "public_state_sufficient", None
        messages=manager_messages(atlas,observation,working,goal,phase,budget)
        menu = None
        if self.cfg.get("memory_choice"):
            menu = candidate_actions(atlas, working, goal, phase, budget, self.tools, audit)
            if len(menu) == 1:
                return None, 0, "no_public_candidate", None
            messages = choice_messages(messages, menu)
        max_input=self.cfg.get("manager_max_input",2048)
        if len(self.policy.encode(messages))>max_input:
            # Refuse to silently drop the goal or pending states. Hand back to the actor.
            return None, 0, "manager_input_budget", None
        n_input=len(self.policy.encode(messages))
        room=min(self.cfg.get("manager_max_new",192),budget.manager_tokens,budget.total_tokens-n_input)
        return messages, room, "total_budget" if room < 1 else None, menu

    def _manager(self,atlas,observation,working,goal,budget,phase,greedy,boundary,costs,forced=None):
        self.last_tool_seconds=0.
        audit = [] if working.get("shared_memory") else None
        messages, room, skipped, menu = self._manager_request(atlas, observation, working, goal, budget, phase, audit)
        if skipped:
            result = {"accepted":True,"finish":True,"reason":skipped}
            if audit:
                result["menu_diagnostics"] = audit
            if skipped == "no_public_candidate":
                result["gap_diagnostic"] = skipped_candidate_reason(working, budget, phase, atlas, goal)
            return None,result
        if forced == "FINISH":
            return None,self.tools.execute(atlas, working, {"operation":"FINISH","expected_revision":atlas.revision}, phase, budget, goal)
        max_input=self.cfg.get("manager_max_input",2048)
        sample = forced
        if sample is None:
            sample = (self.policy.sample_choice(messages, phase, boundary, list(menu), max_input, greedy) if menu
                      else self.policy.sample(messages,"manager",phase,boundary,max_input,room,greedy))
        if forced and (sample.prompt_ids != self.policy.encode(messages) or sample.version != self.policy.versions["manager"]
                       or sample.role != "manager" or sample.phase != phase):
            raise ContractError("Forced candidate does not match the pre-action policy context")
        if menu and (sample.choice_token_ids != self.policy.choice_ids(list(menu)) or sample.text not in menu or
                     sample.completion_ids != self.policy.choice_ids([sample.text])):
            raise ContractError("Sample does not match the versioned choice distribution")
        budget.charge("manager",len(sample.prompt_ids),len(sample.completion_ids))
        costs["manager_input"]+=len(sample.prompt_ids);costs["manager_output"]+=len(sample.completion_ids)
        try:
            tool_start=time.perf_counter()
            proposal=menu[sample.text] if menu else parse_tool(sample.text)
            result=self.tools.execute(atlas,working,proposal,phase,budget,goal)
            if menu:
                result = {**result, "choice":sample.text, "candidate_count":len(menu)}
                if audit:
                    result["menu_diagnostics"] = audit
            if proposal.get("operation")=="SUMMARIZE":working["compressed"]=True
        except (ContractError,ValueError,KeyError,TypeError) as exc:
            result={"accepted":False,"reason":str(exc),"operation":"rejected"}
        self.last_tool_seconds=time.perf_counter()-tool_start
        return sample,result

    def sample_boundary(self, boundary):
        atlas = StateAtlas.restore(boundary.atlas)
        try:
            messages, room, skipped, menu = self._manager_request(atlas, boundary.observation, boundary.working,
                                                                   boundary.goal, boundary.budget, boundary.phase)
            if skipped:
                raise ContractError("Captured manager opportunity cannot be sampled: " + skipped)
            limit = self.cfg.get("manager_max_input", 2048)
            return (self.policy.sample_choice(messages, boundary.phase, boundary.fingerprint(), list(menu), limit) if menu
                    else self.policy.sample(messages, "manager", boundary.phase, boundary.fingerprint(), limit, room, False))
        finally:
            atlas.close()

    def _actor_messages(self,atlas,working,goal,observation,history):
        view=context_view(atlas,working,goal)
        if self.cfg.get("compact_actor_view"):
            # Preserve goal, core claims, pending slots and cited spans. Remove only
            # bookkeeping and duplicate goal text; raw history remains in the atlas.
            view.pop("content_hash", None)
            view.pop("goal", None)
            if "recovery_plan" in view:
                plan = view["recovery_plan"]
                view["recovery_plan"] = {k:plan[k] for k in (
                    "reason", "blocked_commands", "repeated_commands", "held", "unresolved_slots", "candidates") if k in plan}
            history = history[-self.cfg.get("actor_history_turns", 2):]
        rendered_claim_ids = [c["id"] for c in view["claims"]]
        claim_fields = {"entity","predicate","value","support_status","conflicted","observed_at_revision","validity"}
        view["claims"]=[{**{k:v for k,v in c.items() if k in claim_fields},
                         **({"evidence_ids": [r["id"] for r in c["evidence_refs"]]} if working.get("state_graph") else {})}
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
            if working.get("state_graph"):
                content += "\nCurrent observation ID: " + working["observation_id"]
                content += "\nIf the action relies on a cited source, include a separate Evidence: line with its public IDs."
            if working.get("warning"):content+="\n"+working["warning"]
            if view.get("constraint_checklist"):
                content += "\nBefore Buy Now, review the public checklist. Resolve conflicts when possible; pending is unconfirmed."
            if view.get("decision_evidence"):
                content += ("\nBefore your next action, check the listed entity and state slots. A pending slot needs a public check. "
                            "Use the exact observed entity and change a rejected command when the public state has not changed.")
            if view.get("recovery_plan"):
                content += ("\nUse the public recovery candidates when exploration is needed. They are suggestions, not confirmed effects. "
                            "Do not repeat blocked_commands in this unchanged state. Goal mentions do not prove an object has been found.")
            if history:messages[-1]["content"]=content
            else:messages[-1]["content"]+="\n"+content
            if len(self.policy.encode(messages))<=limit:
                sources=[record["id"] for record in view["selected_evidence"]]
                if view["fetched_evidence"]:sources.append(view["fetched_evidence"]["id"])
                if view["summary"]:sources.extend(view["summary"]["source_hashes"])
                if working.get("state_graph"):
                    sources.extend(e["id"] for e in view["recent_actions"]["recent"])
                    sources.extend(eid for row in view["constraint_checklist"] for eid in row["evidence_ids"])
                    sources.extend(ref["id"] for row in view.get("decision_evidence",[]) for ref in row.get("evidence_refs",[]))
                    sources.extend(eid for row in view.get("recovery_plan", {}).get("candidates", []) for eid in row["evidence_ids"])
                saved=atlas.save_view(sorted(set(sources))+rendered_claim_ids,view["summary"])
                return messages,digest(view),saved["id"]
            if history:history.pop(0)
            elif len(current)>600:
                half=max(150,len(current)//4)
                current=current[:half]+"\n[page middle omitted; complete source remains retrievable]\n"+current[-half:]
            else:raise ContractError("Cannot preserve goal and public state within actor input budget")

    def run(self,task_id,episode_id,greedy=False,capture_phase=None,capture_turn=1,boundary=None,forced=None,
            capture_opportunity=False, pair_seed=None, probe_opportunities=()):
        cfg=self.cfg
        if self.actor_teacher and (boundary or capture_phase):
            raise ContractError("Teacher demonstrations are separate from student credit rollouts")
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
            if cfg.get("state_graph"):
                working.update(state_graph=True, compression_after=cfg.get("compression_after", 6),
                               compression_min_chars=cfg.get("compression_min_chars", 1800),
                               domain_predicates=sorted(self.tools.validator.domain_predicates))
                if cfg.get("shared_memory") or cfg.get("disabled_memory_tools"):
                    working.update(shared_memory=cfg.get("shared_memory", False),
                                   disabled_memory_tools=cfg.get("disabled_memory_tools", []))
                if cfg.get("decision_evidence"):
                    working["decision_evidence"] = True
                if cfg.get("decision_recovery"):
                    working["decision_recovery"] = True
                if initial["kind"] == "observation":
                    apply_observation(self.tools, atlas, working, initial, budget, goal)
            costs={"manager_input":0,"manager_output":0,"actor_input":0,"actor_output":0}
            invalid=0;phase_start="state";start_turn=0;prefix_cost=0
        done=False
        completed=False
        termination_reason = "turn_budget"
        initial_costs = dict(costs)
        try:
            for turn in range(start_turn,cfg.get("max_turns",12)):
                if working.get("decision_recovery"):
                    prepare_recovery(self.public_projection, atlas, working, goal)
                if working.get("state_graph") and not (boundary and turn == start_turn):
                    working["state_lookup_used"] = False
                phases=["state","context"]
                if turn==start_turn and phase_start=="context":phases=["context"]
                for phase in phases:
                    target_phase = phase == ("state" if capture_phase=="window" else capture_phase)
                    eligible = capture_phase and target_phase and not boundaries
                    reason = management_reason(atlas, working, goal, phase) if capture_opportunity and eligible else None
                    if capture_opportunity and eligible and probe_opportunities:
                        if phase == "context" and "context_threshold" in probe_opportunities and compression_due(atlas,working):
                            reason = "context_threshold"
                        if reason not in probe_opportunities:
                            reason = None
                        if reason == "context_threshold":
                            _, _, _, menu = self._manager_request(atlas, observation, working, goal, budget, phase)
                            if not any(p["operation"] == "SUMMARIZE" for p in (menu or {}).values()):
                                reason = None
                    capture = (reason and not self._manager_request(atlas, observation, working, goal, budget, phase)[2]
                               if capture_opportunity else turn == capture_turn)
                    if eligible and capture and self.probe_selector:
                        capture = self.probe_selector(atlas, working, goal, observation, phase)
                    if eligible and capture:
                        snap=self.env.snapshot()
                        boundaries.append(Boundary(atlas.scope,atlas.export(),snap,copy.deepcopy(budget),goal,
                            observation,copy.deepcopy(history),capture_phase,dict(self.policy.versions),capture_rng(),
                            copy.deepcopy(costs),invalid,working=copy.deepcopy(working),opportunity=reason))
                    if phase=="state" and phase_start=="window":phase_start="state"
                    local_forced=forced if forced is not None else None
                    if local_forced is not None:forced=None
                    for attempt in range(2 if working.get("state_graph") and phase == "state" else 1):
                        if pair_seed is not None:
                            seed_decision(pair_seed, turn, phase, attempt)
                        sample,result=self._manager(atlas,observation,working,goal,budget,phase,greedy,
                                                     digest([atlas.fingerprint(),working,asdict(budget),phase]),costs,
                                                     local_forced if attempt == 0 else None)
                        if sample:samples.append(sample)
                        if not result["accepted"]:invalid+=1
                        working["last_tool_result"]=result
                        events.append({"phase":phase,"result":result,"revision":atlas.revision,
                                       "sampled":sample is not None and (attempt != 0 or local_forced is None),
                                       "replayed_sample":sample is not None and attempt == 0 and local_forced is not None,
                                       "intervention":attempt == 0 and local_forced == "FINISH",
                                       "tool_wall_seconds":self.last_tool_seconds})
                        if not result["accepted"] or result.get("operation") not in {"RETRIEVE", "FETCH_EVIDENCE"}:
                            break
                phase_start="state"
                decision_rows = review(atlas,working,goal) if working.get("decision_evidence") else []
                messages,view_hash,view_id=self._actor_messages(atlas,working,goal,observation,history)
                room=min(cfg.get("actor_max_new",256),budget.total_tokens-len(self.policy.encode(messages)))
                if room<1:
                    termination_reason = "total_token_budget"
                    break
                if pair_seed is not None:
                    seed_decision(pair_seed, turn, "actor", 0)
                teacher_label = None
                if self.actor_teacher and (not hasattr(self.actor_teacher,"should_intervene") or self.actor_teacher.should_intervene(working)):
                    teacher_label = self.actor_teacher.propose(messages, working["public_state"], self.action_adapter)
                    tokens = self.policy.tokenizer.encode(teacher_label["text"], add_special_tokens=False)
                    if len(tokens) > room:
                        raise ContractError("Teacher action exceeds the student's generation budget")
                    sample = Sample("actor", "execute", self.policy.encode(messages), tokens, [], teacher_label["text"],
                                    self.policy.versions["actor"], digest([atlas.fingerprint(),view_hash]), origin="teacher")
                else:
                    sample=self.policy.sample(messages,"actor","execute",digest([atlas.fingerprint(),view_hash]),
                                              cfg.get("actor_max_input",6144),room,greedy)
                budget.charge("actor",len(sample.prompt_ids),len(sample.completion_ids));budget.turns-=1
                costs["actor_input"]+=len(sample.prompt_ids);costs["actor_output"]+=len(sample.completion_ids)
                samples.append(sample)
                view = atlas.view(view_id)
                required_ids = {ref["id"] for nid in view["selected_ids"] if not nid.startswith("obs_")
                               for ref in atlas.node(nid).get("evidence_refs", [])}
                context_ids = set(required_ids)
                context_ids.update(nid for nid in view["selected_ids"] if nid.startswith("obs_"))
                atlas.record_use(view_id,sample.text,dict(self.policy.versions))
                before=observation
                before_id = working["observation_id"]
                state = working["public_state"]
                recovery_plan = copy.deepcopy(working.get("recovery_plan", {}))
                observation, done, public, action = execute_action(
                    self.env, self.action_adapter, sample.text, before, state, action_mode,
                    recovery_plan.get("blocked_commands", []))
                if public["is_error"] and action["action_status"] != "no_progress_blocked":invalid+=1
                command=public.get("command")
                entity=working["current_entity"]
                receipt_id = None
                if command and not public["is_error"]:
                    next_entity = self.public_projection.evidence_entity(command, entity)
                    if next_entity != entity:
                        working["candidate_since"] = atlas.revision + 1
                        working.pop("confirmation", None)
                    entity = next_entity
                    working["current_entity"]=entity
                if command and (not public["is_error"] or working.get("state_graph") and action["sent"] is True):
                    receipt=atlas.append_public(command,"action_receipt",entity,terminal=bool(done or budget.turns<=0))
                    receipt_id = receipt["id"]
                    if not working.get("state_graph"):
                        attempted={"entity":entity,"predicate":"attempted_action","value":command,"support_status":"supported",
                            "evidence_refs":[{"id":receipt["id"],"start":0,"end":len(command)}],"expected_revision":atlas.revision}
                        checked=self.tools.validator.check(atlas,attempted)
                        atlas.commit(checked,atlas.revision,digest([receipt["id"],"attempted"]))
                record = None
                if action["sent"] is not False or action_mode == "legacy":
                    record = atlas.append_public(observation, "observation", entity, clickables=public["clickables"],
                                                 terminal=bool(done or budget.turns<=0))
                    working["observation_id"] = record["id"]
                if not public["is_error"]:
                    working["public_state"] = self.public_projection.observe(observation, state, command, public["clickables"])
                    if record:
                        apply_observation(self.tools, atlas, working, record, budget, goal)
                        if working.get("shared_memory") and receipt_id and not record.get("terminal"):
                            target = self.public_projection.confirmation_target(command, entity)
                            previous = working.get("confirmation")
                            if target:
                                subject, predicate, value = target
                                previous = {"entity":subject,"predicate":predicate,"value":value,
                                            "attempt_id":receipt_id,"attempt_revision":receipt["revision"]}
                            if previous:
                                working["confirmation"] = confirmation_status(atlas, self.tools, previous)
                next_state = working["public_state"]
                state_before = digest(self.public_projection.progress_key(state))
                state_after = digest(self.public_projection.progress_key(next_state))
                progress, cycle = record_progress(working["interaction_window"], state_before, state_after, command or action["raw_command"],
                                                  before_id, not public["is_error"], done)
                action.update(observation_before_id=before_id, observation_after_id=working["observation_id"],
                              action_receipt_id=receipt_id, public_state_before=state_before, public_state_after=state_after,
                              public_state_delta={k: {"before": state.get(k), "after": next_state.get(k)}
                                                  for k in sorted(state.keys() | next_state.keys())
                                                  if state.get(k) != next_state.get(k)},
                              progress=progress, loop=cycle)
                if working.get("decision_recovery"):
                    information_progress(working, action, before, observation)
                    record_recovery(working, action)
                    action["recovery_plan"] = recovery_plan
                    action["selected_recovery_candidate"] = any(command == c["command"] for c in recovery_plan.get("candidates", []))
                if working.get("decision_evidence"):
                    last = working.get("last_rejected_decision")
                    rejection = action["action_status"] == "environment_invalid" and bool(command)
                    count = (last["count"]+1 if last and last["command"]==command and last["state"]==state_before else 1)
                    working["last_rejected_decision"] = ({"command":command,"state":state_before,"count":count}
                                                         if rejection else None)
                    working["recovery_slots"] = (list(self.public_projection.recovery_slots(command))
                                                 if rejection or action["action_status"] == "binding_failed" else [])
                    if rejection:
                        action["same_state_rejection_count"] = count
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
                if action.get("same_state_rejection_count",0) >= 2:
                    working["warning"] += (" This exact command already failed in the same public state. "
                                           "Check the held object and destination, then choose a different action.")
                if loop_policy == "state" or action_mode != "legacy":
                    working["interaction_feedback"] = ({"message": working["warning"],
                        "evidence_ids": cycle["evidence_ids"] if cycle else [before_id]} if working["warning"] else None)
                history.append({"response":sample.text,"observation":observation,"command":command,"is_error":public["is_error"]})
                events.append({"phase":"execute","view_hash":view_hash,"view_id":view_id,"command":command,
                               **({"teacher_demonstration":teacher_label,"student_messages":messages} if teacher_label else {}),
                               "context_evidence_ids":sorted(context_ids),
                               "required_evidence_ids":sorted(required_ids),
                               "retrieved_evidence_in_view":sorted(context_ids & set(working.get("selected_ids", []))),
                               "cited_evidence_ids": sorted((context_ids | {before_id}) & set(re.findall(
                                   r"obs_\d+_[a-f0-9]+", "\n".join(re.findall(r"^Evidence:[^\n]*", sample.text, re.M | re.I))))),
                               **({"public_confirmation":working["confirmation"]} if working.get("confirmation") else {}),
                               **({"decision_slots":decision_rows} if decision_rows else {}),
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
            outcome.costs["continuation_token_breakdown"] = {key: value - initial_costs[key] for key, value in costs.items()}
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
