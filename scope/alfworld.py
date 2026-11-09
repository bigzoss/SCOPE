"""Optional ALFWorld public-v1 HTTP bridge; no hidden PDDL state enters policy input."""
from __future__ import annotations
import json
from pathlib import Path
import re
import requests
from .contracts import ContractError,InfrastructureError
from .data import SplitRegistry
from .interaction import ActionAdapter, Binding
from .state_graph import ALF_PREDICATES, alfworld_facts
from .io import digest,file_sha

CONTRACT="alfworld-text-public-v1"
SYSTEM=("You are an agent in a household text environment. Complete the public task. Return a brief Thought "
        "and exactly one line Action: command. Commands: look; inventory; go to receptacle; open receptacle; "
        "close receptacle; take object from receptacle; put object in/on receptacle; clean object with sinkbasin; "
        "heat object with microwave; cool object with fridge; use desklamp. Use exact observed object names/numbers. "
        "Use inventory, not check inventory. A task saying look at an object under a lamp is a goal, not an action command. "
        "Check possession and task prerequisites before using a device. "
        "An attempted command is not a confirmed state change. Do not invent observations.")


def parse_command(text):
    if "</think>" in text:text=text.split("</think>",1)[1]
    found=re.findall(r"^\s*Action:\s*(.+?)\s*$",text,re.M|re.I)
    return found[-1].strip() if found and 0<len(found[-1].strip())<=200 else None


_ENTITY = r"[a-z][a-z0-9_]* \d+"
_ENTITY_PATTERN = re.compile(rf"\b{_ENTITY}\b", re.I)
_COMMANDS = tuple(re.compile(pattern) for pattern in (
    r"(?:go to|open|close|use|examine) (.+)",
    r"take (.+) from (.+)", r"put (.+) in/on (.+)",
    r"(?:clean|heat|cool) (.+) with (.+)",
))


class ALFWorldActions(ActionAdapter):
    parse = staticmethod(parse_command)
    state_predicates = ALF_PREDICATES
    public_facts = staticmethod(alfworld_facts)

    def decision_targets(self, state, goal):
        result = []
        for entity in state.get("entities", []):
            kind = entity.rsplit(" ", 1)[0]
            if re.search(r"\b" + re.escape(kind) + r"\b", goal.lower()):
                result.append(("alf:" + entity, "object_location"))
                if entity not in state.get("held", []):
                    result.append(("alf:" + entity, "held"))
        return result[:8]

    def recovery_candidates(self, state, goal):
        # Navigation is grounded in observed receptacles, never in hidden admissible commands.
        locations = {"bed", "desk", "drawer", "cabinet", "shelf", "countertop", "fridge", "microwave",
                     "sinkbasin", "bathtubbasin", "garbagecan", "laundryhamper", "safe", "diningtable",
                     "sidetable", "coffeetable", "dresser", "toilet", "sofa", "armchair", "ottoman", "tvstand"}
        here = state.get("location")
        visited = set(state.get("visited_locations", []))
        if here and state.get("containers", {}).get(here) is False:
            yield {"command": "open " + here, "reason": "inspect_publicly_closed_container"}
        for obj, place in state.get("object_locations", {}).items():
            if place == here and obj not in state.get("held", []) and obj.rsplit(" ", 1)[0] in goal.lower():
                yield {"command": f"take {obj} from {place}", "reason": "inspect_goal_object_at_observed_location"}
        # Candidates describe public preconditions, never assert task completion.
        for obj in state.get("held", []):
            if here and here.rsplit(" ", 1)[0] in locations and state.get("containers", {}).get(here) is not False:
                yield {"command": f"put {obj} in/on {here}", "reason": "place_held_object_at_current_location"}
            for verb, device in (("clean", "sinkbasin"), ("heat", "microwave"), ("cool", "fridge")):
                if here and here.rsplit(" ", 1)[0] == device:
                    yield {"command": f"{verb} {obj} with {here}", "reason": "process_held_object_at_observed_device"}
            for lamp, place in state.get("object_locations", {}).items():
                if place == here and lamp.startswith("desklamp "):
                    yield {"command": f"use {lamp}", "reason": "use_observed_lamp_while_holding_object"}
        for place in sorted(state.get("entities", []), key=lambda x: (x in visited, x)):
            if place != here and place.rsplit(" ", 1)[0] in locations:
                yield {"command": "go to " + place, "reason": "revisit_observed_location" if place in visited else "explore_observed_location"}

    def focus_entities(self, state):
        return ["alf:" + entity for entity in [state.get("location"), *state.get("held", [])] if entity]

    def confirmation_target(self, command, entity):
        change = re.fullmatch(rf"(take|put|open|close|clean|heat|cool) ({_ENTITY})(?: .+)?", command)
        if not change:
            return None
        verb, name = change.groups()
        predicate, value = {
            "take": ("held", True), "put": ("held", False),
            "open": ("container_open", True), "close": ("container_open", False),
            "clean": ("cleaned", True), "heat": ("temperature", "hot"), "cool": ("temperature", "cold"),
        }[verb]
        return "alf:" + name, predicate, value

    def recovery_slots(self, command):
        if not command:
            return ()
        placement = re.fullmatch(rf"put ({_ENTITY}) in/on ({_ENTITY})", command)
        treatment = re.fullmatch(rf"(?:clean|heat|cool) ({_ENTITY}) with ({_ENTITY})", command)
        if placement or treatment:
            obj, container = (placement or treatment).groups()
            return (("alf:"+obj,"held"), ("alf:"+obj,"object_location"), ("alf:"+container,"container_open"))
        return ()

    def normalize(self, command):
        command = " ".join(command.lower().removesuffix(".").split())
        if command == "check inventory":
            return "inventory"
        # 'receptacle' is a schema placeholder, not part of the observed entity name.
        command = re.sub(r"^(go to|open|close) receptacle (?=" + _ENTITY + r"$)", r"\1 ", command)
        return re.sub(r"^(put .+?) (?:in|on) (.+)$", r"\1 in/on \2", command)

    def bind(self, command, state):
        if command in {"look", "inventory"}:
            return Binding("not_required")
        match = next((match for pattern in _COMMANDS if (match := pattern.fullmatch(command))), None)
        if not match:
            return Binding("invalid_syntax", "use an ALFWorld command with all required arguments")
        known = state.get("entities", [])
        for argument in match.groups():
            if argument in known:
                continue
            if _ENTITY_PATTERN.fullmatch(argument):
                return Binding("unobserved", f"{argument!r} has no public observation")
            candidates = tuple(name for name in known if name.rsplit(" ", 1)[0] == argument)[:8]
            status = "ambiguous" if len(candidates) > 1 else "missing"
            return Binding(status, f"{argument!r} needs an exact observed name and number", candidates)
        return Binding("bound")

    def observe(self, observation, previous, command, clickables):
        # The task instruction may mention objects that have not been observed yet.
        text = re.split(r"(?m)^\s*Your task is to\b", observation, maxsplit=1, flags=re.I)[0].lower()
        entities = set(_ENTITY_PATTERN.findall(text))
        state = {**previous, "entities": sorted(entities | set(previous.get("entities", [])))}
        place = re.search(rf"you (?:arrive at|are facing) (?:the )?({_ENTITY})\b", text)
        if place:
            state["location"] = place[1]
            state["visited_locations"] = sorted(set(previous.get("visited_locations", [])) | {place[1]})
        elif "you are in the middle of a room" in text:
            state["location"] = "room"
        carrying = re.search(r"you (?:are carrying|have in your inventory)\s*:?\s*([^\n.]+)", text)
        held = set(previous.get("held", []))
        if "you are not carrying anything" in text:
            state["held"] = []
        elif carrying:
            state["held"] = sorted(set(_ENTITY_PATTERN.findall(carrying[1])))
        taken = re.search(rf"you (?:pick up|take) (?:the )?({_ENTITY})\b", text)
        placed = re.search(rf"you put (?:the )?({_ENTITY}) (?:in|on|into) (?:the )?({_ENTITY})\b", text)
        if taken:
            state["held"] = sorted(held | {taken[1]})
            state["placements"] = {k: v for k, v in previous.get("placements", {}).items() if k != taken[1]}
        elif placed:
            state["held"] = sorted(held - {placed[1]})
            state["placements"] = {**previous.get("placements", {}), placed[1]: placed[2]}
        opened = re.search(rf"you (open|close) (?:the )?({_ENTITY})\b", text)
        containers = dict(previous.get("containers", {}))
        if opened:
            containers[opened[2]] = opened[1] == "open"
        for name, status in re.findall(rf"\b(?:the )?({_ENTITY}) is (open|closed)\b", text):
            containers[name] = status == "open"
        if containers:
            state["containers"] = containers
        treated = re.search(rf"you (clean|heat|cool) (?:the )?({_ENTITY})\b", text)
        if treated:
            treatments = previous.get("treatments", {})
            update = {"clean": True} if treated[1] == "clean" else {"temperature": treated[1]}
            state["treatments"] = {**treatments, treated[2]: {**treatments.get(treated[2], {}), **update}}
        object_locations = dict(previous.get("object_locations", {}))
        for fact in alfworld_facts({"text": observation}):
            if fact.predicate == "object_location":
                object_locations[fact.entity.removeprefix("alf:")] = fact.value.removeprefix("alf:")
        for obj, location in re.findall(rf"\b({_ENTITY}) (?:is )?(?:in/on|on|in|inside) (?:the )?({_ENTITY})\b", text):
            object_locations[obj] = location
        if taken:
            object_locations[taken[1]] = "agent"
        elif placed:
            object_locations[placed[1]] = placed[2]
        if object_locations:
            state["object_locations"] = object_locations
        return state


class ALFWorldRegistry(SplitRegistry):
    def __init__(self,manifest,seed=233):
        path=Path(manifest);data=json.loads(path.read_text())
        if data.get("contract_version")!=CONTRACT:raise ContractError("Unsupported ALFWorld manifest contract")
        mapping={"train":"train","dev":"valid_seen","test":"valid_unseen"}
        self.rows={}
        for destination,source in mapping.items():
            self.rows[destination]=[{**r,"task_id":str(r["task_id"]),"domain":r["task_type"],
                                      "isolation_group":r["game_sha256"]} for r in data["tasks"] if r["split"]==source]
            ids=[r["task_id"] for r in self.rows[destination]]
            if len(ids)!=len(set(ids)):raise ContractError("Duplicate ALFWorld task IDs")
        for left,right in (("train","dev"),("train","test"),("dev","test")):
            for field in ("task_id","isolation_group"):
                if {r[field] for r in self.rows[left]} & {r[field] for r in self.rows[right]}:
                    raise ContractError("ALFWorld exact task/game overlap: "+left+"/"+right)
        self.rows.update(sft_train=[],sft_dev=[])
        groups=sorted({r["isolation_group"] for r in self.rows["train"]},key=lambda g:digest([seed,g]))
        held=set(groups[:max(1,round(len(groups)*.1))])
        self.rows["internal_dev"]=[r for r in self.rows["train"] if r["isolation_group"] in held]
        self.rows["rl_train"]=[r for r in self.rows["train"] if r["isolation_group"] not in held]
        self.hashes={"official_manifest":file_sha(path)}
        self.root=path.parent


class ALFWorldEnvironment:
    actor_system=SYSTEM
    action_adapter = ALFWorldActions()
    def __init__(self,base_url,allowed_tasks,manifest,timeout=120):
        self.base=base_url.rstrip("/");self.allowed=set(map(str,allowed_tasks));self.timeout=timeout
        self.manifest=Path(manifest);self.manifest_sha=file_sha(manifest)
        self.http=requests.Session();self.sid=None;self.observation="";self.metrics={}

    def _post(self,route,payload):
        try:
            response=self.http.post(self.base+route,json=payload,timeout=self.timeout)
            response.raise_for_status();value=response.json()
        except (requests.RequestException,ValueError) as exc:raise InfrastructureError("ALFWorld "+route+": "+str(exc)) from exc
        if not isinstance(value,dict):raise InfrastructureError("ALFWorld returned a non-object response")
        if value.get("contract_version")!=CONTRACT:raise ContractError("ALFWorld public contract mismatch")
        if value.get("manifest_sha256")!=self.manifest_sha:raise ContractError("ALFWorld server manifest mismatch")
        return value

    def health(self):
        if file_sha(self.manifest)!=self.manifest_sha:raise ContractError("ALFWorld manifest changed")
        try:
            response=self.http.get(self.base+"/healthz",timeout=self.timeout);response.raise_for_status();data=response.json()
        except (requests.RequestException,ValueError) as exc:raise InfrastructureError("ALFWorld health check failed: "+str(exc)) from exc
        if data.get("contract_version")!=CONTRACT or data.get("manifest_sha256")!=self.manifest_sha:
            raise ContractError("ALFWorld health contract/manifest mismatch")
        if not data.get("ready"):raise InfrastructureError("ALFWorld service is not ready")
        return {**data,"manifest_unchanged":True,"restore_level":"not_available_in_public_v1"}

    def reset(self,task):
        self.close()
        if str(task) not in self.allowed:raise ContractError("ALFWorld task outside designated split")
        value=self._post("/reset",{"task_id":str(task)})
        self.sid=value["session_id"];self.observation=value["text"]
        match=re.search(r"Your task is to:\s*(.+)",self.observation)
        self.task_goal=match.group(1).strip() if match else self.observation
        self.metrics={"success":0.,"r_strict":0.,"r_loose":0.,"purchased":0.}
        return self.observation

    def step(self,text):
        command=parse_command(text)
        if command is None:return self.observation,False,{"command":None,"is_error":True,"clickables":[]}
        return self.step_command(command)

    def step_command(self, command):
        value=self._post("/step",{"session_id":self.sid,"command":command})
        if type(value.get("success")) is not bool or type(value.get("done")) is not bool:
            raise InfrastructureError("ALFWorld success/done must be explicit booleans")
        self.observation=value["text"];success=value["success"]
        self.metrics={"success":float(success),"r_strict":float(success),"r_loose":float(success),"purchased":0.}
        return self.observation,bool(value.get("done")) or success,{"command":command,
                 "is_error":self.observation.strip().lower().startswith("nothing happens"),"clickables":[]}

    def snapshot(self):
        raise ContractError("ALFWorld public-v1 has no native snapshot API; use frozen evaluation/global training")

    def close(self):
        if self.sid is not None:
            sid,self.sid=self.sid,None
            self._post("/release",{"session_id":sid})
