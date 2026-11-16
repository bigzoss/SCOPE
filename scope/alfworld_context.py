"""Small, source-backed ALFWorld views for the current environment decision."""
import re


_TREATMENT = {
    "clean": ("sinkbasin", "cleaned", "clean", True),
    "cool": ("fridge", "temperature", "temperature", "cool"),
    "heat": ("microwave", "temperature", "temperature", "heat"),
}
_PROCESS_GOAL = re.compile(r"\b(clean|cool|heat|hot|cold)(?:ed)?\s+(?:(?:an?|the|some)\s+)?([a-z][a-z0-9_]+)\b", re.I)
_DESTINATION = re.compile(r"\b(?:put|place)\b[^.]*?\b(?:in/on|into|inside|in|on)\s+(?:the\s+|an?\s+)?([a-z][a-z0-9_]+)\b", re.I)


def treatment_stage(state, goal):
    """Recommend one step supported by the current public household state."""
    process, destination = _PROCESS_GOAL.search(goal), _DESTINATION.search(goal)
    if not process or not destination:
        return None
    process_name, kind = process[1].lower(), process[2].lower()
    verb = {"hot": "heat", "cold": "cool"}.get(process_name, process_name)
    destination = destination[1].lower()
    device, predicate, field, expected = _TREATMENT[verb]
    here = state.get("location")
    entities = state.get("entities", ())
    held = [name for name in state.get("held", ()) if name.rsplit(" ", 1)[0] == kind]
    if len(held) != 1:
        return {"text": f"Next prerequisite: confirm possession of a {kind} before {verb}ing.",
                "slots": [("alf:" + name, "held") for name in entities if name.rsplit(" ", 1)[0] == kind][:2]}
    obj = held[0]
    confirmed = state.get("treatments", {}).get(obj, {}).get(field) == expected
    places = [name for name in entities if name.rsplit(" ", 1)[0] == (destination if confirmed else device)]
    place = here if here in places else places[0] if len(places) == 1 else None
    slots = [("alf:" + obj, "held"), ("alf:" + obj, predicate)]
    command = None
    if not place:
        text = f"Holding {obj}. Find an observed {destination if confirmed else device}."
    elif here != place:
        command = f"go to {place}"
        text = f"Holding {obj}. Go to {place} to {'deliver' if confirmed else verb} it."
    elif not confirmed:
        command = f"{verb} {obj} with {place}"
        text = f"Holding {obj} at {place}. Apply {verb} and check the simulator's confirmation before delivery."
    else:
        slots.append(("alf:" + place, "container_open"))
        command = f"open {place}" if state.get("containers", {}).get(place) is False else f"put {obj} in/on {place}"
        text = f"{verb.capitalize()} confirmed for {obj}. Deliver to {place}."
    return {"text": text, "slots": slots, "command": command, "object": obj,
            "destination": destination, "treatment_confirmed": confirmed}


def treatment_candidates(state, goal, candidates):
    stage = treatment_stage(state, goal)
    if not stage or "object" not in stage:
        return candidates
    obj = stage["object"]
    recommended = stage["command"]
    def allowed(item):
        command = item["command"]
        if not stage["treatment_confirmed"] and command.startswith(f"put {obj} in/on {stage['destination']} "):
            return False
        if stage["treatment_confirmed"] and command.startswith(tuple(f"{verb} {obj} with " for verb in _TREATMENT)):
            return False
        return True
    selected = [item for item in candidates if allowed(item)]
    if recommended:
        selected = [{"command": recommended, "reason": "public_treatment_prerequisite"},
                    *(item for item in selected if item["command"] != recommended)]
    return selected


def _mentions(goal, entity):
    kind = entity.rsplit(" ", 1)[0]
    return re.search(r"\b" + re.escape(kind) + r"\b", goal, re.I) is not None


def decision_card(atlas, working, goal):
    state = working.get("public_state", {})
    here = state.get("location")
    held = state.get("held", [])
    known = [name for name in state.get("entities", []) if _mentions(goal, name)]
    locations = state.get("object_locations", {})
    known.sort(key=lambda name: (name not in held, locations.get(name) != here,
                                  name not in locations, name))
    focus = list(dict.fromkeys([*held, *known[:4]]))[:5]
    relevant = {"alf:" + name for name in focus}
    if here and here != "room":
        relevant.add("alf:" + here)
    relevant.update("alf:" + place for place in state.get("containers", {}) if _mentions(goal, place))
    slot_rank = {"held": 0, "object_location": 1, "container_open": 2, "cleaned": 3, "temperature": 3}
    claims = [claim for claim in atlas.nodes("Claim") if claim["entity"] in relevant and claim["predicate"] in slot_rank]
    claims.sort(key=lambda c: (slot_rank[c["predicate"]], c["entity"].removeprefix("alf:") not in held,
                               -c["observed_at_revision"], c["entity"]))
    rows = []
    for claim in claims[:5]:
        value = claim["value"]
        if isinstance(value, str):
            value = value.removeprefix("alf:")
        ready = claim["support_status"] == "supported" and not claim.get("conflicted")
        label = {"object_location": "last observed location", "held": "holding", "container_open": "open",
                 "cleaned": "cleaned", "temperature": "last observed temperature"}[claim["predicate"]]
        rows.append({"text": f"{claim['entity'].removeprefix('alf:')} · {label}: {value if ready else 'pending'}",
                     "claim": claim, "source_refs": claim["evidence_refs"] if ready else []})
    if not any(locations.get(name) or name in held for name in known):
        rows.append({"text": "Target object's location remains unconfirmed.", "source_refs": []})
    elif here and state.get("containers", {}).get(here) is False:
        rows.append({"text": f"Open {here} before trying to reach its contents.", "source_refs": []})
    return rows


def historical_quote(atlas, working, rows, visible):
    """Return one relevant exact span that is absent from the shown observations."""
    for row in rows:
        for ref in row["source_refs"]:
            if ref["id"] == working.get("observation_id"):
                continue
            observation = atlas.evidence(ref["id"])
            if observation["kind"] != "observation" or observation.get("terminal"):
                continue
            excerpt = observation["text"][ref["start"]:ref["end"]]
            if 0 < len(excerpt) <= 180 and not any(excerpt.casefold() in text.casefold() for text in visible):
                return {**ref, "text": excerpt, "entity": row["claim"]["entity"], "predicate": row["claim"]["predicate"]}
    return None
