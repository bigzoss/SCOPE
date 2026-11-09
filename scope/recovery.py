"""Public-state recovery shared by adapters, online rollout and teacher prefixes.

Candidates are suggestions, never automatically dispatched. No hidden state is read.
"""
import re

from .io import digest


def prepare_recovery(adapter, atlas, working, goal):
    state = working.get("public_state", {})
    key = digest(adapter.progress_key(state))
    failures = working.get("recovery_attempts", [])
    blocked = sorted({r["command"] for r in failures if r["state"] == key and
                      sum(x["state"] == key and x["command"] == r["command"] for x in failures) >= 2})
    suggestions = []
    items = list(adapter.recovery_candidates(state, goal))
    items.sort(key=lambda item: sum(x["state"] == key and x["command"] == item["command"] for x in failures))
    for item in items:
        command = item["command"]
        if command in blocked or adapter.bind(command, state).status not in {"bound", "not_required"}:
            continue
        # Every entity used by a candidate must have appeared in a public observation.
        names = item.get("source_labels", re.findall(r"\b[a-z][a-z0-9_]* \d+\b", command, re.I))
        refs = []
        for name in names:
            record = next((e for e in reversed(atlas.all_evidence()) if e["kind"] == "observation" and
                           (name in e.get("clickables", []) or re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                                     re.split(r"(?im)^\s*Your task is to\b", e["text"])[0], re.I))), None)
            if record is None:
                break
            refs.append(record["id"])
        else:
            suggestions.append({**item, "evidence_ids": sorted(set(refs)), "status": "proposed_not_confirmed"})
        if len(suggestions) >= 8:
            break
    targets = list(adapter.decision_targets(state, goal))
    working["decision_targets"] = targets[:8]
    working["recovery_plan"] = {"state_key": key, "goal": goal,
        "blocked_commands": blocked if suggestions else [], "candidates": suggestions,
        "repeated_commands": blocked,
        "known_entities": state.get("entities", [])[:64],
        "visited_locations": state.get("visited_locations", []),
        "held": state.get("held", []), "unresolved_slots": targets[:8],
        "reason": "repeated_no_progress" if blocked else "public_action_grounding"}
    return working["recovery_plan"]


def record_recovery(working, action):
    """Count failed/no-progress proposals at their pre-action state, including blocked retries."""
    command = action.get("normalized_command") or action.get("raw_command")
    if command and (action.get("action_status") != "accepted" or
                   action.get("progress") == "legal_no_progress" or
                   action.get("information_gain") == "revisited_observation"):
        working["recovery_attempts"] = (working.get("recovery_attempts", []) + [{
            "state": action["public_state_before"], "command": command,
            "reason": action["action_status"], "evidence_id": action["observation_before_id"]}])[-32:]


def information_progress(working, action, before, after):
    """Novel public text is information, not a task reward or a success assertion."""
    seen = working.setdefault("seen_observation_hashes", [])
    old = digest(" ".join(before.split()))
    if old not in seen:
        seen.append(old)
    current = digest(" ".join(after.split()))
    if action["sent"] is False or action["action_status"] != "accepted":
        gain = "no_observation_received"
    else:
        gain = "revisited_observation" if current in seen else "new_observation"
        if current not in seen:
            seen.append(current)
    action["information_gain"] = gain
    action["repeated_invalid_action"] = bool(action.get("loop") and action["action_status"] != "accepted")
    action["state_change_without_new_observation"] = action.get("progress") == "state_changed" and gain == "revisited_observation"
    # A change in public state is deliberately not labelled subgoal completion.
    action["task_progress_verified"] = None


def recovery_diagnostics(events):
    steps = [e for e in events if e.get("phase") == "execute"]
    managers = [e for e in events if e.get("phase") != "execute"]
    from collections import Counter
    return {
        "manager_skip_reasons": dict(Counter(e.get("result", {}).get("reason") for e in managers
                                             if not e.get("sampled") and e.get("result", {}).get("reason"))),
        "sampled_operations": dict(Counter(e.get("result", {}).get("operation") for e in managers if e.get("sampled"))),
        "raw_lookup_calls": sum(e.get("result", {}).get("source") == "raw" for e in managers),
        "repeated_invalid_actions": sum(bool(e.get("repeated_invalid_action")) for e in steps),
        "revisited_observations": sum(e.get("information_gain") == "revisited_observation" for e in steps),
        "state_changes_without_new_observation": sum(bool(e.get("state_change_without_new_observation")) for e in steps),
        "repeated_actions_blocked": sum(e.get("action_status") == "no_progress_blocked" for e in steps),
        "turns_with_recovery_candidates": sum(bool(e.get("recovery_plan", {}).get("candidates")) for e in steps),
        "recovery_candidate_selected": sum(bool(e.get("selected_recovery_candidate")) for e in steps),
        "recovery_candidate_with_state_change": sum(bool(e.get("selected_recovery_candidate")) and
                                                    e.get("progress") == "state_changed" for e in steps),
    }
