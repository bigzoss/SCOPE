"""Budgeted action menus driven by public experiment prerequisites."""
from collections import Counter
import re

from .scienceworld_actions import confirmed_focuses, thermal_subject


def words(text):
    return set(re.findall(r"[a-z]+\d*", text.lower())) - {"the", "a", "to", "in", "on", "of", "with", "and"}


def experiment_stage(goal, history, trial, focus, allowed_focus):
    stage = {"phase": "find_required_object", "target": focus,
             "pending": [], "completed_focus": list(confirmed_focuses(history))}
    if allowed_focus:
        stage.update(phase="explicit_focus" if focus else "submit_measurement", pending=[])
        return stage
    if not trial:
        stage["phase"] = "find_required_object" if focus else "perform_experiment"
        stage["pending"] = ["Locate required objects and confirm action prerequisites in public observations"]
        return stage
    name = trial["subject"].removeprefix("sci:")
    stage["target"] = name
    if focus:
        stage['target'] = focus
        stage["pending"] = ["Locate the next explicitly requested focus object"]
        return stage
    if trial["interval_c"]:
        stage.update(phase="resolve_measurement", pending=["Check whether the observed interval resolves the requested threshold"])
        return stage
    location, powered, refs = None, {}, []
    for index, row in enumerate(history):
        if row.get("is_error"):
            continue
        text = row.get("observation", "").lower()
        # A command alone is never a confirmation. Follow the latest public move.
        moved = re.search(r"you move (?:the )?(.+?) to (?:the )?([^.!\n]+)", text)
        if moved and thermal_subject(moved[1]) == trial["subject"]:
            location = moved[2].strip()
            refs.append(index)
        for device in ("stove", "oven", "hot plate", "blast furnace", "freezer"):
            match = re.search(r"\b" + device + r" (?:is (?:now )?|has been )(activated|deactivated|turned on|turned off|on|off)\b", text)
            if match:
                powered[device] = match[1] in {"activated", "turned on", "on"}
                refs.append(index)
    samples = trial["samples"]
    phase = next((s["phase"] for s in sorted(samples, key=lambda s: -s["turn"]) if s["phase"]), "solid")
    devices = ("freezer",) if phase == "liquid" else ("stove", "oven", "hot plate", "blast furnace")
    device = next((d for d in devices if location and re.search(r"\b" + d + r"\b", location)), None)
    stage.update(location=location, device=device, powered=powered.get(device), public_turns=sorted(set(refs)),
                 material_phase=phase)
    if not device:
        stage.update(phase="place_sample", pending=["Confirm sample placement in a suitable heating/cooling device"])
    elif powered.get(device) is not True:
        stage.update(phase="activate_device", pending=["Confirm that the device is switched on"])
    else:
        stage.update(phase="observe_transition", pending=["Advance the experiment and measure the same sample until phase evidence resolves the threshold"])
    return stage


def select_experiment_menu(commands, goal, history, trial, focus, allowed_focus, limit=8):
    commands = sorted(set(commands))
    stage = experiment_stage(goal, history, trial, focus, allowed_focus)
    recent = Counter(row.get("command") for row in history[-8:])
    visits = Counter(row.get("command") for row in history if (row.get("command") or "").startswith("go to "))
    target = words(stage.get("target") or "")
    # Final answer receptacles must not dominate exploration of an earlier stage.
    objective = re.split(r"\bif\b", goal.lower(), maxsplit=1)[0]
    relevance = words(objective)
    navigation = lambda c: c.startswith("go to ")
    doorway = lambda c: c.startswith("open door")
    inspection = lambda c: c.startswith(("look", "examine ", "read ", "inventory"))
    phase = stage["phase"]
    name = (trial or {}).get("subject", "").removeprefix("sci:")
    device = stage.get("device")

    def preparation(c):
        if not trial or focus:
            return 0
        if phase == "place_sample":
            devices = ("freezer",) if stage.get("material_phase") == "liquid" else ("stove", "oven", "hot plate", "blast furnace")
            return 45 if name in c and c.startswith(("move ", "pour ")) and any(d in c for d in devices) else 0
        if phase == "activate_device":
            return 55 if device in c and c.startswith(("activate ", "turn on ")) else 15 if device in c and inspection(c) else 0
        if phase == "observe_transition":
            last = history[-1].get("command", "") if history else ""
            if c in {"wait", "wait1"}:
                return 38 if last != c else 12
            if name in c and c.startswith("use thermometer on "):
                return 40 if last in {"wait", "wait1"} else 22
            if name in c and inspection(c):
                return 20
        return 0

    def rank(c):
        tokens = words(c)
        score = 6 * len(tokens & target) + len(tokens & relevance) - 8 * recent[c]
        if c in allowed_focus:
            score += 100
        score += preparation(c)
        if navigation(c):
            score += 12 - 5 * visits[c]
            if c.startswith("go to door to "):
                score -= 8
        if doorway(c):
            score += 10
        if c.startswith("connect ") and not re.search(r"\b(?:circuit|electric|conductivity|conductive)\b", goal, re.I):
            score -= 30
        if c.startswith(("move ", "pour ", "mix ")) and phase == "find_required_object":
            score -= 20
        return -score, c

    ranked = sorted(commands, key=rank)
    categories = [lambda c: c in allowed_focus, lambda c: preparation(c) >= 30,
                  navigation, doorway, inspection]
    reserved = [next((c for c in ranked if category(c)), None) for category in categories]
    selected = list(dict.fromkeys(c for c in [*reserved, *ranked] if c))[:limit]
    return selected, {"experiment_stage": stage, "eligible_movement": sum(navigation(c) for c in commands),
                      "offered_movement": [c for c in selected if navigation(c)],
                      "offered_preparation": [c for c in selected if preparation(c) >= 30]}


def select_continuous_experiment_menu(commands, goal, history, trial, focus, allowed_focus, limit=8):
    """Reserve evidence-gathering actions throughout an observed thermal stage."""
    selected, audit = select_experiment_menu(commands, goal, history, trial, focus, allowed_focus, limit)
    stage = audit['experiment_stage']
    audit['stage_candidate_count'] = 0
    if not trial or focus or allowed_focus:
        return selected, audit
    phase = stage['phase']
    name = trial['subject'].removeprefix('sci:')
    device = stage.get('device')
    commands = sorted(set(commands))
    sample = lambda c: name in c and c.startswith(('look at ', 'use thermometer on '))
    measure = [c for c in commands if sample(c) and c.startswith('use thermometer on ')]
    inspect = [c for c in commands if sample(c) and c.startswith('look at ')]
    if phase == 'observe_transition':
        waits = [c for c in commands if c in {'wait1', 'wait'}]
        # The caller already applies calibrated remaining-budget checks.
        if 'wait1' in waits:
            waits = ['wait1']
        core = measure + waits + inspect
        stage['next_evidence'] = 'Measure the same sample, allow heating/cooling time, and inspect its phase.'
        stage['preserve'] = 'Keep the sample in the powered device while gathering transition evidence.'
    elif phase == 'activate_device':
        core = [c for c in commands if device and device in c and c.startswith(('activate ', 'turn on '))]
        core += [c for c in commands if device and device in c and c.startswith('look at ')]
        core += inspect
        stage['next_evidence'] = 'Confirm device power from the observation after activation.'
    elif phase == 'place_sample':
        devices = ('freezer',) if stage.get('material_phase') == 'liquid' else ('stove', 'oven', 'hot plate', 'blast furnace')
        core = [c for c in commands if name in c and c.startswith(('move ', 'pour '))
                and any(re.search(r'\b' + d + r'\b', c) for d in devices)]
        # Navigation remains necessary when no local preparation is executable.
        if not core:
            return selected, audit
        core += inspect + measure
        stage['next_evidence'] = 'Confirm placement of this sample in the selected device.'
    else:
        return selected, audit
    core = list(dict.fromkeys(core))
    if not core:
        return selected, audit
    # Reserve both measurement and waiting, even after repeated measurements.
    selected = core[:limit]
    if 'look around' in commands and len(selected) < limit:
        selected.append('look around')
    audit.update(stage_candidate_count=len(core), offered_preparation=core[:limit],
                 offered_movement=[c for c in selected if c.startswith('go to ')],
                 deferred_command_count=sum(c not in selected for c in commands))
    return selected, audit


def select_progress_experiment_menu(commands, goal, history, trial, focus, allowed_focus, limit=8):
    """Retire repeated identical observations only while preparation is available.

    A physical action (including a failed attempt) reopens inspection. During
    heating/cooling, measurements remain available because time can change state.
    """
    selected, audit = select_continuous_experiment_menu(
        commands, goal, history, trial, focus, allowed_focus, limit)
    stage = audit['experiment_stage']
    audit['no_information_deferred'] = []
    if focus or allowed_focus or stage['phase'] not in {'place_sample', 'activate_device'}:
        return selected, audit
    preparation = [c for c in selected if c.startswith(('move ', 'pour ', 'activate ', 'turn on '))]
    if not preparation:
        return selected, audit
    def observational(command):
        return command.startswith(('look', 'examine ', 'use thermometer on ', 'inventory'))
    window = []
    for row in reversed(history):
        if not observational(row.get('command', '')):
            break
        window.append(row)
    deferred = []
    for command in selected:
        matching = [r for r in window if r.get('command') == command]
        if len(matching) < 2 or any(r.get('is_error') for r in matching[:2]):
            continue
        observations = [' '.join(r.get('observation', '').split()).lower() for r in matching[:2]]
        if observations[0] and observations[0] == observations[1]:
            deferred.append({'command': command, 'reason': 'repeated_identical_public_observation',
                             'count': len(matching)})
    excluded = {r['command'] for r in deferred}
    selected = [c for c in selected if c not in excluded]
    audit['no_information_deferred'] = deferred
    audit['offered_preparation'] = preparation
    return selected, audit
