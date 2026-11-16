"""Public readiness checks for ScienceWorld's potentially terminal focus action."""
import re

from .interaction import ActionAdapter, action_text
from .state_graph import PublicFact


FOCUS = re.compile(r"\bfocus on\s+([^.!?;]+)", re.I)
CONDITIONAL = re.compile(r"\b(?:if|when|after|once|before)\b", re.I)
UNRESOLVED = re.compile(
    r"\b(?:the thing|it|the substance|with the|which|that you|you created|life stages|correct)\b", re.I)


def focus_target(text):
    value = " ".join(text.lower().strip().rstrip(".").split())
    value = re.sub(r"^(?:the|a|an) ", "", value)
    return re.split(r"\s+(?:in|on)\s+", value, maxsplit=1)[0]


def focus_requirements(goal):
    """Return explicitly named focus stages; retain unresolved stages as pending."""
    requirements = []
    for sentence in re.split(r"(?<=[.!?;])\s+", goal):
        for match in FOCUS.finditer(sentence):
            description = match[1].strip()
            pending = CONDITIONAL.search(sentence[:match.start()]) or UNRESOLVED.search(description)
            # A public task can identify its substance before instructing "focus on the substance".
            if (not CONDITIONAL.search(sentence[:match.start()])
                    and description.lower() == "the substance"):
                substance = re.search(
                    r"\b(?:change the state of matter of|boil|melt|freeze) ([^.;!?]+)", goal, re.I)
                if substance:
                    description, pending = substance[1], False
            requirements.append(None if pending else focus_target(description))
    return requirements


def confirmed_focuses(history):
    for turn in history:
        command = turn.get("command") or ""
        if not command.lower().startswith("focus on ") or turn.get("is_error"):
            continue
        match = re.search(r"\byou focus on\s+([^.!?;]+)", turn.get("observation", ""), re.I)
        if match and focus_target(match[1]) == focus_target(command[9:]):
            yield focus_target(match[1])


def focus_guard(command, goal, history):
    if not command.lower().startswith("focus on "):
        return None
    stages = focus_requirements(goal)
    for completed in confirmed_focuses(history):
        if stages and stages[0] == completed:
            stages.pop(0)
    if not stages or stages[0] is None:
        return "Focus target or experimental prerequisite is pending in the public state"
    if focus_target(command[9:]) != stages[0]:
        return "Focus target conflicts with the next explicitly requested object"
    return None


_OBJECT = r"[a-z][a-z0-9 _'-]{0,70}?"
_DEGREES = r"(-?\d+(?:\.\d+)?)\s*(?:degrees?\s+celsius|°\s*c)\b"
_PUBLIC_PATTERNS = {
    "room": re.compile(r"\b(?:this room is called|this outside location is called|you move through the door to) "
                       r"(?:the )?(" + _OBJECT + r")\.", re.I),
    "object_location": re.compile(r"\byou (?:move|put) (?:the )?(" + _OBJECT + r") (?:to|into) (?:the )?(" + _OBJECT + r")\.", re.I),
    "material_state": re.compile(r"\b(?:the )?(" + _OBJECT + r") is (?:now )?(solid|liquid|gaseous|boiling|melting|frozen)\b", re.I),
    "temperature_c": re.compile(r"\b(?:the temperature of (?:the )?(" + _OBJECT + r") is|(?:the )?(" + _OBJECT
                                 + r") (?:has a temperature of|is at))\s+" + _DEGREES, re.I),
    "melting_point_c": re.compile(r"\bthe melting point of (?:the )?(" + _OBJECT + r") is\s+" + _DEGREES, re.I),
    "circuit_connection": re.compile(r"\b(terminal \d+ on " + _OBJECT + r") is (now connected|disconnected) (?:to|from) "
                                     r"(terminal \d+ on " + _OBJECT + r")(?=[.\n]|$)", re.I),
    "device_power": re.compile(r"\b(?:the )?(" + _OBJECT + r") (?:is (?:now )?turned|turns) (on|off)\b", re.I),
}
_THRESHOLD = re.compile(r"\bif the (temperature|melting point) of (?:the )?(" + _OBJECT
                        + r") is (above|below) " + _DEGREES + r"\s*,\s*focus on ([^.!?;]+)", re.I)
_THRESHOLD_REVERSED = re.compile(r"\bif the (" + _OBJECT + r") (temperature) is (above|below) "
                                 + _DEGREES + r"\s*,\s*focus on ([^.!?;]+)", re.I)
_META_COMMANDS = {"reset task", "task", "task description", "help", "teleport", "quit", "check valid actions"}
_INSTRUMENT_RESULT = re.compile(r"^\s*the thermometer measures a temperature of\s+" + _DEGREES + r"\.?\s*$", re.I)
_MEASUREMENT_ACTION = re.compile(r"use (?:the )?thermometer on (?:the )?(" + _OBJECT + r")$", re.I)
_PHASE_NAME = re.compile(r"^(solid|liquid|gaseous) (.+)$", re.I)


def science_entity(value):
    return "sci:" + " ".join(value.lower().strip().removeprefix("the ").split())


def scienceworld_facts(record):
    text = record["text"]
    if re.match(r"\s*(?:task description:|your task is to)", text, re.I):
        return
    instrument_reading = _INSTRUMENT_RESULT.fullmatch(text)
    if instrument_reading:
        action = _MEASUREMENT_ACTION.fullmatch(record.get("action_command") or "")
        if action:
            yield PublicFact(science_entity(action[1]), "temperature_c", instrument_reading[1],
                             *instrument_reading.span(1))
    pattern = r"\b(?:on|in|inside) the (" + _OBJECT + r")\s+(?:is|are)\s*:\s*(?:an?|the) (" + _OBJECT + r")(?=\.|,|\n|$)"
    for match in re.finditer(pattern, text, re.I):
        yield PublicFact(science_entity(match[2]), "object_location", science_entity(match[1]), *match.span())
    for predicate, pattern in _PUBLIC_PATTERNS.items():
        for m in pattern.finditer(text):
            if predicate == "room":
                yield PublicFact("task", predicate, science_entity(m[1]), *m.span())
            elif predicate == "temperature_c":
                # Store exact numeric text as a scalar. Conversion happens only after parsing a typed fact.
                yield PublicFact(science_entity(m[1] or m[2]), predicate, m[3], *m.span())
            elif predicate == "circuit_connection":
                # The circuit edge has its own slot, so another edge on either terminal cannot overwrite it.
                edge = " / ".join(sorted((m[1].lower(), m[3].lower())))
                yield PublicFact(science_entity(edge), predicate, m[2].lower() == "now connected", *m.span())
            else:
                value = science_entity(m[2]) if predicate == "object_location" else m[2].lower()
                yield PublicFact(science_entity(m[1]), predicate, value, *m.span())


def threshold_rules(goal):
    for pattern in (_THRESHOLD, _THRESHOLD_REVERSED):
        for m in pattern.finditer(goal):
            field, material = (m[1], m[2]) if pattern is _THRESHOLD else (m[2], m[1])
            yield (science_entity(material), field.lower().replace(" ", "_") + "_c", m[3].lower(),
                   float(m[4]), focus_target(m[5]))


def thermal_subject(label):
    name = focus_target(label)
    phase = _PHASE_NAME.fullmatch(name)
    return science_entity(phase[2] if phase else name)


def thermal_trial(goal, history):
    """Find solid and liquid readings bound to the same subject by public receipts."""
    melting = [row for row in threshold_rules(goal) if row[1] == "melting_point_c"]
    subjects = {thermal_subject(entity.removeprefix("sci:")) for entity, *_ in melting}
    if len(subjects) != 1:
        return None
    subject = subjects.pop()
    samples = []
    for index, turn in enumerate(history):
        command = turn.get("command") or ""
        reading = _INSTRUMENT_RESULT.fullmatch(turn.get("observation", ""))
        action = _MEASUREMENT_ACTION.fullmatch(command)
        if not reading or not action or turn.get("is_error") or thermal_subject(action[1]) != subject:
            continue
        phase_name = _PHASE_NAME.fullmatch(focus_target(action[1]))
        phase = phase_name[1] if phase_name else None
        if phase is None and index > 0:
            prior = history[index - 1]
            for fact in scienceworld_facts({"text": prior.get("observation", "")}):
                if fact.predicate == "material_state" and thermal_subject(fact.entity.removeprefix("sci:")) == subject:
                    phase = "solid" if fact.value == "frozen" else fact.value
        samples.append({"temperature_c": float(reading[1]), "phase": phase, "turn": index,
                        "command": command, "phase_source_turn": index if phase_name else index - 1 if phase else None})
    solid = [x for x in samples if x["phase"] == "solid"]
    liquid = [x for x in samples if x["phase"] == "liquid"]
    lower = max(solid, key=lambda x: x["temperature_c"], default=None)
    upper = min(liquid, key=lambda x: x["temperature_c"], default=None)
    # Native integer thermometer readings are rounded. Keep the conservative bounds.
    interval = [lower["temperature_c"] - .5, upper["temperature_c"] + .5] if lower and upper else None
    if interval and interval[0] >= interval[1]:
        interval = None
    shown = list({row["turn"]: row for row in [*samples[-2:], lower, upper] if row}.values())
    return {"subject": subject, "samples": shown, "interval_c": interval,
            "transition_observed": bool(lower and upper), "thresholds_c": sorted({row[3] for row in melting})}


def thermal_preparation_priority(command, trial, history):
    """Reserve executable inspection and physical preparation actions in a small menu."""
    if not trial:
        return 0
    name = trial["subject"].removeprefix("sci:")
    samples = trial["samples"]
    phase = next((row["phase"] for row in sorted(samples, key=lambda row: -row["turn"]) if row["phase"]), None)
    if phase is None:
        phase = "liquid" if f"liquid {name}" in command else "solid" if f"solid {name}" in command else None
    device = r"\b(?:freezer)\b" if phase == "liquid" else r"\b(?:stove|oven|hot plate|blast furnace)\b"
    recent = {row.get("command") for row in history[-3:]}
    if command in recent:
        return 0
    if name in command and command.startswith("use thermometer on "):
        return 22 if not samples else 13
    if name in command and command.startswith(("move ", "pour ")) and re.search(device, command):
        return 25
    if re.search(device, command) and command.startswith(("activate ", "turn on ", "open ", "look at ")):
        return 17
    if name in command and command.startswith("look at "):
        return 12
    if samples and command in {"wait", "wait1"}:
        return 8
    return 0


def public_focus_stage(goal, history, state, *, thermal=False):
    stages = focus_requirements(goal)
    completed = list(confirmed_focuses(history))
    for target in completed:
        if stages and stages[0] and (stages[0] == target or thermal and
                                    thermal_subject(stages[0]) == thermal_subject(target)):
            stages.pop(0)
    if stages and stages[0]:
        return stages[0], "explicit_public_stage"
    rules = list(threshold_rules(goal))
    if not stages or not rules:
        return None, "pending_unresolved_target"
    matches = set()
    trial = thermal_trial(goal, history) if thermal else None
    for entity, predicate, comparator, threshold, target in rules:
        value = state.get("facts", {}).get(entity, {}).get(predicate)
        if value is None and thermal:
            canonical = thermal_subject(entity.removeprefix("sci:"))
            value = state.get("facts", {}).get(canonical, {}).get(predicate)
        if value is not None and (float(value) > threshold if comparator == "above" else float(value) < threshold):
            matches.add(target)
        if value is None and predicate == "melting_point_c" and trial and trial["interval_c"]:
            lo, hi = trial["interval_c"]
            beyond_threshold = lo > threshold if comparator == "above" else hi < threshold
            if beyond_threshold:
                matches.add(target)
    if len(matches) == 1:
        target = matches.pop()
        if target not in completed:
            return target, "confirmed_public_measurement"
        return None, "public_focus_already_confirmed"
    return None, "pending_public_measurement"


class ScienceWorldActions(ActionAdapter):
    """Decision slots and versioned facts derived from simulator text."""

    parse = staticmethod(action_text)
    public_facts = staticmethod(scienceworld_facts)
    state_predicates = frozenset(_PUBLIC_PATTERNS)
    independent_decision_slots = True
    observe_with_public_receipt = True
    thermal = False

    def normalize(self, command):
        return " ".join(command.lower().strip().rstrip(".").split())

    def observe(self, observation, previous, command, clickables):
        facts = {key: values.copy() for key, values in previous.get("facts", {}).items()}
        for fact in self.public_facts({"text": observation, "action_command": command}):
            facts.setdefault(fact.entity, {})[fact.predicate] = fact.value
        return {"facts": facts}

    def decision_targets(self, state, goal):
        slots = [(entity, predicate) for entity, predicate, *_ in threshold_rules(goal)]
        if self.thermal:
            for entity, predicate, *_ in threshold_rules(goal):
                if predicate == "melting_point_c":
                    canonical = thermal_subject(entity.removeprefix("sci:"))
                    slots.extend((canonical, slot) for slot in ("temperature_c", "material_state", "object_location"))
        for subject in focus_requirements(goal):
            if subject:
                slots.append((science_entity(subject), "object_location"))
                if re.search(r"\b(?:measure|temperature|melting point)\b", goal, re.I):
                    slots.append((science_entity(subject), "temperature_c"))
        return list(dict.fromkeys(slots))[:8]


class ThermalScienceWorldActions(ScienceWorldActions):
    thermal = True

    @staticmethod
    def public_facts(record):
        for fact in scienceworld_facts(record):
            yield fact
            label = fact.entity.removeprefix("sci:")
            entity = thermal_subject(label)
            if entity != fact.entity and fact.entity.startswith("sci:"):
                yield PublicFact(entity, fact.predicate, fact.value, fact.start, fact.end)
                phase = _PHASE_NAME.fullmatch(focus_target(label))
                if phase and fact.predicate == "temperature_c":
                    yield PublicFact(entity, "material_state", phase[1], fact.start, fact.end)


def science_command_guard(command, goal, state, history, observation, *, thermal=False):
    if command in _META_COMMANDS or command.startswith(("reset ", "teleport ")):
        return "meta_command"
    if command.isdigit() and not re.search(r"(?im)(?:ambiguous|which (?:one|object)|multiple options)", observation):
        return "no_public_disambiguation_request"
    if not command.startswith("focus on "):
        return None
    target, reason = public_focus_stage(goal, history, state, thermal=thermal)
    if target is None:
        return reason
    selected = focus_target(command[len("focus on "):])
    if selected != target and not (thermal and thermal_subject(selected) == thermal_subject(target)):
        return "different_public_focus_target"
    return None
