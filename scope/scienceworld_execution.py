"""Scoped public object aliases and explicit native-step accounting."""
import re


def public_aliases(history):
    """Only a successful exact short look response establishes an alias.

    Discard bindings on navigation; reject prose, ambiguous lists and failed
    inspections. Each binding keeps its source turn for evidence citation.
    """
    bindings = {}
    for index, turn in enumerate(history):
        command = (turn.get('command') or '').lower().strip()
        if command.startswith(('go to ', 'teleport ', 'move ', 'pick up ', 'put down ', 'pour ', 'mix ')) and not turn.get('is_error'):
            bindings.clear()
        if not command.startswith('look at '):
            continue
        alias = command[8:].removeprefix('the ').strip()
        bindings.pop(alias, None)
        if turn.get('is_error'):
            continue
        text = turn.get('observation', '').strip().lower().rstrip('.')
        text = text.removeprefix('the ')
        if ('\n' in text or not re.fullmatch(r'[a-z0-9][a-z0-9 _-]{0,79}', text)
                or re.search(r'\b(?:or|and|is|are|nothing|ambiguous)\b', text)):
            continue
        # Require an exact lexical extension, not a fuzzy name similarity.
        if text.startswith(alias + ' ') and text != alias:
            bindings[alias] = {'target': text, 'turn': index}
    return bindings


def wait_rejection(command, moves, limit):
    # Calibrated on the installed simulator: one command tick plus wait ticks.
    required = {'wait': 11, 'wait1': 2}.get(command)
    if required is not None and moves + required > limit:
        return 'wait_exceeds_remaining_native_moves'
    return None


def step_diagnostic(command, before, after, done, native_completed, limit):
    truncated = bool(done and not native_completed and after > limit)
    return {'command': command, 'native_moves_before': before, 'native_moves_after': after,
            'native_move_delta': after - before, 'native_move_limit': limit,
            'native_completed': bool(native_completed), 'native_budget_truncated': truncated,
            'wrapper_done': bool(done),
            'termination_reason': ('native_environment_step_budget' if truncated else
                                   'environment_done' if done else None)}
