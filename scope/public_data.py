"""Public-channel boundary shared by simulator serving and offline SFT import."""
import re
from .contracts import ContractError

TERMINAL_PUBLIC_TEXT="Task finished. The purchase was submitted; scoring is handled separately."
_EVALUATOR_MARKERS=re.compile(r"Your score\s*\(min|Reward\s+Details|goal_options|price_upper|goal-asin|reward_detail|Target\s*(?:\[SEP\]\s*)+asin",re.I)


def assert_public_text(text):
    if not isinstance(text,str):raise ContractError("Public evidence must be text")
    if _EVALUATOR_MARKERS.search(text):
        raise ContractError("Evaluator/hidden-target text in public channel; recollect with sanitized service")
    return text


def project_transition_text(text,done):
    # Upstream HTML-to-text includes CSS-hidden Target/Reward sections on done_page.
    return TERMINAL_PUBLIC_TEXT if done else assert_public_text(text)
