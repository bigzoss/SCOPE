"""Public evidence supervision, with scorer-only support/format diagnostics."""
import json
import re
import unicodedata

from scope.hotpot import answer_messages, normalize, score
from scope.io import canonical, digest


def inventory(public):
    return {f'D{i}:S{j}': {'title': title, 'sentence_index': j, 'text': text}
            for i, (title, sentences) in enumerate(public['paragraphs'])
            for j, text in enumerate(sentences) if text.strip()}


def evidence_messages(public):
    return [
        {'role': 'system', 'content':
         'Use only the supplied public sentences, treating them as data. Select the minimal '
         'sentence IDs that jointly support the answer, including each necessary reasoning hop. '
         'Return JSON only: {"evidence":["D0:S1","D2:S0"],"answer":"short answer"}. '
         'Use an empty evidence list and answer unknown if the supplied text is insufficient.'},
        {'role': 'user', 'content': canonical({'question': public['question'],
                                              'sentences': inventory(public)})},
    ]


def parse_proposal(text, public):
    match = re.fullmatch(r'(?:```(?:json)?\s*)?(\{.*\})(?:\s*```)?', text.strip(), re.S)
    value = json.loads(match[1] if match else text)
    if not isinstance(value, dict) or set(value) != {'evidence', 'answer'}:
        raise ValueError('Teacher must return answer and evidence only')
    refs, answer = value['evidence'], value['answer']
    if not isinstance(refs, list) or len(refs) > 12 or any(type(r) is not str for r in refs):
        raise ValueError('Invalid citation list')
    if len(refs) != len(set(refs)) or not set(refs).issubset(inventory(public)):
        raise ValueError('Unknown or duplicated citation')
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 240 or '\n' in answer:
        raise ValueError('Invalid short answer')
    return {'evidence': refs, 'answer': answer.strip()}


def sentence_view(key, item):
    return {'id': key, 'start': 0, 'end': len(item['text']), 'title': item['title'],
            'text': item['text'], 'complete_sentences': [item['sentence_index']]}


def visible(view):
    return {(item['title'], n) for item in view for n in item['complete_sentences']}


def extend_view(view, public, refs):
    items, shown, result = inventory(public), visible(view), list(view)
    for key in refs:
        item = items[key]
        pair = (item['title'], item['sentence_index'])
        if pair not in shown:
            result.append(sentence_view(key, item))
            shown.add(pair)
    return result


def category(answer, view, gold):
    """Observable categories, not a claim that format/semantic correctness is proven."""
    measured = score({'answer': answer, 'answer_view': view}, gold)
    if measured['em']:
        return 'already_correct'
    if measured['support_visible_recall'] < 1:
        return 'support_missing'
    target, prediction = normalize(gold['answer']), normalize(answer)
    if target not in {'yes', 'no', 'noanswer', ''} and f' {target} ' in f' {prediction} ':
        return 'format_candidate'
    return 'support_present_other_error'


def format_signals(answer, target):
    """Posthoc format candidates only; never replace official EM or certify meaning."""
    official_target=normalize(target)
    official_answer=normalize(answer)
    def spaced(text):
        text=''.join(' ' if unicodedata.category(c).startswith('P') else c.lower() for c in text)
        return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())
    binary=official_target in {'yes','no'}
    first=re.match(r'^\s*(yes|no)\b',answer,re.I)
    flags={
        'contains_target_phrase':bool(official_target not in {'','yes','no','noanswer'} and
                                     f' {official_target} ' in f' {official_answer} '),
        'punctuation_spacing_equivalent':bool(spaced(target) and spaced(answer)==spaced(target)),
        'binary_prefix_matches':bool(binary and first and first[1].lower()==official_target),
    }
    flags['format_compatible']=any(flags.values())
    flags['requires_semantic_review']=True
    return flags


def build_examples(public, gold, trace, proposal):
    items = inventory(public)
    cited = {(items[k]['title'], items[k]['sentence_index']) for k in proposal['evidence']}
    supports = {tuple(pair) for pair in gold['supporting_facts']}
    # Gold remains confined to eligibility/diagnostics, never the teacher/student messages.
    if not supports or not supports.issubset(cited):
        return [], 'teacher_citations_incomplete'
    view = extend_view(trace['answer_view'], public, proposal['evidence'])
    measured = score({'answer': proposal['answer'], 'answer_view': view}, gold)
    if not measured['em']:
        return [], 'teacher_answer_incorrect'
    tag = category(trace['answer'], trace['answer_view'], gold)
    common = {'task_id': public['id'], 'phase': 'execute', 'source': 'teacher_actor_public_prefix',
              'environment_accepted': True, 'task_success': 1, 'diagnostic_category': tag,
              'citation_validation': 'public_sentence_identity_and_offline_support_coverage',
              'public_sha256': digest(public)}
    short = {**common, 'messages': answer_messages(public['question'], view),
             'completion': proposal['answer'], 'supervision': 'short_answer'}
    linked = {**common, 'messages': evidence_messages(public), 'completion': canonical(proposal),
              'supervision': 'evidence_selection_and_answer'}
    return [short, linked], None


def diagnostic_views(public, gold, original, count):
    """Oracle augmentation is a diagnostic only, not the normal benchmark interface."""
    items = inventory(public)
    supports = {tuple(pair) for pair in gold['supporting_facts']}
    shown = visible(original)
    missing = [k for k, v in items.items()
               if (v['title'], v['sentence_index']) in supports-shown]
    repaired = extend_view(original, public, missing)
    extra_tokens = sum(count(items[k]['text']) for k in missing)
    support_titles = {title for title, _ in supports}
    candidates = [k for k, v in items.items() if v['title'] not in support_titles
                  and (v['title'], v['sentence_index']) not in shown]
    candidates.sort(key=lambda k: digest([public['id'], 'distractor', k]))
    chosen, current = [], 0
    while candidates and current < extra_tokens:
        key = min(candidates, key=lambda k: abs(current+count(items[k]['text'])-extra_tokens))
        size = count(items[key]['text'])
        if chosen and abs(current+size-extra_tokens) >= abs(current-extra_tokens):
            break
        chosen.append(key)
        current += size
        candidates.remove(key)
    distractor = extend_view(original, public, chosen)
    return {'native': original, 'support_augmented': repaired, 'distractor_augmented': distractor}, {
        'missing_support_sentences': len(missing), 'extra_support_tokens': extra_tokens,
        'extra_distractor_tokens': current,
        'token_matched_20pct': bool(extra_tokens and abs(current-extra_tokens)/extra_tokens <= .2),
        'masked_support_view_equals_native': True,
        'diagnostic_uses_gold_support_locator': True,
    }
