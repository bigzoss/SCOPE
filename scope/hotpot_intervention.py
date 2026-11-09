"""Operation × evidence visibility interventions for the frozen HotpotQA reader."""
import json

from .contracts import ContractError
from .hotpot import answer_from_view, answer_messages, answer_view, document_text, normalize, ordered_documents, score
from .io import digest


def subtract(ref, hidden):
    """Subtract exactly one source interval, retaining unaffected fragments in order."""
    if ref["id"] != hidden["id"] or ref["start"] >= hidden["end"] or ref["end"] <= hidden["start"]:
        return [ref]
    return [{**ref,"start":start,"end":end} for start,end in
            ((ref["start"],min(ref["end"],hidden["start"])),(max(ref["start"],hidden["end"]),ref["end"]))
            if start<end]


def _refs(view):
    return [{k:item[k] for k in ("id","start","end")} for item in view]


def _archive(trace, public):
    documents = {}
    for index,(title,sentences) in enumerate(ordered_documents(public,trace["seed"])):
        text, offsets = document_text(title,sentences)
        documents[f"wiki:{index}"] = (title,text,offsets)
    result = {}
    for _,raw in trace["atlas"]["tables"]["evidence"]:
        record = json.loads(raw)
        if record["kind"] != "observation":
            continue
        document = documents.get(record["entity"])
        if (record["scope"]["task"] != public["id"] or document is None or record["text"]!=document[1]
                or record["content_hash"]!=digest(record["text"])):
            raise ContractError("Answer evidence does not belong to the declared public task")
        title,_,offsets = document
        result[record["id"]] = {**record,"title":title,"sentence_offsets":offsets}
    return result


def _changed_refs(a, b, stage):
    actions = [x["actions"][stage] for x in (a,b)]
    refs = [act["menu"][act["choice"]].get("evidence") for act in actions]
    direct = list({digest(r):r for r in refs if r}.values())
    if direct:
        return direct if len(direct)==1 else []
    kept,modified = sorted(actions,key=lambda act:act["operation"]!="KEEP")
    if kept["operation"]!="KEEP" or modified["operation"] not in {"COMPRESS","CLEAR"}:
        return []
    removed = kept["working_after"]
    for ref in modified["working_after"]:
        removed = [fragment for old in removed for fragment in subtract(old,ref)]
    return removed


def _visible_chars(view, witness):
    return sum(max(0,min(r["end"],witness["end"])-max(r["start"],witness["start"]))
               for r in view if r["id"]==witness["id"])


def aligned_credit(total_delta, interaction):
    """Bound local credit by the measured terminal effect and require sign agreement."""
    if total_delta*interaction<=0:
        return 0.
    return min(total_delta,interaction) if total_delta>0 else max(total_delta,interaction)


def continuation_tokens(trace, stage=-1, include_operation=False):
    """Count input and generated tokens for sampled calls after the given stage."""
    def token_count(sample):
        return len(sample["prompt_ids"])+len(sample["completion_ids"])

    first = stage if include_operation and stage >= 0 else stage + 1
    manager = sum(token_count(action["sample"]) for action in trace["actions"][first:] if action.get("sample"))
    return {"manager":manager,"reader":token_count(trace["answer_sample"])}


def compression_efficiency(a, b, stage, score_a, score_b):
    """Only compare cost after preserving both answer quality and full support."""
    costs = [continuation_tokens(trace,stage,include_operation=True) for trace in (a,b)]
    total_a,total_b = (sum(cost.values()) for cost in costs)
    eligible = (a["actions"][stage]["operation"]=="COMPRESS" and b["actions"][stage]["operation"]=="KEEP"
                and all(s["em"]==1 and s["support_visible_recall"]==1 for s in (score_a,score_b)))
    reasons = []
    if a["actions"][stage]["operation"] != "COMPRESS" or b["actions"][stage]["operation"] != "KEEP":
        reasons.append("not_compress_keep_pair")
    if not all(s["em"] == 1 for s in (score_a, score_b)):
        reasons.append("answer_not_preserved_and_correct")
    if not all(s["support_visible_recall"] == 1 for s in (score_a, score_b)):
        reasons.append("support_not_fully_preserved")
    return {"eligible":eligible,"ineligible_reasons":reasons,"includes_operation_cost":True,
            "tokens_a":costs[0],"tokens_b":costs[1],
            "relative_saving":(total_b-total_a)/max(1,total_a,total_b) if eligible else 0.}


def public_support_inventory(public, gold):
    """Measure the source protocol's support ceiling before any memory policy runs."""
    facts = {tuple(pair) for pair in gold["supporting_facts"]}
    present = {(title,index) for title,sentences in public["paragraphs"] for index,sentence in enumerate(sentences) if sentence}
    return {"expected":len(facts),"outside_public":len(facts-present)}


def length_matched_view(policy, question, visible, masked, witness, prompt_ids):
    """Neutralize one visible interval in place and check the full reader token count."""
    if masked==visible:
        return visible
    if policy.encode(answer_messages(question,visible))!=prompt_ids:
        raise ContractError("Stored reader prompt does not match its visible evidence")
    overlaps = [i for i,r in enumerate(visible) if _visible_chars([r],witness)]
    if len(overlaps)!=1:
        return None
    index = overlaps[0]
    row = visible[index]
    start = max(0,witness["start"]-row["start"])
    end = min(len(row["text"]),witness["end"]-row["start"])
    left,right = row["text"][:start],row["text"][end:]

    def replaced(text):
        return [*visible[:index],{**row,"text":left+text+right},*visible[index+1:]]

    expected = len(prompt_ids)
    gap = expected-len(policy.encode(answer_messages(question,replaced(""))))
    # Try ordinary, non-semantic fillers. Match against the actual rendered chat prompt.
    for filler in (" ·"," _"," ~"):
        for adjustment in (0,-1,1,-2,2,-3,3):
            count = gap+adjustment
            if count<1:
                continue
            view = replaced(filler*count)
            if view!=visible and len(policy.encode(answer_messages(question,view)))==expected:
                return view
    return None


def factorial(policy, public, gold, a, b, stage, *, mask_mode="delete"):
    """Read the two existing continuations again with only their witness visibility changed."""
    if mask_mode not in {"delete","length_matched"}:
        raise ContractError("Unknown HotpotQA evidence mask mode")
    first,second = a["actions"][stage],b["actions"][stage]
    if (a["task_id"]!=b["task_id"] or a["task_id"]!=public["id"] or a["seed"]!=b["seed"]
            or a["split"]!=b["split"] or first["boundary"]!=second["boundary"] or first["choice"]==second["choice"]
            or any(x["choice"]!=y["choice"] for x,y in zip(a["actions"][:stage],b["actions"][:stage],strict=True))):
        raise ContractError("Factorial pair did not preserve the same pre-action state")
    archives = [_archive(trace,public) for trace in (a,b)]
    canonical_sources = [{r["entity"]:r["content_hash"] for r in archive.values()} for archive in archives]
    if canonical_sources[0]!=canonical_sources[1]:
        raise ContractError("Factorial branches have different public archives")
    affected = _changed_refs(a,b,stage)
    available = [r for r in affected if _visible_chars(a["answer_view"],r) or _visible_chars(b["answer_view"],r)]
    common = {"stage":first["stage"],"boundary":first["boundary"],"seed":a["seed"],
              "operations":[first["operation"],second["operation"]],"affected_spans":len(affected),"mask_mode":mask_mode}
    if not available:
        return {**common,"eligible":False,"reason":"no_isolatable_visible_witness","evidence_credit":0.}
    ref = max(available,key=lambda r:(max(_visible_chars(a["answer_view"],r),_visible_chars(b["answer_view"],r)),digest(r)))
    source = next(archive[ref["id"]] for archive in archives if ref["id"] in archive)
    if not 0<=ref["start"]<ref["end"]<=len(source["text"]) or source["revision"]>first["atlas_revision"]:
        raise ContractError("Witness refers to an invalid or future public span")
    text = source["text"][ref["start"]:ref["end"]]
    witness = {**ref,"entity":source["entity"],"revision":source["revision"],
               "content_hash":source["content_hash"],"quote_hash":digest(text)}
    plans = []
    for trace,archive in zip((a,b),archives,strict=True):
        arm_source = next(r for r in archive.values() if r["entity"]==source["entity"])
        arm_ref = {**ref,"id":arm_source["id"]}
        visible = trace["answer_view"]
        if answer_view(archive,_refs(visible)) != visible:
            raise ContractError("Reader view does not reproduce its source ranges")
        fragments = [f for r in _refs(visible) for f in subtract(r,arm_ref)]
        masked = answer_view(archive,fragments)
        if _visible_chars(masked,arm_ref):
            raise ContractError("Witness survived the reader mask")
        quote = " ".join(text.casefold().split())
        other = [public["question"], *[field for r in masked for field in (r["title"],r["text"])]]
        if not quote or any(quote in " ".join(field.casefold().split()) for field in other):
            return {**common,"eligible":False,"reason":"witness_duplicated_outside_source_span",
                    "witness":witness,"evidence_credit":0.}
        reader_view = masked
        if mask_mode=="length_matched":
            reader_view = length_matched_view(policy,public["question"],visible,masked,arm_ref,
                                              trace["answer_sample"]["prompt_ids"])
            if reader_view is None:
                return {**common,"eligible":False,"reason":"could_not_match_reader_prompt_length",
                        "witness":witness,"evidence_credit":0.}
        plans.append((trace,visible,masked,reader_view,arm_ref))
    cells = {}
    for key,(trace,visible,masked,reader_view,arm_ref) in zip(("a","b"),plans,strict=True):
        plus = score(trace,gold)
        visible_input = len(trace["answer_sample"]["prompt_ids"])
        visible_output = len(trace["answer_sample"]["completion_ids"])
        cells[key+"_visible"] = {"score":plus,"answer":trace["answer"],"view_hash":digest(visible),
                                "prompt_tokens":visible_input,"completion_tokens":visible_output}
        if reader_view==visible:
            result, prompt_tokens, completion_tokens = trace["answer"],visible_input,visible_output
        else:
            result,sample = answer_from_view(policy,public["question"],reader_view,digest([common,witness,key,reader_view]))
            prompt_tokens,completion_tokens = len(sample.prompt_ids),len(sample.completion_ids)
            if mask_mode=="length_matched" and prompt_tokens!=visible_input:
                raise ContractError("Length-matched reader replay changed the prompt length")
        cells[key+"_masked"] = {"score":score({"answer":result,"answer_view":masked},gold),
                               "answer":result,"answer_changed":normalize(result)!=normalize(trace["answer"]),
                               "view_hash":digest(reader_view),"answer_view":reader_view,
                               "reused_visible":reader_view==visible,
                               "prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,
                               "removed_chars":_visible_chars(visible,arm_ref)}
        if mask_mode=="length_matched":
            cells[key+"_masked"]["scorer_view"] = masked
    delta = cells["a_visible"]["score"]["reward"]-cells["b_visible"]["score"]["reward"]
    residual = cells["a_masked"]["score"]["reward"]-cells["b_masked"]["score"]["reward"]
    interaction = delta-residual
    return {**common,"eligible":True,"witness":witness,"cells":cells,"terminal_delta":delta,
            "masked_delta":residual,"interaction":interaction,"evidence_credit":aligned_credit(delta,interaction)}


def coverage(trace, public, gold):
    """Scorer-side stage coverage. Gold labels never influence the candidate menu."""
    archive = _archive(trace,public)
    facts = {(title,index) for title,index in gold["supporting_facts"]}
    present = {(r["title"],i) for r in archive.values() for i,(start,end) in enumerate(r["sentence_offsets"]) if start<end}
    query_actions = [a for a in trace["actions"] if a["stage"].startswith("retrieve")]
    first_query = next(i for i,a in enumerate(trace["actions"]) if a["stage"].startswith("retrieve"))
    refs = [*trace["actions"][first_query-1]["working_after"],
            *[op["evidence"] for a in query_actions for op in a["menu"].values() if op.get("evidence")]]
    reachable = {(r["title"],i) for r in answer_view(archive,refs) for i in r["complete_sentences"]}
    visible = {(r["title"],i) for r in trace["answer_view"] for i in r["complete_sentences"]}
    missing_menu = (facts & present)-reachable-visible
    result = {"expected":len(facts),"outside_public":len(facts-present),
              "public_not_offered":len(missing_menu),
              "offered_not_delivered":len((facts & reachable)-visible),"delivered":len(facts & visible),
              "all_support_delivered":facts<=visible}
    if "query_extracted_spans" in trace:
        extracted = {(r["title"],i) for r in answer_view(archive,trace["query_extracted_spans"])
                     for i in r["complete_sentences"]}
        result.update(public_not_extracted=len(missing_menu-extracted),extracted_not_offered=len(missing_menu & extracted))
    return result
