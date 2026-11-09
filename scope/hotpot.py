"""AgeMem-style HotpotQA interaction with a public, versioned StateAtlas."""
from collections import Counter
from dataclasses import asdict
import re
import string

from .atlas import StateAtlas
from .contracts import Budget, ContractError, Scope
from .credit import standardize
from .io import canonical, digest
from .retrieval import Retriever, terms
from .tools import ToolExecutor


STAGES = ("write_1", "write_2", "compress", "retrieve_1", "retrieve_2")
NOOP = {"write_1":"SKIP","write_2":"SKIP","compress":"KEEP","retrieve_1":"FINISH","retrieve_2":"FINISH"}
MAX_SPAN = 768


def split_row(row):
    """Policy input and scorer input are separate objects."""
    context = row["context"]
    paragraphs = (list(zip(context["title"], context["sentences"], strict=True)) if isinstance(context,dict)
                  else context)
    facts = row["supporting_facts"]
    supports = (list(zip(facts["title"],facts["sent_id"],strict=True)) if isinstance(facts,dict) else facts)
    return ({"id":str(row.get("id",row.get("_id"))),"question":row["question"],"paragraphs":paragraphs},
            {"answer":row["answer"],"supporting_facts":supports})


def normalize(text):
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(trace, gold):
    answer = normalize(trace["answer"])
    target = normalize(gold["answer"])
    common = sum((Counter(answer.split()) & Counter(target.split())).values())
    special = {"yes","no","noanswer"}
    f1 = 2 * common / (len(answer.split()) + len(target.split())) if common else 0.
    if answer != target and (answer in special or target in special):
        f1 = 0.
    em = int(answer == target)
    shown = {(ref["title"],i) for ref in trace["answer_view"] for i in ref["complete_sentences"]}
    expected = {tuple(pair) for pair in gold["supporting_facts"]}
    return {"em":em,"f1":f1,"reward":(em+f1)/2,
            "support_visible_recall":len(shown & expected)/max(1,len(expected))}


def _excerpt(record, limit=MAX_SPAN, start=0):
    return {"id":record["id"],"start":start,"end":min(start+limit,len(record["text"]))}


def _sentence_excerpt(record, query=""):
    spans = [pair for pair in record["sentence_offsets"] if 0<pair[1]-pair[0]<=MAX_SPAN]
    if not spans:
        return _excerpt(record)
    wanted = set(terms(query))
    best = max(range(len(spans)),key=lambda i:len(wanted & set(terms(record["text"][spans[i][0]:spans[i][1]]))))
    start,end = spans[best]
    for a,b in spans[best+1:]:
        if b-start>MAX_SPAN:
            break
        end=b
    return {"id":record["id"],"start":start,"end":end}


def document_text(title, sentences):
    text, offsets = title + "\n", []
    for sentence in sentences:
        offsets.append((len(text),len(text)+len(sentence)))
        text += sentence + " "
    return text, offsets


def ordered_documents(public, seed):
    return sorted(public["paragraphs"], key=lambda d: digest([seed, public["id"], d[0], d[1]]))


def answer_view(archive, refs):
    return [{**ref,"title":archive[ref["id"]]["title"],
             "text":archive[ref["id"]]["text"][ref["start"]:ref["end"]],
             "complete_sentences":[i for i,(a,b) in enumerate(archive[ref["id"]]["sentence_offsets"])
                                   if ref["start"]<=a<b<=ref["end"]]} for ref in refs]


def answer_messages(question, view):
    return [{"role":"system","content":"Answer from the public excerpts. Return only a concise answer, "
             "or unknown if the excerpts do not support one."},
            {"role":"user","content":canonical({"question":question,"evidence":view})}]


def answer_from_view(policy, question, view, boundary):
    sample = policy.sample(answer_messages(question,view),"actor","answer",boundary,4096,24,greedy=True)
    return (sample.text.splitlines()[0] if sample.text.splitlines() else ""), sample


def _model_messages(stage, menu, public, evidence, kept, sentence_candidates=False, shared_interface=False):
    descriptions = {letter:{**{k:v for k,v in option.items() if k not in {"evidence","expected_revision"}},
                           **({"title":evidence[option["evidence"]["id"]]["title"],
                               "preview":evidence[option["evidence"]["id"]]["text"][
                                   option["evidence"]["start"]:min(option["evidence"]["end"],option["evidence"]["start"]+180)]}
                              if "evidence" in option else {})} for letter,option in menu.items()}
    instructions = {
        "write":"The question arrives later. Store one informative paragraph for later use, or skip.",
        "compress":"A new stream arrived. Keep the working excerpts, shorten them, or clear them. Raw sources remain retrievable.",
        "retrieve":"Select a paragraph to recover missing evidence for the question, or finish this lookup.",
    }
    kind = stage.split("_")[0]
    content = {"stage":stage,("verified_options" if sentence_candidates else "actions"):descriptions,
               "working_memory":[{"title":evidence[e["id"]]["title"],
                                  "text":evidence[e["id"]]["text"][e["start"]:min(e["start"]+140,e["end"])]}
                                 for e in kept]}
    if kind == "retrieve":
        content["question"] = public["question"]
    if shared_interface:
        from .candidate_actions import choice_payload
        options = {}
        for label, option in menu.items():
            op = option["operation"]
            proposal = {"operation": {"STORE":"PROPOSE_STATE_UPDATE", "COMPRESS":"SUMMARIZE",
                         "CLEAR":"FILTER", "KEEP":"FINISH", "SKIP":"FINISH", "RETRIEVE":"FETCH_EVIDENCE"}.get(op, op),
                        "expected_revision": option["expected_revision"]}
            if op in {"STORE", "RETRIEVE"}:
                ref = option["evidence"]
                quote = evidence[ref["id"]]["text"][ref["start"]:ref["end"]]
                if op == "STORE":
                    proposal.update(entity=evidence[ref["id"]]["entity"], predicate="quoted_text", value=quote,
                                    support_status="supported", evidence_refs=[ref])
                else:
                    proposal.update(evidence_id=ref["id"] , start=ref["start"], end=ref["end"], preview=quote[:180])
            elif op == "COMPRESS":
                proposal["items"] = [{"evidence_ref":ref,"text":evidence[ref["id"]]["text"][ref["start"]:ref["end"]]}
                                     for ref in option["refs"]]
            elif op == "CLEAR":
                proposal["selected_ids"] = []
            options[label] = proposal
        return choice_payload({"phase":"state" if kind == "write" else "context", "stage":stage,
            "goal": public["question"] if kind == "retrieve" else "Preserve useful public information for a later query.",
            "revision":next(iter(menu.values()))["expected_revision"], "working_memory":content["working_memory"],
            "management_reason":instructions[kind], "claims":[], "recent_evidence":[]}, options)
    return [{"role":"system","content":instructions[kind]+" Answer with one action letter."},
            {"role":"user","content":canonical(content)}]


def rollout(policy, public, split, seed, *, greedy=False, factual=None, intervention=None,
            intervention_label=None, chooser=None, sentence_candidates=False, record_prompts=False,
            bridge_candidates=False, shared_interface=False):
    """Execute a complete policy trajectory or a matched replay with one forced no-op."""
    if (factual is None) != (intervention is None):
        raise ContractError("A counterfactual needs its factual trajectory and intervention stage")
    if bridge_candidates and not sentence_candidates:
        raise ContractError("Bridge candidates require sentence-aligned excerpts")
    if shared_interface and not sentence_candidates:
        raise ContractError("Shared memory interface requires sentence-aligned excerpts")
    atlas = StateAtlas(Scope(split,public["id"],str(seed)))
    tools = ToolExecutor()
    budget = Budget(retrievals=2)
    archive, kept, actions, samples = {}, [], [], []
    query_spans = {}
    documents = ordered_documents(public,seed)
    try:
        for index,stage in enumerate(STAGES):
            options = [{"operation":NOOP[stage]}]
            if stage.startswith("write"):
                lo = index * len(documents)//2
                hi = (index+1)*len(documents)//2
                for di,(title,sentences) in enumerate(documents[lo:hi],lo):
                    text, offsets = document_text(title,sentences)
                    record = atlas.append_public(text,entity=f"wiki:{di}")
                    archive[record["id"]] = {**record,"title":title,"sentence_offsets":offsets}
                    options.append({"operation":"STORE","evidence":(
                        _sentence_excerpt(archive[record["id"]]) if sentence_candidates else _excerpt(record))})
            elif stage == "compress":
                if kept:
                    if shared_interface:
                        options.extend(compression_options(archive, kept))
                        options.append({"operation":"CLEAR"})
                    else:
                        options.extend([{"operation":"COMPRESS"},{"operation":"CLEAR"}])
            else:
                if stage=="retrieve_1":
                    atlas.append_public(public["question"],"requirement")
                    if bridge_candidates:
                        query_spans = {key:_sentence_excerpt(record,public["question"]) for key,record in archive.items()}
                ranked = Retriever().search(atlas,public["question"],top_k=8)
                visible_text = " ".join(archive[ref["id"]]["text"][ref["start"]:ref["end"]] for ref in kept).casefold()
                linked = [record for record in archive.values() if len(record["title"])>=4 and record["title"].casefold() in visible_text
                          and not any(r["id"]==record["id"] for r in kept)]
                if bridge_candidates:
                    linked = [r for r in linked if re.search(r"(?<!\w)"+re.escape(r["title"].casefold())+r"(?!\w)",visible_text)]
                    overlap = set(terms(public["question"]))
                    linked.sort(key=lambda r:(-len(overlap & set(terms(r["text"]))),r["id"]))
                ranked = list({r["id"]:r for r in [*linked[:2 if bridge_candidates else 1],*ranked]
                               if r["id"] in archive}.values())
                for record in ranked:
                    ref = query_spans.get(record["id"])
                    if ref is None:
                        ref = (_sentence_excerpt(archive[record["id"]],public["question"])
                               if sentence_candidates else _excerpt(record))
                    if any(r["id"]==ref["id"] and r["start"]<=ref["start"] and r["end"]>=ref["end"] for r in kept):
                        continue
                    options.append({"operation":"RETRIEVE","evidence":ref})
                    if len(options)==(7 if sentence_candidates else 5):
                        break
            revision = atlas.revision
            options.sort(key=lambda op:digest([seed,stage,op]))
            menu = {letter:{**op,"expected_revision":revision} for letter,op in zip("ABCDEFGH",options)}
            messages = _model_messages(stage,menu,public,archive,kept,sentence_candidates,shared_interface)
            boundary = digest([atlas.fingerprint(),kept,menu,messages])
            sample = None
            decision = None
            if factual and index<=intervention:
                original = factual["actions"][index]
                if original["boundary"]!=boundary:
                    raise ContractError("Counterfactual did not reconstruct the exact public prefix")
                label = (original["choice"] if index<intervention else (intervention_label or next(
                    k for k,v in menu.items() if v["operation"]==NOOP[stage])))
            elif len(menu)==1:
                label = next(iter(menu))
            elif chooser:
                public_options = {k:{"source_id":ref["id"],"start":ref["start"],"end":ref["end"],
                    "title":archive[ref["id"]]["title"],"quote":archive[ref["id"]]["text"][ref["start"]:ref["end"]]}
                    for k,op in menu.items() if (ref:=op.get("evidence"))}
                working_sources = [{**ref,"title":archive[ref["id"]]["title"],
                                    "quote":archive[ref["id"]]["text"][ref["start"]:ref["end"]]} for ref in kept]
                decision = chooser(messages,menu,public_options,working_sources)
                label = decision["choice"]
            else:
                policy.torch.manual_seed(int(digest([seed,stage])[:8],16))
                sample = policy.sample_choice(messages,stage,boundary,list(menu),2048,greedy)
                label = sample.text
            if label not in menu:
                raise ContractError("Proposed choice is not in the verified action menu")
            choice = menu[label]
            op = choice["operation"]
            result = None
            if op in {"STORE","RETRIEVE"}:
                ref = choice["evidence"]
                record = atlas.evidence(ref["id"])
                if op=="RETRIEVE":
                    result = tools.execute(atlas,{},{"operation":"FETCH_EVIDENCE","expected_revision":revision,
                        "evidence_id":ref["id"],"start":ref["start"],"end":ref["end"]},"context",budget,"")
                else:
                    result = tools.execute(atlas,{},{"operation":"PROPOSE_STATE_UPDATE","expected_revision":revision,
                        "entity":record["entity"],"predicate":"quoted_text","value":record["text"][ref["start"]:ref["end"]],
                        "support_status":"supported","evidence_refs":[ref]},"state",budget,"")
                kept = [old for old in kept if old["id"]!=ref["id"]] + [ref]
            elif op=="CLEAR":
                kept = []
            elif op=="COMPRESS":
                short = choice.get("refs") or [{**ref,"end":max((end for start,end in archive[ref["id"]]["sentence_offsets"]
                          if ref["start"]<=start<end<=min(ref["start"]+320,ref["end"])),default=ref["end"])}
                         if sentence_candidates else {**ref,"end":min(ref["start"]+320,ref["end"])} for ref in kept]
                result = tools.execute(atlas,{},{"operation":"SUMMARIZE","expected_revision":revision,
                    "items":[{"evidence_ref":ref,"text":archive[ref["id"]]["text"][ref["start"]:ref["end"]]} for ref in short]},
                    "context",budget,"")
                kept = short
            atlas.append_public(op,"action_receipt")
            actions.append({"stage":stage,"boundary":boundary,"atlas_revision":revision,"menu":menu,"choice":label,
                            "operation":op,"tool_result":result,"working_after":list(kept),
                            "sample":asdict(sample) if sample else None,
                            **({"student_messages":messages} if record_prompts else {}),
                            **({"teacher_decision":decision} if decision else {})})
            samples.append(sample)
        shown = answer_view(archive,kept)
        view = atlas.save_view([ref["id"] for ref in kept])
        answer, sample = answer_from_view(policy,public["question"],shown,digest(view))
        atlas.record_use(view["id"],sample.text,dict(policy.versions))
        trace = {"task_id":public["id"],"seed":seed,"split":split,"intervention":intervention,"actions":actions,
                 "answer_view":shown,"answer":answer,"answer_sample":asdict(sample),"atlas":atlas.export(),
                 **({"query_extracted_spans":list(query_spans.values())} if bridge_candidates else {})}
        return trace,samples
    finally:
        atlas.close()


def compression_options(archive, kept):
    """Let the policy choose content, not only a fixed truncation toggle; never see the future question."""
    variants = []
    for strategy in ("first_sentences", "last_sentences"):
        refs = []
        for ref in kept:
            spans = [(a,b) for a,b in archive[ref["id"]]["sentence_offsets"] if
                     ref["start"] <= a < b <= ref["end"] and b-a <= 320]
            if not spans:
                refs.append(dict(ref))
                continue
            start, end = spans[0] if strategy == "first_sentences" else spans[-1]
            refs.append({"id":ref["id"], "start":start, "end":end})
        if refs != kept and not any(x["refs"] == refs for x in variants):
            variants.append({"operation":"COMPRESS", "strategy":strategy, "refs":refs})
    return variants


def weights(groups, stage, local_weight):
    """Group-normalized trajectory credit plus matched, signed local outcome credit."""
    group_adv = standardize([g["score"]["reward"] for g in groups])
    weighted, differences = [], []
    for group,global_adv in zip(groups,group_adv,strict=True):
        control = group.get("control_score")
        local = group.get("evidence_credit", group["score"]["reward"] - control["reward"] if control else 0.)
        differences.append(local)
        extra = local_weight * local + group.get("compression_credit",0.)
        live = [(index,s) for index,s in enumerate(group["samples"]) if s is not None]
        for index,sample in live:
            sample.advantage = global_adv + (extra if index==stage else 0.)
            sample.credit_scope = ("hotpot:evidence" if "evidence_credit" in group else "hotpot:paired"
                                   ) if index==stage and control else "hotpot:trajectory"
            if sample.advantage:
                weighted.append((sample,1/(len(groups)*len(live))))
    return weighted,group_adv,differences
