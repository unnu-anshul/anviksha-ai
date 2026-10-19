import json

from agent import db_hook, evidence, llm, pdf_parser, tools
from agent.schemas import (CLAIM_TYPES, MAX_ATTEMPTS, MAX_CLAIMS, NOT_CHECKABLE_TYPES,
                           STATUS_LABELS, new_claim)

SYSTEM = (
    "You are ANVIKSHA, a careful research-verification agent. "
    "Use ONLY numbers that appear in the supplied paper text or tables. "
    "Never invent or estimate values. Answer with valid JSON only."
)

CAPABILITIES = (
    "The prototype CAN: recompute averages/sums/weighted means from table values, check "
    "improvement arithmetic, F1 from precision+recall, percentages from counts, significance "
    "from summary statistics (mean, std, n), confidence intervals, p-values from t and df. "
    "The prototype CANNOT: train models, run GPUs, reproduce full experiments, or judge "
    "qualitative claims."
)


def _show(claims):
    return json.dumps(
        [{"id": c["id"], "text": c["text"], "claim_type": c["claim_type"],
          "location": c["location"], "numbers": c["numbers"]} for c in claims],
        indent=1)


# ---------------------------------------------------------------- 1. analyze
def analyze(state):
    pages, tables = pdf_parser.load(state["pdf_bytes"])
    text = pdf_parser.full_text(pages)
    prompt = (
        "TASK: Extract the verifiable numerical and statistical claims from this paper.\n"
        'Return JSON: {"claims":[{"id":"C1","text":"the claim as written","claim_type":"...",'
        '"page":1,"location":"Table 1 or Section","numbers":{"label":number_or_list}}]}\n'
        f"Rules: at most {MAX_CLAIMS} claims. Copy numbers exactly as printed (do not round, "
        "do not recompute). Include table values needed to check the claim inside 'numbers'. "
        "Also list claims that cannot be verified by calculation (training results, hardware "
        "speed) and label them honestly. "
        f"claim_type must be one of: {', '.join(CLAIM_TYPES)}.\n\n"
        f"PAPER TEXT:\n{text[:40000]}\n\nTABLES:\n{pdf_parser.tables_as_text(tables)[:15000]}"
    )
    data = llm.ask_json(prompt, SYSTEM, task="analyze")
    raw = data.get("claims", [])[:MAX_CLAIMS]
    claims = [new_claim(r, i + 1) for i, r in enumerate(raw)]
    return {
        "pages": pages, "tables": tables, "full_text": text, "claims": claims,
        "events": [f"Parsed {len(pages)} page(s), found {len(tables)} table(s).",
                   f"Extracted {len(claims)} claim(s)."],
    }


# ---------------------------------------------------------------- 2. classify
def classify(state):
    claims = state["claims"]
    prompt = (
        "TASK: Independently decide for each claim whether it is checkable by this prototype.\n"
        f"{CAPABILITIES}\n"
        'Return JSON: {"results":[{"id":"C1","checkable":true,"reason":"one sentence"}]}\n\n'
        f"CLAIMS:\n{_show(claims)}"
    )
    data = llm.ask_json(prompt, SYSTEM, task="classify")
    by_id = {r.get("id"): r for r in data.get("results", [])}
    events = []
    for c in claims:
        r = by_id.get(c["id"], {})
        checkable = bool(r.get("checkable", False))
        reason = r.get("reason") or "No classification returned."
        if c["claim_type"] in NOT_CHECKABLE_TYPES:       # hard safety rule
            checkable = False
            reason = reason or f"Claim type '{c['claim_type']}' is outside prototype scope."
        c["checkable"], c["classify_reason"] = checkable, reason
        if not checkable:
            c["status"] = "not_checkable"
            c["explanation"] = reason
            c["limitations"] = ["Outside the numerical/statistical scope of the prototype."]
        events.append(f"{c['id']}: {'checkable' if checkable else 'NOT checkable'} - {reason}")
    return {"claims": claims, "events": events}


# ---------------------------------------------------------------- 3. evidence
def find_evidence(state):
    ev = evidence.discover(state["pages"], state["full_text"])
    lines = [f"Title guess: {ev['title_guess'][:80]}",
             f"Crossref hits: {len(ev['crossref'])}, OpenAlex hits: {len(ev['openalex'])}",
             f"Repos linked in paper: {len(ev['repos_in_paper'])}, "
             f"GitHub search hits: {len(ev['github_search'])}"]
    lines += [f"Evidence warning: {e}" for e in ev["errors"]]
    return {"evidence": ev, "events": lines}


# ---------------------------------------------------------------- 4. plan
def plan(state):
    claims = state["claims"]
    pending = [c for c in claims
               if c["checkable"] and c["status"] is None
               and (c["plan"] is None or c["needs_replan"])]
    if not pending:
        return {"events": ["Planner: nothing to plan."]}
    items = []
    for c in pending:
        items.append({"id": c["id"], "text": c["text"], "claim_type": c["claim_type"],
                      "numbers": c["numbers"], "location": c["location"],
                      "previous_attempts": c["history"], "feedback": c["feedback"]})
    prompt = (
        "TASK: For each claim choose ONE verification method and fill its parameters.\n"
        f"AVAILABLE METHODS:\n{tools.CATALOG}\n"
        'Return JSON: {"plans":[{"id":"C1","method":"name or none","params":{...},'
        '"reason":"why this method"}]}\n'
        "Rules: take numbers ONLY from the claim's numbers or the tables below. If the data "
        "needed is not in the paper, return method 'none' and explain. If previous_attempts "
        "or feedback exist, choose a DIFFERENT approach or fix the parameters.\n\n"
        f"CLAIMS:\n{json.dumps(items, indent=1)}\n\n"
        f"TABLES:\n{pdf_parser.tables_as_text(state['tables'])[:15000]}"
    )
    data = llm.ask_json(prompt, SYSTEM, task="plan")
    by_id = {p.get("id"): p for p in data.get("plans", [])}
    events = []
    for c in pending:
        p = by_id.get(c["id"])
        if not p or p.get("method") not in tools.REGISTRY:
            reason = (p or {}).get("reason") or "Planner found no supported method."
            c["status"], c["checkable"] = "not_checkable", False
            c["explanation"] = f"No supported verification method: {reason}"
            c["limitations"] = ["Required data or method not available in the prototype."]
            c["needs_replan"] = False
            events.append(f"{c['id']}: planner returned no usable method -> not checkable.")
            continue
        c["plan"] = {"method": p["method"], "params": p.get("params", {}),
                     "reason": p.get("reason", "")}
        c["pending_exec"], c["needs_replan"] = True, False
        events.append(f"{c['id']}: plan = {p['method']} ({p.get('reason', '')[:80]})")
    return {"claims": claims, "events": events}


# ---------------------------------------------------------------- 5. execute
def execute(state):
    claims, events = state["claims"], []
    for c in claims:
        if not c["pending_exec"]:
            continue
        c["pending_exec"], c["pending_review"] = False, True
        c["attempts"] += 1
        plan_ = c["plan"] or {}
        try:
            c["result"] = tools.run_method(plan_.get("method"), plan_.get("params"))
            c["error"] = None
            events.append(f"{c['id']}: attempt {c['attempts']} OK -> {c['result']['summary']}")
        except Exception as e:
            c["result"], c["error"] = None, f"{type(e).__name__}: {e}"
            events.append(f"{c['id']}: attempt {c['attempts']} FAILED -> {c['error']}")
        c["history"].append({"attempt": c["attempts"], "method": plan_.get("method"),
                             "params": plan_.get("params"), "error": c["error"]})
    return {"claims": claims, "events": events}


# ---------------------------------------------------------------- 6. critic
def critic(state):
    claims, events, to_review = state["claims"], [], []
    for c in claims:
        if not c["pending_review"]:
            continue
        c["pending_review"] = False
        if c["error"]:
            if c["attempts"] < MAX_ATTEMPTS:
                c["needs_replan"] = True
                c["feedback"] = f"Previous attempt failed: {c['error']}"
                events.append(f"{c['id']}: critic -> replan (attempt {c['attempts']} failed).")
            else:
                c["status"] = "not_checkable"
                c["explanation"] = (f"Verification failed after {c['attempts']} attempts: "
                                    f"{c['error']}")
                c["limitations"] = ["Needed data was missing or unusable."]
                events.append(f"{c['id']}: critic -> giving up, marked not checkable.")
        else:
            to_review.append(c)

    if to_review:
        payload = [{"id": c["id"], "claim": c["text"], "plan": c["plan"],
                    "result": c["result"]} for c in to_review]
        prompt = (
            "TASK: Act as a strict reviewer. For each item decide whether the computation "
            "actually tests the claim (right values, right method, no mixed-up rows or units).\n"
            'Return JSON: {"reviews":[{"id":"C1","agrees":true,"issues":["..."],'
            '"explanation":"2-3 plain sentences for the final report, mention the values used",'
            '"limitations":["..."]}]}\n\n'
            f"ITEMS:\n{json.dumps(payload, indent=1, default=str)}"
        )
        try:
            data = llm.ask_json(prompt, SYSTEM, task="critic")
            reviews = {r.get("id"): r for r in data.get("reviews", [])}
        except Exception as e:
            reviews = {}
            events.append(f"Critic LLM unavailable ({e}); using deterministic review only.")
        for c in to_review:
            r = reviews.get(c["id"], {"agrees": True, "issues": [], "explanation": "",
                                      "limitations": ["Independent LLM review unavailable."]})
            c["review"] = r
            if r.get("agrees", True):
                # numbers decide the status, never the LLM
                c["status"] = "reproduced" if c["result"]["match"] else "discrepancy"
                c["explanation"] = r.get("explanation") or c["result"]["summary"]
                c["limitations"] = list(r.get("limitations", [])) + [
                    "Compared using a rounding-aware tolerance."]
                events.append(f"{c['id']}: critic agrees -> {STATUS_LABELS[c['status']]}.")
            elif c["attempts"] < MAX_ATTEMPTS:
                c["needs_replan"] = True
                c["feedback"] = "Reviewer issues: " + "; ".join(r.get("issues", []))
                events.append(f"{c['id']}: critic disagrees -> replan.")
            else:
                c["status"] = "unverified"
                c["explanation"] = ("Reviewer doubts the check: "
                                    + "; ".join(r.get("issues", [])))
                c["limitations"] = ["Check could not be confirmed by the review step."]
                events.append(f"{c['id']}: critic disagrees at max attempts -> unverified.")
    return {"claims": claims, "events": events}


def route_after_critic(state):
    again = any(c["needs_replan"] and c["status"] is None for c in state["claims"])
    return "plan" if again else "report"


# ---------------------------------------------------------------- 7. report
def report(state):
    claims = state["claims"]
    for c in claims:                      # safety net: nothing leaves without a status
        if c["status"] is None:
            c["status"] = "unverified"
            c["explanation"] = c["explanation"] or "No conclusion was reached."
    counts = {k: sum(1 for c in claims if c["status"] == k) for k in STATUS_LABELS}
    rep = {
        "paper": state["pdf_name"],
        "summary": counts,
        "claims": [{
            "id": c["id"], "text": c["text"], "claim_type": c["claim_type"],
            "page": c["page"], "location": c["location"], "numbers": c["numbers"],
            "checkable": c["checkable"], "classify_reason": c["classify_reason"],
            "plan": c["plan"], "attempts": c["attempts"], "history": c["history"],
            "result": c["result"], "status": c["status"],
            "status_label": STATUS_LABELS[c["status"]],
            "explanation": c["explanation"], "limitations": c["limitations"],
            "review": c["review"],
        } for c in claims],
        "evidence": state.get("evidence", {}),
        "agent_log": state.get("events", []),
        "global_limitations": [
            "Prototype verifies numerical/statistical claims only.",
            "No external repository code was executed.",
            "Claims are extracted by an LLM and may miss or misread items; "
            "always check the source.",
        ],
    }
    rep["run_id"] = db_hook.save(rep)
    return {"report": rep, "events": [f"Report ready (run #{rep['run_id']})."]}
