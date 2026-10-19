import operator
from typing import Annotated, TypedDict

MAX_ATTEMPTS = 2      # how many times the agent may try to verify one claim
MAX_CLAIMS = 12       # cost control: claims analysed per paper

STATUS_LABELS = {
    "reproduced": "Reproduced",
    "discrepancy": "Discrepancy found",
    "not_checkable": "Not checkable within prototype limits",
    "unverified": "Unverified (inconclusive)",
}

CLAIM_TYPES = [
    "table_aggregate",      # mean / sum / weighted mean of reported table values
    "improvement",          # X improves over Y by Z (points or percent)
    "metric_consistency",   # F1 from precision+recall, accuracy from counts
    "statistical_test",     # significance claims, p-values
    "confidence_interval",  # reported CI
    "equation",             # symbolic or numeric equation check
    "training_result",      # needs full model training -> not checkable
    "hardware_performance", # speed on specific GPU etc -> not checkable
    "qualitative",          # opinion or non-numeric claim
    "other",
]

NOT_CHECKABLE_TYPES = {"training_result", "hardware_performance", "qualitative"}


class AgentState(TypedDict, total=False):
    pdf_name: str
    pdf_bytes: bytes
    pages: list
    tables: list
    full_text: str
    claims: list
    evidence: dict
    events: Annotated[list, operator.add]   # every node appends log lines
    report: dict


def new_claim(raw, idx):
    ctype = raw.get("claim_type")
    return {
        "id": str(raw.get("id") or f"C{idx}"),
        "text": str(raw.get("text", "")).strip(),
        "claim_type": ctype if ctype in CLAIM_TYPES else "other",
        "page": raw.get("page"),
        "location": str(raw.get("location") or ""),
        "numbers": raw.get("numbers") if isinstance(raw.get("numbers"), dict) else {},
        "checkable": None,
        "classify_reason": "",
        "plan": None,
        "attempts": 0,
        "history": [],
        "result": None,
        "error": None,
        "feedback": "",
        "review": None,
        "status": None,
        "explanation": "",
        "limitations": [],
        "needs_replan": False,
        "pending_exec": False,
        "pending_review": False,
    }
