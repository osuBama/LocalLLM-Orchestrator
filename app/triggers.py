"""Should this interaction be sent to the memory model? (spec §19)

Deterministic heuristics, English + Portuguese. Replaceable by a classifier
later: keep the signature should_extract(task) -> (bool, reasons).
"""
from __future__ import annotations

import re

from .schemas import InteractionTask

_RULES: dict[str, re.Pattern] = {
    "correction": re.compile(
        r"\b(that'?s (wrong|incorrect|not right)|you'?re wrong|not what i (asked|meant)|"
        r"actually,|no,? (it|that|this) (is|was|isn'?t)|i said|está errado|não é isso|"
        r"estás errado|corrig)", re.I),
    "success": re.compile(
        r"\b(works now|it works|working now|fixed|solved|resolved|success(ful(ly)?)?|"
        r"that did it|funcion(a|ou)|resolvid[oa]|já dá)", re.I),
    "failure": re.compile(
        r"\b(error|failed|failure|fails|exception|traceback|doesn'?t work|not working|"
        r"broken|crash(ed|es)?|timed? ?out|refused|denied|404|500|erro|falh(a|ou)|não funciona)", re.I),
    "config_change": re.compile(
        r"\b(changed|configured|set (it )?to|updated|upgraded|installed|uninstalled|moved|"
        r"renamed|enabled|disabled|switched|migrated|alterei|configurei|instalei|mudei|atualizei)\b", re.I),
    "constraint_or_preference": re.compile(
        r"\b(always|never|must(n'?t)?|do not|don'?t ever|from now on|i prefer|requirement|"
        r"constraint|sempre|nunca|não quero|prefiro|obrigatório)\b", re.I),
    "objective": re.compile(
        r"\b(goal|objective|next step|milestone|the plan is|we need to|todo|to-do|objetivo|"
        r"próximo passo|o plano)\b", re.I),
    "discovery": re.compile(
        r"\b(turns out|found (out|that)|discovered|it seems|root cause|the cause|apparently|"
        r"descobri|afinal|a causa)\b", re.I),
    "decision": re.compile(
        r"\b(decided|decision|we'?ll (go with|use)|let'?s (go with|use)|chose|going with|"
        r"decidi|vamos usar|escolhi)\b", re.I),
    "file_change": re.compile(
        r"\b(created|modified|wrote|saved|edited|deleted) (the |a |an )?(file|script|config|"
        r"folder|directory)|\bcriei o ficheiro|\bficheiro\b", re.I),
}

_TRIVIAL = re.compile(r"^\s*(hi|hello|hey|thanks?|thank you|ok(ay)?|cool|nice|great|olá|ola|"
                      r"obrigad[oa]|bom dia|boa tarde|boa noite)[\s!.?]*$", re.I)


def should_extract(task: InteractionTask, mode: str = "heuristic") -> tuple[bool, list[str]]:
    if mode == "always":
        return True, ["mode=always"]
    reasons: list[str] = []
    if task.flags:
        reasons.append("flagged_by_primary")
    if task.tool_events:
        reasons.append("tool_used")
    user = task.user_message or ""
    if _TRIVIAL.match(user) and not task.tool_events and not task.flags:
        return False, ["trivial"]
    text = f"{user}\n{task.assistant_response or ''}"
    for name, pat in _RULES.items():
        # Corrections and preferences come from the user; the rest can come from either side.
        source = user if name in ("correction", "constraint_or_preference") else text
        if pat.search(source):
            reasons.append(name)
    return bool(reasons), reasons
