"""Taxonomy-neutral repair signatures and mechanism audit prompt."""

from __future__ import annotations

import difflib
import json
from typing import Any


MECHANISM_MATCH_AUDIT_SYSTEM = """You audit whether a candidate historical repair
signature is useful for verifying a proposed security hypothesis. You receive only
the two neutral descriptions. Do not infer a label for the target function, do not
use taxonomy-family equality as evidence, and do not assume the hypothesis is the
actual patch target.

Assign one label per item:
- 2: the same failed safety condition is present in both descriptions. The risky
  operation and the trigger/invariant/protection relation align closely enough that
  the candidate repair is a direct mechanism-level reference.
- 1: the immediate failure differs, but the candidate supplies a concrete,
  transferable verification or repair principle that would materially help assess
  the hypothesis.
- 0: only a broad weakness/CWE-like category, generic defensive advice, or surface
  wording overlaps; the candidate does not materially guide validation.

Return JSON only. Every item must be decided exactly once:
{
  "judgments": [
    {
      "audit_id": "id",
      "label": 0,
      "root_cause_alignment": "short explanation",
      "trigger_invariant_alignment": "short explanation",
      "transferable_principle": "short explanation",
      "reason": "short decisive reason"
    }
  ]
}"""


SIGNATURE_SYSTEM = """Extract one primary Mechanism Signature from a vulnerability
repair pair. The signature represents the earliest security-relevant failed safety
condition repaired by the actual code change. The vulnerable and fixed functions,
their diff, and a CVE description are available. The code change is primary
evidence; metadata may be incomplete or misleading.

Use neutral technical language. Do NOT name a CWE, vulnerability category,
mechanism family, taxonomy label, or broad subtype. Do not infer helper or caller
contracts that are not visible. Do not turn tests, formatting, renames, or unrelated
changes into a vulnerability. When the change is unclear, state the uncertainty.

The before and after excerpts must be short, exact, nonempty substrings of the
respective functions. Return JSON only.

Schema:
{
  "adjudicability": "clear|uncertain|unrelated_patch",
  "operation": "dangerous operation in reusable semantic language",
  "risk_object": "object acted upon",
  "trigger_condition": "specific bad-state relation",
  "violated_invariant": "safety relation required before the operation",
  "failed_protection": "guard, ownership, state, or error action that failed",
  "repair_action": "what the patch changes to establish the invariant",
  "evidence_before": "short exact vulnerable source excerpt",
  "evidence_after": "short exact fixed source excerpt",
  "uncertainty": "none or the missing context",
  "reason": "one diff-grounded sentence"
}"""


def repair_diff(record: dict[str, Any], context: int = 3) -> str:
    return "\n".join(
        difflib.unified_diff(
            str(record["func_vuln"]).splitlines(),
            str(record["func_safe"]).splitlines(),
            fromfile="func_vuln",
            tofile="func_safe",
            n=context,
            lineterm="",
        )
    )


def signature_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "vulnerable_function": record["func_vuln"],
        "fixed_function": record["func_safe"],
        "repair_diff": repair_diff(record),
        "cve_description": record.get("cve_desc") or "",
    }
    return [
        {"role": "system", "content": SIGNATURE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def signature_view(value: dict[str, Any], kind: str) -> dict[str, Any]:
    """Return only fields permitted to taxonomy classification."""
    shared = {
        key: value.get(key, "")
        for key in (
            "operation",
            "risk_object",
            "trigger_condition",
            "violated_invariant",
            "uncertainty",
        )
    }
    if kind == "hypothesis":
        shared["protection"] = value.get("suspected_missing_protection", "")
        shared["evidence"] = value.get("supporting_evidence", [])
    else:
        shared["protection"] = value.get("failed_protection", "")
        shared["evidence"] = {
            "before": value.get("evidence_before", ""),
            "after": value.get("evidence_after", ""),
        }
    return shared
