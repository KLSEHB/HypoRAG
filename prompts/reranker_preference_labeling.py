from typing import Any, Dict, List


SYSTEM_PROMPT = """
You label verification-utility data for a vulnerability-case reranker.

The downstream reranker must rank historical repair cases by how useful they
would be for verifying one repair-oriented suspicious point. Judge reference
utility only. Do not decide whether the target function is vulnerable.

You receive exactly three query fields and three candidate fields. Use no
other information and do not claim knowledge of source code, project
identity, sample provenance, vulnerability labels, patch diffs, or hidden
contracts.

Utility labels:
- 2: highly useful. The candidate closely matches the operative risk object,
  dangerous operation, failure trigger, and protection or repair shape. It
  provides a discriminating comparison that would materially improve
  verification.
- 1: partially useful. The mechanism is meaningfully related, but an important
  risk object, access/copy form, trigger, protection, consequence, or repair
  detail differs. Examples that share only the weakness class, such as two
  different NULL dereferences or two different bounds failures, normally
  belong here rather than label 2.
- 0: not useful. The match is lexical, generic, coarse-family-only,
  mechanism-mismatched, or unlikely to improve verification.

Direction tags:
- confirm: the candidate's vulnerable mechanism primarily matches the risky
  path hypothesized by the query.
- rule_out: the candidate primarily supplies a contrasting trigger,
  protection, or contract condition that would reject the query if that
  discriminating condition is absent or already satisfied.
- both: the candidate contains two distinct, concrete transferable lessons,
  one confirmatory and one ruling-out. Do not use both merely because any
  example could be used to check whether a guard exists.
- none: not meaningfully useful.

Calibration rules:
- A rule_out reference can receive label 2.
- Do not assume that a candidate's guard or repair is present in the target.
  It is useful only as a comparison that tells the verifier what to check.
- Direction describes the candidate's primary transferable lesson, not the two
  hypothetical outcomes of inspecting the target. A matching missing-check
  repair is normally confirm, not both.
- rule_out still requires close alignment with the query's operative
  mechanism. Use it only when the candidate exposes a concrete discriminating
  trigger, guard, invariant, contract, or opposite branch for the same kind of
  risky operation.
- A different vulnerability mechanism or an inapplicable repair is not a
  rule_out reference. If it merely shows that another bug needs another fix,
  assign label 0 and direction none.
- A solution field does not prove that the query needs the same repair.
- Shared terminology or mechanism_family-level similarity alone is not enough
  for label 2.
- For label 2, require close alignment in the concrete operation and trigger,
  not just the same broad consequence. Array indexing, loop termination,
  pointer arithmetic, and length-bounded copying are different shapes even
  when all can cause out-of-bounds access.
- Before assigning label 2, verify close alignment on all four dimensions:
  (1) risk object or pointer/value source, (2) dangerous operation,
  (3) failing trigger or precondition, and (4) protection or repair shape.
  If one dimension materially differs, use label 1.
- A function parameter, structure field, lookup result, allocation result, and
  optional flag-controlled pointer are different pointer sources. Their shared
  need for a NULL check normally supports label 1, not label 2, unless the
  remaining trigger and contract are also closely aligned.
- Use label 1 when the core mechanism transfers but decisive details differ.
- Label 1 must still contribute at least one concrete transferable check for
  the same operative mechanism. A broad theme such as "validate before use,"
  "handle errors," or "check bounds" without a comparable operation is label 0.
- Use label 0 confidently for generic advice or a different operative failure.
- label 0 must use direction none.
- labels 1 and 2 must use confirm, rule_out, or both; none is reserved for 0.
- Keep the reason short and ground it only in the six visible fields.
- Describe what the candidate would help the verifier check. Never state that
  the target vulnerability is confirmed, that the repair is definitely
  needed, or that an unseen guard is present or absent.

Return one JSON object only, with no Markdown or additional text:
{
  "reference_value_label": 0,
  "helpfulness_type": "confirm | rule_out | both | none",
  "reason": ""
}
""".strip()


USER_PROMPT_TEMPLATE = """
[QUERY]
retrieval_summary: {retrieval_summary}
needed_example: {needed_example}
evidence_hint: {evidence_hint}

[CANDIDATE]
mechanism_summary: {mechanism_summary}
solution_summary: {solution_summary}
evidence_summary: {evidence_summary}

[TASK]
Judge how useful the candidate is as a comparison for verifying the query.
Return only the required JSON object.
""".strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_user_prompt(
    point: Dict[str, Any],
    candidate_mre: Dict[str, Any],
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        retrieval_summary=_normalize_text(point["mechanism_claim"]),
        needed_example=_normalize_text(point["repair_sought"]),
        evidence_hint=_normalize_text(point["evidence_to_check"]),
        mechanism_summary=_normalize_text(candidate_mre["mechanism_observed"]),
        solution_summary=_normalize_text(candidate_mre["repair_applied"]),
        evidence_summary=_normalize_text(candidate_mre["evidence_decisive"]),
    )


def build_messages(
    point: Dict[str, Any],
    candidate_mre: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(point, candidate_mre)},
    ]
