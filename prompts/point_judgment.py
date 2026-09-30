from typing import Any, Dict, Sequence


SYSTEM_PROMPT = """
You verify exactly one suspicious vulnerability point against the current code.

Use retrieved historical examples only as weak mechanism hints. They are never proof that the current code is vulnerable.

Return JSON only. Do not include analysis text, Markdown, code fences, or a [FINAL_JSON] marker.

Decision labels:
- supported: the current code itself establishes a concrete, security-relevant, mechanism-complete vulnerability path for the exact suspicious point.
- unsupported: the risky operation is absent, the trigger is not established, visible protection/invariants block the mechanism, or the point is generic/off-target.
- insufficient_evidence: caller contract, helper/API contract, framework invariant, macro behavior, allocator behavior, or external context is decisive but not visible from this function.

Strict supported gate:
- You may choose supported only when all of these are true:
  1. risky_operation_present is yes
  2. trigger_condition_present is yes
  3. protection_status is missing or insufficient
  4. support_evidence contains concrete current-function facts, not historical examples
  5. support_evidence identifies a security impact, not only incorrect behavior, missing style checks, or possible robustness issues
- Do not choose supported for a claim whose main basis is "no explicit check", "could be", "caller might pass bad input", "helper might fail", "debug assert is compiled out", generic OOM behavior, or a historical analogy.

Caller contract and debug assertion discipline:
- A caller-supplied argument being theoretically invalid is not a trigger. If the bad state requires a caller to violate an unstated precondition, choose insufficient_evidence unless the current function visibly accepts untrusted/external input and owns validation.
- Do not infer nullability, size validity, shape validity, ownership transfer, or ordering obligations merely from the absence of a local check.
- A debug-only assertion is evidence of an intended invariant, not proof of a reachable production vulnerability. If the release-build trigger depends on unknown caller behavior, choose insufficient_evidence.
- Checking one parameter for NULL does not prove every other pointer parameter may be NULL.

Helper/API/framework discipline:
- Do not assume helper/API/framework/container/macro semantics from names alone.
- If a framework normally constructs an object, fills builtin fields, validates tensor shapes, owns memory, sends protocol alerts, or establishes map/key invariants, and that contract is not visible here, choose insufficient_evidence rather than supported.
- If visible usage suggests a checked builder, guarded accessor, bounded type, status gate, ownership discipline, or range restriction, choose unsupported unless the current code shows the remaining failing path.

Security-impact discipline:
- For resource leaks, require attacker-triggered repeated accumulation, lost security-sensitive resource, or a visible denial-of-service path. A one-shot or bounded leak is insufficient_evidence or unsupported.
- For allocation failure/OOM claims, require a concrete unsafe dereference, use-after-free, uninitialized output, or security-relevant protocol state failure in the current function. Generic OOM failure handling is insufficient.
- For protocol/error-handling claims, missing a log, alert, or cleanup style is not enough; the current function must show a concrete externally exploitable state or security impact.
- For semantic mismatch or duplicate-input claims, require a sensitive authorization, validation bypass, memory-safety effect, or externally exploitable state. Incorrect comparison alone is not enough.
- For output initialization, supported is allowed only when the current function visibly returns success while leaving a caller-visible output unset or stale.

Off-target discipline:
- Treat the priority field of the suspicious point only as a hint. It cannot substitute for current-code proof.
- If the suspicious point is plausible but unrelated to the evidence_hint, too broad for the visible code, or only a different possible issue, choose unsupported or insufficient_evidence.
- Multiple weak/generic clues must not be combined into supported.

Mechanism-specific discipline:
- Bounds/size/shape: require a specific size/index/count/offset/dimension flowing into allocation, copy, parser read/write, or memory access on a concrete path. Visible range limits, widening, checked builders, or capacity checks count as protection.
- Pointer/lookup: require a specific dereference or lookup result use before a relevant validity check and a visible reason the pointer/key can be invalid here. Do not infer invalid lookup semantics from API names alone.
- Error handling: require a visible success/failure path where output, cleanup, or propagation is wrong and security-relevant.
- Lifetime/concurrency: require a visible free/unref/delete/escaped alias/callback/refcount/race-prone transition on the same path.
- Input/state/permission: require a concrete accepted malformed/unauthorized state reaching a sensitive operation.
- Information exposure: require a concrete outward copy, return, log, permission, or readable data path.

Evidence writing rules:
- support_evidence must be short paraphrases of current-code facts supporting the judgment.
- blocking_evidence must be short paraphrases of guards, checks, validation, cleanup, synchronization, invariants, helper/API usage semantics, ownership discipline, range restrictions, or equivalent current-code facts that block the mechanism.
- Avoid raw code excerpts with backslashes or complex string escapes; paraphrase instead.

Exactly one evidence field per verdict:
- supported: fill support_evidence. Leave blocking_evidence and missing_context empty.
- unsupported: fill blocking_evidence. Leave support_evidence and missing_context empty.
- insufficient_evidence: fill missing_context, and leave support_evidence and blocking_evidence empty.
- missing_context must name the decisive information that is not visible in this function, such as a caller contract, a helper/API contract, a framework invariant, macro behavior, or allocator behavior. Do not restate the suspicious point, and do not speculate about what that information would turn out to be.

Retrieved-example attribution:
- matched_example_ranks must list the "Example rank" numbers of the retrieved case analyses that actually informed this judgment.
- Include a rank only if that case supplied the mechanism framing, the protection pattern, or the check you looked for. Do not list every retrieved example by default.
- Use an empty list when no retrieved example influenced the judgment. This is a legitimate outcome and must not be padded.

Confidence calibration:
- high requires the risky operation, trigger condition, security impact, and missing/insufficient protection to be directly visible in the current function.
- medium may use limited inference from visible current-code facts.
- low means the judgment depends on uncertain caller/helper/API behavior or external context.

Return exactly this JSON shape:
{
  "verdict": "supported | unsupported | insufficient_evidence",
  "confidence": "high | medium | low",
  "risky_operation_present": "yes | no | unknown",
  "trigger_condition_present": "yes | no | unknown",
  "protection_status": "missing | insufficient | present | unknown",
  "support_evidence": [],
  "blocking_evidence": [],
  "missing_context": "",
  "matched_example_ranks": []
}
""".strip()


CASE_MAPPED_SYSTEM_PROMPT = SYSTEM_PROMPT + """


Case-to-code mapping requirement:
Before choosing the verdict, process every retrieved example separately. For
each example, extract one concrete safety principle from its repaired pattern
or verification guidance, then map that principle to the current function.
The mapping must name the relevant current-code operation, variable, guard, or
missing fact. A generic statement that the two cases share a family is not a
mapping.

The historical example remains only a source of an inspection principle. It
must never substitute for current-code evidence, establish a trigger that is
not visible, or by itself justify `supported`.

For every example rank supplied by the user, return one object in
`case_mappings`:
- `example_rank`: the supplied rank.
- `case_principle`: the specific check, invariant, or protection pattern from
  that example.
- `current_code_mapping`: how that principle maps to the suspicious operation
  in the current function, using concrete current-code facts where possible.
- `mapping_status`: exactly one of `established`, `blocked`,
  `missing_context`, or `not_applicable`.
- `influence`: exactly one of `supports`, `refutes`, `context_only`, or
  `none`. `supports` means the current code establishes the analogous failing
  relation; it does not mean the historical case proves the verdict.

After all mappings, choose the verdict using the existing strict supported
gate. The final support/blocking/missing-context evidence must still be about
the current function.

Return exactly this JSON shape:
{
  "verdict": "supported | unsupported | insufficient_evidence",
  "confidence": "high | medium | low",
  "risky_operation_present": "yes | no | unknown",
  "trigger_condition_present": "yes | no | unknown",
  "protection_status": "missing | insufficient | present | unknown",
  "support_evidence": [],
  "blocking_evidence": [],
  "missing_context": "",
  "matched_example_ranks": [],
  "case_mappings": [
    {
      "example_rank": 1,
      "case_principle": "",
      "current_code_mapping": "",
      "mapping_status": "established | blocked | missing_context | not_applicable",
      "influence": "supports | refutes | context_only | none"
    }
  ]
}
""".strip()


CASE_MAPPED_VISIBLE_EVIDENCE_SYSTEM_PROMPT = CASE_MAPPED_SYSTEM_PROMPT + """


Visible-evidence closure:
Do not select `insufficient_evidence` merely because some caller, helper, or
environmental information is absent. Before using that verdict, identify the
specific missing fact and ask whether it is genuinely necessary to establish
the claimed failure.

Treat the following as decidable from the current function when they are
explicitly shown: data/control flow, literal bounds, arithmetic on visible
operands, signedness or width stated by the code, fixed local-buffer extent,
and a visible guard that does or does not cover the stated trigger. If those
facts establish the complete failing relation and security impact, choose
`supported` even when unrelated external implementation details are absent.

Conversely, retain `insufficient_evidence` when the alleged failure depends on
an unseen helper's bounds/ownership behavior, an unseen container or structure
capacity, a caller-only reachability precondition, privileged environment
permissions, or an external protocol contract. Do not invent those facts from
a retrieved example.

For each case mapping, make `missing_context` concrete: name the exact missing
fact and explain why the visible current-code relation cannot establish the
failure without it. A generic statement that a caller or helper is unseen is
not sufficient.
""".strip()


CASE_MAPPED_VISIBLE_CARDINALITY_SYSTEM_PROMPT = (
    CASE_MAPPED_VISIBLE_EVIDENCE_SYSTEM_PROMPT
    + """


Concrete extent and cardinality rules:
Treat an operation's requested extent as directly established when the current
statement visibly passes both (1) the address of a fixed local destination and
(2) an explicit byte/element count whose visible arithmetic can wrap, underflow,
or exceed that destination. Do not require the implementation of an ordinary
read/write primitive merely to recognize this requested out-of-bounds extent,
unless the current code itself shows that the primitive clamps or rejects it.

Likewise, a direct array access may be supported without seeing the allocation
site when the current function visibly shows all of the following: an explicit
field giving that array's valid entry count, an index bounded by an independent
larger cardinality, and no local relation proving the index is below the entry
count. In that specific situation, do not call the missing allocation layout
decisive. The conclusion must still name the two competing cardinalities and
the concrete unchecked access.

These are narrow local-evidence exceptions. They do not apply when the claimed
failure instead depends on an unseen helper's semantics, an unspecified
container capacity, a caller-only precondition, or an unshown protocol rule.
"""
).strip()


CASE_MAPPED_LOCAL_ARITHMETIC_SYSTEM_PROMPT = (
    CASE_MAPPED_VISIBLE_CARDINALITY_SYSTEM_PROMPT
    + """


Local arithmetic and representation closure:
Treat an arithmetic or representation failure as decidable from the current
function when all of the following are visible: (1) a length, offset, index,
or count has a wider or signed representation, or is directly subtracted from
or added to another visible value; (2) no visible local guard excludes the
bad numeric range; and (3) the resulting narrowed, wrapped, or underflowed
value immediately controls a local allocation size, copy extent, array
subscript, or pointer offset. In this narrow case, do not require an unseen
caller to provide a concrete runtime value merely to recognize the failing
numeric relation. State the exact numeric condition and the concrete unsafe
consumer in support_evidence.

This exception is limited to visible C/C++ arithmetic and representation
semantics. It does not permit `supported` when the decisive fact is an unseen
helper or container contract, allocation extent, parser behavior, framework
invariant, privilege state, or API return domain. It also does not apply to a
division, loop, error path, or abstract state concern unless the visible code
already establishes a concrete memory-safety or security-impacting operation.
"""
).strip()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(value: Any, max_chars: int) -> str:
    text = _normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_example_for_prompt(example_record: Dict[str, Any]) -> str:
    analysis_guide = example_record.get("analysis_guide", {})
    vulnerable_pattern = analysis_guide.get("vulnerable_pattern", {})
    fixed_pattern = analysis_guide.get("fixed_pattern", {})

    parts = [
        f"Project: {_normalize_text(example_record.get('project', ''))}",
        f"File: {_normalize_text(example_record.get('file_name', ''))}",
        "",
        "[Case explanation]",
        _truncate_text(analysis_guide.get("case_explanation", ""), 700),
        "",
        "[Minimal vulnerable pattern]",
        _truncate_text(vulnerable_pattern.get("code", ""), 650),
        f"Explanation: {_truncate_text(vulnerable_pattern.get('explanation', ''), 450)}",
        "",
        "[Minimal fixed pattern]",
        _truncate_text(fixed_pattern.get("code", ""), 650),
        f"Explanation: {_truncate_text(fixed_pattern.get('explanation', ''), 450)}",
        "",
        "[Guidance for similar suspicious points]",
        _truncate_text(analysis_guide.get("guidance_for_similar_suspicion", ""), 700),
    ]
    return "\n".join(part for part in parts if _normalize_text(part))


def build_user_prompt(
    raw_record: Dict[str, Any],
    code: str,
    point: Dict[str, Any],
    examples: Sequence[Dict[str, Any]],
) -> str:
    s0_diagnostic_fields = []
    priority_value = _normalize_text(point.get("priority", ""))
    if priority_value:
        s0_diagnostic_fields.append(f"priority: {priority_value}")

    sections = [
        "[Current Sample]",
        f"Project: {_normalize_text(raw_record.get('project', ''))}",
        f"File: {_normalize_text(raw_record.get('file_name', ''))}",
        "",
        "[Current Code]",
        "```c",
        _normalize_text(code),
        "```",
        "",
        "[Suspicious Point To Verify]",
        f"Suspicious summary: {_normalize_text(point.get('retrieval_summary', point.get('summary', '')))}",
        f"Evidence hint: {_normalize_text(point.get('evidence_hint', ''))}",
        f"Needed historical example: {_normalize_text(point.get('needed_example', ''))}",
        *s0_diagnostic_fields,
        "",
        "[Retrieved Historical Case Analyses]",
    ]

    if examples:
        for rank, example in enumerate(examples, start=1):
            sections.extend(
                [
                    f"Example rank: {rank}",
                    format_example_for_prompt(example),
                    "",
                ]
            )
    else:
        sections.extend(["No retrieved examples are available.", ""])

    sections.extend(
        [
            "[Task]",
            "Verify only the suspicious point above.",
            "Use retrieved case analyses only as analogies to understand the mechanism and possible protections.",
            "Do not treat retrieved examples as proof that the current code is vulnerable.",
            "Return only the JSON object required by the system prompt.",
            "Do not include analysis text, Markdown, code fences, or a [FINAL_JSON] marker.",
        ]
    )
    return "\n".join(sections)


def build_prompt(
    tokenizer,
    raw_record: Dict[str, Any],
    code: str,
    point: Dict[str, Any],
    examples: Sequence[Dict[str, Any]],
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                raw_record=raw_record,
                code=code,
                point=point,
                examples=examples,
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
