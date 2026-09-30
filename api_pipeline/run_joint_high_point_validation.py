#!/usr/bin/env python3
"""Point-level validation for the joint-high hypothesis retrieval experiment.

This is deliberately downstream of retrieval.  It never regenerates hypotheses,
M/R/E fields, retrieval rankings, or reranker scores.  For the unique knowledge
units selected by the tuned reranker it first materializes four *guidance-only*
fields from the neutral repair signature.  The point verifier then sees the
current vulnerable function, one generated hypothesis, and only those four
guidance fields for its already selected top cases.

Patch-grounded hypothesis labels are used only after model inference to score
the experiment.  They are never included in either generation or verification
prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.common import (  # noqa: E402
    build_client,
    call_chat_completion_with_usage,
    extract_json_object,
    load_api_key,
)
from prompts.point_judgment import (  # noqa: E402
    CASE_MAPPED_SYSTEM_PROMPT,
    CASE_MAPPED_LOCAL_ARITHMETIC_SYSTEM_PROMPT,
    CASE_MAPPED_VISIBLE_CARDINALITY_SYSTEM_PROMPT,
    CASE_MAPPED_VISIBLE_EVIDENCE_SYSTEM_PROMPT,
    SYSTEM_PROMPT as S5_SYSTEM_PROMPT,
)


GUIDANCE_FIELDS = (
    "case_explanation",
    "vulnerable_pattern",
    "repaired_pattern",
    "verification_guidance",
)
VALID_VERDICTS = {"supported", "unsupported", "insufficient_evidence"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def latest_success(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "success":
            latest[str(row[key])] = row
    return latest


def redact_exception(exc: Exception, api_key: str) -> str:
    return str(exc).replace(api_key, "[REDACTED]")


def usage_sum(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "success":
            continue
        for attempt in row.get("attempts") or []:
            usage = attempt.get("usage") or {}
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(field), (int, float)):
                    total[field] += int(usage[field])
    return dict(total)


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(args.selected_path)
    if not rows:
        raise RuntimeError(f"No tuned-reranker selections found: {args.selected_path}")
    duplicate = len({(row["hypothesis_id"], int(row["candidate_idx"])) for row in rows}) != len(rows)
    if duplicate:
        raise RuntimeError("Selected reranker rows contain duplicate hypothesis/candidate pairs")
    return sorted(rows, key=lambda row: (str(row["hypothesis_id"]), int(row["rerank_rank"])))


def generated_hypotheses(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in latest_success(read_jsonl(args.hypotheses_path), "key").values():
        idx = int(row["idx"])
        for point in row["parsed"]["hypotheses"]:
            signature_id = f"test:{idx}:h{int(point['id'])}"
            if signature_id in result:
                raise RuntimeError(f"Duplicate generated hypothesis: {signature_id}")
            result[signature_id] = point
    if not result:
        raise RuntimeError(f"No successful hypothesis output found: {args.hypotheses_path}")
    return result


def raw_test_records(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    records = {int(row["idx"]): row for row in read_jsonl(args.test_path)}
    if not records:
        raise RuntimeError(f"No test records found: {args.test_path}")
    return records


def patch_gold_labels(args: argparse.Namespace) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in read_jsonl(args.patch_gold_path):
        if row.get("status") != "reviewed":
            continue
        signature_id = str(row.get("signature_id") or "")
        label = row.get("final_match_label")
        if not signature_id or label not in {0, 1, 2}:
            continue
        result[signature_id] = int(label)
    if not result:
        raise RuntimeError(f"No patch-grounded review labels found: {args.patch_gold_path}")
    return result


def guidance_messages(candidate: dict[str, Any]) -> list[dict[str, str]]:
    """Produce validation-only repair guidance from the frozen neutral signature."""
    schema = {field: "one self-contained concise sentence or short paragraph" for field in GUIDANCE_FIELDS}
    system = "\n\n".join((
        "You create verification guidance from ONE neutralized historical repair signature.",
        "A later verifier will see only the four requested fields, a current function, and a suspicious hypothesis. The verifier will not see the original repair code, diff, project name, CVE, or the complete historical case. Therefore make every field concrete and self-contained.",
        "Use only relations present in the supplied neutral signature. Do not invent API contracts, exploitability, patch details, or facts absent from the signature. These fields are mechanism hints, never proof that another function is vulnerable.",
        "Field requirements:\n"
        "- case_explanation: State the historical operation, risk object, failed safety relation, consequence, and why its repair mattered.\n"
        "- vulnerable_pattern: State the pre-repair unsafe relation, the trigger, and the consequential operation.\n"
        "- repaired_pattern: State the guard, state change, or invariant-establishing repair and the unsafe relation it prevents.\n"
        "- verification_guidance: State actionable checks in another function: the operation, values or state, safety relation, and local guard that could confirm or reject a similar concern.",
        "Before producing the final JSON, reason thoroughly and privately about the supplied signature. Keep private reasoning out of the final output. Each field may be up to 120 words. Return JSON only, with exactly this schema:\n"
        + json.dumps(schema, ensure_ascii=False),
    ))
    user = {
        "candidate_signature_id": candidate["candidate_signature_id"],
        "neutral_repair_signature": candidate["candidate_signature"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def validate_guidance(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("guidance output is not a JSON object")
    result: dict[str, str] = {}
    for field in GUIDANCE_FIELDS:
        text = value.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"missing nonempty {field}")
        if len(text.split()) > 120 or len(text) > 1500:
            raise ValueError(f"{field} exceeds the guidance-field bound")
        result[field] = text.strip()
    return result


def point_messages(
    raw_record: dict[str, Any],
    hypothesis: dict[str, Any],
    examples: list[tuple[dict[str, Any], dict[str, str]]],
    verification_prompt_mode: str,
) -> list[dict[str, str]]:
    sections = [
        "[Current Sample]",
        f"Project: {raw_record.get('project', '')}",
        f"File: {raw_record.get('file_name', '')}",
        "",
        "[Current Code]",
        "```c",
        str(raw_record.get("func_vuln") or ""),
        "```",
        "",
        "[Suspicious Point To Verify]",
        f"Code anchor: {hypothesis.get('code_anchor', '')}",
        f"Suspicious summary: {hypothesis.get('mechanism_claim', hypothesis.get('operation', ''))}",
        f"Evidence hint: {hypothesis.get('evidence_to_check', '')}",
        f"Needed historical example: {hypothesis.get('repair_sought', '')}",
        f"Mechanism family: {hypothesis.get('mechanism_family', '')}",
        f"Operation: {hypothesis.get('operation', '')}",
        f"Risk object: {hypothesis.get('risk_object', '')}",
        f"Trigger condition: {hypothesis.get('trigger_condition', '')}",
        f"Violated invariant: {hypothesis.get('violated_invariant', '')}",
        f"Uncertainty: {hypothesis.get('uncertainty', '')}",
        "",
        "[Retrieved Historical Case Analyses]",
    ]
    for rank, (candidate, guidance) in enumerate(examples, 1):
        sections.extend((
            f"Example rank: {rank}",
            "[Case explanation]",
            guidance["case_explanation"],
            "[Vulnerable pattern]",
            guidance["vulnerable_pattern"],
            "[Repaired pattern]",
            guidance["repaired_pattern"],
            "[Guidance for similar suspicious points]",
            guidance["verification_guidance"],
            "",
        ))
    sections.extend((
        "[Task]",
        "Verify only the suspicious point above against the current function.",
        "Historical cases may inform the mechanism and checks to inspect, but are not proof that the current code is vulnerable.",
        "Return only the JSON object required by the system prompt.",
    ))
    system_prompt = {
        "baseline": S5_SYSTEM_PROMPT,
        "case-mapped": CASE_MAPPED_SYSTEM_PROMPT,
        "case-mapped-visible-evidence": CASE_MAPPED_VISIBLE_EVIDENCE_SYSTEM_PROMPT,
        "case-mapped-visible-cardinality": CASE_MAPPED_VISIBLE_CARDINALITY_SYSTEM_PROMPT,
        "case-mapped-local-arithmetic": CASE_MAPPED_LOCAL_ARITHMETIC_SYSTEM_PROMPT,
    }[verification_prompt_mode]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(sections)},
    ]


def validate_point_output(
    value: Any,
    point_id: int,
    *,
    expected_example_ranks: list[int],
    require_case_mappings: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("point output is not a JSON object")
    verdict = str(value.get("verdict") or "").strip()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"invalid point verdict: {verdict!r}")
    confidence = str(value.get("confidence") or "").strip()
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"invalid point confidence: {confidence!r}")
    result = {
        "point_id": int(value.get("point_id", point_id)),
        "verdict": verdict,
        "confidence": confidence,
        "risky_operation_present": str(value.get("risky_operation_present") or "unknown").strip(),
        "trigger_condition_present": str(value.get("trigger_condition_present") or "unknown").strip(),
        "protection_status": str(value.get("protection_status") or "unknown").strip(),
        "support_evidence": [str(item) for item in value.get("support_evidence", [])],
        "blocking_evidence": [str(item) for item in value.get("blocking_evidence", [])],
        "missing_context": str(value.get("missing_context") or ""),
        "matched_example_ranks": [
            int(item) for item in value.get("matched_example_ranks", [])
            if str(item).strip().lstrip("-").isdigit()
        ],
    }
    if result["point_id"] != point_id:
        result["point_id"] = point_id
    if require_case_mappings:
        raw_mappings = value.get("case_mappings")
        if not isinstance(raw_mappings, list):
            raise ValueError("case-mapped output must include case_mappings")
        mappings = []
        valid_statuses = {"established", "blocked", "missing_context", "not_applicable"}
        valid_influences = {"supports", "refutes", "context_only", "none"}
        for item in raw_mappings:
            if not isinstance(item, dict):
                raise ValueError("case_mappings must contain JSON objects")
            rank = item.get("example_rank")
            if not str(rank).strip().lstrip("-").isdigit():
                raise ValueError("case mapping has an invalid example_rank")
            status = str(item.get("mapping_status") or "").strip()
            influence = str(item.get("influence") or "").strip()
            if status not in valid_statuses:
                raise ValueError(f"invalid mapping_status: {status!r}")
            if influence not in valid_influences:
                raise ValueError(f"invalid mapping influence: {influence!r}")
            mappings.append({
                "example_rank": int(rank),
                "case_principle": str(item.get("case_principle") or "").strip(),
                "current_code_mapping": str(item.get("current_code_mapping") or "").strip(),
                "mapping_status": status,
                "influence": influence,
            })
        if sorted(item["example_rank"] for item in mappings) != sorted(expected_example_ranks):
            raise ValueError("case_mappings must cover every retrieved example exactly once")
        result["case_mappings"] = sorted(mappings, key=lambda item: item["example_rank"])
    return result


def call_with_retries(
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    timeout: int,
    retries: int,
    validator: Any,
    response_format: dict[str, str] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    client = build_client(api_base, api_key).with_options(max_retries=0)
    try:
        for attempt in range(1, retries + 1):
            raw_output: str | None = None
            usage: dict[str, Any] | None = None
            try:
                raw_output, usage = call_chat_completion_with_usage(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    max_new_tokens=max_new_tokens,
                    timeout=timeout,
                    reasoning_effort="high",
                    thinking_type="enabled",
                    response_format=response_format,
                )
                parsed = validator(extract_json_object(raw_output))
                attempts.append({"attempt": attempt, "status": "success", "raw_output": raw_output, "usage": usage})
                return parsed, attempts
            except Exception as exc:
                failure = {"attempt": attempt, "status": "failed", "error": redact_exception(exc, api_key)}
                if raw_output is not None:
                    failure["raw_output"] = raw_output
                if usage is not None:
                    failure["usage"] = usage
                attempts.append(failure)
                completion_tokens = int((usage or {}).get("completion_tokens") or 0)
                if completion_tokens >= max_new_tokens:
                    raise RuntimeError(
                        f"output budget exhausted at {max_new_tokens} tokens; "
                        "do not retry until the budget configuration changes"
                    ) from exc
                if "insufficient balance" in failure["error"].lower() or "error code: 402" in failure["error"].lower():
                    raise RuntimeError(
                        "provider returned insufficient balance; do not retry until balance is replenished"
                    ) from exc
                if attempt < retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(attempts[-1]["error"])
    finally:
        client.close()


def generate_guidance(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    path = args.output_dir / "knowledge_guidance.jsonl"
    cached = latest_success(read_jsonl(path), "key")
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        signature_id = str(row["candidate_signature_id"])
        prior = unique.get(signature_id)
        if prior is not None and prior["candidate_signature"] != row["candidate_signature"]:
            raise RuntimeError(f"inconsistent neutral signature for {signature_id}")
        unique[signature_id] = row
    tasks = []
    for signature_id in sorted(unique):
        candidate = unique[signature_id]
        messages = guidance_messages(candidate)
        task = {
            "key": f"guidance:{signature_id}",
            "candidate_signature_id": signature_id,
            "candidate_idx": int(candidate["candidate_idx"]),
            "candidate_signature": candidate["candidate_signature"],
            "messages": messages,
            "input_hash": stable_hash(messages),
        }
        prior = cached.get(task["key"])
        if prior and prior.get("input_hash") != task["input_hash"]:
            raise RuntimeError(f"stale guidance cache for {task['key']}")
        if task["key"] not in cached:
            tasks.append(task)
    if args.limit:
        tasks = tasks[:args.limit]
    print(json.dumps({"stage": "guidance", "cached": len(cached), "unique_selected": len(unique), "pending": len(tasks)}), flush=True)
    if not tasks:
        return
    api_key = load_api_key(None, args.api_key_env)

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        result = {
            **task,
            "model_name": args.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_type": "enabled",
            "reasoning_effort": "high",
            "max_new_tokens": args.guidance_max_new_tokens,
        }
        try:
            parsed, attempts = call_with_retries(
                api_base=args.api_base,
                api_key=api_key,
                model_name=args.model_name,
                messages=task["messages"],
                max_new_tokens=args.guidance_max_new_tokens,
                timeout=args.timeout,
                retries=args.retries,
                validator=validate_guidance,
                response_format={"type": "json_object"},
            )
            return {**result, "status": "success", "parsed": parsed, "attempts": attempts}
        except Exception as exc:
            return {**result, "status": "failed", "error": redact_exception(exc, api_key), "attempts": []}

    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(path, row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "guidance", "completed": completed, "remaining": len(tasks) - completed, "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"{failures} guidance tasks failed; successful rows were persisted")


def guidance_outputs(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    rows = latest_success(read_jsonl(args.guidance_dir / "knowledge_guidance.jsonl"), "key")
    output = {str(row["candidate_signature_id"]): row["parsed"] for row in rows.values()}
    expected = {str(row["candidate_signature_id"]) for row in selected}
    missing = sorted(expected - set(output))
    if missing:
        raise RuntimeError(f"Guidance incomplete: {len(output)}/{len(expected)}; e.g. {missing[:3]}")
    return output


def verify_points(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    guidance = guidance_outputs(args, selected)
    hypotheses = generated_hypotheses(args)
    records = raw_test_records(args)
    cached = latest_success(read_jsonl(args.output_dir / "point_verification.jsonl"), "key")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["hypothesis_id"])].append(row)
    tasks = []
    for hypothesis_id in sorted(grouped):
        candidates = sorted(grouped[hypothesis_id], key=lambda row: int(row["rerank_rank"]))
        hypothesis = hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise RuntimeError(f"Missing generated hypothesis: {hypothesis_id}")
        query_idx = int(candidates[0]["query_idx"])
        record = records.get(query_idx)
        if record is None:
            raise RuntimeError(f"Missing raw test record idx={query_idx}")
        examples = [(candidate, guidance[str(candidate["candidate_signature_id"])]) for candidate in candidates]
        messages = point_messages(record, hypothesis, examples, args.verification_prompt_mode)
        key = f"point:{hypothesis_id}"
        task = {
            "key": key,
            "hypothesis_id": hypothesis_id,
            "query_idx": query_idx,
            "hypothesis": hypothesis,
            "selected_candidates": [
                {
                    "rerank_rank": int(candidate["rerank_rank"]),
                    "candidate_idx": int(candidate["candidate_idx"]),
                    "candidate_signature_id": candidate["candidate_signature_id"],
                    "reranker_score": candidate["reranker_score"],
                }
                for candidate in candidates
            ],
            "messages": messages,
            "input_hash": stable_hash(messages),
        }
        prior = cached.get(key)
        if prior and prior.get("input_hash") != task["input_hash"]:
            raise RuntimeError(f"stale point-verification cache for {key}")
        if key not in cached:
            tasks.append(task)
    if args.limit:
        tasks = tasks[:args.limit]
    print(json.dumps({"stage": "point_verification", "cached": len(cached), "expected": len(grouped), "pending": len(tasks)}), flush=True)
    if not tasks:
        return
    api_key = load_api_key(None, args.api_key_env)

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        result = {
            **task,
            "model_name": args.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_type": "enabled",
            "reasoning_effort": "high",
            "max_new_tokens": args.verification_max_new_tokens,
        }
        try:
            point_id = int(task["hypothesis"].get("id", -1))
            parsed, attempts = call_with_retries(
                api_base=args.api_base,
                api_key=api_key,
                model_name=args.model_name,
                messages=task["messages"],
                max_new_tokens=args.verification_max_new_tokens,
                timeout=args.timeout,
                retries=args.retries,
                validator=lambda value: validate_point_output(
                    value,
                    point_id,
                    expected_example_ranks=list(range(1, len(candidates) + 1)),
                    require_case_mappings=args.verification_prompt_mode != "baseline",
                ),
                response_format={"type": "json_object"},
            )
            return {**result, "status": "success", "parsed": parsed, "attempts": attempts}
        except Exception as exc:
            return {**result, "status": "failed", "error": redact_exception(exc, api_key), "attempts": []}

    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(args.output_dir / "point_verification.jsonl", row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "point_verification", "completed": completed, "remaining": len(tasks) - completed, "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"{failures} point-verification tasks failed; successful rows were persisted")


def summarize(args: argparse.Namespace, selected: list[dict[str, Any]]) -> None:
    expected_hypotheses = {str(row["hypothesis_id"]) for row in selected}
    guidance_rows = read_jsonl(args.guidance_dir / "knowledge_guidance.jsonl")
    point_rows = read_jsonl(args.output_dir / "point_verification.jsonl")
    points = latest_success(point_rows, "key")
    gold = patch_gold_labels(args)
    verdicts: Counter[str] = Counter()
    group_rows: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    no_gold: list[str] = []
    for hypothesis_id in sorted(expected_hypotheses):
        row = points.get(f"point:{hypothesis_id}")
        if row is None:
            continue
        verdict = str(row["parsed"]["verdict"])
        verdicts[verdict] += 1
        label = gold.get(hypothesis_id)
        if label is None:
            no_gold.append(hypothesis_id)
        else:
            group_rows[label].append(row)

    def group_summary(label: int) -> dict[str, Any]:
        rows = group_rows[label]
        value = Counter(str(row["parsed"]["verdict"]) for row in rows)
        supported = value["supported"]
        rejected = len(rows) - supported
        result: dict[str, Any] = {
            "patch_gold_label": label,
            "total": len(rows),
            "verdict_counts": dict(value),
            "supported": supported,
            "not_supported": rejected,
        }
        if label == 2:
            result["correct_supported"] = supported
            result["correct_rate"] = supported / len(rows) if rows else None
        elif label == 0:
            result["correct_rejected"] = rejected
            result["correct_rate"] = rejected / len(rows) if rows else None
        else:
            result["interpretation"] = "transferable/non-target hypothesis: reported separately, not forced into a binary patch-target correctness claim"
        return result

    strict_known = group_rows[0] + group_rows[2]
    strict_correct = (
        sum(row["parsed"]["verdict"] == "supported" for row in group_rows[2])
        + sum(row["parsed"]["verdict"] != "supported" for row in group_rows[0])
    )
    expected_candidates = {str(row["candidate_signature_id"]) for row in selected}
    result = {
        "experiment": "joint_high_point_validation",
        "configuration": {
            "hypothesis_source": str(args.hypotheses_path.resolve()),
            "selected_top3": str(args.selected_path.resolve()),
            "patch_gold_review": str(args.patch_gold_path.resolve()),
            "model_name": args.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_type": "enabled",
            "reasoning_effort": "high",
            "verification_prompt_mode": args.verification_prompt_mode,
            "knowledge_guidance_source": str(args.guidance_dir.resolve()),
            "guidance_fields": list(GUIDANCE_FIELDS),
            "knowledge_visibility_to_verifier": "four guidance fields only; no repair code, diff, project, or CVE metadata",
            "verification_visibility": (
                "current vulnerable function, one generated hypothesis, and "
                + args.selection_description
            ),
        },
        "coverage": {
            "selected_pairs": len(selected),
            "selected_hypotheses": len(expected_hypotheses),
            "selected_functions": len({int(row["query_idx"]) for row in selected}),
            "unique_selected_knowledge": len(expected_candidates),
            "guidance_successes": len(latest_success(guidance_rows, "key")),
            "point_verification_successes": len(points),
            "point_verification_failures": len(expected_hypotheses - {str(row["hypothesis_id"]) for row in points.values()}),
            "patch_gold_available": sum(len(group_rows[label]) for label in (0, 1, 2)),
            "patch_gold_unavailable": len(no_gold),
        },
        "verdict_counts": dict(verdicts),
        "point_verification_by_patch_gold_label": {
            "label_2_exact_patch_target": group_summary(2),
            "label_1_transferable_non_target": group_summary(1),
            "label_0_unrelated_or_unsupported": group_summary(0),
        },
        "strict_patch_target_binary": {
            "denominator": len(strict_known),
            "correct": strict_correct,
            "accuracy": strict_correct / len(strict_known) if strict_known else None,
            "definition": "label=2 must be supported; label=0 must be unsupported or insufficient_evidence. label=1 is deliberately excluded because it is semantically related but not a patch-grounded target.",
        },
        "usage": {
            "guidance_generation": usage_sum(guidance_rows),
            "point_verification": usage_sum(point_rows),
        },
    }
    write_json(args.output_dir / "summary.json", result)
    lines = [
        "# Joint-High Point Validation", "",
        f"- Selected hypotheses: {result['coverage']['selected_hypotheses']}",
        f"- Unique selected knowledge cases: {result['coverage']['unique_selected_knowledge']}",
        f"- Point-verification successes/failures: {result['coverage']['point_verification_successes']}/{result['coverage']['point_verification_failures']}",
        f"- Exact target support: {result['point_verification_by_patch_gold_label']['label_2_exact_patch_target']['correct_supported']}/{result['point_verification_by_patch_gold_label']['label_2_exact_patch_target']['total']}",
        f"- Label-0 rejection: {result['point_verification_by_patch_gold_label']['label_0_unrelated_or_unsupported']['correct_rejected']}/{result['point_verification_by_patch_gold_label']['label_0_unrelated_or_unsupported']['total']}",
        f"- Strict patch-target binary: {strict_correct}/{len(strict_known)} ({result['strict_patch_target_binary']['accuracy']:.1%})" if strict_known else "- Strict patch-target binary: n/a",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("guidance", "verify", "summarize", "run"))
    parser.add_argument(
        "--hypotheses-path", type=Path,
        default=ROOT / "data" / "hypothesis_thinking_joint_50_focused_high_jsonmode_full_v1" / "hypotheses.jsonl",
    )
    parser.add_argument(
        "--selected-path", type=Path,
        default=ROOT / "data" / "hypothesis_thinking_joint_50_focused_high_jsonmode_full_v1_retrieval_eval" / "reranker" / "preference_reranker_50pct_opt_top3.jsonl",
    )
    parser.add_argument(
        "--selection-description",
        default="tuned-reranker-selected top cases",
        help="Human-readable provenance for selected cases in the summary.",
    )
    parser.add_argument(
        "--patch-gold-path", type=Path,
        default=ROOT / "data" / "hypothesis_thinking_joint_50_focused_high_jsonmode_full_v1_gold_eval_v1" / "match_manual_review.jsonl",
    )
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "raw" / "primevul_test_merged.jsonl")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "hypothesis_thinking_joint_50_focused_high_point_validation_v1",
    )
    parser.add_argument(
        "--guidance-dir",
        type=Path,
        default=None,
        help="Directory containing reusable knowledge_guidance.jsonl; defaults to --output-dir.",
    )
    parser.add_argument(
        "--verification-prompt-mode",
        choices=(
            "baseline",
            "case-mapped",
            "case-mapped-visible-evidence",
            "case-mapped-visible-cardinality",
            "case-mapped-local-arithmetic",
        ),
        default="baseline",
    )
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--model-name", default="your-model-id")
    parser.add_argument("--guidance-max-new-tokens", type=int, default=393216)
    parser.add_argument("--verification-max-new-tokens", type=int, default=393216)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.guidance_dir is None:
        args.guidance_dir = args.output_dir

    selected = selected_rows(args)
    if args.command in {"guidance", "run"}:
        generate_guidance(args, selected)
    if args.command in {"verify", "run"}:
        verify_points(args, selected)
    if args.command in {"summarize", "run"}:
        summarize(args, selected)


if __name__ == "__main__":
    main()
