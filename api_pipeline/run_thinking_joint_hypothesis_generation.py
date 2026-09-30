#!/usr/bin/env python3
"""Generate direct, thinking-enabled hypotheses with frozen taxonomy and M/R/E.

This is intentionally a new experiment.  It does not overwrite the frozen neutral
hypotheses used by the taxonomy-search and retrieval ablations.  Each request sees
only one vulnerable function and emits zero to three locally supported hypotheses,
with one shared frozen-taxonomy assignment and M/R/E retrieval fields per point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.common import build_client, call_chat_completion_with_usage, extract_json_object, load_api_key  # noqa: E402
from api_pipeline.run_taxonomy_route_retrieval import ROUTES, taxonomy_definition  # noqa: E402
from api_pipeline.source_text import exact_source_excerpt, trim  # noqa: E402


DEFAULT_SOURCE_DIR = Path("data/taxonomy_other_absorbs_lifetime_error_protocol_15_dev")
DEFAULT_OUTPUT_DIR = Path("data/hypothesis_thinking_joint_50_v1")
DEFAULT_TAXONOMY = "allocation_state_representation_other_absorbs_lifetime_error_protocol_15"
DEFAULT_TEST_PATH = Path("data/raw/primevul_test_merged.jsonl")
PROMPT_VERSION_RECALL = "recall_v2"
PROMPT_VERSIONS = (PROMPT_VERSION_RECALL,)
DEFAULT_PROMPT_VERSION = PROMPT_VERSION_RECALL


# This is the route-style portion of PROMPT_STYLE verbatim. Its preceding
# "one retrieval representation" framing is specific to the old one-item
# route task and conflicts with the direct 0--3 hypothesis task here.
DIRECT_MRE_STYLE = """Style contract shared by every route: write one factual, reusable sentence of at
most 55 words.  Use the vocabulary operation, object, trigger/invariant, guard,
or repair action when supported.  Preserve important distinctions such as cursor
remaining extent versus fixed capacity, arithmetic-created extent versus direct
transfer, and semantic scalar domain versus object/type validity.  Avoid generic
phrases such as 'validate input' or 'prevent vulnerability', broad CWE labels, and
project-specific narrative.  Return JSON only."""


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
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def latest_success(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "success":
            result[str(row["key"])] = row
    return result


def joint_system(
    taxonomy: dict[str, Any],
    analysis_profile: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    objectives = []
    fields = []
    for route, config in ROUTES.items():
        fields.append(config["hypothesis_field"])
        objectives.append(f"{route.upper()} route: {config['hypothesis_objective']}")

    hypothesis_schema = {
        "id": 1,
        "code_anchor": "short exact nonempty source excerpt",
        "operation": "dangerous operation in reusable semantic language",
        "risk_object": "object acted upon",
        "trigger_condition": "specific bad-state relation",
        "violated_invariant": "safety relation required before the operation",
    }
    hypothesis_schema.update({
        "uncertainty": "none or the missing local fact",
        "mechanism_family": "one allowed family",
        **{field: "one concise sentence" for field in fields},
    })
    schema = {"hypotheses": [hypothesis_schema]}
    taxonomy_payload = {
        "allowed_families": taxonomy["families"],
        "precedence": taxonomy["precedence"],
    }
    if analysis_profile == "thorough":
        thinking_instruction = (
            "Before producing the final JSON, first perform a full-function candidate-discovery pass. Trace "
            "control-flow reachability and identify security-relevant operations, risk objects, trigger "
            "conditions, and required invariants. For each candidate, determine whether visible protections "
            "clearly establish the required invariant for the same values, object, execution path, and operation. "
            "Do not discard a candidate merely because a protection exists; preserve locally plausible candidates "
            "when protection sufficiency remains uncertain. Then select up to three distinct candidates that "
            "maximize coverage of plausible failure points, assign taxonomy families, and formulate M/R/E. Do not "
            "finish early for apparently simple code. Keep that analysis in the thinking channel, then emit the "
            "complete JSON object only."
        )
    elif analysis_profile == "focused":
        thinking_instruction = (
            "Before producing the final JSON, privately make one complete candidate-discovery pass across the "
            "function. Identify distinct security-relevant operations, their risk objects, locally plausible "
            "triggers, and required invariants. Check whether visible protections clearly establish each invariant "
            "for the same values, object, execution path, and operation; retain a distinct candidate when that "
            "sufficiency remains uncertain. Do not retain candidates lacking a concrete locally plausible unsafe "
            "relation, but do not discard distinct candidates merely because a nearby protection exists. Select up "
            "to three candidates that maximize coverage, then assign taxonomy families and formulate M/R/E. Keep "
            "this analysis private and emit the complete JSON object only."
        )
    else:
        raise ValueError(f"unknown analysis profile: {analysis_profile}")

    task_instruction = """Analyze the given function and identify zero to three concrete potentially dangerous operation points as security hypotheses. A hypothesis does not need to prove that a vulnerability definitely exists. It must identify a concrete, locally plausible failed safety condition worth subsequent retrieval and verification.

Before selecting final hypotheses, privately scan the whole function for security-relevant operations and state transitions that could become unsafe under locally plausible conditions. Consider memory access or transfer, size and offset arithmetic, parsing and cursor movement, indexing, allocation, lifetime or ownership transitions, representation conversion, state or permission checks, and error-sensitive operations when present. Then select up to three distinct candidates that provide the strongest coverage of plausible failure points.

Every final hypothesis must identify a concrete operation, risk object, specific trigger condition, safety invariant required before that operation, and a short exact code anchor copied from the function. Ground a hypothesis in a concrete operation plus a plausible unsafe relation among visible values, objects, states, or control flow; do not propose one solely because a generic validation check is absent. Treat unseen caller or helper behavior as uncertainty, not as proof that a candidate is safe or unsafe.

Do not discard a candidate merely because a check, guard, bound, clamp, reset, or error path exists nearby. Determine whether it establishes the required invariant for the same values, object, execution path, and operation. Discard a candidate only when the visible code clearly establishes that invariant at the operation. If the protection may be incomplete or its sufficiency cannot be determined locally, retain the candidate and state the uncertainty. A debug assert alone is neither proof of a defect nor a production guarantee.

When multiple slots are used, maximize coverage of distinct dangerous operations or distinct failed invariants. Do not spend separate slots on branch-by-branch or near-duplicate variants of the same concern. Do not stay conservative by collapsing distinct, locally plausible candidates into one.

First identify each candidate failure condition from the code itself. Only then assign exactly one primary mechanism_family using the frozen taxonomy definitions and precedence; taxonomy assignment must not suppress, merge, or reshape an otherwise locally plausible hypothesis. Choose OTHER only when no remaining specific family fits. Finally write M/R/E for the selected hypothesis."""

    return "\n\n".join((
        task_instruction,
        thinking_instruction,
        DIRECT_MRE_STYLE,
        "Frozen taxonomy:\n" + json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        "Route objectives (identical to the existing M/R/E experiment):\n" + "\n".join(objectives),
        "Schema:\n" + json.dumps(schema, ensure_ascii=False, indent=2),
    ))


def validate(
    value: Any,
    code: str,
    allowed_families: set[str],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(f"unknown prompt version: {prompt_version}")
    if not isinstance(value, dict) or not isinstance(value.get("hypotheses"), list):
        raise ValueError("response requires a hypotheses list")
    hypotheses = value["hypotheses"]
    if len(hypotheses) > 3:
        raise ValueError("at most three hypotheses are allowed")
    identifiers: set[int] = set()
    required = (
        "code_anchor", "operation", "risk_object", "trigger_condition",
        "violated_invariant", "uncertainty",
        "mechanism_claim", "repair_sought", "evidence_to_check",
    )
    for item in hypotheses:
        if not isinstance(item, dict):
            raise ValueError("hypothesis must be an object")
        identifier = item.get("id")
        if not isinstance(identifier, int) or identifier in identifiers:
            raise ValueError("hypothesis ids must be unique integers")
        identifiers.add(identifier)
        for field in required:
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"missing hypothesis field: {field}")
        if item.get("mechanism_family") not in allowed_families:
            raise ValueError(f"invalid mechanism_family: {item.get('mechanism_family')!r}")
        anchor = exact_source_excerpt(item["code_anchor"], code)
        if anchor is None:
            raise ValueError("code_anchor is not an exact source excerpt")
        item["code_anchor"] = anchor
        for field in ("mechanism_claim", "repair_sought", "evidence_to_check"):
            if len(item[field].split()) > 55 or len(item[field]) > 700:
                raise ValueError(f"{field} exceeds the shared M/R/E style bound")
            item[field] = item[field].strip()
    return {"hypotheses": hypotheses}


def selected_records(args: argparse.Namespace) -> tuple[list[int], dict[int, dict[str, Any]], int]:
    manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = [int(idx) for idx in manifest["test_ids"]]
    if args.ids:
        requested = [int(value) for value in args.ids.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(ids))
        if unknown:
            raise RuntimeError(f"--ids are not in the frozen source manifest: {unknown}")
        requested_set = set(requested)
        ids = [idx for idx in ids if idx in requested_set]
    records = {int(row["idx"]): row for row in read_jsonl(args.test_path)}
    missing = [idx for idx in ids if idx not in records]
    if missing:
        raise RuntimeError(f"test manifest references missing records: {missing}")
    return ids, records, int(manifest["request_code_chars"])


def prepare(args: argparse.Namespace) -> None:
    ids, _, request_limit = selected_records(args)
    taxonomy = taxonomy_definition(args.source_dir, args.taxonomy)
    manifest = {
        "experiment": "thinking_enabled_direct_joint_hypothesis_generation_v1",
        "source_dir": str(args.source_dir),
        "source_manifest_sha256": hashlib.sha256((args.source_dir / "manifest.json").read_bytes()).hexdigest(),
        "test_ids": ids,
        "taxonomy": args.taxonomy,
        "taxonomy_definition": taxonomy,
        "generation": {
            "model_name": args.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_type": "enabled",
            "reasoning_effort": args.reasoning_effort,
            "response_format": {"type": "json_object"} if args.json_mode else None,
            "max_new_tokens": args.max_new_tokens,
            "analysis_profile": args.analysis_profile,
            "prompt_version": args.prompt_version,
            "prompt_hash": stable_hash(joint_system(taxonomy, args.analysis_profile, args.prompt_version)),
            "mre_contract": "Definitions and the 55-word style limit are reused verbatim from ROUTES and PROMPT_STYLE.",
        },
        "input_contract": {
            "visible_to_model": "func_vuln only",
            "function_char_limit": request_limit,
            "max_hypotheses": 3,
        },
    }
    target = args.output_dir / "experiment_manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("existing manifest differs; choose a fresh output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(target, manifest)
    print(json.dumps({"stage": "prepare", "functions": len(ids), "taxonomy": args.taxonomy}))


def generate(args: argparse.Namespace) -> None:
    ids, records, request_limit = selected_records(args)
    taxonomy = taxonomy_definition(args.source_dir, args.taxonomy)
    allowed_families = set(taxonomy["families"])
    system = joint_system(taxonomy, args.analysis_profile, args.prompt_version)
    cache_path = args.output_dir / "hypotheses.jsonl"
    cache = latest_success(read_jsonl(cache_path))
    tasks = []
    for idx in ids:
        code = trim(str(records[idx]["func_vuln"]), request_limit)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps({"idx": idx, "function": code}, ensure_ascii=False)}]
        task = {"key": f"test:{idx}", "idx": idx, "messages": messages, "input_hash": stable_hash(messages)}
        previous = cache.get(task["key"])
        if previous and previous.get("input_hash") != task["input_hash"]:
            raise RuntimeError(f"stale cache for {task['key']}; choose a new output directory")
        if task["key"] not in cache:
            tasks.append(task)
    if args.limit:
        tasks = tasks[:args.limit]
    print(json.dumps({"stage": "generate", "cached": len(cache), "pending": len(tasks)}), flush=True)
    if not tasks:
        return
    api_key = load_api_key(None, args.api_key_env)

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        client = build_client(args.api_base, api_key).with_options(max_retries=0)
        result: dict[str, Any] = {
            **task, "model_name": args.model_name, "temperature": 0.0, "top_p": 1.0,
            "thinking_type": "enabled", "reasoning_effort": args.reasoning_effort,
            "response_format": {"type": "json_object"} if args.json_mode else None,
            "max_new_tokens": args.max_new_tokens,
            "attempts": [],
        }
        messages = task["messages"]
        try:
            for attempt in range(1, args.retries + 1):
                event: dict[str, Any] = {"attempt": attempt, "started_at": time.time(), "request_messages": messages}
                raw: str | None = None
                usage: dict[str, Any] | None = None
                try:
                    raw, usage = call_chat_completion_with_usage(
                        client, args.model_name, messages, 0.0, 1.0, args.max_new_tokens,
                        args.timeout, args.reasoning_effort, "enabled",
                        response_format={"type": "json_object"} if args.json_mode else None,
                    )
                    # Preserve provider output and usage even when schema validation fails.
                    event.update(raw_output=raw, usage=usage)
                    parsed = validate(
                        extract_json_object(raw),
                        str(records[task["idx"]]["func_vuln"]),
                        allowed_families,
                        args.prompt_version,
                    )
                    event.update(status="success")
                    result["attempts"].append(event)
                    return {**result, "status": "success", "raw_output": raw, "parsed": parsed, "usage": usage}
                except Exception as exc:
                    event.update(status="failed", error=str(exc).replace(api_key, "[REDACTED]"))
                    result["attempts"].append(event)
                    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
                    if completion_tokens >= args.max_new_tokens:
                        result["retryable_after_budget_change"] = True
                        return {**result, "status": "failed", "error": event["error"]}
                    if "insufficient balance" in event["error"].lower() or "error code: 402" in event["error"].lower():
                        result["retryable_after_balance_replenished"] = True
                        return {**result, "status": "failed", "error": event["error"]}
                    if attempt < args.retries:
                        messages = list(messages) + [{
                            "role": "user",
                            "content": "Return the complete JSON for the same function. Keep private reasoning private; every code_anchor must be an exact source excerpt, all M/R/E fields must be nonempty and follow their stated definitions.",
                        }]
                        time.sleep(min(20.0, 2.0 ** attempt))
            return {**result, "status": "failed", "error": result["attempts"][-1]["error"]}
        finally:
            client.close()

    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
        futures = [executor.submit(worker, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(cache_path, row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "generate", "completed": completed, "remaining": len(tasks) - completed, "idx": row["idx"], "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"generation had {failures} failures; successful rows are resumable")


def summarize(args: argparse.Namespace) -> None:
    ids, _, _ = selected_records(args)
    rows = latest_success(read_jsonl(args.output_dir / "hypotheses.jsonl"))
    if set(rows) != {f"test:{idx}" for idx in ids}:
        raise RuntimeError(f"generation incomplete: {len(rows)}/{len(ids)}")
    points = [point for row in rows.values() for point in row["parsed"]["hypotheses"]]
    usage: Counter[str] = Counter()
    for row in rows.values():
        row_usage = row.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(row_usage.get(key) or 0)
        reasoning_tokens = int((row_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        usage["reasoning_tokens"] += reasoning_tokens
        usage["visible_output_tokens"] += max(0, int(row_usage.get("completion_tokens") or 0) - reasoning_tokens)
    write_json(args.output_dir / "generation_summary.json", {
        "functions": len(ids),
        "hypotheses": len(points),
        "zero_hypothesis_functions": sum(not rows[f"test:{idx}"]["parsed"]["hypotheses"] for idx in ids),
        "families": dict(sorted(Counter(point["mechanism_family"] for point in points).items())),
        "usage_final_success_attempts_only": dict(usage),
    })
    print(json.dumps(json.loads((args.output_dir / "generation_summary.json").read_text(encoding="utf-8")), ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    parser.add_argument("--model-name", default="your-model-id")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=393216)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    parser.add_argument("--analysis-profile", choices=("thorough", "focused"), default="thorough")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--json-mode", action="store_true",
                        help="Request OpenAI-compatible JSON mode from the provider.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Optional comma-separated frozen test idx values for an isolated pilot.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(function=prepare)
    subparsers.add_parser("generate").set_defaults(function=generate)
    subparsers.add_parser("summarize").set_defaults(function=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
