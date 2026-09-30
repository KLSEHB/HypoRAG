"""Evaluate code-only retrieval baselines on the frozen taxonomy holdout.

This script deliberately keeps retrieval and evaluation separate.  Random,
BM25, and dense BGE retrieve from vulnerable-function source alone.  A strict
offline audit then receives each target and candidate repair diff and decides
whether they implement the same immediate failed safety condition.  The latter
matches the mechanism-match unit used by the paper's code-retrieval table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_pipeline.common import (  # noqa: E402
    build_client,
    call_chat_completion_with_usage,
    extract_json_object,
    load_api_key,
)
from api_pipeline.retrieval_core import (  # noqa: E402
    encode_texts,
    load_embedding_model,
    resolve_device,
)
from api_pipeline.run_code_retrieval_comparison import CodeBm25  # noqa: E402


METHODS = ("random", "bm25_code", "dense_code")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", re.UNICODE)

STRICT_REPAIR_AUDIT_SYSTEM = """You are a strict evaluator of mechanism match
between two vulnerability repair pairs.  Each item provides target and candidate
metadata plus their vulnerable-to-fixed repair diffs.  The diffs are the primary
evidence; metadata may be incomplete or misleading.

Set is_match=true only when the repairs address the same immediate failed safety
condition: the risky operation and the specific trigger/invariant/protection
relation align closely enough that the candidate is a direct mechanism-level
reference for the target.  Set false when they share only a broad CWE-like label,
an out-of-bounds or validation theme, surface wording, or generic defensive advice.
Do not grant a match merely because both add a check.  If the diff evidence is
insufficient to establish a direct mechanism match, return false.

Return JSON only:
{"judgments":[{"audit_id":"id","is_match":false,"reason":"short decisive reason"}]}"""

TRANSFER_REPAIR_AUDIT_SYSTEM = """You audit the verification utility of a
candidate historical repair for a target vulnerability repair. Each item provides
the target and candidate metadata plus vulnerable-to-fixed repair diffs. The diffs
are primary evidence; metadata may be incomplete or misleading.

Assign exactly one label per item:
- 2: the same immediate failed safety condition is present. Risky operation and
  trigger/invariant/protection relation align closely enough for a direct
  mechanism-level reference.
- 1: the immediate failure differs, but the candidate provides a concrete,
  transferable verification or repair principle that would materially help assess
  the target. It must be more specific than sharing a broad CWE-like category.
- 0: only broad weakness, generic defensive advice, surface wording, or an
  unrelated operation overlaps; it does not materially guide verification.

Do not give credit merely because both patches add a check. If no concrete lesson
transfers from the candidate's trigger and protection relation, use 0.

Return JSON only:
{"judgments":[{"audit_id":"id","label":0,"root_cause_alignment":"short explanation","trigger_invariant_alignment":"short explanation","transferable_principle":"short explanation","reason":"short decisive reason"}]}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def record_cves(record: dict[str, Any]) -> set[str]:
    raw = record.get("cve")
    values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    return {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip() and str(value).strip().lower() != "none"
    }


def code_token_set(record: dict[str, Any]) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(record.get("func_vuln") or ""))}


def token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def repair_diff(record: dict[str, Any]) -> str:
    import difflib

    return "\n".join(
        difflib.unified_diff(
            str(record.get("func_vuln") or "").splitlines(),
            str(record.get("func_safe") or "").splitlines(),
            fromfile="func_vuln",
            tofile="func_safe",
            n=3,
            lineterm="",
        )
    )


def load_selected_records(path: Path, ids: Sequence[int]) -> list[dict[str, Any]]:
    wanted = {int(value) for value in ids}
    by_idx = {int(row["idx"]): row for row in read_jsonl(path)}
    missing = sorted(wanted - set(by_idx))
    if missing:
        raise ValueError(f"Missing selected record IDs in {path.name}: {missing[:10]}")
    return [by_idx[int(value)] for value in ids]


def eligible_candidates(query: dict[str, Any], candidates: Sequence[dict[str, Any]], threshold: float) -> list[int]:
    query_project = str(query.get("project") or "")
    query_cves = record_cves(query)
    query_tokens = code_token_set(query)
    eligible: list[int] = []
    for position, candidate in enumerate(candidates):
        if str(candidate.get("project") or "") == query_project:
            continue
        if query_cves & record_cves(candidate):
            continue
        if token_jaccard(query_tokens, code_token_set(candidate)) >= threshold:
            continue
        eligible.append(position)
    return eligible


def output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir)


def frozen_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(Path(args.holdout_dir, "manifest.json").read_text(encoding="utf-8"))
    queries = load_selected_records(Path(args.test_path), manifest["test_ids"])
    candidates = load_selected_records(Path(args.train_path), manifest["train_ids"])
    if len(queries) != 50 or len(candidates) != 500:
        raise ValueError(f"Expected frozen 50/500 holdout, got {len(queries)}/{len(candidates)}")
    return queries, candidates, manifest


def make_pair(method: str, query: dict[str, Any], candidate: dict[str, Any], score: float | None, eligible_count: int) -> dict[str, Any]:
    audit_id = stable_hash("independent-code-baseline", query["idx"], candidate["idx"])[:24]
    return {
        "retrieval_id": stable_hash("independent-code-baseline", method, query["idx"])[:24],
        "audit_id": audit_id,
        "method": method,
        "query_idx": int(query["idx"]),
        "candidate_idx": int(candidate["idx"]),
        "rank": 1,
        "score": score,
        "eligible_candidate_count": eligible_count,
        "query": {
            "idx": int(query["idx"]),
            "project": query.get("project"),
            "cve": query.get("cve"),
            "cwe": query.get("cwe"),
            "cve_desc": query.get("cve_desc"),
            "func_vuln": query.get("func_vuln"),
            "repair_diff": repair_diff(query),
        },
        "candidate": {
            "idx": int(candidate["idx"]),
            "project": candidate.get("project"),
            "cve": candidate.get("cve"),
            "cwe": candidate.get("cwe"),
            "cve_desc": candidate.get("cve_desc"),
            "func_vuln": candidate.get("func_vuln"),
            "repair_diff": repair_diff(candidate),
        },
    }


def retrieve(args: argparse.Namespace) -> None:
    out = output_dir(args)
    queries, candidates, holdout = frozen_records(args)
    eligibility = [eligible_candidates(query, candidates, args.clone_threshold) for query in queries]
    if any(not rows for rows in eligibility):
        raise RuntimeError("At least one query has no eligible candidate")

    random_pairs: list[dict[str, Any]] = []
    for query, eligible in zip(queries, eligibility):
        position = min(eligible, key=lambda value: (stable_hash(args.seed, "random", query["idx"], candidates[value]["idx"]), value))
        random_pairs.append(make_pair("random", query, candidates[position], None, len(eligible)))

    bm25 = CodeBm25([str(row.get("func_vuln") or "") for row in candidates])
    bm25_pairs: list[dict[str, Any]] = []
    for query, eligible in zip(queries, eligibility):
        best_position, score = max(
            ((position, bm25_score) for position, bm25_score in enumerate(bm25_scores(bm25, str(query.get("func_vuln") or ""))) if position in set(eligible)),
            key=lambda item: (item[1], -item[0]),
        )
        bm25_pairs.append(make_pair("bm25_code", query, candidates[best_position], score, len(eligible)))

    device = resolve_device(args.device)
    tokenizer, model = load_embedding_model(Path(args.embedding_model_path), device)
    try:
        candidate_embeddings = encode_in_batches(tokenizer, model, [str(row.get("func_vuln") or "") for row in candidates], args, device)
        query_embeddings = encode_in_batches(tokenizer, model, [str(row.get("func_vuln") or "") for row in queries], args, device)
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    candidate_matrix = torch.tensor(candidate_embeddings, dtype=torch.float32)
    dense_pairs: list[dict[str, Any]] = []
    for query, vector, eligible in zip(queries, query_embeddings, eligibility):
        scores = candidate_matrix @ torch.tensor(vector, dtype=torch.float32)
        best_position = max(eligible, key=lambda position: (float(scores[position]), -position))
        dense_pairs.append(make_pair("dense_code", query, candidates[best_position], float(scores[best_position]), len(eligible)))

    all_pairs = [*random_pairs, *bm25_pairs, *dense_pairs]
    write_jsonl(out / "retrieval_pairs.jsonl", all_pairs)
    write_json(
        out / "retrieval_manifest.json",
        {
            "experiment": "independent_holdout_code_retrieval_baselines",
            "holdout_source": str(Path(args.holdout_dir).resolve()),
            "query_count": len(queries),
            "candidate_count": len(candidates),
            "methods": list(METHODS),
            "top_k": 1,
            "query_input": "complete vulnerable function source",
            "candidate_input": "complete vulnerable function source",
            "candidate_exclusions": {
                "same_project": True,
                "same_cve": True,
                "token_jaccard_gte": args.clone_threshold,
            },
            "random_seed": args.seed,
            "embedding_model_path": str(Path(args.embedding_model_path).resolve()),
            "embedding_max_length": args.embedding_max_length,
            "device": device,
            "holdout_test_ids": holdout["test_ids"],
            "holdout_train_ids": holdout["train_ids"],
        },
    )
    write_json(
        out / "retrieval_summary.json",
        {
            "pair_count": len(all_pairs),
            "eligible_candidate_count": {
                "min": min(len(value) for value in eligibility),
                "max": max(len(value) for value in eligibility),
                "mean": sum(len(value) for value in eligibility) / len(eligibility),
            },
            "per_method": {method: sum(row["method"] == method for row in all_pairs) for method in METHODS},
        },
    )
    print(json.dumps({"status": "retrieval_complete", "pairs": len(all_pairs), "device": device}))


def bm25_scores(index: CodeBm25, query: str) -> list[float]:
    """Score all documents using the exact BM25 definition of the old table."""
    from collections import Counter

    from api_pipeline.run_code_retrieval_comparison import code_terms

    query_counts = Counter(code_terms(query))
    scores = [0.0] * len(index.doc_terms)
    for term, query_frequency in query_counts.items():
        inverse_frequency = index.idf.get(term)
        if inverse_frequency is None:
            continue
        for position, term_counts in enumerate(index.doc_terms):
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + index.k1 * (1.0 - index.b + index.b * index.lengths[position] / max(index.average_length, 1.0))
            scores[position] += query_frequency * inverse_frequency * frequency * (index.k1 + 1.0) / denominator
    return scores


def encode_in_batches(tokenizer: Any, model: Any, texts: Sequence[str], args: argparse.Namespace, device: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), args.embedding_batch_size):
        vectors.extend(encode_texts(tokenizer, model, texts[start:start + args.embedding_batch_size], args.embedding_max_length, device))
        print(json.dumps({"stage": "dense_encode", "completed": min(start + args.embedding_batch_size, len(texts)), "total": len(texts)}), flush=True)
    return vectors


def latest_success(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    return {str(row["batch_key"]): row for row in rows if row.get("status") == "success"}


def audit_messages(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    visible = [
        {
            "audit_id": row["audit_id"],
            "target": {
                "cwe": row["query"].get("cwe"),
                "cve_description": row["query"].get("cve_desc"),
                "repair_diff": row["query"]["repair_diff"],
            },
            "candidate": {
                "cwe": row["candidate"].get("cwe"),
                "cve_description": row["candidate"].get("cve_desc"),
                "repair_diff": row["candidate"]["repair_diff"],
            },
        }
        for row in batch
    ]
    return [
        {"role": "system", "content": STRICT_REPAIR_AUDIT_SYSTEM},
        {"role": "user", "content": json.dumps({"items": visible}, ensure_ascii=False)},
    ]


def validate_audit(raw: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = extract_json_object(raw)
    judgments = value.get("judgments")
    expected = {row["audit_id"] for row in batch}
    if not isinstance(judgments, list) or len(judgments) != len(batch):
        raise ValueError("audit must return one judgment per input")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for judgment in judgments:
        audit_id = judgment.get("audit_id") if isinstance(judgment, dict) else None
        if audit_id not in expected or audit_id in seen:
            raise ValueError("audit returned an unknown or duplicate audit_id")
        if not isinstance(judgment.get("is_match"), bool) or not isinstance(judgment.get("reason"), str) or not judgment["reason"].strip():
            raise ValueError("audit judgment must contain is_match and reason")
        seen.add(audit_id)
        result.append({"audit_id": audit_id, "is_match": judgment["is_match"], "reason": judgment["reason"]})
    return result


def validate_transfer_audit(raw: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = extract_json_object(raw)
    judgments = value.get("judgments")
    expected = {row["audit_id"] for row in batch}
    if not isinstance(judgments, list) or len(judgments) != len(batch):
        raise ValueError("transfer audit must return one judgment per input")
    required = ("root_cause_alignment", "trigger_invariant_alignment", "transferable_principle", "reason")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for judgment in judgments:
        audit_id = judgment.get("audit_id") if isinstance(judgment, dict) else None
        if audit_id not in expected or audit_id in seen:
            raise ValueError("transfer audit returned an unknown or duplicate audit_id")
        if judgment.get("label") not in {0, 1, 2} or any(not isinstance(judgment.get(field), str) or not judgment[field].strip() for field in required):
            raise ValueError("transfer audit judgment is incomplete")
        seen.add(audit_id)
        result.append({"audit_id": audit_id, **{field: judgment[field] for field in (*required, "label")}})
    return result


def request_audit(batch_key: str, batch: list[dict[str, Any]], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    messages = audit_messages(batch)
    result: dict[str, Any] = {
        "batch_key": batch_key,
        "audit_ids": [row["audit_id"] for row in batch],
        "input_hash": stable_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "model_name": args.audit_model_name,
        "thinking_type": "disabled",
        "temperature": 0.0,
        "top_p": 1.0,
        "attempts": [],
    }
    client = build_client(args.api_base, api_key).with_options(max_retries=0)
    try:
        for attempt in range(1, args.retries + 1):
            try:
                raw, usage = call_chat_completion_with_usage(
                    client, args.audit_model_name, messages, 0.0, 1.0, args.audit_max_new_tokens,
                    args.timeout, "high", "disabled",
                )
                judgments = validate_audit(raw, batch)
                result["attempts"].append({"attempt": attempt, "status": "success", "raw_output": raw, "usage": usage})
                return {**result, "status": "success", "judgments": judgments}
            except Exception as exc:
                result["attempts"].append({"attempt": attempt, "status": "failed", "error": str(exc).replace(api_key, "[REDACTED]")})
                if attempt < args.retries:
                    time.sleep(min(12.0, 2.0 ** attempt))
        return {**result, "status": "failed", "error": result["attempts"][-1]["error"]}
    finally:
        client.close()


def request_transfer_audit(batch_key: str, batch: list[dict[str, Any]], args: argparse.Namespace, api_key: str) -> dict[str, Any]:
    messages = audit_messages(batch)
    messages[0] = {"role": "system", "content": TRANSFER_REPAIR_AUDIT_SYSTEM}
    result: dict[str, Any] = {
        "batch_key": batch_key,
        "audit_ids": [row["audit_id"] for row in batch],
        "input_hash": stable_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        "model_name": args.audit_model_name,
        "thinking_type": "disabled",
        "temperature": 0.0,
        "top_p": 1.0,
        "attempts": [],
    }
    client = build_client(args.api_base, api_key).with_options(max_retries=0)
    try:
        for attempt in range(1, args.retries + 1):
            try:
                raw, usage = call_chat_completion_with_usage(
                    client, args.audit_model_name, messages, 0.0, 1.0, args.audit_max_new_tokens,
                    args.timeout, "high", "disabled",
                )
                judgments = validate_transfer_audit(raw, batch)
                result["attempts"].append({"attempt": attempt, "status": "success", "raw_output": raw, "usage": usage})
                return {**result, "status": "success", "judgments": judgments}
            except Exception as exc:
                result["attempts"].append({"attempt": attempt, "status": "failed", "error": str(exc).replace(api_key, "[REDACTED]")})
                if attempt < args.retries:
                    time.sleep(min(12.0, 2.0 ** attempt))
        return {**result, "status": "failed", "error": result["attempts"][-1]["error"]}
    finally:
        client.close()


def audit(args: argparse.Namespace) -> None:
    out = output_dir(args)
    pairs = read_jsonl(out / "retrieval_pairs.jsonl")
    unique = {row["audit_id"]: row for row in pairs}
    batches = [list(sorted(unique.values(), key=lambda row: row["audit_id"]))[start:start + args.audit_batch_size]
               for start in range(0, len(unique), args.audit_batch_size)]
    target = out / "audit_assisted.jsonl"
    cached = latest_success(target)
    tasks = []
    for number, batch in enumerate(batches):
        batch_key = f"audit:{number}:{stable_hash(*(row['audit_id'] for row in batch))[:12]}"
        if batch_key not in cached:
            tasks.append((batch_key, batch))
    api_key = load_api_key(None, args.api_key_env)
    print(json.dumps({"stage": "audit", "unique_pairs": len(unique), "cached": len(cached), "pending": len(tasks)}), flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.audit_workers, max(1, len(tasks)))) as executor:
        futures = {executor.submit(request_audit, key, batch, args, api_key): key for key, batch in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(target, row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "audit", "completed": completed, "remaining": len(tasks) - completed, "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"Audit failed for {failures} batches; rerun to resume")
    export_ai_labels(args)


def audit_transfer(args: argparse.Namespace) -> None:
    out = output_dir(args)
    pairs = read_jsonl(out / "retrieval_pairs.jsonl")
    unique = {row["audit_id"]: row for row in pairs}
    ordered = list(sorted(unique.values(), key=lambda row: row["audit_id"]))
    batches = [ordered[start:start + args.audit_batch_size] for start in range(0, len(ordered), args.audit_batch_size)]
    target = out / "audit_transfer_assisted.jsonl"
    cached = latest_success(target)
    tasks = []
    for number, batch in enumerate(batches):
        batch_key = f"transfer:{number}:{stable_hash(*(row['audit_id'] for row in batch))[:12]}"
        if batch_key not in cached:
            tasks.append((batch_key, batch))
    api_key = load_api_key(None, args.api_key_env)
    print(json.dumps({"stage": "transfer_audit", "unique_pairs": len(unique), "cached": len(cached), "pending": len(tasks)}), flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.audit_workers, max(1, len(tasks)))) as executor:
        futures = {executor.submit(request_transfer_audit, key, batch, args, api_key): key for key, batch in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(target, row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "transfer_audit", "completed": completed, "remaining": len(tasks) - completed, "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"Transfer audit failed for {failures} batches; rerun to resume")
    export_transfer_labels(args)


def export_transfer_labels(args: argparse.Namespace) -> None:
    out = output_dir(args)
    expected = {row["audit_id"] for row in read_jsonl(out / "retrieval_pairs.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    for batch in latest_success(out / "audit_transfer_assisted.jsonl").values():
        for judgment in batch["judgments"]:
            labels[judgment["audit_id"]] = {**judgment, "review_status": "ai_assisted_unreviewed"}
    if set(labels) != expected:
        raise RuntimeError(f"Transfer labels incomplete: missing={len(expected - set(labels))}, extra={len(set(labels) - expected)}")
    write_jsonl(out / "audit_transfer_labels.jsonl", [labels[key] for key in sorted(labels)])
    print(json.dumps({"stage": "transfer_audit_export", "labels": len(labels),
                      "strict": sum(row["label"] == 2 for row in labels.values()),
                      "transfer": sum(row["label"] in {1, 2} for row in labels.values())}))


def export_ai_labels(args: argparse.Namespace) -> None:
    out = output_dir(args)
    expected = {row["audit_id"] for row in read_jsonl(out / "retrieval_pairs.jsonl")}
    labels: dict[str, dict[str, Any]] = {}
    for batch in latest_success(out / "audit_assisted.jsonl").values():
        for judgment in batch["judgments"]:
            labels[judgment["audit_id"]] = {**judgment, "review_status": "ai_assisted_unreviewed"}
    if set(labels) != expected:
        raise RuntimeError(f"Audit labels incomplete: missing={len(expected - set(labels))}, extra={len(set(labels) - expected)}")
    write_jsonl(out / "audit_labels_ai.jsonl", [labels[key] for key in sorted(labels)])
    positives = [row for row in read_jsonl(out / "retrieval_pairs.jsonl") if labels[row["audit_id"]]["is_match"]]
    write_jsonl(out / "audit_positive_review_queue.jsonl", positives)
    print(json.dumps({"stage": "audit_export", "labels": len(labels), "ai_positive_pairs": len(positives)}))


def apply_manual_review(args: argparse.Namespace) -> None:
    """Overlay explicit human decisions on the full assisted-label set."""
    out = output_dir(args)
    assisted = {row["audit_id"]: row for row in read_jsonl(out / "audit_labels_ai.jsonl")}
    decisions = {row["audit_id"]: row for row in read_jsonl(Path(args.manual_decisions))}
    unknown = sorted(set(decisions) - set(assisted))
    if unknown:
        raise ValueError(f"Manual review contains unknown audit IDs: {unknown[:5]}")
    for audit_id, decision in decisions.items():
        if not isinstance(decision.get("is_match"), bool) or not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
            raise ValueError(f"Manual review decision is incomplete: {audit_id}")
        assisted[audit_id] = {
            "audit_id": audit_id,
            "is_match": decision["is_match"],
            "reason": decision["reason"],
            "review_status": "manual_reviewed",
        }
    write_jsonl(out / "audit_labels_reviewed.jsonl", [assisted[key] for key in sorted(assisted)])
    print(json.dumps({"stage": "manual_review", "reviewed": len(decisions), "labels": len(assisted)}))


def summarize(args: argparse.Namespace) -> None:
    out = output_dir(args)
    pairs = read_jsonl(out / "retrieval_pairs.jsonl")
    labels_path = Path(args.labels) if args.labels else out / "audit_labels_ai.jsonl"
    labels = {row["audit_id"]: row for row in read_jsonl(labels_path)}
    if {row["audit_id"] for row in pairs} != set(labels):
        raise RuntimeError("Labels do not exactly cover retrieval pairs")
    result: dict[str, Any] = {
        "metric": "strict mechanism match: same immediate failed safety condition, evaluated from target/candidate repair diffs",
        "labels_path": str(labels_path.resolve()),
        "manual_reviewed": any(row.get("review_status") == "manual_reviewed" for row in labels.values()),
        "methods": {},
    }
    for method in METHODS:
        rows = [row for row in pairs if row["method"] == method]
        matches = sum(bool(labels[row["audit_id"]]["is_match"]) for row in rows)
        result["methods"][method] = {"strict_match_count": matches, "n": len(rows), "strict_match_rate": matches / len(rows) if rows else None}
    write_json(out / "summary.json", result)
    lines = [
        "# Independent Holdout Code Retrieval Baselines",
        "",
        "| Retriever | Strict Mech. Match |",
        "| --- | ---: |",
    ]
    labels_for_table = {"random": "Random", "bm25_code": "BM25-Code", "dense_code": "Dense-Code"}
    for method in METHODS:
        data = result["methods"][method]
        lines.append(f"| {labels_for_table[method]} | {data['strict_match_count']}/{data['n']} ({data['strict_match_rate']:.1%}) |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def summarize_transfer(args: argparse.Namespace) -> None:
    out = output_dir(args)
    pairs = read_jsonl(out / "retrieval_pairs.jsonl")
    labels = {row["audit_id"]: row for row in read_jsonl(out / "audit_transfer_labels.jsonl")}
    if {row["audit_id"] for row in pairs} != set(labels):
        raise RuntimeError("Transfer labels do not exactly cover retrieval pairs")
    result: dict[str, Any] = {
        "metric": "transferable mechanism match: label 1 or 2, evaluated from target/candidate repair diffs",
        "strict_metric": "strict mechanism match: label 2",
        "manual_review_scope": "The existing binary strict audit retains its separate manual review. Transfer labels follow the same three-level assisted-audit protocol as the taxonomy experiment.",
        "methods": {},
    }
    for method in METHODS:
        rows = [row for row in pairs if row["method"] == method]
        strict = sum(labels[row["audit_id"]]["label"] == 2 for row in rows)
        transfer = sum(labels[row["audit_id"]]["label"] in {1, 2} for row in rows)
        result["methods"][method] = {
            "strict_match_count": strict,
            "transferable_match_count": transfer,
            "n": len(rows),
            "strict_match_rate": strict / len(rows) if rows else None,
            "transferable_match_rate": transfer / len(rows) if rows else None,
        }
    write_json(out / "transfer_summary.json", result)
    lines = [
        "# Independent Holdout Transferable Mechanism Match",
        "",
        "| Retriever | Transferable Mech. Match |",
        "| --- | ---: |",
    ]
    labels_for_table = {"random": "Random", "bm25_code": "BM25-Code", "dense_code": "Dense-Code"}
    for method in METHODS:
        data = result["methods"][method]
        lines.append(f"| {labels_for_table[method]} | {data['transferable_match_count']}/{data['n']} ({data['transferable_match_rate']:.1%}) |")
    (out / "TRANSFER_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("retrieve", "audit", "audit-transfer", "export-ai-labels", "export-transfer-labels", "apply-manual-review", "summarize", "summarize-transfer"))
    value.add_argument("--output_dir", default="data/taxonomy_search_v14_independent_code_baselines")
    value.add_argument("--holdout_dir", default="data/taxonomy_search_v14_independent_confirmation")
    value.add_argument("--train_path", default="data/raw/primevul_train_merged.jsonl")
    value.add_argument("--test_path", default="data/raw/primevul_test_merged.jsonl")
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--clone_threshold", type=float, default=0.8)
    value.add_argument("--embedding_model_path", default="models/embedding/encoder")
    value.add_argument("--embedding_max_length", type=int, default=4096)
    value.add_argument("--embedding_batch_size", type=int, default=8)
    value.add_argument("--device", default="auto")
    value.add_argument("--api_base", default="http://localhost:8000/v1")
    value.add_argument("--api_key_env", default="LLM_API_KEY")
    value.add_argument("--audit_model_name", default="your-model-id")
    value.add_argument("--audit_batch_size", type=int, default=4)
    value.add_argument("--audit_workers", type=int, default=4)
    value.add_argument("--audit_max_new_tokens", type=int, default=768)
    value.add_argument("--timeout", type=int, default=180)
    value.add_argument("--retries", type=int, default=3)
    value.add_argument("--labels", default="")
    value.add_argument("--manual_decisions", default="")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "retrieve":
        retrieve(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "audit-transfer":
        audit_transfer(args)
    elif args.command == "export-ai-labels":
        export_ai_labels(args)
    elif args.command == "export-transfer-labels":
        export_transfer_labels(args)
    elif args.command == "apply-manual-review":
        if not args.manual_decisions:
            raise ValueError("--manual_decisions is required")
        apply_manual_review(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        summarize_transfer(args)


if __name__ == "__main__":
    main()
