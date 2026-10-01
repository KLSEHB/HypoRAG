"""Preference-data helpers for the current M/R/E reranker interface."""

import math
import re
from typing import Any, Iterable, Mapping, Sequence


def pair_id(query_id: str, candidate_idx: int) -> str:
    return f"{query_id}::{candidate_idx}"


def reranker_query(point: Mapping[str, Any]) -> str:
    return "\n".join([
        f"retrieval_summary: {point['mechanism_claim']}",
        f"needed_example: {point['repair_sought']}",
        f"evidence_hint: {point['evidence_to_check']}",
    ])


def reranker_candidate(fields: Mapping[str, Any]) -> str:
    return "\n".join([
        f"mechanism_summary: {fields['mechanism_observed']}",
        f"solution_summary: {fields['repair_applied']}",
        f"evidence_summary: {fields['evidence_decisive']}",
    ])


def latest_successful_labels(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest = {}
    for row in rows:
        if row.get("status") == "success":
            latest[str(row["pair_id"])] = row
    return latest


def parse_teacher_output(raw_text: str) -> dict[str, Any]:
    from api_pipeline.common import extract_json_object

    parsed = extract_json_object(raw_text)
    label = int(parsed["reference_value_label"])
    direction = str(parsed["helpfulness_type"]).strip().lower().replace("-", "_")
    reason = str(parsed["reason"]).strip()
    if label not in (0, 1, 2):
        raise ValueError(f"Invalid reference_value_label: {label}")
    if direction not in ("confirm", "rule_out", "both", "none"):
        raise ValueError(f"Invalid helpfulness_type: {direction}")
    if (label == 0) != (direction == "none"):
        raise ValueError("label 0 requires none; labels 1/2 require a direction")
    if not reason:
        raise ValueError("reason must not be empty")
    boundary_pattern = re.compile(
        r"\b(proves?)\s+(that\s+)?(the\s+)?(target|query)(\s+code)?\s+"
        r"(is|has)\s+(vulnerable|safe|a vulnerability)"
        r"|\b(definitely|clearly)\s+needs?\s+(the\s+)?(fix|repair)"
        r"|\b(target|query)\s+(is|has)\s+(vulnerable|safe|a vulnerability)",
        flags=re.IGNORECASE,
    )
    if boundary_pattern.search(reason):
        raise ValueError("reason crosses the reference-utility information boundary")
    return {
        "reference_value_label": label,
        "helpfulness_type": direction,
        "reason": reason,
    }


def ndcg(relevances: Sequence[int], k: int) -> float:
    gains = sum(
        (2 ** int(value) - 1) / math.log2(rank + 2)
        for rank, value in enumerate(relevances[:k])
    )
    ideal = sorted((int(value) for value in relevances), reverse=True)
    ideal_gains = sum(
        (2 ** value - 1) / math.log2(rank + 2)
        for rank, value in enumerate(ideal[:k])
    )
    return gains / ideal_gains if ideal_gains else 0.0


def build_pairwise_rows(
    pools: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    latest = latest_successful_labels(labels)
    output: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    record_splits: dict[int, str] = {}
    for pool in pools:
        query = pool["query"]
        query_id = str(query["query_id"])
        split = str(query["dataset_split"])
        if split not in output:
            raise ValueError(f"Unknown dataset split: {split}")
        record_idx = int(query["idx"])
        previous = record_splits.setdefault(record_idx, split)
        if previous != split:
            raise ValueError(f"Repair record {record_idx} appears in both {previous} and {split}")

        candidates = []
        seen_indices = set()
        for candidate in pool.get("candidates", []):
            candidate_idx = int(candidate["candidate_idx"])
            if candidate_idx in seen_indices:
                raise ValueError(f"Duplicate candidate {candidate_idx} in {query_id}")
            seen_indices.add(candidate_idx)
            label = latest.get(pair_id(query_id, candidate_idx))
            if label is not None:
                value = int(label["reference_value_label"])
                if value not in (0, 1, 2):
                    raise ValueError(f"Invalid utility label: {value}")
                candidates.append((candidate, label, value))

        for positive, positive_label, high in candidates:
            for negative, negative_label, low in candidates:
                if high <= low:
                    continue
                output[split].append({
                    "query_id": query_id,
                    "dataset_split": split,
                    "query": reranker_query(query),
                    "positive": reranker_candidate(positive["candidate_mre"]),
                    "negative": reranker_candidate(negative["candidate_mre"]),
                    "positive_candidate_idx": int(positive["candidate_idx"]),
                    "negative_candidate_idx": int(negative["candidate_idx"]),
                    "positive_label": high,
                    "negative_label": low,
                    "weight": high - low,
                    "positive_helpfulness_type": positive_label.get("helpfulness_type", "none"),
                    "negative_helpfulness_type": negative_label.get("helpfulness_type", "none"),
                })
    return output
