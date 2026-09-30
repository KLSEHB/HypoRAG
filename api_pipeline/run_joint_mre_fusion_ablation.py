#!/usr/bin/env python3
"""Compare two fusion rules over the frozen joint-M/R/E route rankings.

The input experiment has one shared taxonomy family per hypothesis and three
family-filtered dense top-3 lists.  This ablation changes only how those lists
are fused; it reuses the same LLM-assisted labels for every selected pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.retrieval_core import (  # noqa: E402
    load_reranker_model,
    rerank_candidates,
    resolve_device,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def route_rankings(input_dir: Path, routes: tuple[str, ...]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for route in routes:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(input_dir / "retrieval" / route / "retrieval_pairs.jsonl"):
            grouped[str(row["hypothesis_id"])].append(row)
        for hypothesis_id, rows in grouped.items():
            rows.sort(key=lambda row: (int(row["rank"]), int(row["candidate_idx"])))
            if len(rows) != 3:
                raise ValueError(f"{route}/{hypothesis_id} has {len(rows)}, expected exactly three route hits")
        result[route] = dict(grouped)
    expected = set(result[routes[0]])
    if any(set(result[route]) != expected for route in routes[1:]):
        raise ValueError("the three route rankings do not contain the same hypotheses")
    return result


def candidate_payload(rows: list[dict[str, Any]], method: str, selection_reason: str, score: float | None) -> dict[str, Any]:
    first = rows[0]
    route_hits = [
        {
            "route": row["route"],
            "rank": int(row["rank"]),
            "embedding_score": float(row["score"]),
        }
        for row in sorted(rows, key=lambda row: (str(row["route"]), int(row["rank"])))
    ]
    return {
        "fusion_method": method,
        "selection_reason": selection_reason,
        "fusion_score": score,
        "hypothesis_id": first["hypothesis_id"],
        "query_idx": int(first["query_idx"]),
        "candidate_idx": int(first["candidate_idx"]),
        "candidate_signature_id": first["candidate_signature_id"],
        "family": first["family"],
        "candidate_family": first["candidate_family"],
        "family_filter_fallback": bool(first.get("family_filter_fallback", False)),
        "pool_size": int(first["pool_size"]),
        "eligible_global": int(first["eligible_global"]),
        "audit_id": first["audit_id"],
        "hypothesis": first["hypothesis"],
        "candidate_signature": first["candidate_signature"],
        "route_hits": route_hits,
    }


def rrf_top3(
    rankings: dict[str, dict[str, list[dict[str, Any]]]], routes: tuple[str, ...], hypothesis_id: str, rrf_k: int
) -> list[dict[str, Any]]:
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        for row in rankings[route][hypothesis_id]:
            by_candidate[int(row["candidate_idx"])].append(row)
    scored = []
    for candidate_idx, rows in by_candidate.items():
        score = sum(1.0 / (rrf_k + int(row["rank"])) for row in rows)
        scored.append((candidate_idx, score, rows))
    scored.sort(key=lambda value: (-value[1], value[0]))
    if len(scored) < 3:
        raise ValueError(f"RRF has fewer than three unique candidates for {hypothesis_id}")
    return [candidate_payload(rows, "rrf60_top3", "rrf", score) for _, score, rows in scored[:3]]


def route_top1_merge(
    rankings: dict[str, dict[str, list[dict[str, Any]]]], routes: tuple[str, ...], hypothesis_id: str
) -> list[dict[str, Any]]:
    """Return the exact deduplicated union of one top-1 hit from each route."""
    selected: list[dict[str, Any]] = []
    chosen: set[int] = set()
    for route in routes:
        row = rankings[route][hypothesis_id][0]
        if int(row["candidate_idx"]) in chosen:
            continue
        selected.append(candidate_payload([row], "route_top1_merge", "route_top1", None))
        chosen.add(int(row["candidate_idx"]))
    return selected


def route_top3_union(
    rankings: dict[str, dict[str, list[dict[str, Any]]]], routes: tuple[str, ...], hypothesis_id: str
) -> list[dict[str, Any]]:
    """Return the exact deduplicated union of all three route top-3 lists."""
    by_candidate: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for route in routes:
        for row in rankings[route][hypothesis_id]:
            by_candidate[int(row["candidate_idx"])].append(row)
    return [
        candidate_payload(rows, "route_top3_union", "route_top3_union", None)
        for _, rows in sorted(by_candidate.items())
    ]


def latest_joint_outputs(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("status") == "success":
            latest[str(row["signature_id"])] = row["parsed"]
    return latest


def reranker_query(fields: dict[str, Any]) -> str:
    return "\n".join([
        f"retrieval_summary: {fields['mechanism_claim']}",
        f"needed_example: {fields['repair_sought']}",
        f"evidence_hint: {fields['evidence_to_check']}",
    ])


def reranker_candidate(fields: dict[str, Any]) -> str:
    return "\n".join([
        f"mechanism_summary: {fields['mechanism_observed']}",
        f"solution_summary: {fields['repair_applied']}",
        f"evidence_summary: {fields['evidence_decisive']}",
    ])


def base_rerank_union_top3(
    union_rows: list[dict[str, Any]], input_dir: Path, model_path: Path,
    device_name: str, max_length: int, batch_size: int,
) -> list[dict[str, Any]]:
    hypotheses = latest_joint_outputs(input_dir / "generation" / "joint_hypothesis.jsonl")
    knowledge = latest_joint_outputs(input_dir / "generation" / "joint_knowledge.jsonl")
    by_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in union_rows:
        by_hypothesis[row["hypothesis_id"]].append(row)
    device = resolve_device(device_name)
    tokenizer, model = load_reranker_model(model_path, device)
    try:
        result: list[dict[str, Any]] = []
        for hypothesis_id in sorted(by_hypothesis):
            candidates = by_hypothesis[hypothesis_id]
            query_fields = hypotheses[hypothesis_id]
            scores = rerank_candidates(
                tokenizer, model, reranker_query(query_fields),
                [reranker_candidate(knowledge[row["candidate_signature_id"]]) for row in candidates],
                max_length=max_length, batch_size=batch_size, device=device,
            )
            ranked = sorted(
                zip(candidates, scores), key=lambda value: (-float(value[1]), int(value[0]["candidate_idx"]))
            )[:3]
            if len(ranked) != 3:
                raise ValueError(f"{hypothesis_id} has fewer than three union candidates")
            for rank, (row, score) in enumerate(ranked, 1):
                result.append({
                    **row,
                    "fusion_method": "base_reranker_union_top3",
                    "rerank_rank": rank,
                    "reranker_score": float(score),
                })
        return result
    finally:
        del model


def summarize(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]], expected_per_hypothesis: int | None) -> dict[str, Any]:
    if len(rows) != len({(row["hypothesis_id"], row["candidate_idx"]) for row in rows}):
        raise ValueError("fusion output contains a duplicate hypothesis-candidate pair")
    if any(row["audit_id"] not in labels for row in rows):
        missing = sum(row["audit_id"] not in labels for row in rows)
        raise ValueError(f"{missing} selected pairs lack a cached audit label")
    enriched = [{**row, "label": int(labels[row["audit_id"]]["label"])} for row in rows]
    per_hypothesis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        per_hypothesis[row["hypothesis_id"]].append(row)
        per_family[row["family"]].append(row)
    if expected_per_hypothesis is not None and any(len(value) != expected_per_hypothesis for value in per_hypothesis.values()):
        raise ValueError(f"each hypothesis must contribute exactly {expected_per_hypothesis} fusion pairs")
    label_counts = Counter(row["label"] for row in enriched)
    family_summary = []
    for family, family_rows in sorted(per_family.items(), key=lambda item: (-len(item[1]), item[0])):
        family_hypotheses = {row["hypothesis_id"] for row in family_rows}
        strict = sum(row["label"] == 2 for row in family_rows)
        transferable = sum(row["label"] >= 1 for row in family_rows)
        family_summary.append({
            "family": family,
            "hypothesis_count": len(family_hypotheses),
            "pair_count": len(family_rows),
            "strict_match_count": strict,
            "strict_match_rate": strict / len(family_rows),
            "transferable_match_count": transferable,
            "transferable_match_rate": transferable / len(family_rows),
        })
    return {
        "pair_count": len(enriched),
        "hypothesis_count": len(per_hypothesis),
        "function_count": len({row["query_idx"] for row in enriched}),
        "strict_match_count": label_counts[2],
        "strict_match_rate": label_counts[2] / len(enriched),
        "transferable_match_count": label_counts[1] + label_counts[2],
        "transferable_match_rate": (label_counts[1] + label_counts[2]) / len(enriched),
        "label_counts": {str(label): label_counts[label] for label in (0, 1, 2)},
        "strict_hypothesis_success": sum(any(row["label"] == 2 for row in values) for values in per_hypothesis.values()) / len(per_hypothesis),
        "transferable_hypothesis_success": sum(any(row["label"] >= 1 for row in values) for values in per_hypothesis.values()) / len(per_hypothesis),
        "global_fallback_pair_count": sum(row["family_filter_fallback"] for row in enriched),
        "family_breakdown": family_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "taxonomy_other_absorbs_lifetime_error_protocol_15_joint_mre_thinking_full_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "taxonomy_other_absorbs_lifetime_error_protocol_15_joint_mre_thinking_full_v1" / "fusion_rrf60_vs_route_top1_merge_v1")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-model-path", type=Path, default=ROOT / "models" / "reranker" / "base" / "cross-encoder")
    parser.add_argument("--reranker-label", default="base_bge_reranker_v2_m3")
    parser.add_argument("--reranker-device", default="auto")
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.rrf_k < 0:
        parser.error("--rrf-k must be non-negative")

    routes = ("mechanism", "repair", "evidence")
    rankings = route_rankings(args.input_dir, routes)
    labels = {row["audit_id"]: row for row in read_jsonl(args.input_dir / "audit" / "audit_labels_assisted.jsonl")}
    hypothesis_ids = sorted(rankings[routes[0]])
    rrf_rows = [row for hypothesis_id in hypothesis_ids for row in rrf_top3(rankings, routes, hypothesis_id, args.rrf_k)]
    top1_rows = [row for hypothesis_id in hypothesis_ids for row in route_top1_merge(rankings, routes, hypothesis_id)]
    top3_union_rows = [row for hypothesis_id in hypothesis_ids for row in route_top3_union(rankings, routes, hypothesis_id)]
    base_reranker_rows = base_rerank_union_top3(
        top3_union_rows, args.input_dir, args.reranker_model_path, args.reranker_device,
        args.reranker_max_length, args.reranker_batch_size,
    )

    rrf_summary = summarize(rrf_rows, labels, expected_per_hypothesis=3)
    top1_summary = summarize(top1_rows, labels, expected_per_hypothesis=None)
    top3_union_summary = summarize(top3_union_rows, labels, expected_per_hypothesis=None)
    base_reranker_summary = summarize(base_reranker_rows, labels, expected_per_hypothesis=3)
    comparison = {
        "input_experiment": str(args.input_dir.resolve()),
        "shared_generation": "joint M/R/E fields plus one shared family; thinking enabled",
        "audit": "reuses the exact cached LLM-assisted three-level labels from the input experiment",
        "comparison_unit": "unique hypothesis-candidate pairs; RRF returns top-3 per hypothesis, while top-1 merge returns the exact deduplicated union without backfill",
        "rrf": {"rrf_k": args.rrf_k, "route_rank_depth": 3, **rrf_summary},
        "route_top1_merge": {
            "route_order": list(routes),
            "deduplication": "deduplicate candidate IDs within each hypothesis",
            "backfill": "none; duplicate top-1 candidate IDs are kept once and no later-ranked hit is substituted",
            **top1_summary,
        },
        "route_top3_union": {
            "deduplication": "deduplicate candidate IDs within each hypothesis across the three route top-3 lists",
            "selection": "retain every unique candidate in the union; no post-fusion truncation",
            **top3_union_summary,
        },
        f"{args.reranker_label}_after_route_top3_union": {
            "model_path": str(args.reranker_model_path.resolve()),
            "input_mapping": {
                "query": "mechanism_claim, repair_sought, evidence_to_check",
                "candidate": "mechanism_observed, repair_applied, evidence_decisive",
            },
            "max_length": args.reranker_max_length,
            "candidate_set": "deduplicated union of the three route top-3 lists per hypothesis",
            "selection": "reranker top-3 per hypothesis",
            **base_reranker_summary,
        },
    }
    write_jsonl(args.output_dir / "rrf60_top3_pairs.jsonl", rrf_rows)
    write_jsonl(args.output_dir / "route_top1_merge_pairs.jsonl", top1_rows)
    write_jsonl(args.output_dir / "route_top3_union_pairs.jsonl", top3_union_rows)
    write_jsonl(args.output_dir / "reranked_union_top3_pairs.jsonl", base_reranker_rows)
    write_json(args.output_dir / "summary.json", comparison)
    lines = [
        "# Joint M/R/E Fusion Ablation",
        "",
        "| Fusion | Unit | Strict Mech. Match | Transferable Match |",
        "| --- | --- | ---: | ---: |",
        f"| RRF-60 | Hypothesis-candidate top-3 | {rrf_summary['strict_match_count']}/{rrf_summary['pair_count']} ({rrf_summary['strict_match_rate']:.1%}) | {rrf_summary['transferable_match_count']}/{rrf_summary['pair_count']} ({rrf_summary['transferable_match_rate']:.1%}) |",
        f"| Per-route top-1 merge | Deduplicated hypothesis-candidate union | {top1_summary['strict_match_count']}/{top1_summary['pair_count']} ({top1_summary['strict_match_rate']:.1%}) | {top1_summary['transferable_match_count']}/{top1_summary['pair_count']} ({top1_summary['transferable_match_rate']:.1%}) |",
        f"| Per-route top-3 union | Deduplicated hypothesis-candidate union | {top3_union_summary['strict_match_count']}/{top3_union_summary['pair_count']} ({top3_union_summary['strict_match_rate']:.1%}) | {top3_union_summary['transferable_match_count']}/{top3_union_summary['pair_count']} ({top3_union_summary['transferable_match_rate']:.1%}) |",
        f"| {args.reranker_label} after route-top-3 union | Hypothesis-candidate top-3 | {base_reranker_summary['strict_match_count']}/267 ({base_reranker_summary['strict_match_rate']:.1%}) | {base_reranker_summary['transferable_match_count']}/267 ({base_reranker_summary['transferable_match_rate']:.1%}) |",
        "",
        "Duplicate top-1 candidate IDs are retained once; no top-2/top-3 backfill is used.",
        "All labels are reused from the same LLM-assisted three-level audit cache; no new audit calls were made.",
    ]
    (args.output_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
