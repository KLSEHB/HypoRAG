"""Build and Teacher-label preference pools for the frozen M/R/E pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.common import build_client, call_chat_completion_with_usage
from api_pipeline import run_formal_hyporag as formal
from api_pipeline.reranker_preference import latest_successful_labels, pair_id, parse_teacher_output
from api_pipeline.run_independent_code_retrieval_baselines import eligible_candidates
from api_pipeline.run_taxonomy_route_retrieval import ROUTES
from prompts.reranker_preference_labeling import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, build_messages


def stable_int(seed: int, *parts: Any) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def record_splits(indices: list[int], seed: int) -> dict[int, str]:
    shuffled = sorted(set(indices))
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.8)
    dev_end = train_end + int(len(shuffled) * 0.1)
    return {
        idx: "train" if pos < train_end else "dev" if pos < dev_end else "test"
        for pos, idx in enumerate(shuffled)
    }


def selection_score(query_id: str, seed: int) -> float:
    return stable_int(seed, query_id, "selection") / (2**64 - 1)


def teacher_prompt_hash() -> str:
    payload = SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return formal.read_jsonl(path)


def select_pool_candidates(
    query: dict[str, Any], merged: dict[int, list[dict[str, Any]]],
    knowledge: dict[int, dict[str, Any]], eligible_ids: list[int],
    seed: int, quotas: tuple[int, int, int], cross_family_negatives: int,
) -> list[dict[str, Any]]:
    strata: dict[str, list[int]] = defaultdict(list)
    for idx, hits in merged.items():
        best = min(int(hit["rank"]) for hit in hits)
        name = "rank_1_5" if best <= 5 else "rank_6_10" if best <= 10 else "rank_11_20"
        strata[name].append(idx)
    selected: dict[int, dict[str, Any]] = {}
    for name, quota in zip(("rank_1_5", "rank_6_10", "rank_11_20"), quotas):
        ids = sorted(strata[name])
        random.Random(stable_int(seed, query["query_id"], name)).shuffle(ids)
        for idx in ids[:quota]:
            selected[idx] = {"candidate_source": "retrieval", "route_hits": merged[idx]}

    source_idx = int(query["idx"])
    if source_idx in knowledge:
        if source_idx in selected:
            selected[source_idx]["candidate_source"] = "retrieval_same_source"
        else:
            selected[source_idx] = {"candidate_source": "same_source", "route_hits": merged.get(source_idx, [])}

    query_family = str(query["mechanism_family"])
    different = [
        idx for idx in eligible_ids
        if idx not in selected and knowledge[idx]["mre"]["mechanism_family"] != query_family
    ]
    random.Random(stable_int(seed, query["query_id"], "cross_family")).shuffle(different)
    for idx in different[:cross_family_negatives]:
        selected[idx] = {"candidate_source": "cross_family", "route_hits": []}

    candidates = []
    for idx, entry in selected.items():
        hits = entry["route_hits"]
        best_rank = min((int(hit["rank"]) for hit in hits), default=None)
        candidates.append({
            "candidate_idx": idx,
            "candidate_mre": knowledge[idx]["mre"],
            "candidate_source": entry["candidate_source"],
            "same_source": idx == source_idx,
            "best_route_rank": best_rank,
            "rank_stratum": (
                "injected" if best_rank is None else
                "rank_1_5" if best_rank <= 5 else
                "rank_6_10" if best_rank <= 10 else "rank_11_20"
            ),
            "route_hits": hits,
        })
    return sorted(candidates, key=lambda row: (
        row["candidate_source"] == "cross_family",
        row["best_route_rank"] if row["best_route_rank"] is not None else 10000,
        row["candidate_idx"],
    ))


def build_pools(args: argparse.Namespace) -> None:
    directory = args.formal_output_dir
    manifest = formal.read_json(directory / "experiment_manifest.json")
    if manifest["taxonomy"]["name"] != formal.DEFAULT_TAXONOMY:
        raise ValueError("Candidate pools require the released frozen taxonomy")
    formal_args = argparse.Namespace(
        output_dir=directory, train_path=args.train_path,
        exclusion_manifest=args.exclusion_manifest,
    )
    records = formal.selected_train(formal_args)
    if formal.stable_hash(records) != manifest["train"]["records_sha256"]:
        raise ValueError("Training records differ from the formal experiment manifest")
    knowledge = formal.knowledge_units(formal_args)
    if not knowledge:
        raise ValueError("No complete knowledge packages are available")
    index = formal.read_json(formal.index_manifest_path(formal_args))
    if str(args.embedding_model_path.resolve()) != str(Path(index["embedding_model_path"]).resolve()):
        raise ValueError("Use the same encoder path as the formal M/R/E index")

    s0_path = directory / "reranker_distill" / "s0_train_hypotheses.jsonl"
    latest_s0 = formal.latest(read_jsonl(s0_path))
    expected_keys = {
        f"train-s0:{int(record['idx'])}:{side}"
        for record in records for side in ("vuln", "safe")
    }
    if set(latest_s0) != expected_keys:
        raise ValueError(f"Train S0 is incomplete: {len(latest_s0)}/{len(expected_keys)} sides")
    if any(row["status"] not in ("success", "excluded") for row in latest_s0.values()):
        raise ValueError("Train S0 has unfinished task statuses")
    config = {
        "pool_builder_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        ).hexdigest(),
        "formal_manifest_hash": formal.stable_hash(manifest),
        "knowledge_fingerprint": index["knowledge_fingerprint"],
        "train_s0_hash": formal.stable_hash([
            (key, row["status"], row.get("input_hash"), row.get("parsed"))
            for key, row in sorted(latest_s0.items())
        ]),
        "seed": args.seed, "split_seed": args.split_seed,
        "retrieval_topk": args.retrieval_topk,
        "quotas": [args.rank_1_5, args.rank_6_10, args.rank_11_20],
        "cross_family_negatives": args.cross_family_negatives,
        "clone_threshold": args.clone_threshold,
        "embedding_model_path": str(args.embedding_model_path.resolve()),
        "embedding_max_length": args.embedding_max_length,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "candidate_pool_metadata.json"
    if metadata_path.exists() and formal.read_json(metadata_path) != config:
        raise ValueError("Candidate pool configuration changed; choose another output directory")
    if not metadata_path.exists():
        formal.write_json(metadata_path, config)

    query_splits = record_splits([int(record["idx"]) for record in records], args.split_seed)
    queries = []
    by_idx = {int(record["idx"]): record for record in records}
    for key, row in sorted(latest_s0.items()):
        if row["status"] != "success":
            continue
        idx, side = int(row["idx"]), str(row["side"])
        for point in row["parsed"]["hypotheses"]:
            queries.append({
                **point, "query_id": f"train:{idx}:{side}:h{int(point['id'])}",
                "idx": idx, "side": side, "dataset_split": query_splits[idx],
            })
    pool_path = args.output_dir / "candidate_pools.jsonl"
    existing = {str(row["query"]["query_id"]): row for row in read_jsonl(pool_path)}
    if len(existing) != len(read_jsonl(pool_path)):
        raise ValueError("Duplicate query pools in append-only output")
    pending = [query for query in queries if query["query_id"] not in existing]
    if not pending:
        print(json.dumps({"stage": "build-pools", "complete": len(existing), "pending": 0}))
        return
    candidates = [knowledge[idx] for idx in sorted(knowledge)]
    candidate_indices = [item["idx"] for item in candidates]
    device = formal.resolve_device(args.device)
    tokenizer, model = formal.load_embedding_model(args.embedding_model_path, device)
    try:
        vectors = formal.load_index_vectors(formal_args, knowledge)
        embed_args = argparse.Namespace(
            embedding_batch_size=args.embedding_batch_size,
            embedding_max_length=args.embedding_max_length,
        )
        query_vectors = {
            route: formal.encode_all(
                tokenizer, model,
                [str(query[config["hypothesis_field"]]) for query in pending],
                embed_args, device,
            )
            for route, config in ROUTES.items()
        }
        for position, query in enumerate(pending):
            record = by_idx[int(query["idx"])]
            source = {**record, "func_vuln": record[f"func_{query['side']}"]}
            eligible_positions = eligible_candidates(
                source, [item["record"] for item in candidates], args.clone_threshold,
            )
            eligible_ids = [candidate_indices[pos] for pos in eligible_positions]
            merged: dict[int, list[dict[str, Any]]] = defaultdict(list)
            family = str(query["mechanism_family"])
            for route in ROUTES:
                pool = eligible_positions if family == "OTHER" else [
                    pos for pos in eligible_positions
                    if candidates[pos]["mre"]["mechanism_family"] == family
                ]
                fallback = not pool
                if fallback:
                    pool = eligible_positions
                scores = vectors[route] @ query_vectors[route][position]
                ranked = sorted(pool, key=lambda pos: (-float(scores[pos]), candidate_indices[pos]))[:args.retrieval_topk]
                for rank, candidate_pos in enumerate(ranked, start=1):
                    merged[candidate_indices[candidate_pos]].append({
                        "route": route, "rank": rank, "score": float(scores[candidate_pos]),
                        "family_filter_fallback": fallback,
                    })
            candidate_rows = select_pool_candidates(
                query, merged, knowledge, eligible_ids, args.seed,
                (args.rank_1_5, args.rank_6_10, args.rank_11_20),
                args.cross_family_negatives,
            )
            append_jsonl(pool_path, {"query": query, "candidates": candidate_rows})
            if (position + 1) % 100 == 0:
                print(json.dumps({"stage": "build-pools", "added": position + 1, "pending": len(pending)}), flush=True)
    finally:
        del model
    print(json.dumps({"stage": "build-pools", "total": len(existing) + len(pending)}))


def select_groups(args: argparse.Namespace) -> None:
    pools = read_jsonl(args.output_dir / "candidate_pools.jsonl")
    metadata = formal.read_json(args.output_dir / "candidate_pool_metadata.json")
    selection_meta = {
        "pool_metadata_hash": formal.stable_hash(metadata),
        "query_ids_hash": formal.stable_hash(sorted(str(row["query"]["query_id"]) for row in pools)),
        "selection_seed": args.selection_seed, "selection_ratio": args.selection_ratio,
        "teacher_prompt_hash": teacher_prompt_hash(),
    }
    config_path = args.output_dir / "query_group_selection.json"
    if config_path.exists() and formal.read_json(config_path) != selection_meta:
        raise ValueError("Query selection or pools changed; choose another output directory")
    if not config_path.exists():
        formal.write_json(config_path, selection_meta)
    manifest_path = args.output_dir / "query_group_manifest.jsonl"
    previous = {row["query_id"]: row for row in read_jsonl(manifest_path)}
    selected = []
    for pool in pools:
        query_id = str(pool["query"]["query_id"])
        score = selection_score(query_id, args.selection_seed)
        if score >= args.selection_ratio:
            continue
        candidate_count = len(pool["candidates"])
        old = previous.get(query_id, {})
        if old and int(old["candidate_count"]) != candidate_count:
            raise ValueError(f"Candidate count changed for {query_id}")
        selected.append({
            "query_id": query_id, "selection_seed": args.selection_seed,
            "selection_score": score, "selection_ratio": args.selection_ratio,
            "candidate_count": candidate_count,
            "status": old.get("status", "selected"),
            "successful_labels": int(old.get("successful_labels", 0)),
            "prompt_hash": teacher_prompt_hash(),
        })
    atomic_jsonl(manifest_path, sorted(selected, key=lambda row: row["query_id"]))
    print(json.dumps({"stage": "select-groups", "selected": len(selected), "total": len(pools)}))


def label_one(
    query: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace,
    prompt_hash: str,
) -> dict[str, Any]:
    messages = build_messages(query, candidate["candidate_mre"])
    base = {
        "pair_id": pair_id(str(query["query_id"]), int(candidate["candidate_idx"])),
        "query_id": query["query_id"], "candidate_idx": int(candidate["candidate_idx"]),
        "dataset_split": query["dataset_split"],
        "candidate_source": candidate["candidate_source"],
        "rank_stratum": candidate.get("rank_stratum", "unknown"),
        "same_source": bool(candidate["same_source"]),
        "prompt_hash": prompt_hash,
        "teacher_input": {
            "query": {key: query[key] for key in ("mechanism_claim", "repair_sought", "evidence_to_check")},
            "candidate": {key: candidate["candidate_mre"][key] for key in (
                "mechanism_observed", "repair_applied", "evidence_decisive",
            )},
        },
    }
    attempts = []
    try:
        client = build_client(args.api_base, args.api_key).with_options(max_retries=0)
    except Exception as exc:
        return {**base, "status": "failed", "attempts": [{"attempt": 0, "error": str(exc)}]}
    try:
        for attempt in range(1, args.retries + 1):
            try:
                raw, usage = call_chat_completion_with_usage(
                    client=client, model_name=args.model_name, messages=messages,
                    temperature=args.temperature, top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens, timeout=args.timeout,
                    reasoning_effort=args.reasoning_effort,
                    thinking_type=args.thinking_type,
                    chat_template_family=args.chat_template_family,
                    response_format={"type": "json_object"},
                )
                parsed = parse_teacher_output(raw)
                return {**base, "status": "success", **parsed, "raw_output": raw, "usage": usage}
            except Exception as exc:
                attempts.append({"attempt": attempt, "error": str(exc)})
                if attempt < args.retries:
                    time.sleep(min(30, 2**attempt))
        return {**base, "status": "failed", "attempts": attempts}
    finally:
        client.close()


def label_groups(args: argparse.Namespace) -> None:
    selection = formal.read_json(args.output_dir / "query_group_selection.json")
    if selection["teacher_prompt_hash"] != teacher_prompt_hash():
        raise ValueError("Teacher prompt changed since query selection")
    pools = {str(row["query"]["query_id"]): row for row in read_jsonl(args.output_dir / "candidate_pools.jsonl")}
    manifest_path = args.output_dir / "query_group_manifest.jsonl"
    manifest = {str(row["query_id"]): row for row in read_jsonl(manifest_path)}
    label_path = args.output_dir / "teacher_labels.jsonl"
    labels = latest_successful_labels(
        row for row in read_jsonl(label_path)
        if row.get("prompt_hash") == teacher_prompt_hash()
    )
    if not args.api_key:
        raise ValueError("Set LLM_API_KEY or pass --api-key")
    snapshot = args.output_dir / "prompt_snapshots" / f"teacher_{teacher_prompt_hash()}.txt"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        snapshot.write_text(SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE + "\n", encoding="utf-8")
    pending = sorted(manifest.values(), key=lambda row: (row["selection_score"], row["query_id"]))
    if args.max_query_groups > 0:
        pending = pending[:args.max_query_groups]
    for position, group in enumerate(pending, start=1):
        query_id = str(group["query_id"])
        pool = pools[query_id]
        candidates = pool["candidates"]
        if len(candidates) != int(group["candidate_count"]):
            raise ValueError(f"Candidate pool changed for {query_id}")
        missing = [candidate for candidate in candidates if pair_id(query_id, int(candidate["candidate_idx"])) not in labels]
        group["status"] = "in_progress" if missing else "complete"
        group["successful_labels"] = len(candidates) - len(missing)
        atomic_jsonl(manifest_path, sorted(manifest.values(), key=lambda row: row["query_id"]))
        if missing:
            with ThreadPoolExecutor(max_workers=min(args.max_workers, len(missing))) as executor:
                futures = [
                    executor.submit(label_one, pool["query"], candidate, args, teacher_prompt_hash())
                    for candidate in missing
                ]
                for future in as_completed(futures):
                    row = future.result()
                    append_jsonl(label_path, row)
                    if row["status"] == "success":
                        labels[str(row["pair_id"])] = row
        group["successful_labels"] = sum(
            pair_id(query_id, int(candidate["candidate_idx"])) in labels for candidate in candidates
        )
        group["status"] = "complete" if group["successful_labels"] == len(candidates) else "partial_failed"
        atomic_jsonl(manifest_path, sorted(manifest.values(), key=lambda row: row["query_id"]))
        print(json.dumps({"stage": "label-groups", "group": position, "total": len(pending),
                          "query_id": query_id, "successful": group["successful_labels"],
                          "candidate_count": len(candidates), "status": group["status"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build-pools")
    build.add_argument("--formal-output-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--train-path", type=Path, required=True)
    build.add_argument("--exclusion-manifest", type=Path, default=ROOT / "config" / "excluded_records.json")
    build.add_argument("--embedding-model-path", type=Path, required=True)
    build.add_argument("--embedding-max-length", type=int, default=1024)
    build.add_argument("--embedding-batch-size", type=int, default=32)
    build.add_argument("--retrieval-topk", type=int, default=20)
    build.add_argument("--rank-1-5", type=int, default=4)
    build.add_argument("--rank-6-10", type=int, default=3)
    build.add_argument("--rank-11-20", type=int, default=3)
    build.add_argument("--cross-family-negatives", type=int, default=2)
    build.add_argument("--clone-threshold", type=float, default=0.80)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--split-seed", type=int, default=42)
    build.add_argument("--device", default="auto")
    select = subcommands.add_parser("select-groups")
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--selection-ratio", type=float, default=0.5)
    select.add_argument("--selection-seed", type=int, default=42)
    label = subcommands.add_parser("label-groups")
    label.add_argument("--output-dir", type=Path, required=True)
    label.add_argument("--api-base", default="http://localhost:8000/v1")
    label.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    label.add_argument("--model-name", required=True)
    label.add_argument("--temperature", type=float, default=0.0)
    label.add_argument("--top-p", type=float, default=1.0)
    label.add_argument("--thinking-type", choices=("disabled", "enabled"), default="disabled")
    label.add_argument("--reasoning-effort", default="high")
    label.add_argument("--chat-template-family", choices=("default", "template_kwargs"), default="default")
    label.add_argument("--max-new-tokens", type=int, default=8192)
    label.add_argument("--timeout", type=int, default=300)
    label.add_argument("--retries", type=int, default=3)
    label.add_argument("--max-workers", type=int, default=4)
    label.add_argument("--max-query-groups", type=int, default=-1)
    args = parser.parse_args()
    if args.command == "build-pools":
        if args.retrieval_topk < 20:
            parser.error("--retrieval-topk must be at least 20 for the three rank strata")
        build_pools(args)
    elif args.command == "select-groups":
        if not 0 < args.selection_ratio <= 1:
            parser.error("--selection-ratio must be in (0, 1]")
        select_groups(args)
    else:
        if args.max_workers < 1 or args.retries < 1:
            parser.error("--max-workers and --retries must be positive")
        label_groups(args)


if __name__ == "__main__":
    main()
