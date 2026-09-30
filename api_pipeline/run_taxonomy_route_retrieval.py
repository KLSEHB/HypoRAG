#!/usr/bin/env python3
"""Evaluate taxonomy-routed M/R/E retrieval fields on frozen neutral inputs.

Each route is generated independently, but its field and taxonomy family are
emitted in one API response.  Thus a route measures the representation and the
route-specific routing decision together.  This script never regenerates S0
hypotheses or S1 repair signatures; it only consumes the frozen 89/481 source.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from api_pipeline.run_independent_code_retrieval_baselines import (  # noqa: E402
    eligible_candidates,
    frozen_records,
)
from api_pipeline.repair_signature import (  # noqa: E402
    MECHANISM_MATCH_AUDIT_SYSTEM,
    signature_view,
)


ROUTES: dict[str, dict[str, str]] = {
    "mechanism": {
        "hypothesis_field": "mechanism_claim",
        "knowledge_field": "mechanism_observed",
        "hypothesis_objective": (
            "State the alleged failed safety mechanism: the concrete operation, "
            "risk object, and trigger/invariant relation that would need checking. "
            "Do not prescribe a repair."
        ),
        "knowledge_objective": (
            "State the failed safety mechanism evidenced by the repair: the concrete "
            "operation, risk object, and trigger/invariant relation. Do not describe "
            "the repair procedure except where needed to identify that mechanism."
        ),
    },
    "repair": {
        "hypothesis_field": "repair_sought",
        "knowledge_field": "repair_applied",
        "hypothesis_objective": (
            "State the specific validation, guard, ownership action, or invariant-"
            "establishing repair principle that a useful historical case should provide."
        ),
        "knowledge_objective": (
            "State the concrete repair action and the invariant it establishes, using "
            "only the repair signature evidence."
        ),
    },
    "evidence": {
        "hypothesis_field": "evidence_to_check",
        "knowledge_field": "evidence_decisive",
        "hypothesis_objective": (
            "State the local evidence or relation that must be checked to confirm or "
            "reject this hypothesis: operation, relevant values/state, and required guard."
        ),
        "knowledge_objective": (
            "State the decisive before/after evidence and the concrete relation it "
            "establishes; focus on what makes the repair case useful for verification."
        ),
    },
}

PROMPT_STYLE = """You create one concise retrieval representation from a frozen,
taxonomy-neutral security description.  Jointly choose the primary mechanism
family and write the requested route field in the SAME JSON response.  Do not
first assign a family and then mechanically restate its definition.

The supplied description is the only evidence.  Do not infer missing source,
caller, helper, patch, label, CWE, CVE, project, or API facts.  Follow the given
taxonomy definitions and precedence exactly.  Choose OTHER only when no remaining
specific family describes the earliest failed safety condition.

Style contract shared by every route: write one factual, reusable sentence of at
most 55 words.  Use the vocabulary operation, object, trigger/invariant, guard,
or repair action when supported.  Preserve important distinctions such as cursor
remaining extent versus fixed capacity, arithmetic-created extent versus direct
transfer, and semantic scalar domain versus object/type validity.  Avoid generic
phrases such as 'validate input' or 'prevent vulnerability', broad CWE labels, and
project-specific narrative.  Return JSON only."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def stable_hash(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


def latest_success(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "success":
            result[str(row[key])] = row
    return result


def source_dir(args: argparse.Namespace) -> Path:
    return Path(args.source_dir)


def output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_dir)


def taxonomy_definition(source: Path, taxonomy: str) -> dict[str, Any]:
    catalog = json.loads((source / "taxonomy_catalog.json").read_text(encoding="utf-8"))
    if taxonomy not in catalog:
        raise ValueError(f"unknown taxonomy in source catalog: {taxonomy}")
    return catalog[taxonomy]


def frozen_items(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the immutable neutral 89-H and 481-K representations."""
    hypothesis_rows = latest_success(read_jsonl(source / "hypotheses.jsonl"), "key")
    signature_rows = latest_success(read_jsonl(source / "signatures.jsonl"), "key")

    hypotheses: list[dict[str, Any]] = []
    for row in hypothesis_rows.values():
        for point in row["parsed"]["hypotheses"]:
            hypotheses.append({
                "kind": "hypothesis",
                "signature_id": f"test:{int(row['idx'])}:h{int(point['id'])}",
                "source_idx": int(row["idx"]),
                "neutral": signature_view(point, "hypothesis"),
            })

    signatures: list[dict[str, Any]] = []
    for row in signature_rows.values():
        parsed = row["parsed"]
        if parsed.get("adjudicability") == "unrelated_patch":
            continue
        signatures.append({
            "kind": "knowledge",
            "signature_id": f"train:{int(row['idx'])}:primary",
            "source_idx": int(row["idx"]),
            "neutral": signature_view(parsed, "repair_signature"),
        })
    hypotheses.sort(key=lambda value: value["signature_id"])
    signatures.sort(key=lambda value: value["signature_id"])
    if len(hypotheses) != 89 or len(signatures) != 481:
        raise RuntimeError(f"expected frozen 89/481 inputs, got {len(hypotheses)}/{len(signatures)}")
    return hypotheses, signatures


def route_messages(
    item: dict[str, Any], route: str, taxonomy: dict[str, Any]
) -> tuple[list[dict[str, str]], str]:
    config = ROUTES[route]
    is_hypothesis = item["kind"] == "hypothesis"
    field = config["hypothesis_field"] if is_hypothesis else config["knowledge_field"]
    objective = config["hypothesis_objective"] if is_hypothesis else config["knowledge_objective"]
    taxonomy_payload = {
        "allowed_families": taxonomy["families"],
        "precedence": taxonomy["precedence"],
    }
    system = "\n\n".join((
        PROMPT_STYLE,
        "Frozen taxonomy:\n" + json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        f"Route objective: {objective}",
        "Schema:\n{"
        f"\"mechanism_family\":\"one allowed family\",\"{field}\":\"one concise sentence\""
        "}",
    ))
    user = {
        "signature_id": item["signature_id"],
        "source_kind": item["kind"],
        "neutral_description": item["neutral"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ], field


def joint_route_messages(
    item: dict[str, Any], taxonomy: dict[str, Any], profile: str = "original"
) -> tuple[list[dict[str, str]], list[str]]:
    """Generate M/R/E and their one shared routing family in a single call."""
    is_hypothesis = item["kind"] == "hypothesis"
    objectives: list[str] = []
    fields: list[str] = []
    for route, config in ROUTES.items():
        fields.append(config["hypothesis_field"] if is_hypothesis else config["knowledge_field"])
        objectives.append(
            f"{route.upper()} route: " + (
                config["hypothesis_objective"] if is_hypothesis else config["knowledge_objective"]
            )
        )
    taxonomy_payload = {
        "allowed_families": taxonomy["families"],
        "precedence": taxonomy["precedence"],
    }
    schema = {"mechanism_family": "one allowed family", **{
        field: "one concise sentence" for field in fields
    }}
    common_sections = [
        PROMPT_STYLE,
        "Frozen taxonomy:\n" + json.dumps(taxonomy_payload, ensure_ascii=False, indent=2),
        "Route objectives (their definitions are unchanged from the independent route prompts):\n"
        + "\n".join(objectives),
    ]
    if profile == "original":
        common_sections.extend((
            "Return one shared mechanism_family for all three route fields. "
            "The family must describe the earliest failed safety condition, not merely "
            "the wording preferred by one route.",
            "Each field must answer its own route objective. Do not make M, R, and E "
            "three near-duplicate sentences or copy a family definition into all fields.",
        ))
    elif profile == "definition_aligned_thinking":
        common_sections.append(
            "Before producing the final JSON, perform thorough private analysis: "
            "identify the concrete operation, risk object, trigger or violated invariant; "
            "apply the taxonomy definitions and precedence; then separately determine the "
            "mechanism, repair, and evidence representations from the supplied description. "
            "Do not finish early for apparently simple cases. Keep the analysis in the "
            "thinking channel and reserve enough generation budget for a complete final JSON; "
            "the final answer must contain JSON only."
        )
    else:
        raise ValueError(f"unknown joint prompt profile: {profile}")
    common_sections.append("Schema:\n" + json.dumps(schema, ensure_ascii=False))
    system = "\n\n".join(common_sections)
    user = {
        "signature_id": item["signature_id"],
        "source_kind": item["kind"],
        "neutral_description": item["neutral"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ], fields


def validate_route_result(value: Any, allowed_families: set[str], field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("route response must be an object")
    family = value.get("mechanism_family")
    text = value.get(field)
    if family not in allowed_families:
        raise ValueError(f"invalid mechanism_family: {family!r}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"missing route field: {field}")
    if len(text.split()) > 75 or len(text) > 700:
        raise ValueError(f"route field exceeds style bound: {field}")
    return {"mechanism_family": family, field: text.strip()}


def validate_joint_route_result(
    value: Any, allowed_families: set[str], fields: list[str]
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("joint route response must be an object")
    family = value.get("mechanism_family")
    if family not in allowed_families:
        raise ValueError(f"invalid mechanism_family: {family!r}")
    result = {"mechanism_family": family}
    for field in fields:
        text = value.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"missing joint route field: {field}")
        if len(text.split()) > 75 or len(text) > 700:
            raise ValueError(f"joint route field exceeds style bound: {field}")
        result[field] = text.strip()
    return result


def manifest_value(args: argparse.Namespace) -> dict[str, Any]:
    source = source_dir(args)
    hypotheses, signatures = frozen_items(source)
    taxonomy = taxonomy_definition(source, args.taxonomy)
    field_prompt_hashes: dict[str, dict[str, str]] = {}
    if args.generation_mode == "joint":
        field_prompt_hashes["joint_mre"] = {}
        for item in (hypotheses[0], signatures[0]):
            messages, _ = joint_route_messages(item, taxonomy, args.joint_prompt_profile)
            field_prompt_hashes["joint_mre"][item["kind"]] = fingerprint(messages[0])
    else:
        for route in ROUTES:
            field_prompt_hashes[route] = {}
            for item in (hypotheses[0], signatures[0]):
                messages, _ = route_messages(item, route, taxonomy)
                field_prompt_hashes[route][item["kind"]] = fingerprint(messages[0])
    result = {
        "experiment": "taxonomy_route_retrieval_v1",
        "source_experiment": str(source.resolve()),
        "source_manifest_sha256": fingerprint(json.loads((source / "manifest.json").read_text(encoding="utf-8"))),
        "source_taxonomy_catalog_sha256": fingerprint(json.loads((source / "taxonomy_catalog.json").read_text(encoding="utf-8"))),
        "taxonomy": args.taxonomy,
        "taxonomy_definition": taxonomy,
        "frozen_inputs": {"hypotheses": len(hypotheses), "knowledge": len(signatures)},
        "routes": {
            route: {
                "hypothesis_field": config["hypothesis_field"],
                "knowledge_field": config["knowledge_field"],
            }
            for route, config in ROUTES.items()
        },
        "generation": {
            "model_name": args.model_name,
            "temperature": 0.0,
            "top_p": 1.0,
            "thinking_type": args.generation_thinking_type,
            "reasoning_effort": args.generation_reasoning_effort,
            "prompt_hashes": field_prompt_hashes,
            "contract": "Each route jointly emits its route field and mechanism_family in one prompt.",
        },
        "retrieval": {
            "encoder": str(Path(args.embedding_model_path).resolve()),
            "encoder_max_length": args.embedding_max_length,
            "candidate_exclusion": json.loads((source / "manifest.json").read_text(encoding="utf-8"))["candidate_exclusion"],
            "selection": "route-specific family-filtered dense top-3; OTHER uses eligible global pool",
        },
        "audit": {
            "protocol": "original taxonomy three-level LLM-assisted initial audit",
            "reuse_labels": str((source / "audit_labels_assisted.jsonl").resolve()),
            "audit_model_name": args.audit_model_name,
            "thinking_type": "disabled",
        },
    }
    if args.generation_mode == "joint":
        result["generation"]["mode"] = "joint_mre_shared_family"
        result["generation"]["joint_prompt_profile"] = args.joint_prompt_profile
    return result


def prepare(args: argparse.Namespace) -> None:
    out = output_dir(args)
    value = manifest_value(args)
    target = out / "experiment_manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != value:
        raise ValueError("existing route experiment manifest differs; choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(target, value)
    write_json(out / "taxonomy_catalog.json", {args.taxonomy: value["taxonomy_definition"]})
    print(json.dumps({"stage": "prepare", **value["frozen_inputs"]}, ensure_ascii=False))


def generation_path(args: argparse.Namespace, route: str, kind: str) -> Path:
    name = f"joint_{kind}" if args.generation_mode == "joint" else f"{route}_{kind}"
    return output_dir(args) / "generation" / f"{name}.jsonl"


def request_route(
    task: dict[str, Any], args: argparse.Namespace, api_key: str, allowed_families: set[str]
) -> dict[str, Any]:
    max_new_tokens = int(task.get("max_new_tokens") or args.generation_max_new_tokens)
    result = {
        **task,
        "model_name": args.model_name,
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking_type": args.generation_thinking_type,
        "reasoning_effort": args.generation_reasoning_effort,
        "max_new_tokens": max_new_tokens,
        "attempts": [],
    }
    messages = task["messages"]
    client = build_client(args.api_base, api_key).with_options(max_retries=0)
    try:
        for attempt in range(1, args.retries + 1):
            event: dict[str, Any] = {"attempt": attempt, "started_at": time.time(), "request_messages": messages}
            try:
                raw, usage = call_chat_completion_with_usage(
                    client, args.model_name, messages, 0.0, 1.0,
                    max_new_tokens, args.timeout,
                    args.generation_reasoning_effort, args.generation_thinking_type,
                )
                parsed = (
                    validate_joint_route_result(extract_json_object(raw), allowed_families, task["fields"])
                    if task["generation_mode"] == "joint"
                    else validate_route_result(extract_json_object(raw), allowed_families, task["field"])
                )
                event.update(status="success", raw_output=raw, usage=usage)
                result["attempts"].append(event)
                return {**result, "status": "success", "parsed": parsed}
            except Exception as exc:
                event.update(status="failed", error=str(exc).replace(api_key, "[REDACTED]"))
                result["attempts"].append(event)
                if attempt < args.retries:
                    messages = list(task["messages"]) + [{
                        "role": "user",
                        "content": (
                            "Regenerate the complete JSON for the same neutral description. "
                            "Use exactly one listed family and every required nonempty concise route field; "
                            "do not change the evidence policy."
                        ),
                    }]
                    time.sleep(min(20.0, 2.0 ** attempt))
        return {**result, "status": "failed", "error": result["attempts"][-1]["error"]}
    finally:
        client.close()


def generate(args: argparse.Namespace) -> None:
    prepare(args)
    out = output_dir(args)
    source = source_dir(args)
    taxonomy = taxonomy_definition(source, args.taxonomy)
    allowed = set(taxonomy["families"])
    hypotheses, signatures = frozen_items(source)
    api_key: str | None = None
    generation_units = (
        [("joint_mre", kind, items) for kind, items in (("hypothesis", hypotheses), ("knowledge", signatures))]
        if args.generation_mode == "joint"
        else [(route, kind, items) for route in args.routes for kind, items in (("hypothesis", hypotheses), ("knowledge", signatures))]
    )
    for route, kind, items in generation_units:
            path = generation_path(args, route, kind)
            cached = latest_success(read_jsonl(path), "key")
            tasks: list[dict[str, Any]] = []
            for item in items:
                if args.generation_mode == "joint":
                    messages, fields = joint_route_messages(item, taxonomy, args.joint_prompt_profile)
                    field = None
                else:
                    messages, field = route_messages(item, route, taxonomy)
                    fields = None
                key = f"{kind}:{item['signature_id']}"
                task = {
                    "key": key,
                    "route": route,
                    "kind": kind,
                    "signature_id": item["signature_id"],
                    "source_idx": item["source_idx"],
                    "field": field,
                    "fields": fields,
                    "generation_mode": args.generation_mode,
                    "messages": messages,
                    "input_hash": fingerprint(messages),
                    "prompt_hash": fingerprint(messages[0]),
                    "max_new_tokens": args.retry_max_new_tokens or args.generation_max_new_tokens,
                }
                old = cached.get(key)
                if old and (
                    old.get("input_hash") != task["input_hash"]
                    or old.get("model_name") != args.model_name
                    or (
                        not args.retry_max_new_tokens
                        and int(old.get("max_new_tokens") or 0) < task["max_new_tokens"]
                    )
                ):
                    raise ValueError(f"stale route-generation cache: {key}")
                tasks.append(task)
            pending = [task for task in tasks if task["key"] not in cached]
            if args.limit:
                pending = pending[:args.limit]
            print(json.dumps({"stage": "generate", "route": route, "kind": kind,
                              "cached": len(cached), "pending": len(pending)}), flush=True)
            if not pending:
                continue
            if api_key is None:
                api_key = load_api_key(None, args.api_key_env)
            failures = 0
            with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {
                    executor.submit(request_route, task, args, api_key, allowed): task
                    for task in pending
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    append_jsonl(path, row)
                    failures += row["status"] != "success"
                    print(json.dumps({"stage": "generate", "route": route, "kind": kind,
                                      "completed": completed, "remaining": len(pending) - completed,
                                      "status": row["status"]}), flush=True)
            if failures:
                raise RuntimeError(f"{route}/{kind}: {failures} failed rows; successful rows retained")
    write_usage_summary(args)


def route_outputs(args: argparse.Namespace, route: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    out = output_dir(args)
    hypotheses, signatures = frozen_items(source_dir(args))
    source_route = "joint_mre" if args.generation_mode == "joint" else route
    hrows = latest_success(read_jsonl(generation_path(args, source_route, "hypothesis")), "key")
    krows = latest_success(read_jsonl(generation_path(args, source_route, "knowledge")), "key")
    expected_h = {item["signature_id"] for item in hypotheses}
    expected_k = {item["signature_id"] for item in signatures}
    outputs_h = {row["signature_id"]: row["parsed"] for row in hrows.values()}
    outputs_k = {row["signature_id"]: row["parsed"] for row in krows.values()}
    if set(outputs_h) != expected_h or set(outputs_k) != expected_k:
        raise RuntimeError(
            f"route {route} outputs incomplete: hypotheses={len(outputs_h)}/{len(expected_h)}, "
            f"knowledge={len(outputs_k)}/{len(expected_k)}"
        )
    return outputs_h, outputs_k


def encode_all(tokenizer: Any, model: Any, texts: list[str], args: argparse.Namespace, device: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), args.embedding_batch_size):
        vectors.extend(encode_texts(
            tokenizer, model, texts[start:start + args.embedding_batch_size],
            args.embedding_max_length, device,
        ))
    return np.asarray(vectors, dtype=np.float32)


def retrieve(args: argparse.Namespace) -> None:
    prepare(args)
    source = source_dir(args)
    out = output_dir(args)
    hypotheses, signatures = frozen_items(source)
    raw_args = argparse.Namespace(
        holdout_dir=str(source), train_path=args.train_path, test_path=args.test_path,
    )
    query_rows, train_rows, manifest = frozen_records(raw_args)
    query_by_idx = {int(row["idx"]): row for row in query_rows}
    train_by_idx = {int(row["idx"]): row for row in train_rows}
    candidate_rows = [train_by_idx[item["source_idx"]] for item in signatures]
    candidate_position = {int(row["idx"]): position for position, row in enumerate(candidate_rows)}
    if len(candidate_position) != len(candidate_rows):
        raise RuntimeError("duplicate repair candidate IDs")

    device = resolve_device(args.device)
    tokenizer, model = load_embedding_model(Path(args.embedding_model_path), device)
    try:
        for route in args.routes:
            houtputs, koutputs = route_outputs(args, route)
            hfield = ROUTES[route]["hypothesis_field"]
            kfield = ROUTES[route]["knowledge_field"]
            candidate_vectors = encode_all(
                tokenizer, model, [koutputs[item["signature_id"]][kfield] for item in signatures], args, device
            )
            hypothesis_vectors = encode_all(
                tokenizer, model, [houtputs[item["signature_id"]][hfield] for item in hypotheses], args, device
            )
            pairs: list[dict[str, Any]] = []
            pool_rows: list[dict[str, Any]] = []
            for hypothesis, vector in zip(hypotheses, hypothesis_vectors):
                houtput = houtputs[hypothesis["signature_id"]]
                family = houtput["mechanism_family"]
                query = query_by_idx[hypothesis["source_idx"]]
                eligible = eligible_candidates(query, candidate_rows, args.clone_threshold)
                family_filter_fallback = False
                if family == "OTHER":
                    pool = eligible
                else:
                    pool = [
                        position for position in eligible
                        if koutputs[signatures[position]["signature_id"]]["mechanism_family"] == family
                    ]
                if not pool:
                    # A sparse diagnostic sample can contain no knowledge item from a
                    # valid, specific family. Preserve the generated family and record
                    # a deterministic global fallback rather than dropping the query.
                    pool = eligible
                    family_filter_fallback = True
                if not pool:
                    raise RuntimeError(f"route {route} has no eligible pool for {hypothesis['signature_id']}")
                scores = candidate_vectors @ vector
                selected = sorted(
                    pool,
                    key=lambda position: (-float(scores[position]), int(candidate_rows[position]["idx"]), position),
                )[:args.top_k]
                pool_rows.append({
                    "hypothesis_id": hypothesis["signature_id"],
                    "source_idx": hypothesis["source_idx"],
                    "family": family,
                    "family_filter_fallback": family_filter_fallback,
                    "eligible_global": len(eligible),
                    "pool_size": len(pool),
                    "returned": len(selected),
                })
                for rank, position in enumerate(selected, 1):
                    candidate = candidate_rows[position]
                    signature = signatures[position]
                    audit_id = stable_hash(
                        "taxonomy-search-audit", hypothesis["signature_id"], int(candidate["idx"])
                    )[:24]
                    pairs.append({
                        "retriever": f"taxonomy_route_dense_{route}",
                        "route": route,
                        "taxonomy": args.taxonomy,
                        "hypothesis_id": hypothesis["signature_id"],
                        "query_idx": hypothesis["source_idx"],
                        "candidate_idx": int(candidate["idx"]),
                        "candidate_signature_id": signature["signature_id"],
                        "family": family,
                        "family_filter_fallback": family_filter_fallback,
                        "candidate_family": koutputs[signature["signature_id"]]["mechanism_family"],
                        "rank": rank,
                        "score": float(scores[position]),
                        "pool_size": len(pool),
                        "eligible_global": len(eligible),
                        "audit_id": audit_id,
                        "hypothesis": hypothesis["neutral"],
                        "candidate_signature": signature["neutral"],
                        "query_route_text": houtput[hfield],
                        "candidate_route_text": koutputs[signature["signature_id"]][kfield],
                    })
            if len(pairs) != sum(row["returned"] for row in pool_rows):
                raise AssertionError("route retrieval pair count does not match pool returns")
            route_dir = out / "retrieval" / route
            write_jsonl(route_dir / "retrieval_pairs.jsonl", pairs)
            write_jsonl(route_dir / "pool_stats.jsonl", pool_rows)
            write_json(route_dir / "retrieval_manifest.json", {
                "route": route,
                "taxonomy": args.taxonomy,
                "query_field": hfield,
                "candidate_field": kfield,
                "route_family_assignment": (
                    "one shared family jointly generated with M/R/E"
                    if args.generation_mode == "joint"
                    else "jointly generated with its route field"
                ),
                "candidate_exclusions": manifest["candidate_exclusion"],
                "selection": "dense top-3 after route-specific family filtering; OTHER and an otherwise empty specific family use the global eligible pool",
                "embedding_model": str(Path(args.embedding_model_path).resolve()),
                "embedding_max_length": args.embedding_max_length,
                "hypothesis_count": len(hypotheses),
                "knowledge_count": len(signatures),
            })
            write_json(route_dir / "retrieval_summary.json", {
                "pair_count": len(pairs),
                "hypothesis_count": len(hypotheses),
                "knowledge_count": len(signatures),
                "pool_min": min(row["pool_size"] for row in pool_rows),
                "pool_median": sorted(row["pool_size"] for row in pool_rows)[len(pool_rows) // 2],
                "pool_max": max(row["pool_size"] for row in pool_rows),
                "under_three_hypotheses": sum(row["returned"] < args.top_k for row in pool_rows),
            })
            print(json.dumps({"stage": "retrieve", "route": route, "pairs": len(pairs)}), flush=True)
    finally:
        del model


def all_pairs(args: argparse.Namespace, routes: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    selected = routes or ROUTES.keys()
    result: dict[str, dict[str, Any]] = {}
    for route in selected:
        for row in read_jsonl(output_dir(args) / "retrieval" / route / "retrieval_pairs.jsonl"):
            prior = result.get(row["audit_id"])
            if prior is not None and (
                prior["hypothesis"] != row["hypothesis"] or prior["candidate_signature"] != row["candidate_signature"]
            ):
                raise RuntimeError(f"inconsistent duplicate audit pair: {row['audit_id']}")
            result[row["audit_id"]] = row
    return result


def audit_messages(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    visible = [{
        "audit_id": row["audit_id"],
        "hypothesis": row["hypothesis"],
        "candidate_signature": row["candidate_signature"],
    } for row in batch]
    return [
        {"role": "system", "content": MECHANISM_MATCH_AUDIT_SYSTEM},
        {"role": "user", "content": json.dumps({"items": visible}, ensure_ascii=False)},
        {
            "role": "user",
            "content": (
                "For this batch, copy each audit_id exactly, character for character, "
                "from this closed list. Do not add, remove, complete, or normalize any "
                "characters: " + json.dumps([row["audit_id"] for row in batch])
            ),
        },
    ]


def validate_judgments(raw: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = extract_json_object(raw)
    judgments = value.get("judgments") if isinstance(value, dict) else None
    expected = {row["audit_id"] for row in batch}
    required = ("root_cause_alignment", "trigger_invariant_alignment", "transferable_principle", "reason")
    if not isinstance(judgments, list) or len(judgments) != len(batch):
        raise ValueError("audit must return exactly one judgment per pair")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for judgment in judgments:
        if not isinstance(judgment, dict) or judgment.get("audit_id") not in expected or judgment["audit_id"] in seen:
            raise ValueError("audit returned unknown or duplicate audit_id")
        if judgment.get("label") not in {0, 1, 2}:
            raise ValueError("audit label must be 0, 1, or 2")
        if any(not isinstance(judgment.get(field), str) or not judgment[field].strip() for field in required):
            raise ValueError("audit judgment contains an empty rationale field")
        seen.add(judgment["audit_id"])
        result.append({"audit_id": judgment["audit_id"], **{
            field: judgment[field] for field in ("label", *required)
        }})
    return result


def request_audit(
    batch_key: str, batch: list[dict[str, Any]], args: argparse.Namespace, api_key: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_key": batch_key,
        "audit_ids": [row["audit_id"] for row in batch],
        "model_name": args.audit_model_name,
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking_type": "disabled",
        "attempts": [],
    }
    client = build_client(args.api_base, api_key).with_options(max_retries=0)
    messages = audit_messages(batch)
    try:
        for attempt in range(1, args.retries + 1):
            try:
                raw, usage = call_chat_completion_with_usage(
                    client, args.audit_model_name, messages, 0.0, 1.0,
                    args.audit_max_new_tokens, args.timeout, "high", "disabled",
                )
                result["attempts"].append({
                    "attempt": attempt, "status": "success", "raw_output": raw, "usage": usage,
                })
                return {**result, "status": "success", "judgments": validate_judgments(raw, batch)}
            except Exception as exc:
                result["attempts"].append({
                    "attempt": attempt, "status": "failed", "error": str(exc).replace(api_key, "[REDACTED]"),
                })
                if attempt < args.retries:
                    time.sleep(min(20.0, 2.0 ** attempt))
        return {**result, "status": "failed", "error": result["attempts"][-1]["error"]}
    finally:
        client.close()


def seed_audit_labels(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    source = source_dir(args)
    rows = read_jsonl(source / "audit_labels_assisted.jsonl")
    result = {str(row["audit_id"]): row for row in rows}
    if any(row.get("label") not in {0, 1, 2} for row in result.values()):
        raise ValueError("source audit labels contain an invalid label")
    return result


def completed_audit_labels(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    result = seed_audit_labels(args)
    for batch in latest_success(read_jsonl(output_dir(args) / "audit" / "audit_assisted.jsonl"), "batch_key").values():
        for judgment in batch["judgments"]:
            prior = result.get(judgment["audit_id"])
            if prior and prior.get("label") != judgment.get("label"):
                raise RuntimeError(f"conflicting audit label: {judgment['audit_id']}")
            result[judgment["audit_id"]] = {**judgment, "review_status": "ai_assisted_unreviewed"}
    return result


def export_audit_labels(args: argparse.Namespace) -> None:
    pairs = all_pairs(args)
    labels = completed_audit_labels(args)
    missing = set(pairs) - set(labels)
    if missing:
        raise RuntimeError(f"route audit labels incomplete: {len(missing)} missing")
    write_jsonl(output_dir(args) / "audit" / "audit_labels_assisted.jsonl", [
        labels[audit_id] for audit_id in sorted(pairs)
    ])


def audit(args: argparse.Namespace) -> None:
    pairs = all_pairs(args, args.routes)
    labels = completed_audit_labels(args)
    pending_ids = sorted(set(pairs) - set(labels))
    batches = [
        pending_ids[start:start + args.audit_batch_size]
        for start in range(0, len(pending_ids), args.audit_batch_size)
    ]
    print(json.dumps({"stage": "audit", "unique_pairs": len(pairs),
                      "reused_or_cached": len(pairs) - len(pending_ids), "new": len(pending_ids)}), flush=True)
    if not batches:
        export_audit_labels(args)
        return
    api_key = load_api_key(None, args.api_key_env)
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.audit_workers, len(batches))) as executor:
        futures = {
            executor.submit(
                request_audit,
                f"audit:{number}:{stable_hash(*batch_ids)[:12]}",
                [pairs[audit_id] for audit_id in batch_ids], args, api_key,
            ): batch_ids
            for number, batch_ids in enumerate(batches)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(output_dir(args) / "audit" / "audit_assisted.jsonl", row)
            failures += row["status"] != "success"
            print(json.dumps({"stage": "audit", "completed": completed,
                              "remaining": len(batches) - completed, "status": row["status"]}), flush=True)
    if failures:
        raise RuntimeError(f"{failures} audit batches failed; rerun safely resumes")
    export_audit_labels(args)
    write_usage_summary(args)


def route_summary(args: argparse.Namespace, route: str, labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = source_dir(args)
    taxonomy = taxonomy_definition(source, args.taxonomy)
    hypotheses, signatures = frozen_items(source)
    houtputs, koutputs = route_outputs(args, route)
    pairs = read_jsonl(output_dir(args) / "retrieval" / route / "retrieval_pairs.jsonl")
    expected = {row["audit_id"] for row in pairs}
    if not expected <= set(labels):
        raise RuntimeError(f"summary labels missing for route {route}")
    label_counts = Counter(int(labels[row["audit_id"]]["label"]) for row in pairs)
    by_hypothesis: dict[str, list[int]] = defaultdict(list)
    by_function: dict[int, list[int]] = defaultdict(list)
    by_family: dict[str, list[int]] = defaultdict(list)
    for row in pairs:
        label = int(labels[row["audit_id"]]["label"])
        by_hypothesis[row["hypothesis_id"]].append(label)
        by_function[int(row["query_idx"])].append(label)
        by_family[row["family"]].append(label)
    h_counts = Counter(houtputs[item["signature_id"]]["mechanism_family"] for item in hypotheses)
    k_counts = Counter(koutputs[item["signature_id"]]["mechanism_family"] for item in signatures)
    family_breakdown = []
    for family in sorted(taxonomy["families"], key=lambda name: (-h_counts[name], name)):
        values = by_family[family]
        transferable = sum(label in {1, 2} for label in values)
        strict = sum(label == 2 for label in values)
        family_breakdown.append({
            "family": family,
            "hypothesis_count": h_counts[family],
            "hypothesis_rate": h_counts[family] / len(hypotheses),
            "knowledge_count": k_counts[family],
            "knowledge_rate": k_counts[family] / len(signatures),
            "retrieval_pair_count": len(values),
            "strict_match_count": strict,
            "strict_match_rate": strict / len(values) if values else None,
            "transferable_match_count": transferable,
            "transferable_match_rate": transferable / len(values) if values else None,
            "label_counts": {str(label): values.count(label) for label in (0, 1, 2)},
        })
    strict = label_counts[2]
    transfer = label_counts[1] + label_counts[2]
    result = {
        "taxonomy": args.taxonomy,
        "route": route,
        "query_field": ROUTES[route]["hypothesis_field"],
        "candidate_field": ROUTES[route]["knowledge_field"],
        "generation_mode": args.generation_mode,
        "pair_count": len(pairs),
        "hypothesis_count": len(hypotheses),
        "function_count": len({item["source_idx"] for item in hypotheses}),
        "strict_match_count": strict,
        "strict_match_rate": strict / len(pairs) if pairs else None,
        "transferable_match_count": transfer,
        "transferable_match_rate": transfer / len(pairs) if pairs else None,
        "strict_hypothesis_success_at_3": sum(any(label == 2 for label in by_hypothesis[item["signature_id"]]) for item in hypotheses) / len(hypotheses),
        "transferable_hypothesis_success_at_3": sum(any(label in {1, 2} for label in by_hypothesis[item["signature_id"]]) for item in hypotheses) / len(hypotheses),
        "strict_function_any_match": sum(any(label == 2 for label in values) for values in by_function.values()) / len(by_function),
        "transferable_function_any_match": sum(any(label in {1, 2} for label in values) for values in by_function.values()) / len(by_function),
        "label_counts": {str(label): label_counts[label] for label in (0, 1, 2)},
        "family_breakdown": family_breakdown,
    }
    route_dir = output_dir(args) / "retrieval" / route
    write_json(route_dir / "summary.json", result)
    lines = [
        f"# {route.title()} Route: Family-Filtered Dense Retrieval",
        "",
        "| Unit | Strict Mech. Match | Transferable Match |",
        "| --- | ---: | ---: |",
        f"| Hypothesis-candidate top-3 | {strict}/{len(pairs)} ({strict / len(pairs):.1%}) | {transfer}/{len(pairs)} ({transfer / len(pairs):.1%}) |",
        "",
        "The route field and route-specific family were jointly generated in one prompt. Labels are the original three-level audit protocol: strict=2; transferable=1 or 2.",
    ]
    (route_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def write_usage_summary(args: argparse.Namespace) -> None:
    usage = Counter()
    stages: dict[str, Any] = {}
    for path in sorted((output_dir(args) / "generation").glob("*.jsonl")):
        rows = read_jsonl(path)
        local = Counter()
        for row in rows:
            for attempt in row.get("attempts", []):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    local[key] += int(attempt.get("usage", {}).get(key) or 0)
        usage.update(local)
        stages[f"generation/{path.name}"] = dict(local)
    for path in (output_dir(args) / "audit" / "audit_assisted.jsonl",):
        if not path.exists():
            continue
        local = Counter()
        for row in read_jsonl(path):
            for attempt in row.get("attempts", []):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    local[key] += int(attempt.get("usage", {}).get(key) or 0)
        usage.update(local)
        stages["audit"] = dict(local)
    write_json(output_dir(args) / "usage_summary.json", {"total": dict(usage), "stages": stages})


def summarize(args: argparse.Namespace) -> None:
    labels = completed_audit_labels(args)
    results = {route: route_summary(args, route, labels) for route in args.routes}
    random_source = json.loads((source_dir(args) / "summary.json").read_text(encoding="utf-8"))
    random = random_source["taxonomies"][args.taxonomy]
    bm25_path = Path(args.bm25_summary)
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8")) if bm25_path.exists() else None
    comparison = {
        "taxonomy": args.taxonomy,
        "random_top3": {
            "strict_match_count": round(random["strict_pair_rate"] * random["pair_count"]),
            "strict_match_rate": random["strict_pair_rate"],
            "transferable_match_count": round(random["transfer_pair_rate"] * random["pair_count"]),
            "transferable_match_rate": random["transfer_pair_rate"],
            "pair_count": random["pair_count"],
        },
        "bm25_code_top3": bm25,
        "dense_routes": results,
        "warning": (
            "M/R/E use one shared jointly generated family in this joint-generation ablation."
            if args.generation_mode == "joint"
            else "M/R/E routes use route-specific jointly generated families; compare their routing distributions as well as retrieval scores."
        ),
    }
    write_json(output_dir(args) / "comparison.json", comparison)
    lines = [
        "# Frozen Taxonomy: Single-Route Retrieval Comparison",
        "",
        "| Retriever | Unit | Strict Mech. Match | Transferable Match |",
        "| --- | --- | ---: | ---: |",
        f"| Family Random | Hypothesis-candidate top-3 | {comparison['random_top3']['strict_match_count']}/{random['pair_count']} ({random['strict_pair_rate']:.1%}) | {comparison['random_top3']['transferable_match_count']}/{random['pair_count']} ({random['transfer_pair_rate']:.1%}) |",
    ]
    if bm25:
        lines.append(
            f"| Family BM25-Code | Hypothesis-candidate top-3 | {bm25['strict_match_count']}/{bm25['pair_count']} ({bm25['strict_match_rate']:.1%}) | {bm25['transferable_match_count']}/{bm25['pair_count']} ({bm25['transferable_match_rate']:.1%}) |"
        )
    for route, result in results.items():
        lines.append(
            f"| Family Dense-{route.title()} | Hypothesis-candidate top-3 | {result['strict_match_count']}/{result['pair_count']} ({result['strict_match_rate']:.1%}) | {result['transferable_match_count']}/{result['pair_count']} ({result['transferable_match_rate']:.1%}) |"
        )
    (output_dir(args) / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_usage_summary(args)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("command", choices=("prepare", "generate", "retrieve", "audit", "summarize", "run"))
    value.add_argument("--source-dir", default="data/taxonomy_other_absorbs_lifetime_error_protocol_15_dev")
    value.add_argument("--output-dir", default="data/taxonomy_other_absorbs_lifetime_error_protocol_15_route_retrieval_v1")
    value.add_argument("--taxonomy", default="allocation_state_representation_other_absorbs_lifetime_error_protocol_15")
    value.add_argument("--routes", nargs="+", choices=tuple(ROUTES), default=tuple(ROUTES))
    value.add_argument(
        "--generation-mode", choices=("independent", "joint"), default="independent",
        help="independent: one field/family call per route; joint: one M/R/E/shared-family call per item.",
    )
    value.add_argument(
        "--joint-prompt-profile",
        choices=("original", "definition_aligned_thinking"),
        default="original",
        help="Prompt formulation for joint generation; ignored by independent mode.",
    )
    value.add_argument("--train-path", default="data/raw/primevul_train_merged.jsonl")
    value.add_argument("--test-path", default="data/raw/primevul_test_merged.jsonl")
    value.add_argument("--clone-threshold", type=float, default=0.80)
    value.add_argument("--top-k", type=int, default=3)
    value.add_argument("--embedding-model-path", default="models/embedding/encoder")
    value.add_argument("--embedding-max-length", type=int, default=1024)
    value.add_argument("--embedding-batch-size", type=int, default=8)
    value.add_argument("--device", default="auto")
    value.add_argument("--api-base", default="http://localhost:8000/v1")
    value.add_argument("--api-key-env", default="LLM_API_KEY")
    value.add_argument("--model-name", default="your-model-id")
    value.add_argument(
        "--generation-thinking-type", choices=("disabled", "enabled"), default="disabled",
        help="Whether field/family generation uses the provider thinking channel.",
    )
    value.add_argument(
        "--generation-reasoning-effort", default="high",
        help="Reasoning-effort request for field/family generation.",
    )
    value.add_argument("--generation-max-new-tokens", type=int, default=512)
    value.add_argument(
        "--retry-max-new-tokens", type=int, default=0,
        help="For generate only, use this larger cap solely for missing items while reusing successes.",
    )
    value.add_argument("--workers", type=int, default=6)
    value.add_argument("--limit", type=int, default=0, help="Optional cap per route/kind generation batch; 0 means no cap.")
    value.add_argument("--audit-model-name", default="your-model-id")
    value.add_argument("--audit-max-new-tokens", type=int, default=1024)
    value.add_argument("--audit-batch-size", type=int, default=2)
    value.add_argument("--audit-workers", type=int, default=4)
    value.add_argument("--timeout", type=int, default=180)
    value.add_argument("--retries", type=int, default=3)
    value.add_argument("--bm25-summary", default="data/taxonomy_other_absorbs_lifetime_error_protocol_15_within_family_bm25_code/summary.json")
    return value


def main() -> None:
    args = parser().parse_args()
    if (
        args.workers < 1
        or args.audit_workers < 1
        or args.top_k < 1
        or args.retries < 1
        or args.retry_max_new_tokens < 0
    ):
        parser.error("invalid workers, top-k, or retries")
    if args.command in {"prepare", "run"}:
        prepare(args)
    if args.command in {"generate", "run"}:
        generate(args)
    if args.command in {"retrieve", "run"}:
        retrieve(args)
    if args.command in {"audit", "run"}:
        audit(args)
    if args.command in {"summarize", "run"}:
        summarize(args)


if __name__ == "__main__":
    main()
