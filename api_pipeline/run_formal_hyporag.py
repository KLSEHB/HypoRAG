#!/usr/bin/env python3
"""Resumable formal HypoRAG experiment with an OpenAI-compatible backend.

This runner keeps each experiment in its own output directory.  It has two independent
LLM chains: knowledge generation and test-side S0.  The later retrieval/S5
stages join their append-only outputs without regenerating successful work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import chromadb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.common import (  # noqa: E402
    build_client,
    call_chat_completion_with_usage,
    extract_json_object,
    render_messages_as_text,
)
from api_pipeline.retrieval_core import (  # noqa: E402
    encode_texts,
    load_embedding_model,
    load_reranker_model,
    rerank_candidates,
    resolve_device,
)
from api_pipeline.run_independent_code_retrieval_baselines import eligible_candidates  # noqa: E402
from api_pipeline.run_joint_high_point_validation import (  # noqa: E402
    GUIDANCE_FIELDS,
    guidance_messages,
    point_messages,
    validate_guidance,
    validate_point_output,
)
from api_pipeline.run_joint_mre_fusion_ablation import reranker_candidate, reranker_query  # noqa: E402
from api_pipeline.run_taxonomy_route_retrieval import (  # noqa: E402
    ROUTES,
    joint_route_messages,
    validate_joint_route_result,
)
from api_pipeline.run_thinking_joint_hypothesis_generation import (  # noqa: E402
    PROMPT_VERSIONS,
    joint_system,
    validate as validate_hypotheses,
)
from api_pipeline.repair_signature import signature_messages, signature_view  # noqa: E402


DEFAULT_TAXONOMY = "allocation_state_representation_other_absorbs_lifetime_error_protocol_15"
DEFAULT_OUTPUT = ROOT / "data" / "formal_hyporag"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def latest(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("key") or "")
        if key:
            result[key] = row
    return result


def successful(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: row for key, row in latest(rows).items() if row.get("status") == "success"}


def raw_records(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    result = []
    for row in rows:
        if not str(row.get("func_vuln") or "").strip() or not str(row.get("func_safe") or "").strip():
            raise RuntimeError(f"raw record {row.get('idx')} lacks a function side")
        result.append(row)
    return result


def excluded_train_ids(path: Path) -> set[int]:
    value = read_json(path)
    return {int(item["idx"]) for item in value.get("excluded_records", [])}


def taxonomy_definition(args: argparse.Namespace) -> dict[str, Any]:
    catalog = read_json(args.taxonomy_json)
    if args.taxonomy not in catalog:
        raise RuntimeError(f"taxonomy {args.taxonomy!r} absent from {args.taxonomy_json}")
    value = catalog[args.taxonomy]
    if not isinstance(value.get("families"), dict) or not value.get("precedence"):
        raise RuntimeError("invalid taxonomy definition")
    return value


class InputTooLong(RuntimeError):
    pass


class PromptBudgeter:
    """Request the largest legal completion, using a local tokenizer when available."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.tokenizer = None
        if args.tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_path, trust_remote_code=True, local_files_only=True
            )
        self.max_model_len = args.max_model_len
        self.safety_tokens = args.context_safety_tokens
        self.minimum = args.minimum_completion_tokens

    def input_tokens(self, messages: list[dict[str, str]]) -> int:
        try:
            if self.tokenizer is None:
                # LLM API does not expose its tokenizer locally. This is a
                # conservative estimate used only to avoid exceeding the context.
                return max(1, len(render_messages_as_text(messages)) // 4)
            tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "xhigh"},
            )
            return len(tokens)
        except Exception:
            return len(self.tokenizer(render_messages_as_text(messages), add_special_tokens=False)["input_ids"])

    def completion_budget(self, messages: list[dict[str, str]]) -> tuple[int, int]:
        prompt_tokens = self.input_tokens(messages)
        available = self.max_model_len - prompt_tokens - self.safety_tokens
        if available < self.minimum:
            raise InputTooLong(
                f"prompt uses {prompt_tokens} tokens; only {available} completion tokens remain in "
                f"max_model_len={self.max_model_len}"
            )
        return prompt_tokens, available


def task_usage(rows: Iterable[dict[str, Any]], successes_only: bool) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        if successes_only and row.get("status") != "success":
            continue
        for attempt in row.get("attempts") or []:
            if successes_only and attempt.get("status") != "success":
                continue
            usage = attempt.get("usage") or {}
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(field)
                if isinstance(value, (int, float)):
                    total[field] += int(value)
    return dict(total)


def validate_signature(value: Any, record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("signature is not a JSON object")
    if value.get("adjudicability") not in {"clear", "uncertain", "unrelated_patch"}:
        raise ValueError("invalid signature adjudicability")
    fields = (
        "operation", "risk_object", "trigger_condition", "violated_invariant", "failed_protection",
        "repair_action", "evidence_before", "evidence_after", "uncertainty", "reason",
    )
    result = {"adjudicability": value["adjudicability"]}
    for field in fields:
        text = value.get(field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"signature missing {field}")
        result[field] = text.strip()
    for field, source_field in (("evidence_before", "func_vuln"), ("evidence_after", "func_safe")):
        source = str(record[source_field])
        if result[field] not in source:
            raise ValueError(f"signature {field} is not an exact source excerpt")
    return result


def task_runner(
    args: argparse.Namespace,
    stage: str,
    path: Path,
    tasks: list[dict[str, Any]],
    validator: Callable[[Any, dict[str, Any]], Any],
) -> None:
    if args.pilot_count:
        tasks = tasks[:args.pilot_count]
    cache = latest(read_jsonl(path))
    pending: list[dict[str, Any]] = []
    for task in tasks:
        old = cache.get(task["key"])
        if old and old.get("input_hash") != task["input_hash"]:
            raise RuntimeError(f"stale {stage} output for {task['key']}; choose a new output directory")
        if old is None or (
            args.resume_failed
            and old.get("status") == "excluded"
            and old.get("exclusion_reason") != "input_too_long"
        ):
            pending.append(task)
    print(json.dumps({"stage": stage, "cached": len(cache), "pending": len(pending)}), flush=True)
    if not pending:
        return

    budgeter = PromptBudgeter(args)

    def worker(task: dict[str, Any]) -> dict[str, Any]:
        base = {
            **task,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking_type": "enabled",
            "reasoning_effort": "xhigh",
            "chat_template_family": args.chat_template_family,
            "attempts": [],
        }
        messages = task["messages"]
        try:
            prompt_tokens, max_tokens = budgeter.completion_budget(messages)
        except InputTooLong as exc:
            return {**base, "status": "excluded", "exclusion_reason": "input_too_long", "error": str(exc)}
        base["prompt_token_estimate"] = prompt_tokens
        base["max_new_tokens"] = max_tokens
        client = build_client(args.api_base, args.api_key).with_options(max_retries=0)
        try:
            for attempt in range(1, args.retries + 1):
                raw: str | None = None
                usage: dict[str, Any] | None = None
                try:
                    prompt_tokens, max_tokens = budgeter.completion_budget(messages)
                    raw, usage = call_chat_completion_with_usage(
                        client=client,
                        model_name=args.model_name,
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=max_tokens,
                        timeout=args.timeout,
                        reasoning_effort=args.reasoning_effort,
                        thinking_type="enabled",
                        chat_template_family=args.chat_template_family,
                        response_format={"type": "json_object"},
                    )
                    parsed = validator(extract_json_object(raw), task)
                    base["attempts"].append({"attempt": attempt, "status": "success", "raw_output": raw, "usage": usage})
                    return {**base, "status": "success", "parsed": parsed}
                except Exception as exc:
                    event: dict[str, Any] = {"attempt": attempt, "status": "failed", "error": str(exc)}
                    if raw is not None:
                        event["raw_output"] = raw
                    if usage is not None:
                        event["usage"] = usage
                    base["attempts"].append(event)
                    if attempt < args.retries:
                        messages = [*messages, {"role": "user", "content": (
                            "Return the complete required JSON object for the same input. "
                            "Keep reasoning private and do not emit prose, Markdown, or partial JSON."
                        )}]
                        base["prompt_token_estimate"] = budgeter.input_tokens(messages)
                        time.sleep(min(30.0, 2.0 ** attempt))
            return {**base, "status": "excluded", "exclusion_reason": "generation_or_schema_failure", "error": base["attempts"][-1]["error"]}
        finally:
            client.close()

    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = [executor.submit(worker, task) for task in pending]
        for completed, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_jsonl(path, row)
            print(json.dumps({
                "stage": stage, "completed": completed, "remaining": len(pending) - completed,
                "key": row["key"], "status": row["status"],
            }), flush=True)


def selected_train(args: argparse.Namespace) -> list[dict[str, Any]]:
    excluded = excluded_train_ids(args.exclusion_manifest)
    rows = [row for row in raw_records(args.train_path) if int(row["idx"]) not in excluded]
    if len(rows) != 3765:
        raise RuntimeError(f"expected 3765 train records after exclusions, found {len(rows)}")
    return rows


def selected_test(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = raw_records(args.test_path)
    if len(rows) != 433:
        raise RuntimeError(f"expected 433 test pairs, found {len(rows)}")
    return rows


def manifest_value(args: argparse.Namespace) -> dict[str, Any]:
    taxonomy = taxonomy_definition(args)
    train = selected_train(args)
    test = selected_test(args)
    signature_example = {
        "adjudicability": "clear",
        "operation": "placeholder operation",
        "risk_object": "placeholder object",
        "trigger_condition": "placeholder trigger",
        "violated_invariant": "placeholder invariant",
        "failed_protection": "placeholder protection",
        "repair_action": "placeholder repair",
        "evidence_before": "placeholder before",
        "evidence_after": "placeholder after",
        "uncertainty": "placeholder uncertainty",
        "reason": "placeholder reason",
    }
    route_messages, _ = joint_route_messages(
        {"signature_id": "placeholder", "kind": "knowledge", "neutral": signature_view(signature_example, "knowledge")},
        taxonomy,
        "definition_aligned_thinking",
    )
    guidance_prompt = guidance_messages({
        "candidate_signature_id": "placeholder", "candidate_signature": signature_example,
    })
    point_prompt = point_messages(
        {"project": "", "file_name": "", "func_vuln": "void placeholder(void) {}"},
        {
            "code_anchor": "placeholder", "mechanism_claim": "placeholder",
            "evidence_to_check": "placeholder", "repair_sought": "placeholder",
            "mechanism_family": "OTHER", "operation": "placeholder", "risk_object": "placeholder",
            "trigger_condition": "placeholder", "violated_invariant": "placeholder", "uncertainty": "placeholder",
        },
        [( {}, {field: "placeholder" for field in GUIDANCE_FIELDS})],
        args.verification_prompt_mode,
    )
    return {
        "experiment": "formal_hyporag",
        "selection_note": "Full 433-pair result includes the 50 functions used during development for taxonomy and prompt selection.",
        "train": {
            "total_raw_records": 3772,
            "excluded_records": read_json(args.exclusion_manifest)["excluded_records"],
            "usable_records": len(train),
            "records_sha256": stable_hash(train),
        },
        "test": {"pairs": len(test), "records_sha256": stable_hash(test)},
        "taxonomy": {"name": args.taxonomy, "definition": taxonomy, "sha256": stable_hash(taxonomy)},
        "model": {
            "name": args.model_name,
            "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
            "model_revision": args.model_revision,
            "vllm_version": args.vllm_version,
            "api_base": args.api_base,
            "dtype": "bfloat16",
            "thinking_type": "enabled",
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_model_len": args.max_model_len,
            "completion_budget": "largest legal per-request value after prompt length and safety margin",
        },
        "knowledge": {
            "signature_then": "separate retrieval package and guidance package",
            "retrieval_package": "shared family plus M/R/E",
            "guidance_fields": list(GUIDANCE_FIELDS),
        },
        "prompt_hashes": {
            "signature_system": stable_hash(signature_messages(train[0])[0]["content"]),
            "retrieval_package_system": stable_hash(route_messages[0]["content"]),
            "guidance_package_system": stable_hash(guidance_prompt[0]["content"]),
            "s0_system": stable_hash(joint_system(taxonomy, "focused", args.s0_prompt_version)),
            "s5_system": stable_hash(point_prompt[0]["content"]),
        },
        "online": {
            "s0_prompt_version": args.s0_prompt_version,
            "s0_analysis_profile": "focused",
            "max_hypotheses_per_function": 3,
            "dense_routes": list(ROUTES),
            "dense_top_k_per_route": args.dense_top_k,
            "reranker_model": str(args.reranker_model_path),
            "reranker_top_k": args.reranker_top_k,
            "point_validation_prompt": args.verification_prompt_mode,
            "aggregation": "deterministic_boolean_or",
        },
        "candidate_exclusion": {"same_project": True, "same_cve": True, "token_jaccard_gte": args.clone_threshold},
    }


def prepare(args: argparse.Namespace) -> None:
    value = manifest_value(args)
    target = args.output_dir / "experiment_manifest.json"
    if target.exists() and read_json(target) != value:
        raise RuntimeError("existing formal manifest differs; choose another output directory")
    write_json(target, value)
    write_json(args.output_dir / "taxonomy.json", value["taxonomy"])
    print(json.dumps({"stage": "prepare", "knowledge": 3765, "test_pairs": 433}), flush=True)


def signature_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "knowledge" / "signatures.jsonl"


def retrieval_knowledge_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "knowledge" / "retrieval_package.jsonl"


def guidance_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "knowledge" / "guidance_package.jsonl"


def stage_output_path(args: argparse.Namespace, stage: str, normal_path: Path) -> Path:
    if args.pilot_count:
        return args.output_dir / "pilots" / f"{stage}.jsonl"
    return normal_path


def generate_signatures(args: argparse.Namespace) -> None:
    tasks = []
    for record in selected_train(args):
        messages = signature_messages(record)
        tasks.append({
            "key": f"signature:{int(record['idx'])}", "idx": int(record["idx"]), "record": record,
            "messages": messages, "input_hash": stable_hash(messages),
        })
    task_runner(args, "knowledge_signatures", stage_output_path(args, "knowledge_signatures", signature_path(args)), tasks,
                lambda value, task: validate_signature(value, task["record"]))


def signature_outputs(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    rows = successful(read_jsonl(signature_path(args)))
    result = {int(row["idx"]): row for row in rows.values()}
    return result


def generate_retrieval_package(args: argparse.Namespace) -> None:
    taxonomy = taxonomy_definition(args)
    allowed = set(taxonomy["families"])
    signatures = signature_outputs(args)
    tasks = []
    for idx, row in sorted(signatures.items()):
        item = {"signature_id": f"knowledge:{idx}", "kind": "knowledge", "neutral": signature_view(row["parsed"], "knowledge")}
        messages, fields = joint_route_messages(item, taxonomy, "definition_aligned_thinking")
        tasks.append({
            "key": f"retrieval:{idx}", "idx": idx, "signature_id": item["signature_id"], "fields": fields,
            "messages": messages, "input_hash": stable_hash(messages),
        })
    task_runner(args, "knowledge_retrieval_package", stage_output_path(args, "knowledge_retrieval_package", retrieval_knowledge_path(args)), tasks,
                lambda value, task: validate_joint_route_result(value, allowed, task["fields"]))


def generate_guidance_package(args: argparse.Namespace) -> None:
    signatures = signature_outputs(args)
    tasks = []
    for idx, row in sorted(signatures.items()):
        candidate = {"candidate_signature_id": f"knowledge:{idx}", "candidate_signature": row["parsed"]}
        messages = guidance_messages(candidate)
        tasks.append({
            "key": f"guidance:{idx}", "idx": idx, "signature_id": candidate["candidate_signature_id"],
            "messages": messages, "input_hash": stable_hash(messages),
        })
    task_runner(args, "knowledge_guidance_package", stage_output_path(args, "knowledge_guidance_package", guidance_path(args)), tasks,
                lambda value, _task: validate_guidance(value))


def generate_s0(args: argparse.Namespace) -> None:
    taxonomy = taxonomy_definition(args)
    allowed = set(taxonomy["families"])
    system = joint_system(taxonomy, "focused", args.s0_prompt_version)
    tasks = []
    for record in selected_test(args):
        for side, field in (("vuln", "func_vuln"), ("safe", "func_safe")):
            code = str(record[field])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"idx": int(record["idx"]), "side": side, "function": code}, ensure_ascii=False)},
            ]
            tasks.append({
                "key": f"s0:{int(record['idx'])}:{side}", "idx": int(record["idx"]), "side": side,
                "code": code, "messages": messages, "input_hash": stable_hash(messages),
            })
    task_runner(args, "s0", stage_output_path(args, "s0", args.output_dir / "online" / "s0_hypotheses.jsonl"), tasks,
                lambda value, task: validate_hypotheses(value, task["code"], allowed, args.s0_prompt_version))


def generate_train_s0(args: argparse.Namespace) -> None:
    """Generate current-taxonomy hypotheses for preference-data queries."""
    taxonomy = taxonomy_definition(args)
    allowed = set(taxonomy["families"])
    system = joint_system(taxonomy, "focused", args.s0_prompt_version)
    tasks = []
    for record in selected_train(args):
        for side, field in (("vuln", "func_vuln"), ("safe", "func_safe")):
            code = str(record[field])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "idx": int(record["idx"]), "side": side, "function": code,
                }, ensure_ascii=False)},
            ]
            tasks.append({
                "key": f"train-s0:{int(record['idx'])}:{side}",
                "idx": int(record["idx"]), "side": side, "code": code,
                "messages": messages, "input_hash": stable_hash(messages),
            })
    task_runner(
        args, "train_s0", stage_output_path(
            args, "train_s0", args.output_dir / "reranker_distill" / "s0_train_hypotheses.jsonl",
        ),
        tasks,
        lambda value, task: validate_hypotheses(
            value, task["code"], allowed, args.s0_prompt_version,
        ),
    )


def knowledge_units(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    train_by_idx = {int(row["idx"]): row for row in selected_train(args)}
    signatures = signature_outputs(args)
    retrieval = successful(read_jsonl(retrieval_knowledge_path(args)))
    guidance = successful(read_jsonl(guidance_path(args)))
    result: dict[int, dict[str, Any]] = {}
    for idx, record in train_by_idx.items():
        signature = signatures.get(idx)
        r = retrieval.get(f"retrieval:{idx}")
        g = guidance.get(f"guidance:{idx}")
        if signature is None or r is None or g is None:
            continue
        result[idx] = {
            "idx": idx, "record": record, "signature": signature["parsed"],
            "mre": r["parsed"], "guidance": g["parsed"],
        }
    return result


def knowledge_fingerprint(knowledge: dict[int, dict[str, Any]]) -> str:
    """Fingerprint exactly the generated retrieval package selected for indexing."""
    return stable_hash([
        {
            "idx": idx,
            "family": item["mre"]["mechanism_family"],
            "mechanism": item["mre"]["mechanism_observed"],
            "repair": item["mre"]["repair_applied"],
            "evidence": item["mre"]["evidence_decisive"],
        }
        for idx, item in sorted(knowledge.items())
    ])


def index_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / "chromadb"


def index_manifest_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "knowledge" / "index_manifest.json"


def build_indices(args: argparse.Namespace) -> None:
    """Materialize and validate the three frozen M/R/E Chroma collections.

    Collection construction is incremental so an interrupted embedding job only
    adds missing identifiers. It never deletes an existing formal index.
    """
    knowledge = knowledge_units(args)
    if not knowledge:
        raise RuntimeError("no complete knowledge units are available for indexing")
    if len(knowledge) != 3765:
        print(json.dumps({
            "stage": "index", "warning": "knowledge coverage below target after explicit exclusions",
            "usable": len(knowledge), "target": 3765,
        }), flush=True)
    ids = [str(idx) for idx in sorted(knowledge)]
    fingerprint = knowledge_fingerprint(knowledge)
    directory = index_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(directory))
    device = resolve_device(args.device)
    tokenizer, model = load_embedding_model(args.embedding_model_path, device)
    route_fields = {
        "mechanism": "mechanism_observed",
        "repair": "repair_applied",
        "evidence": "evidence_decisive",
    }
    try:
        for route, field in route_fields.items():
            metadata = {
                "hnsw:space": "cosine",
                "knowledge_fingerprint": fingerprint,
                "route": route,
            }
            collection = client.get_or_create_collection(name=route, metadata=metadata)
            current_metadata = collection.metadata or {}
            if current_metadata.get("knowledge_fingerprint") not in {None, fingerprint}:
                raise RuntimeError(
                    f"existing {route} collection has a different knowledge fingerprint; "
                    "choose another formal output directory"
                )
            existing = set(collection.get(include=[])["ids"])
            unexpected = existing - set(ids)
            if unexpected:
                raise RuntimeError(f"{route} index contains unexpected identifiers: {sorted(unexpected)[:5]}")
            missing = [idx for idx in ids if idx not in existing]
            for start in range(0, len(missing), args.embedding_batch_size):
                batch_ids = missing[start:start + args.embedding_batch_size]
                batch = [knowledge[int(idx)] for idx in batch_ids]
                documents = [str(item["mre"][field]) for item in batch]
                embeddings = encode_texts(
                    tokenizer, model, documents, args.embedding_max_length, device
                )
                collection.add(
                    ids=batch_ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=[
                        {
                            "idx": int(idx),
                            "mechanism_family": str(item["mre"]["mechanism_family"]),
                        }
                        for idx, item in zip(batch_ids, batch)
                    ],
                )
                print(json.dumps({
                    "stage": "index", "route": route,
                    "indexed": min(start + len(batch_ids), len(missing)), "missing_total": len(missing),
                }), flush=True)
            if collection.count() != len(ids):
                raise RuntimeError(f"{route} collection count {collection.count()} != {len(ids)}")
    finally:
        del model
    value = {
        "index_dir": str(directory),
        "knowledge_count": len(ids),
        "knowledge_fingerprint": fingerprint,
        "embedding_model_path": str(args.embedding_model_path),
        "embedding_max_length": args.embedding_max_length,
        "collections": {route: client.get_collection(route).count() for route in route_fields},
    }
    write_json(index_manifest_path(args), value)
    print(json.dumps({"stage": "index", **value}), flush=True)


def load_index_vectors(
    args: argparse.Namespace, knowledge: dict[int, dict[str, Any]]
) -> dict[str, np.ndarray]:
    manifest = read_json(index_manifest_path(args))
    expected = knowledge_fingerprint(knowledge)
    if manifest.get("knowledge_fingerprint") != expected:
        raise RuntimeError("Chroma index knowledge fingerprint does not match generated knowledge")
    ids = [str(idx) for idx in sorted(knowledge)]
    client = chromadb.PersistentClient(path=str(index_dir(args)))
    vectors: dict[str, np.ndarray] = {}
    for route in ROUTES:
        collection = client.get_collection(route)
        if collection.count() != len(ids):
            raise RuntimeError(f"Chroma {route} count does not match usable knowledge")
        payload = collection.get(ids=ids, include=["embeddings"])
        found = {
            str(identifier): embedding
            for identifier, embedding in zip(payload["ids"], payload["embeddings"])
        }
        if set(found) != set(ids):
            raise RuntimeError(f"Chroma {route} does not contain the expected knowledge IDs")
        vectors[route] = np.asarray([found[idx] for idx in ids], dtype=np.float32)
    return vectors


def s0_outputs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return successful(read_jsonl(args.output_dir / "online" / "s0_hypotheses.jsonl"))


def encode_all(tokenizer: Any, model: Any, texts: list[str], args: argparse.Namespace, device: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), args.embedding_batch_size):
        vectors.extend(encode_texts(tokenizer, model, texts[start:start + args.embedding_batch_size], args.embedding_max_length, device))
    return np.asarray(vectors, dtype=np.float32)


def retrieve_and_rerank(args: argparse.Namespace) -> None:
    output = args.output_dir / "online" / "reranked_top3.jsonl"
    if output.exists():
        print(json.dumps({"stage": "retrieve_rerank", "status": "already_materialized"}), flush=True)
        return
    knowledge = knowledge_units(args)
    if not knowledge:
        raise RuntimeError("knowledge is empty")
    s0 = s0_outputs(args)
    tests = selected_test(args)
    candidates = [knowledge[idx] for idx in sorted(knowledge)]
    device = resolve_device(args.device)
    embed_tokenizer, embed_model = load_embedding_model(args.embedding_model_path, device)
    route_rows: list[dict[str, Any]] = []
    union: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    try:
        candidate_vectors = load_index_vectors(args, knowledge)
        test_by_idx = {int(record["idx"]): record for record in tests}
        for key, s0_row in sorted(s0.items()):
            record = test_by_idx[int(s0_row["idx"])]
            query_record = {**record, "func_vuln": s0_row["code"]}
            eligible = eligible_candidates(query_record, [item["record"] for item in candidates], args.clone_threshold)
            for point in s0_row["parsed"]["hypotheses"]:
                hypothesis_id = f"test:{int(s0_row['idx'])}:{s0_row['side']}:h{int(point['id'])}"
                family = point["mechanism_family"]
                for route, config in ROUTES.items():
                    query_vector = encode_all(embed_tokenizer, embed_model, [point[config["hypothesis_field"]]], args, device)[0]
                    pool = eligible if family == "OTHER" else [
                        pos for pos in eligible if candidates[pos]["mre"]["mechanism_family"] == family
                    ]
                    fallback = not pool
                    if fallback:
                        pool = eligible
                    if not pool:
                        append_jsonl(args.output_dir / "online" / "retrieval_exclusions.jsonl", {
                            "hypothesis_id": hypothesis_id, "idx": int(s0_row["idx"]), "side": s0_row["side"],
                            "status": "excluded", "reason": "no_eligible_candidates", "route": route,
                        })
                        continue
                    scores = candidate_vectors[route] @ query_vector
                    selected = sorted(pool, key=lambda pos: (-float(scores[pos]), int(candidates[pos]["idx"])))[:args.dense_top_k]
                    for rank, pos in enumerate(selected, 1):
                        candidate = candidates[pos]
                        row = {
                            "hypothesis_id": hypothesis_id, "idx": int(s0_row["idx"]), "side": s0_row["side"],
                            "point": point, "family": family, "route": route, "route_rank": rank,
                            "candidate_idx": candidate["idx"], "candidate_signature_id": f"knowledge:{candidate['idx']}",
                            "candidate_mre": candidate["mre"], "candidate_guidance": candidate["guidance"],
                            "dense_score": float(scores[pos]), "pool_size": len(pool), "eligible_global": len(eligible),
                            "family_filter_fallback": fallback,
                        }
                        route_rows.append(row)
                        existing = union[hypothesis_id].get(candidate["idx"])
                        if existing is None:
                            row["route_hits"] = [{"route": route, "rank": rank, "score": float(scores[pos])}]
                            union[hypothesis_id][candidate["idx"]] = row
                        else:
                            existing["route_hits"].append({"route": route, "rank": rank, "score": float(scores[pos])})
    finally:
        del embed_model

    rerank_tokenizer, rerank_model = load_reranker_model(args.reranker_model_path, device)
    reranked: list[dict[str, Any]] = []
    try:
        for hypothesis_id, choices_by_idx in sorted(union.items()):
            choices = list(choices_by_idx.values())
            point = choices[0]["point"]
            scores = rerank_candidates(
                rerank_tokenizer, rerank_model, reranker_query(point),
                [reranker_candidate(choice["candidate_mre"]) for choice in choices],
                args.reranker_max_length, args.reranker_batch_size, device,
            )
            ordered = sorted(zip(choices, scores), key=lambda item: (-float(item[1]), int(item[0]["candidate_idx"])))
            for rank, (choice, score) in enumerate(ordered[:args.reranker_top_k], 1):
                reranked.append({**choice, "rerank_rank": rank, "reranker_score": float(score)})
    finally:
        del rerank_model
    write_jsonl(args.output_dir / "online" / "route_top10.jsonl", route_rows)
    write_jsonl(output, reranked)
    write_json(args.output_dir / "online" / "retrieval_summary.json", {
        "dense_top_k_per_route": args.dense_top_k, "route_pairs": len(route_rows),
        "deduplicated_hypotheses": len(union), "reranked_pairs": len(reranked),
        "reranker_top_k": args.reranker_top_k,
    })
    print(json.dumps({"stage": "retrieve_rerank", "route_pairs": len(route_rows), "reranked_pairs": len(reranked), "hypotheses": len(union)}), flush=True)


def selected_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.output_dir / "online" / "reranked_top3.jsonl"):
        grouped[str(row["hypothesis_id"])].append(row)
    for value in grouped.values():
        value.sort(key=lambda row: int(row["rerank_rank"]))
    return dict(grouped)


def verify_points(args: argparse.Namespace) -> None:
    tests = {int(row["idx"]): row for row in selected_test(args)}
    s0 = s0_outputs(args)
    selected = selected_candidates(args)
    tasks = []
    for hypothesis_id, candidates in sorted(selected.items()):
        first = candidates[0]
        raw = {**tests[int(first["idx"])], "func_vuln": s0[f"s0:{int(first['idx'])}:{first['side']}"]["code"]}
        messages = point_messages(raw, first["point"], [(row, row["candidate_guidance"]) for row in candidates], args.verification_prompt_mode)
        tasks.append({
            "key": f"s5:{hypothesis_id}", "hypothesis_id": hypothesis_id, "idx": int(first["idx"]), "side": first["side"],
            "point": first["point"], "selected_candidate_count": len(candidates),
            "messages": messages, "input_hash": stable_hash(messages),
        })
    task_runner(args, "s5", stage_output_path(args, "s5", args.output_dir / "online" / "s5_point_verification.jsonl"), tasks,
                lambda value, task: validate_point_output(
                    value, int(task["point"]["id"]),
                    expected_example_ranks=list(range(1, int(task["selected_candidate_count"]) + 1)),
                    require_case_mappings=True,
                ))


def aggregate(args: argparse.Namespace) -> None:
    s0 = s0_outputs(args)
    s5 = successful(read_jsonl(args.output_dir / "online" / "s5_point_verification.jsonl"))
    selected = selected_candidates(args)
    records: list[dict[str, Any]] = []
    for record in selected_test(args):
        idx = int(record["idx"])
        pair = {"idx": idx, "status": "success", "sides": {}}
        for side in ("vuln", "safe"):
            generated = s0.get(f"s0:{idx}:{side}")
            if generated is None:
                pair["status"] = "excluded"
                pair["sides"][side] = {"reason": "s0_missing_or_excluded"}
                continue
            points = generated["parsed"]["hypotheses"]
            verified = []
            missing = []
            for point in points:
                hypothesis_id = f"test:{idx}:{side}:h{int(point['id'])}"
                if hypothesis_id not in selected:
                    missing.append(hypothesis_id)
                    continue
                row = s5.get(f"s5:{hypothesis_id}")
                if row is None:
                    missing.append(hypothesis_id)
                    continue
                verified.append(row)
            if missing:
                pair["status"] = "excluded"
                pair["sides"][side] = {"reason": "retrieval_or_s5_incomplete", "missing_hypotheses": missing}
                continue
            supported = [row["key"] for row in verified if row["parsed"].get("verdict") == "supported"]
            pair["sides"][side] = {
                "final_label": "vulnerable" if supported else "non_vulnerable",
                "hypothesis_count": len(points), "supported_hypotheses": supported,
            }
        if pair["status"] == "success":
            vuln = pair["sides"]["vuln"]["final_label"]
            safe = pair["sides"]["safe"]["final_label"]
            pair["pair_outcome"] = (
                "Both-R" if vuln == "vulnerable" and safe == "non_vulnerable" else
                "Both-W" if vuln == "non_vulnerable" and safe == "vulnerable" else
                "Both-s" if vuln == "non_vulnerable" else "Both-v"
            )
        records.append(pair)
    write_jsonl(args.output_dir / "online" / "deterministic_or_pairs.jsonl", records)
    print(json.dumps({"stage": "aggregate", "valid": sum(row["status"] == "success" for row in records), "excluded": sum(row["status"] != "success" for row in records)}), flush=True)


def stage_snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current = list(latest(rows).values())
    exclusions = Counter(
        str(row.get("exclusion_reason") or "unknown")
        for row in current if row.get("status") == "excluded"
    )
    return {
        "latest_records": len(current),
        "success": sum(row.get("status") == "success" for row in current),
        "excluded": sum(row.get("status") == "excluded" for row in current),
        "exclusion_reasons": dict(exclusions),
    }


def validate_formal(args: argparse.Namespace) -> None:
    """Write explicit coverage/consistency checks without hiding exclusions."""
    knowledge = knowledge_units(args)
    status_rows = {
        "knowledge_signatures": read_jsonl(signature_path(args)),
        "knowledge_retrieval_package": read_jsonl(retrieval_knowledge_path(args)),
        "knowledge_guidance_package": read_jsonl(guidance_path(args)),
        "s0": read_jsonl(args.output_dir / "online" / "s0_hypotheses.jsonl"),
        "s5": read_jsonl(args.output_dir / "online" / "s5_point_verification.jsonl"),
    }
    checks: list[str] = []
    if len(knowledge) != len(set(knowledge)):
        checks.append("knowledge_idx_not_unique")
    if not index_manifest_path(args).exists():
        checks.append("index_manifest_missing")
        index_value: dict[str, Any] = {}
    else:
        index_value = read_json(index_manifest_path(args))
        counts = index_value.get("collections") or {}
        if any(int(counts.get(route, -1)) != len(knowledge) for route in ROUTES):
            checks.append("chroma_collection_count_mismatch")
        if index_value.get("knowledge_fingerprint") != knowledge_fingerprint(knowledge):
            checks.append("chroma_knowledge_fingerprint_mismatch")
    expected_s0 = 2 * len(selected_test(args))
    s0_success = successful(status_rows["s0"])
    if len(s0_success) > expected_s0:
        checks.append("s0_count_exceeds_test_sides")
    route_rows = read_jsonl(args.output_dir / "online" / "route_top10.jsonl")
    route_counts: Counter[tuple[str, str]] = Counter(
        (str(row["hypothesis_id"]), str(row["route"])) for row in route_rows
    )
    reranked = read_jsonl(args.output_dir / "online" / "reranked_top3.jsonl")
    rerank_counts: Counter[str] = Counter(str(row["hypothesis_id"]) for row in reranked)
    if any(count > args.dense_top_k for count in route_counts.values()):
        checks.append("route_top_k_exceeded")
    if any(count > args.reranker_top_k for count in rerank_counts.values()):
        checks.append("reranker_top_k_exceeded")
    pairs = read_jsonl(args.output_dir / "online" / "deterministic_or_pairs.jsonl")
    if pairs and len(pairs) != 433:
        checks.append("pair_output_count_mismatch")
    output = {
        "knowledge_target": 3765,
        "knowledge_complete_units": len(knowledge),
        "knowledge_excluded_from_target": 3765 - len(knowledge),
        "chroma": index_value,
        "expected_s0_sides": expected_s0,
        "route_top_k": args.dense_top_k,
        "reranker_top_k": args.reranker_top_k,
        "route_rows": len(route_rows),
        "reranked_rows": len(reranked),
        "pair_records": len(pairs),
        "pair_valid": sum(row.get("status") == "success" for row in pairs),
        "pair_excluded": sum(row.get("status") != "success" for row in pairs),
        "stage_status": {name: stage_snapshot(rows) for name, rows in status_rows.items()},
        "validation_errors": checks,
        "validated": not checks,
    }
    write_json(args.output_dir / "formal_validation.json", output)
    print(json.dumps({"stage": "validate", "validated": not checks, "errors": checks}), flush=True)


def summarize(args: argparse.Namespace) -> None:
    pairs = read_jsonl(args.output_dir / "online" / "deterministic_or_pairs.jsonl")
    valid = [row for row in pairs if row.get("status") == "success"]
    outcomes: Counter[str] = Counter(row["pair_outcome"] for row in valid)
    tp = outcomes["Both-R"] + outcomes["Both-v"]
    fp = outcomes["Both-W"] + outcomes["Both-v"]
    tn = outcomes["Both-R"] + outcomes["Both-s"]
    fn = outcomes["Both-W"] + outcomes["Both-s"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    stages = {
        "knowledge_signatures": read_jsonl(signature_path(args)),
        "knowledge_retrieval_package": read_jsonl(retrieval_knowledge_path(args)),
        "knowledge_guidance_package": read_jsonl(guidance_path(args)),
        "s0": read_jsonl(args.output_dir / "online" / "s0_hypotheses.jsonl"),
        "s5": read_jsonl(args.output_dir / "online" / "s5_point_verification.jsonl"),
    }
    total_all: Counter[str] = Counter()
    total_success: Counter[str] = Counter()
    for rows in stages.values():
        total_all.update(task_usage(rows, successes_only=False))
        total_success.update(task_usage(rows, successes_only=True))
    n = len(valid)
    summary = {
        "configuration": read_json(args.output_dir / "experiment_manifest.json"),
        "coverage": {"requested_pairs": 433, "valid_pairs": n, "excluded_pairs": len(pairs) - n, "complete": n == 433},
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "ACC": (tp + tn) / (2 * n) if n else 0.0,
        "PRECISION": precision,
        "RECALL": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "pair_acc": outcomes["Both-R"] / n if n else 0.0,
        "NPDS": (outcomes["Both-R"] - outcomes["Both-W"]) / n if n else 0.0,
        "pair_outcomes": {name: {"count": outcomes[name], "rate": outcomes[name] / n if n else 0.0} for name in ("Both-R", "Both-W", "Both-s", "Both-v")},
        "all_pred_non_vulnerable_rate": outcomes["Both-s"] / n if n else 0.0,
        "all_pred_vulnerable_rate": outcomes["Both-v"] / n if n else 0.0,
        "usage": {"all_attempts": dict(total_all), "successful_attempts_only": dict(total_success)},
        "stage_status": {name: stage_snapshot(rows) for name, rows in stages.items()},
        "formal_validation": read_json(args.output_dir / "formal_validation.json") if (args.output_dir / "formal_validation.json").exists() else None,
    }
    write_json(args.output_dir / "final_results.json", summary)
    lines = [
        "# Local LLM Formal HypoRAG", "",
        "Selected-configuration full evaluation; the 433-pair set retains the 50 development functions.",
        f"coverage: {n}/433", f"Both-R/W/s/v: {outcomes['Both-R']}/{outcomes['Both-W']}/{outcomes['Both-s']}/{outcomes['Both-v']}",
        f"NPDS: {summary['NPDS']:.4f}",
        f"ACC / Precision / Recall / F1: {summary['ACC']:.4f} / {precision:.4f} / {recall:.4f} / {summary['F1']:.4f}",
        f"token usage: {dict(total_all)}",
    ]
    (args.output_dir / "final_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "summarize", "valid_pairs": n, "NPDS": summary["NPDS"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "signatures", "retrieval-package", "guidance-package", "index", "s0", "train-s0", "retrieve", "s5", "aggregate", "validate", "summarize", "run"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "raw" / "primevul_train_merged.jsonl")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "raw" / "primevul_test_merged.jsonl")
    parser.add_argument("--exclusion-manifest", type=Path, default=ROOT / "config" / "excluded_records.json")
    parser.add_argument("--taxonomy-json", type=Path, default=ROOT / "config" / "taxonomy_catalog.json")
    parser.add_argument("--taxonomy", default=DEFAULT_TAXONOMY)
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--model-name", default="your-model-id")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--model-revision", default="unknown")
    parser.add_argument("--vllm-version", default="unknown")
    parser.add_argument("--max-model-len", type=int, default=128000)
    parser.add_argument("--context-safety-tokens", type=int, default=256)
    parser.add_argument("--minimum-completion-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reasoning-effort", choices=("high", "max", "xhigh"), default="high")
    parser.add_argument("--chat-template-family", choices=("default", "template_kwargs"), default="default")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume-failed", action="store_true", help="retry prior non-length exclusions")
    parser.add_argument("--pilot-count", type=int, default=0, help="run a no-formal-output request preflight")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--s0-prompt-version", choices=PROMPT_VERSIONS, default="recall_v2")
    parser.add_argument("--embedding-model-path", type=Path, default=ROOT / "models" / "embedding" / "encoder")
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--clone-threshold", type=float, default=0.80)
    parser.add_argument("--reranker-model-path", type=Path, default=ROOT / "models" / "reranker" / "tuned" / "cross-encoder")
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-top-k", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verification-prompt-mode", default="case-mapped-local-arithmetic", choices=("baseline", "case-mapped", "case-mapped-visible-evidence", "case-mapped-visible-cardinality", "case-mapped-local-arithmetic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[tuple[str, Callable[[argparse.Namespace], None]]] = [
        ("prepare", prepare), ("signatures", generate_signatures), ("retrieval-package", generate_retrieval_package),
        ("guidance-package", generate_guidance_package), ("index", build_indices), ("s0", generate_s0), ("train-s0", generate_train_s0), ("retrieve", retrieve_and_rerank),
        ("s5", verify_points), ("aggregate", aggregate), ("validate", validate_formal), ("summarize", summarize),
    ]
    for name, function in steps:
        if args.command == name or (args.command == "run" and name != "train-s0"):
            function(args)


if __name__ == "__main__":
    main()
