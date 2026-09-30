"""Build a reproducible four-way code retrieval comparison for PrimeVul.

The experiment is deliberately independent of the RAG pipeline.  Queries are
test-set vulnerable functions, while the candidate corpus contains only
training-set vulnerable functions.  No labels or repair-side code are used by
any retrieval method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_pipeline.common import (  # noqa: E402
    build_api_url,
    build_client,
    call_chat_completion_with_usage,
    load_api_key,
)
from api_pipeline.retrieval_core import (  # noqa: E402
    encode_texts,
    load_embedding_model,
    resolve_device,
)


METHODS = ("random", "bm25_code", "dense_code", "dense_summary")
SEMANTIC_SYSTEM_PROMPT = """You create retrieval representations for source code.
Read exactly one C or C++ function and write a concise English semantic summary.
Describe: function purpose; important inputs, state, and data objects; key control or
data-flow operations; memory/resource/API-sensitive operations; visible validation or
error handling; and any risk-relevant mechanism that is directly apparent from code.
Do not classify the function as vulnerable or safe. Do not mention CVE, CWE, patches,
datasets, projects, labels, or external facts. Do not propose fixes. Return plain text
only, at most 220 words.
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_code(value: Any) -> str:
    return str(value or "").strip()


def record_key(split: str, idx: int) -> str:
    return f"{split}:{idx}"


def code_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["idx"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["code"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compact_record(raw: Dict[str, Any], split: str) -> Dict[str, Any]:
    idx = int(raw["idx"])
    code = normalize_code(raw.get("func_vuln"))
    return {
        "record_key": record_key(split, idx),
        "split": split,
        "idx": idx,
        "project": str(raw.get("project") or ""),
        "file_name": str(raw.get("file_name") or ""),
        "code": code,
    }


def load_dataset(path: Path, split: str) -> List[Dict[str, Any]]:
    seen: set[int] = set()
    result: List[Dict[str, Any]] = []
    for raw in read_jsonl(path):
        record = compact_record(raw, split)
        if not record["code"]:
            continue
        if record["idx"] in seen:
            raise ValueError(f"Duplicate {split} idx: {record['idx']}")
        seen.add(record["idx"])
        result.append(record)
    return result


def source_manifest(train: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "train_candidate_count": len(train),
        "test_eligible_count": len(test),
        "train_corpus_sha256": code_fingerprint(train),
        "test_source_sha256": code_fingerprint(test),
    }


def create_query_manifest(args: argparse.Namespace) -> List[Dict[str, Any]]:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "query_manifest.jsonl"
    experiment_path = output_dir / "experiment_manifest.json"
    train = load_dataset(Path(args.train_path), "train")
    test = load_dataset(Path(args.test_path), "test")
    if len(test) < args.query_count:
        raise ValueError(f"Only {len(test)} eligible test functions for {args.query_count} queries")
    source = source_manifest(train, test)

    if manifest_path.exists() and not args.overwrite:
        queries = read_jsonl(manifest_path)
        existing = json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
        required = {
            "seed": args.seed,
            "query_count": args.query_count,
            "train_corpus_sha256": source["train_corpus_sha256"],
            "test_source_sha256": source["test_source_sha256"],
        }
        if all(existing.get(key) == value for key, value in required.items()) and len(queries) == args.query_count:
            return queries
        raise RuntimeError("Existing query manifest does not match this experiment; use --overwrite")

    rng = random.Random(args.seed)
    sampled = rng.sample(test, args.query_count)
    queries = [
        {
            **record,
            "selection_order": position,
            "selection_seed": args.seed,
            "query_manifest_version": 1,
        }
        for position, record in enumerate(sampled, start=1)
    ]
    write_jsonl(manifest_path, queries)
    write_json(
        experiment_path,
        {
            "experiment": "primevul_code_retrieval_comparison",
            "query_manifest_version": 1,
            "seed": args.seed,
            "query_count": args.query_count,
            "source": source,
            "candidate_definition": "all non-empty training-set func_vuln functions",
            "query_definition": "random sample of non-empty test-set func_vuln functions",
            "methods": list(METHODS),
        },
    )
    return queries


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CODE_TOKEN = re.compile(
    r"0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?|"
    r"(?:==|!=|<=|>=|->|\+\+|--|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=)|"
    r"[A-Za-z_][A-Za-z0-9_]*"
)
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def code_terms(code: str) -> List[str]:
    terms: List[str] = []
    for token in CODE_TOKEN.findall(code):
        lower = token.lower()
        terms.append(lower)
        if IDENTIFIER.fullmatch(token):
            for piece in CAMEL_BOUNDARY.sub(" ", token).replace("_", " ").split():
                piece = piece.lower()
                if piece and piece != lower:
                    terms.append(piece)
    return terms


class CodeBm25:
    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(code_terms(code)) for code in documents]
        self.lengths = [sum(counter.values()) for counter in self.doc_terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        document_frequency: Counter[str] = Counter()
        for counter in self.doc_terms:
            document_frequency.update(counter.keys())
        total = len(self.doc_terms)
        self.idf = {
            term: math.log(1.0 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def best(self, query: str) -> tuple[int, float]:
        query_counts = Counter(code_terms(query))
        scores = [0.0] * len(self.doc_terms)
        for term, query_frequency in query_counts.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, term_counts in enumerate(self.doc_terms):
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                scores[index] += query_frequency * idf * frequency * (self.k1 + 1.0) / denominator
        best_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
        return best_index, float(scores[best_index])


def derived_seed(seed: int, query_key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{query_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def result_row(
    method: str,
    query: Dict[str, Any],
    candidate: Dict[str, Any],
    score: float | None,
    score_type: str,
    experiment: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method,
        "query_idx": query["idx"],
        "candidate_idx": candidate["idx"],
        "rank": 1,
        "score": score,
        "score_type": score_type,
        "query": {key: query[key] for key in ("record_key", "split", "idx", "project", "file_name", "code")},
        "candidate": {key: candidate[key] for key in ("record_key", "split", "idx", "project", "file_name", "code")},
        "selection_seed": experiment["seed"],
        "query_manifest_version": experiment["query_manifest_version"],
        "train_corpus_sha256": experiment["source"]["train_corpus_sha256"],
    }
    if extra:
        row.update(extra)
    return row


def write_method_results(output_dir: Path, method: str, rows: Sequence[Dict[str, Any]]) -> None:
    if len(rows) != 100:
        raise ValueError(f"Expected 100 {method} rows, got {len(rows)}")
    write_jsonl(output_dir / f"{method}_results.jsonl", rows)


def run_local(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    queries = create_query_manifest(args)
    experiment = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    candidates = load_dataset(Path(args.train_path), "train")
    if len(candidates) != experiment["source"]["train_candidate_count"]:
        raise RuntimeError("Training candidate corpus changed after query manifest creation")

    random_rows: List[Dict[str, Any]] = []
    for query in queries:
        sampled = candidates[derived_seed(args.seed, query["record_key"]) % len(candidates)]
        random_rows.append(
            result_row(
                "random", query, sampled, None, "uniform_random", experiment,
                {"derived_selection_seed": derived_seed(args.seed, query["record_key"])},
            )
        )
    write_method_results(output_dir, "random", random_rows)

    bm25 = CodeBm25([candidate["code"] for candidate in candidates])
    bm25_rows = []
    for query in queries:
        index, score = bm25.best(query["code"])
        bm25_rows.append(result_row("bm25_code", query, candidates[index], score, "bm25", experiment))
    write_method_results(output_dir, "bm25_code", bm25_rows)

    device = resolve_device(args.device)
    tokenizer, model = load_embedding_model(Path(args.embedding_model_path), device)
    all_records = [*candidates, *queries]
    token_lengths: Dict[str, int] = {}
    for record in all_records:
        token_lengths[record["record_key"]] = len(tokenizer(record["code"], add_special_tokens=True)["input_ids"])
    embeddings: Dict[str, List[float]] = {}
    for start in range(0, len(all_records), args.embedding_batch_size):
        batch = all_records[start : start + args.embedding_batch_size]
        vectors = encode_texts(
            tokenizer, model, [record["code"] for record in batch], args.embedding_max_length, device
        )
        embeddings.update({record["record_key"]: vector for record, vector in zip(batch, vectors)})
        print(f"[INFO] Encoded raw code {min(start + len(batch), len(all_records))}/{len(all_records)}")
    candidate_matrix = torch.tensor([embeddings[row["record_key"]] for row in candidates], dtype=torch.float32)
    bge_rows = []
    for query in queries:
        scores = candidate_matrix @ torch.tensor(embeddings[query["record_key"]], dtype=torch.float32)
        index = int(torch.argmax(scores).item())
        bge_rows.append(
            result_row(
                "dense_code", query, candidates[index], float(scores[index].item()), "cosine_similarity", experiment,
                {
                    "embedding_model_path": str(Path(args.embedding_model_path)),
                    "embedding_max_length": args.embedding_max_length,
                    "query_code_truncated": token_lengths[query["record_key"]] > args.embedding_max_length,
                    "candidate_code_truncated": token_lengths[candidates[index]["record_key"]] > args.embedding_max_length,
                },
            )
        )
    write_method_results(output_dir, "dense_code", bge_rows)
    write_json(
        output_dir / "dense_code_metadata.json",
        {
            "embedding_model_path": str(Path(args.embedding_model_path)),
            "embedding_dimension": len(next(iter(embeddings.values()))),
            "embedding_max_length": args.embedding_max_length,
            "embedding_batch_size": args.embedding_batch_size,
            "device": device,
            "query_truncated_count": sum(token_lengths[q["record_key"]] > args.embedding_max_length for q in queries),
            "candidate_truncated_count": sum(token_lengths[c["record_key"]] > args.embedding_max_length for c in candidates),
        },
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    semantic_inputs = []
    for role, records in (("candidate", candidates), ("query", queries)):
        for record in records:
            semantic_inputs.append(
                {
                    "record_key": record["record_key"],
                    "split": record["split"],
                    "idx": record["idx"],
                    "role": role,
                    "code": record["code"],
                }
            )
    write_jsonl(output_dir / "semantic_inputs.jsonl", semantic_inputs)
    print(f"[INFO] Local methods complete. Semantic inputs: {len(semantic_inputs)}")


def prepare_semantic_inputs(args: argparse.Namespace) -> None:
    """Materialize server-side LLM inputs from an already frozen query manifest."""
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "query_manifest.jsonl"
    experiment_path = output_dir / "experiment_manifest.json"
    if not manifest_path.exists() or not experiment_path.exists():
        raise FileNotFoundError("Run the local command once to freeze query_manifest.jsonl")
    queries = read_jsonl(manifest_path)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    candidates = load_dataset(Path(args.train_path), "train")
    if code_fingerprint(candidates) != experiment["source"]["train_corpus_sha256"]:
        raise RuntimeError("Training candidate corpus does not match the frozen query manifest")
    semantic_inputs = []
    for role, records in (("candidate", candidates), ("query", queries)):
        for record in records:
            semantic_inputs.append(
                {
                    "record_key": record["record_key"],
                    "split": record["split"],
                    "idx": record["idx"],
                    "role": role,
                    "code": record["code"],
                }
            )
    write_jsonl(output_dir / "semantic_inputs.jsonl", semantic_inputs)
    print(f"[INFO] Materialized semantic inputs: {len(semantic_inputs)}")


def truncate_for_semantic(code: str, max_input_chars: int) -> tuple[str, bool]:
    if len(code) <= max_input_chars:
        return code, False
    front = max_input_chars * 2 // 3
    back = max_input_chars - front
    return (
        code[:front]
        + "\n\n/* [middle of function omitted only because of model context limit] */\n\n"
        + code[-back:],
        True,
    )


def latest_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    return {str(row[key]): row for row in rows}


def call_semantic_item(item: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    code, truncated = truncate_for_semantic(item["code"], args.semantic_max_input_chars)
    messages = [
        {"role": "system", "content": SEMANTIC_SYSTEM_PROMPT},
        {"role": "user", "content": "```c\n" + code + "\n```"},
    ]
    raw_outputs: List[str] = []
    last_error: Exception | None = None
    client = build_client(build_api_url(args.api_base), load_api_key(args.api_key, args.api_key_env))
    for attempt in range(1, args.retries + 1):
        try:
            raw, usage = call_chat_completion_with_usage(
                client=client,
                model_name=args.model_name,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.semantic_max_new_tokens,
                timeout=args.timeout,
                reasoning_effort=args.reasoning_effort,
                thinking_type="disabled",
                chat_template_family="template_kwargs",
            )
            raw_outputs.append(raw)
            summary = raw.strip()
            if not summary:
                raise RuntimeError("empty semantic summary")
            return {
                "record_key": item["record_key"],
                "status": "success",
                "semantic_summary": summary,
                "raw_output": raw,
                "semantic_input_truncated": truncated,
                "semantic_input_chars": len(code),
                "usage": usage,
                "model_name": args.model_name,
            }
        except Exception as exc:  # retry includes vLLM transport failures
            last_error = exc
            if attempt < args.retries:
                time.sleep(args.retry_sleep_seconds)
    return {
        "record_key": item["record_key"],
        "status": "failed",
        "error": str(last_error),
        "raw_outputs": raw_outputs,
        "semantic_input_truncated": truncated,
        "semantic_input_chars": len(code),
        "model_name": args.model_name,
    }


def generate_semantics(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    inputs = read_jsonl(output_dir / "semantic_inputs.jsonl")
    summary_path = output_dir / "semantic_summaries.jsonl"
    latest = latest_by_key(read_jsonl(summary_path) if summary_path.exists() else [], "record_key")
    pending = [row for row in inputs if latest.get(row["record_key"], {}).get("status") != "success"]
    print(f"[INFO] Semantic summaries complete={len(inputs) - len(pending)}/{len(inputs)} pending={len(pending)}")
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.semantic_workers)) as executor:
            futures = {executor.submit(call_semantic_item, item, args): item["record_key"] for item in pending}
            for position, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                append_jsonl(summary_path, row)
                print(f"[INFO] Semantic {position}/{len(pending)} {row['record_key']} status={row['status']}")
    latest = latest_by_key(read_jsonl(summary_path), "record_key")
    failed = [key for key in (row["record_key"] for row in inputs) if latest.get(key, {}).get("status") != "success"]
    write_json(
        output_dir / "semantic_generation_status.json",
        {
            "input_count": len(inputs),
            "success_count": len(inputs) - len(failed),
            "failed_count": len(failed),
            "failed_record_keys": failed,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking_type": "disabled",
            "semantic_max_new_tokens": args.semantic_max_new_tokens,
            "semantic_max_input_chars": args.semantic_max_input_chars,
        },
    )
    if args.strict and failed:
        raise RuntimeError(f"Semantic generation incomplete: {len(failed)} failures")


def semantic_retrieve(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    experiment = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    queries = read_jsonl(output_dir / "query_manifest.jsonl")
    candidates = load_dataset(Path(args.train_path), "train")
    summaries = latest_by_key(read_jsonl(output_dir / "semantic_summaries.jsonl"), "record_key")
    missing = [row["record_key"] for row in [*queries, *candidates] if summaries.get(row["record_key"], {}).get("status") != "success"]
    if missing:
        raise RuntimeError(f"Cannot retrieve semantic results; missing {len(missing)} summaries")
    device = resolve_device(args.device)
    tokenizer, model = load_embedding_model(Path(args.embedding_model_path), device)
    all_records = [*candidates, *queries]
    semantic_texts = [summaries[row["record_key"]]["semantic_summary"] for row in all_records]
    embeddings: Dict[str, List[float]] = {}
    for start in range(0, len(all_records), args.embedding_batch_size):
        batch = all_records[start : start + args.embedding_batch_size]
        vectors = encode_texts(
            tokenizer, model, semantic_texts[start : start + len(batch)], args.embedding_max_length, device
        )
        embeddings.update({row["record_key"]: vector for row, vector in zip(batch, vectors)})
        print(f"[INFO] Encoded semantic summaries {min(start + len(batch), len(all_records))}/{len(all_records)}")
    candidate_matrix = torch.tensor([embeddings[row["record_key"]] for row in candidates], dtype=torch.float32)
    rows = []
    for query in queries:
        scores = candidate_matrix @ torch.tensor(embeddings[query["record_key"]], dtype=torch.float32)
        index = int(torch.argmax(scores).item())
        candidate = candidates[index]
        rows.append(
            result_row(
                "dense_summary", query, candidate, float(scores[index].item()), "cosine_similarity", experiment,
                {
                    "semantic_model_name": args.model_name,
                    "semantic_summary_query": summaries[query["record_key"]]["semantic_summary"],
                    "semantic_summary_candidate": summaries[candidate["record_key"]]["semantic_summary"],
                    "query_semantic_input_truncated": summaries[query["record_key"]].get("semantic_input_truncated", False),
                    "candidate_semantic_input_truncated": summaries[candidate["record_key"]].get("semantic_input_truncated", False),
                    "embedding_model_path": str(Path(args.embedding_model_path)),
                    "embedding_max_length": args.embedding_max_length,
                },
            )
        )
    write_method_results(output_dir, "dense_summary", rows)
    write_json(
        output_dir / "dense_summary_metadata.json",
        {
            "semantic_model_name": args.model_name,
            "embedding_model_path": str(Path(args.embedding_model_path)),
            "embedding_dimension": len(next(iter(embeddings.values()))),
            "embedding_max_length": args.embedding_max_length,
            "semantic_input_count": len(all_records),
            "semantic_input_truncated_count": sum(
                bool(summaries[row["record_key"]].get("semantic_input_truncated")) for row in all_records
            ),
        },
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def merge_and_validate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    experiment = json.loads((output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    queries = read_jsonl(output_dir / "query_manifest.jsonl")
    expected_query_ids = [row["idx"] for row in queries]
    rows: List[Dict[str, Any]] = []
    problems: List[str] = []
    for method in METHODS:
        path = output_dir / f"{method}_results.jsonl"
        if not path.exists():
            problems.append(f"missing result file: {path.name}")
            continue
        method_rows = read_jsonl(path)
        if len(method_rows) != len(queries):
            problems.append(f"{method}: expected {len(queries)} rows, got {len(method_rows)}")
        if [row.get("query_idx") for row in method_rows] != expected_query_ids:
            problems.append(f"{method}: query ordering/content differs from manifest")
        if any(row.get("rank") != 1 or row.get("candidate", {}).get("split") != "train" for row in method_rows):
            problems.append(f"{method}: invalid rank or non-training candidate")
        rows.extend(method_rows)
    if len(rows) != len(queries) * len(METHODS):
        problems.append(f"canonical row count is {len(rows)}, expected {len(queries) * len(METHODS)}")
    write_jsonl(output_dir / "canonical_retrieval_pairs.jsonl", rows)
    csv_path = output_dir / "retrieval_overview.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("method,query_idx,candidate_idx,score,score_type,query_project,candidate_project\n")
        for row in rows:
            handle.write(
                ",".join(
                    str(value).replace(",", " ")
                    for value in (
                        row["method"], row["query_idx"], row["candidate_idx"], row["score"], row["score_type"],
                        row["query"]["project"], row["candidate"]["project"],
                    )
                ) + "\n"
            )
    markdown = [
        "# PrimeVul Code Retrieval Comparison",
        "",
        f"- Queries: {len(queries)} test-set vulnerable functions",
        f"- Candidates: {experiment['source']['train_candidate_count']} training-set vulnerable functions",
        f"- Seed: {experiment['seed']}",
        f"- Canonical retrieval pairs: {len(rows)}",
        "",
        "| Method | Rows |",
        "| --- | ---: |",
    ]
    for method in METHODS:
        markdown.append(f"| `{method}` | {sum(row['method'] == method for row in rows)} |")
    if problems:
        markdown.extend(["", "## Validation failures", *[f"- {item}" for item in problems]])
    (output_dir / "retrieval_overview.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(output_dir / "validation.json", {"passed": not problems, "problems": problems, "canonical_count": len(rows)})
    if problems:
        raise RuntimeError("Retrieval comparison validation failed: " + "; ".join(problems))
    print(f"[INFO] Validation passed: {len(rows)} canonical retrieval pairs")


def create_query_subset(args: argparse.Namespace) -> None:
    """Copy all method results for the first N frozen manifest queries."""
    output_dir = Path(args.output_dir)
    queries = read_jsonl(output_dir / "query_manifest.jsonl")
    rows = read_jsonl(output_dir / "canonical_retrieval_pairs.jsonl")
    selected_queries = sorted(queries, key=lambda row: int(row["selection_order"]))[: args.subset_query_count]
    if len(selected_queries) != args.subset_query_count:
        raise ValueError(f"Only {len(selected_queries)} queries are available")
    selected_idx = {int(row["idx"]) for row in selected_queries}
    subset_rows = [row for row in rows if int(row["query_idx"]) in selected_idx]
    expected = args.subset_query_count * len(METHODS)
    per_query = Counter(int(row["query_idx"]) for row in subset_rows)
    if len(subset_rows) != expected or set(per_query) != selected_idx or any(count != len(METHODS) for count in per_query.values()):
        raise RuntimeError("Canonical source does not contain one result per method for every selected query")
    suffix = f"first{args.subset_query_count}queries"
    write_jsonl(output_dir / f"canonical_retrieval_pairs_{suffix}.jsonl", subset_rows)
    write_jsonl(output_dir / f"query_manifest_{suffix}.jsonl", selected_queries)
    write_json(
        output_dir / f"canonical_retrieval_pairs_{suffix}_manifest.json",
        {
            "source": "canonical_retrieval_pairs.jsonl",
            "selection_rule": "ascending frozen query_manifest.selection_order",
            "selection_order_start": 1,
            "selection_order_end": args.subset_query_count,
            "query_count": args.subset_query_count,
            "methods_per_query": len(METHODS),
            "canonical_pair_count": len(subset_rows),
            "query_indices": [int(row["idx"]) for row in selected_queries],
        },
    )
    print(f"[INFO] Wrote {len(subset_rows)} rows for {len(selected_queries)} fixed queries")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("local", "prepare-semantic", "semantic", "semantic-retrieve", "merge", "subset"))
    parser.add_argument("--train_path", default="data/raw/primevul_train_merged.jsonl")
    parser.add_argument("--test_path", default="data/raw/primevul_test_merged.jsonl")
    parser.add_argument("--output_dir", default="data/code_retrieval_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_count", type=int, default=100)
    parser.add_argument("--subset_query_count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embedding_model_path", default="models/embedding/encoder")
    parser.add_argument("--embedding_max_length", type=int, default=4096)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_key_env", default="LLM_API_KEY")
    parser.add_argument("--model_name", default="your-model-id")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep_seconds", type=float, default=3.0)
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--semantic_max_new_tokens", type=int, default=384)
    parser.add_argument("--semantic_max_input_chars", type=int, default=120000)
    parser.add_argument("--semantic_workers", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "local":
        run_local(args)
    elif args.command == "prepare-semantic":
        prepare_semantic_inputs(args)
    elif args.command == "semantic":
        generate_semantics(args)
    elif args.command == "semantic-retrieve":
        semantic_retrieve(args)
    elif args.command == "subset":
        create_query_subset(args)
    else:
        merge_and_validate(args)


if __name__ == "__main__":
    main()
