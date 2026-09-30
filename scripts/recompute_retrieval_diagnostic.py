"""Recompute the released filtered top-1 surface and mechanism metrics."""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_c


ROOT = Path(__file__).resolve().parents[1] / "retrieval_diagnostic"
METHODS = ("random_code", "bm25_code", "dense_code", "dense_summary", "hypothesis_latest")
TOKEN = re.compile(r"[A-Za-z_]\w*|\d+")
PARSER = Parser(Language(tree_sitter_c.language()))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def row_key(row: dict) -> tuple:
    return (row["method"], row["query_idx"], row.get("hypothesis_id"), row["candidate_idx"])


def token_jaccard(query: str, candidate: str) -> float:
    left, right = set(TOKEN.findall(query)), set(TOKEN.findall(candidate))
    return len(left & right) / len(left | right) if left or right else 0.0


@lru_cache(maxsize=1024)
def called_functions(code: str) -> frozenset[str]:
    source = code.encode("utf-8")
    stack = [PARSER.parse(source).root_node]
    calls = set()
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            function_node = node.child_by_field_name("function")
            if function_node is not None:
                identifiers = []
                inner_stack = [function_node]
                while inner_stack:
                    child = inner_stack.pop()
                    if child.type in {"identifier", "field_identifier"}:
                        identifiers.append(child)
                    inner_stack.extend(reversed(child.children))
                if identifiers:
                    last = max(identifiers, key=lambda item: item.start_byte)
                    calls.add(source[last.start_byte:last.end_byte].decode("utf-8", errors="replace"))
        stack.extend(reversed(node.children))
    return frozenset(calls)


def callee_overlap(query: str, candidate: str) -> float | None:
    left = called_functions(query)
    return len(left & called_functions(candidate)) / len(left) if left else None


def recompute(root: Path = ROOT) -> list[dict]:
    pairs = read_jsonl(root / "retrieval_pairs.jsonl")
    audits = read_jsonl(root / "mechanism_audit.jsonl")
    assert len(pairs) == len(audits) == 341
    assert len({row_key(row) for row in pairs}) == len({row_key(row) for row in audits}) == 341
    assert {row_key(row) for row in pairs} == {row_key(row) for row in audits}
    assert len({row["query_idx"] for row in pairs}) == 50
    assert Counter(row["method"] for row in pairs) == {
        "random_code": 50, "bm25_code": 50, "dense_code": 50,
        "dense_summary": 50, "hypothesis_latest": 141,
    }
    query_ids = {row["query_idx"] for row in pairs}
    audit_by_key = {row_key(row): row for row in audits}

    for row in pairs:
        assert not row["same_project"] and not row["same_cve"]
        assert row["token_jaccard"] < 0.8
        assert row["query_func_vuln"] and row["candidate_func_vuln"]
        audit = audit_by_key[row_key(row)]
        assert row["token_jaccard"] == audit["token_jaccard"]
        assert row["target_callee_overlap"] == audit["target_callee_overlap"]
        assert abs(token_jaccard(row["query_func_vuln"], row["candidate_func_vuln"])
                   - row["token_jaccard"]) < 1e-12
        assert callee_overlap(row["query_func_vuln"], row["candidate_func_vuln"]) == row["target_callee_overlap"]
        assert str(audit["audit"].get("reason", "")).strip()

    for method in METHODS:
        assert {row["query_idx"] for row in pairs if row["method"] == method} == query_ids

    result = []
    for method in METHODS:
        rows = [row for row in audits if row["method"] == method]
        labels = Counter(row["audit"]["label"] for row in rows)
        assert sum(labels.values()) == len(rows)
        result.append({
            "method": method,
            "query_unit": rows[0]["unit"],
            "n": len(rows),
            "mean_token_jaccard": sum(row["token_jaccard"] for row in rows) / len(rows),
            "mean_target_callee_overlap": sum(row["target_callee_overlap"] for row in rows) / len(rows),
            "label_counts": {str(label): labels[label] for label in (0, 1, 2)},
            "transferable_match_count": labels[1] + labels[2],
            "strict_match_count": labels[2],
        })
    return result


if __name__ == "__main__":
    stored = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    computed = recompute()
    assert computed == stored
    for row in computed:
        print(f"{row['method']:<20} {row['n']:>3} "
              f"{row['mean_token_jaccard']:.4f} "
              f"{row['mean_target_callee_overlap']:.4f} "
              f"{row['transferable_match_count']}/{row['n']}")
