"""Export the fixed-50 filtered top-1 retrieval study without model metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


METHODS = ("random_code", "bm25_code", "dense_code", "dense_summary", "hypothesis_latest")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row: dict) -> tuple:
    return (row["method"], row["query_idx"], row.get("hypothesis_id"), row["candidate_idx"])


def export(source: Path, audit_source: Path, train_path: Path, test_path: Path,
           candidate_ids_path: Path, output: Path) -> None:
    manifest = json.loads((source / "manifest_full50.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(source / "retrieval_pairs_full50.jsonl")
    audits = read_jsonl(audit_source / "audit_labeled.jsonl")
    train = {row["idx"]: row for row in read_jsonl(train_path)}
    test = {row["idx"]: row for row in read_jsonl(test_path)}
    candidate_ids = json.loads(candidate_ids_path.read_text(encoding="utf-8"))
    assert len(candidate_ids) == len(set(candidate_ids)) == manifest["candidate_count"] == 3611
    assert set(candidate_ids) <= set(train)
    assert len(pairs) == len(audits) == 341
    assert Counter(row["method"] for row in pairs) == Counter({method: 50 for method in METHODS[:4]}) + Counter({METHODS[4]: 141})
    assert len({key(row) for row in pairs}) == 341
    audit_by_key = {key(row): row for row in audits}
    assert len(audit_by_key) == 341 and set(audit_by_key) == {key(row) for row in pairs}
    assert len(set(manifest["test_ids"])) == 50

    exported_pairs = []
    exported_audits = []
    for row in pairs:
        audit = audit_by_key[key(row)]
        query = test[row["query_idx"]]
        candidate = train[row["candidate_idx"]]
        assert row["candidate_idx"] in candidate_ids
        assert row["query_idx"] in manifest["test_ids"]
        assert row["same_project"] is False and row["same_cve"] is False
        assert row["token_jaccard"] < 0.80
        assert query["project"].lower() != candidate["project"].lower()
        assert query.get("cve") != candidate.get("cve")
        assert row["token_jaccard"] == audit["token_jaccard"]
        assert row["target_callee_overlap"] == audit["target_callee_overlap"]
        assert audit["audit"]["label"] in (0, 1, 2)
        exported_pairs.append({
            **row,
            "query_func_vuln": query["func_vuln"],
            "candidate_func_vuln": candidate["func_vuln"],
        })
        exported_audits.append(audit)

    summaries = read_jsonl(source / "semantic_summaries_full50.jsonl")
    required_summary_keys = {f"train:{idx}" for idx in candidate_ids}
    required_summary_keys.update(f"test:{idx}" for idx in manifest["test_ids"])
    summaries = [row for row in summaries if row["record_key"] in required_summary_keys]
    assert len(summaries) == len(required_summary_keys)
    sanitized_summaries = [
        {"record_key": row["record_key"], "status": row["status"],
         "semantic_summary": row["semantic_summary"]}
        for row in summaries
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "retrieval_pairs.jsonl", exported_pairs)
    write_jsonl(output / "mechanism_audit.jsonl", exported_audits)
    write_jsonl(output / "semantic_query_summaries.jsonl", sanitized_summaries)
    (output / "candidate_ids.json").write_text(json.dumps(candidate_ids, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = []
    for method in METHODS:
        rows = [row for row in exported_audits if row["method"] == method]
        labels = Counter(row["audit"]["label"] for row in rows)
        summary.append({
            "method": method,
            "query_unit": rows[0]["unit"],
            "n": len(rows),
            "mean_token_jaccard": sum(row["token_jaccard"] for row in rows) / len(rows),
            "mean_target_callee_overlap": sum(row["target_callee_overlap"] for row in rows) / len(rows),
            "label_counts": {str(label): labels[label] for label in (0, 1, 2)},
            "transferable_match_count": labels[1] + labels[2],
            "strict_match_count": labels[2],
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for row in summary:
        print(f"{row['method']}: {row['n']} rows, J={row['mean_token_jaccard']:.4f}, "
              f"calls={row['mean_target_callee_overlap']:.4f}, "
              f"transferable={row['transferable_match_count']}/{row['n']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "retrieval_diagnostic")
    args = parser.parse_args()
    export(args.source, args.audit_source, args.train, args.test,
           args.candidate_ids, args.output)
