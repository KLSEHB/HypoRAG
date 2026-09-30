"""Check the released retrieval audit and basic artifact hygiene."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from recompute_retrieval_diagnostic import ROOT as DIAGNOSTIC, read_jsonl, recompute


ROOT = Path(__file__).resolve().parents[1]


def verify_table() -> None:
    computed = recompute()
    stored = json.loads((DIAGNOSTIC / "summary.json").read_text(encoding="utf-8"))
    if computed != stored:
        raise AssertionError("The released summary differs from the 341 audit rows")
    expected = {
        "random_code": (50, 4), "bm25_code": (50, 10),
        "dense_code": (50, 6), "dense_summary": (50, 10),
        "hypothesis_latest": (141, 114),
    }
    for row in computed:
        if (row["n"], row["transferable_match_count"]) != expected[row["method"]]:
            raise AssertionError(f"Unexpected match count for {row['method']}")
        print(f"{row['method']}: J={row['mean_token_jaccard']:.4f}, "
              f"calls={row['mean_target_callee_overlap']:.4f}, "
              f"transferable={row['transferable_match_count']}/{row['n']}")
    manifest = json.loads((DIAGNOSTIC / "manifest.json").read_text(encoding="utf-8"))
    if len(set(manifest["test_ids"])) != 50 or manifest["hypothesis_top1"] != 141:
        raise AssertionError("Incorrect query manifest")
    candidate_ids = json.loads((DIAGNOSTIC / "candidate_ids.json").read_text(encoding="utf-8"))
    if not (len(candidate_ids) == len(set(candidate_ids)) == manifest["candidate_count"]):
        raise AssertionError("Candidate ID count differs from the manifest")
    if len(read_jsonl(DIAGNOSTIC / "semantic_query_summaries.jsonl")) != 3661:
        raise AssertionError("Expected 3611 candidate and 50 query summaries")
    if any(row["candidate_idx"] not in set(candidate_ids)
           for row in read_jsonl(DIAGNOSTIC / "retrieval_pairs.jsonl")):
        raise AssertionError("Retrieved candidate not in the frozen corpus")


def verify_hygiene() -> None:
    forbidden = re.compile(r"sk-[A-Za-z0-9]{12,}|/hpcfs/fhome/|C:\\Users\\", re.I)
    for path in [*ROOT.rglob("*.py"), *ROOT.rglob("*.md"), *ROOT.rglob("*.jsonl")]:
        source = path.read_text(encoding="utf-8")
        if path != Path(__file__).resolve() and forbidden.search(source):
            raise AssertionError(f"Sensitive or host-specific string in {path}")
        if path.suffix == ".py":
            ast.parse(source, filename=str(path))
    if any(ROOT.rglob("*.safetensors")):
        raise AssertionError("Model weights must not be released")
    if not (ROOT / "appendix.pdf").is_file():
        raise AssertionError("appendix.pdf is missing")


if __name__ == "__main__":
    verify_table()
    verify_hygiene()
    print("Release verification passed")
