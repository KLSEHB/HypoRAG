"""Build pairwise reranker data from completed M/R/E query pools and teacher labels."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_pipeline.reranker_preference import build_pairwise_rows, pair_id


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pools-path", type=Path, required=True)
    parser.add_argument("--teacher-labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pools = read_jsonl(args.candidate_pools_path)
    labels = read_jsonl(args.teacher_labels_path)
    if not pools:
        raise ValueError("Candidate pool file is empty")
    successful: dict[str, dict] = {}
    for label in labels:
        if label.get("status") != "success":
            continue
        key = str(label["pair_id"])
        previous = successful.get(key)
        if previous is not None and int(previous["reference_value_label"]) != int(label["reference_value_label"]):
            raise ValueError(f"Conflicting successful labels for {key}")
        successful[key] = label
    query_ids = set()
    for pool in pools:
        query_id = str(pool["query"]["query_id"])
        if query_id in query_ids:
            raise ValueError(f"Duplicate query pool: {query_id}")
        query_ids.add(query_id)
        for candidate in pool.get("candidates", []):
            key = pair_id(query_id, int(candidate["candidate_idx"]))
            if key not in successful:
                raise ValueError(f"Incomplete teacher labels for {query_id}: missing {key}")

    pairwise = build_pairwise_rows(pools, list(successful.values()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in pairwise.items():
        write_jsonl(args.output_dir / f"pairwise_{split}.jsonl", rows)
    print(json.dumps({split: len(rows) for split, rows in pairwise.items()}))


if __name__ == "__main__":
    main()
