import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from api_pipeline.common import read_jsonl
from api_pipeline.train_reranker import evaluate_all, load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a reranker on canonical Teacher preference data."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--pairwise_path",
        default="data/reranker_distill/pairwise_test.jsonl",
    )
    parser.add_argument(
        "--candidate_pools_path",
        default="data/reranker_distill/candidate_pools.jsonl",
    )
    parser.add_argument(
        "--teacher_labels_path",
        default="data/reranker_distill/teacher_labels_canonical.jsonl",
    )
    parser.add_argument("--dataset_split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model, precision_name = load_model(
        Path(args.model_path),
        device,
        args.precision,
    )
    metrics = evaluate_all(
        tokenizer=tokenizer,
        model=model,
        pair_rows=read_jsonl(Path(args.pairwise_path)),
        pools=read_jsonl(Path(args.candidate_pools_path)),
        labels=read_jsonl(Path(args.teacher_labels_path)),
        dataset_split=args.dataset_split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    result = {
        "model_path": str(Path(args.model_path).resolve()),
        "device": device,
        "precision": precision_name,
        "dataset_split": args.dataset_split,
        "metrics": metrics,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
