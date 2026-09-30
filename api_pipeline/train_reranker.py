import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api_pipeline.common import read_jsonl
from api_pipeline.reranker_preference import (
    latest_successful_labels,
    ndcg,
    pair_id,
    reranker_candidate,
    reranker_query,
)
from api_pipeline.retrieval_core import (
    rerank_candidates,
)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_precision(device: str, precision: str) -> tuple[str, torch.dtype]:
    if precision == "auto":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        if device.startswith("cuda"):
            return "fp16", torch.float16
        return "fp32", torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return precision, mapping[precision]


def load_model(model_path: Path, device: str, precision: str):
    precision_name, dtype = resolve_precision(device, precision)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if device.startswith("cuda"):
        kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        **kwargs,
    ).to(device)
    return tokenizer, model, precision_name


def encode_pairs(tokenizer, pairs, max_length: int, device: str):
    return tokenizer(
        list(pairs),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8,
    ).to(device)


def summarize_pairwise_results(
    rows: Sequence[Dict[str, Any]],
    margins: Sequence[float],
) -> Dict[str, float]:
    if not rows:
        return {}
    if len(rows) != len(margins):
        raise ValueError("rows and margins must have the same length")

    credits_by_query: Dict[str, List[float]] = defaultdict(list)
    weighted_correct = 0.0
    total_weight = 0.0
    credits: List[float] = []
    for row, margin in zip(rows, margins):
        credit = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        weight = float(row["weight"])
        query_id = str(row["query_id"])
        credits.append(credit)
        credits_by_query[query_id].append(credit)
        weighted_correct += weight * credit
        total_weight += weight

    per_query_accuracy = [
        sum(query_credits) / len(query_credits)
        for query_credits in credits_by_query.values()
    ]
    return {
        "pairwise_accuracy": sum(credits) / len(credits),
        "macro_preference_accuracy": (
            sum(per_query_accuracy) / len(per_query_accuracy)
        ),
        "weighted_pairwise_accuracy": (
            weighted_correct / total_weight if total_weight else 0.0
        ),
        "mean_margin": sum(margins) / len(margins),
        "num_pairs": float(len(rows)),
        "num_queries": float(len(credits_by_query)),
    }


@torch.no_grad()
def evaluate_pairwise(
    tokenizer,
    model,
    rows: Sequence[Dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    if not rows:
        return {}
    model.eval()
    margins: List[float] = []
    for batch in batched(rows, batch_size):
        positive_pairs = [[row["query"], row["positive"]] for row in batch]
        negative_pairs = [[row["query"], row["negative"]] for row in batch]
        combined = positive_pairs + negative_pairs
        inputs = encode_pairs(tokenizer, combined, max_length, device)
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        batch_size_actual = len(batch)
        margins_tensor = scores[:batch_size_actual] - scores[batch_size_actual:]
        margins.extend(float(margin) for margin in margins_tensor.cpu().tolist())
    return summarize_pairwise_results(rows, margins)


@torch.no_grad()
def evaluate_ranking(
    tokenizer,
    model,
    pools: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    dataset_split: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, float]:
    latest = latest_successful_labels(labels)
    reciprocal_ranks: List[float] = []
    ndcg_values: Dict[int, List[float]] = {1: [], 3: [], 5: []}
    recalls: Dict[int, List[float]] = {1: [], 3: [], 5: []}
    cross_family_correct = 0
    cross_family_total = 0
    direction_rr: Dict[str, List[float]] = defaultdict(list)

    for pool in pools:
        query = pool["query"]
        if query.get("dataset_split") != dataset_split:
            continue
        labeled_candidates = []
        for candidate in pool.get("candidates", []):
            label = latest.get(
                pair_id(str(query["query_id"]), int(candidate["candidate_idx"]))
            )
            if label is not None:
                labeled_candidates.append((candidate, label))
        if len(labeled_candidates) < 2:
            continue
        query_text = reranker_query(query)
        candidate_texts = [
            reranker_candidate(candidate["candidate_mre"])
            for candidate, _ in labeled_candidates
        ]
        scores = rerank_candidates(
            tokenizer=tokenizer,
            model=model,
            query=query_text,
            candidate_texts=candidate_texts,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        ranked = sorted(
            zip(labeled_candidates, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        relevances = [
            int(label["reference_value_label"])
            for ((_, label), _) in ranked
        ]
        label2_positions = [
            rank
            for rank, ((_, label), _) in enumerate(ranked, start=1)
            if int(label["reference_value_label"]) == 2
        ]
        if label2_positions:
            reciprocal_ranks.append(1.0 / min(label2_positions))
            total_label2 = len(label2_positions)
            for k in recalls:
                recalls[k].append(
                    sum(position <= k for position in label2_positions) / total_label2
                )
            for direction in ("confirm", "rule_out", "both"):
                positions = [
                    rank
                    for rank, ((_, label), _) in enumerate(ranked, start=1)
                    if int(label["reference_value_label"]) == 2
                    and label["helpfulness_type"] == direction
                ]
                if positions:
                    direction_rr[direction].append(1.0 / min(positions))
        for k in ndcg_values:
            ndcg_values[k].append(ndcg(relevances, k))

        non_cross_scores = [
            score
            for ((candidate, _), score) in ranked
            if candidate.get("candidate_source") != "cross_family"
        ]
        if non_cross_scores:
            best_non_cross = max(non_cross_scores)
            for ((candidate, label), score) in ranked:
                if candidate.get("candidate_source") != "cross_family":
                    continue
                if int(label["reference_value_label"]) != 0:
                    continue
                cross_family_total += 1
                cross_family_correct += int(score < best_non_cross)

    result: Dict[str, float] = {
        "num_ranked_queries": float(len(ndcg_values[1])),
        "mrr_label2": (
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else 0.0
        ),
        "cross_family_negative_accuracy": (
            cross_family_correct / cross_family_total
            if cross_family_total
            else 0.0
        ),
    }
    for k, values in ndcg_values.items():
        result[f"ndcg@{k}"] = sum(values) / len(values) if values else 0.0
    for k, values in recalls.items():
        result[f"recall_label2@{k}"] = (
            sum(values) / len(values) if values else 0.0
        )
    for direction, values in direction_rr.items():
        result[f"mrr_{direction}"] = sum(values) / len(values)
    return result


def evaluate_all(
    tokenizer,
    model,
    pair_rows: Sequence[Dict[str, Any]],
    pools: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    dataset_split: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    return {
        "pairwise": evaluate_pairwise(
            tokenizer,
            model,
            pair_rows,
            max_length,
            batch_size,
            device,
        ),
        "ranking": evaluate_ranking(
            tokenizer,
            model,
            pools,
            labels,
            dataset_split,
            max_length,
            batch_size,
            device,
        ),
    }


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_path = Path(args.output_path)
    if output_path.exists() and any(output_path.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Refusing to overwrite existing model directory: {output_path}")
    if output_path.exists() and args.overwrite_output:
        import shutil

        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(Path(args.train_path))
    eval_rows = read_jsonl(Path(args.eval_path)) if Path(args.eval_path).exists() else []
    if args.max_train_pairs > 0:
        train_rows = train_rows[: args.max_train_pairs]
    if args.max_eval_pairs > 0:
        eval_rows = eval_rows[: args.max_eval_pairs]
    if not train_rows:
        raise ValueError("No pairwise training rows are available")
    pools = (
        read_jsonl(Path(args.candidate_pools_path))
        if Path(args.candidate_pools_path).exists()
        else []
    )
    labels = (
        read_jsonl(Path(args.teacher_labels_path))
        if Path(args.teacher_labels_path).exists()
        else []
    )

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer, model, precision_name = load_model(
        Path(args.reranker_model_path),
        device,
        args.precision,
    )
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    baseline_metrics = evaluate_all(
        tokenizer,
        model,
        eval_rows,
        pools,
        labels,
        args.eval_split,
        args.max_length,
        args.eval_batch_size,
        device,
    )
    print("[INFO] Baseline metrics:")
    print(json.dumps(baseline_metrics, indent=2))
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_rows) / args.train_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_steps = max(1, optimizer_steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    started = time.perf_counter()
    epoch_metrics: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        shuffled = list(train_rows)
        rng.shuffle(shuffled)
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_number, batch in enumerate(
            batched(shuffled, args.train_batch_size),
            start=1,
        ):
            positive_pairs = [[row["query"], row["positive"]] for row in batch]
            negative_pairs = [[row["query"], row["negative"]] for row in batch]
            inputs = encode_pairs(
                tokenizer,
                positive_pairs + negative_pairs,
                args.max_length,
                device,
            )
            autocast_enabled = device.startswith("cuda") and precision_name != "fp32"
            autocast_dtype = (
                torch.bfloat16 if precision_name == "bf16" else torch.float16
            )
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                scores = model(**inputs, return_dict=True).logits.view(-1).float()
                actual = len(batch)
                margins = scores[:actual] - scores[actual:]
                weights = torch.tensor(
                    [float(row["weight"]) for row in batch],
                    device=device,
                )
                loss = (weights * F.softplus(-margins)).mean()
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            total_loss += float(loss.detach().cpu()) * actual
            seen += actual
            should_step = (
                batch_number % args.gradient_accumulation_steps == 0
                or batch_number == batches_per_epoch
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % args.log_every_steps == 0:
                    print(
                        f"[INFO] epoch={epoch} optimizer_step={optimizer_step}/"
                        f"{total_steps} loss={total_loss/max(1, seen):.6f}"
                    )

        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, seen),
            "optimizer_step": optimizer_step,
            "eval": evaluate_all(
                tokenizer,
                model,
                eval_rows,
                pools,
                labels,
                args.eval_split,
                args.max_length,
                args.eval_batch_size,
                device,
            ),
        }
        epoch_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    summary = {
        "device": device,
        "precision": precision_name,
        "epochs": args.epochs,
        "optimizer_steps": optimizer_step,
        "elapsed_seconds": time.perf_counter() - started,
        "train_pair_count": len(train_rows),
        "eval_pair_count": len(eval_rows),
        "baseline_metrics": baseline_metrics,
        "epoch_metrics": epoch_metrics,
        "training_args": vars(args),
    }
    (output_path / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    reloaded_tokenizer, reloaded_model, _ = load_model(
        output_path,
        device,
        args.precision,
    )
    reload_metrics = evaluate_pairwise(
        reloaded_tokenizer,
        reloaded_model,
        eval_rows[: min(32, len(eval_rows))],
        args.max_length,
        args.eval_batch_size,
        device,
    )
    (output_path / "reload_check.json").write_text(
        json.dumps(reload_metrics, indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Model saved and reloaded successfully: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a BGE cross-encoder with weighted pairwise preferences."
    )
    parser.add_argument(
        "--train_path",
        default="data/reranker_distill/pairwise_train.jsonl",
    )
    parser.add_argument(
        "--eval_path",
        default="data/reranker_distill/pairwise_dev.jsonl",
    )
    parser.add_argument(
        "--candidate_pools_path",
        default="data/reranker_distill/candidate_pools.jsonl",
    )
    parser.add_argument(
        "--teacher_labels_path",
        default="data/reranker_distill/teacher_labels_canonical.jsonl",
    )
    parser.add_argument("--eval_split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument(
        "--reranker_model_path",
        default="models/reranker/base/cross-encoder",
    )
    parser.add_argument(
        "--output_path",
        default="models/reranker/tuned/preference-reranker",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_train_pairs", type=int, default=-1)
    parser.add_argument("--max_eval_pairs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_output", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
