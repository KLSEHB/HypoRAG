import gc
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from modelscope import AutoModel as ModelScopeAutoModel
from modelscope import AutoTokenizer as ModelScopeAutoTokenizer
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer as HfAutoTokenizer,
)


ROUTE_FIELDS: Dict[str, str] = {
    "mechanism": "mechanism_summary",
    "solution": "solution_summary",
    "evidence": "evidence_summary",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_entries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "idx": int(raw_record["idx"]),
            "mechanism_family": normalize_text(raw_record["retrieval_index"]["mechanism_family"]),
            "mechanism_summary": normalize_text(raw_record["retrieval_index"]["mechanism_summary"]),
            "solution_summary": normalize_text(raw_record["retrieval_index"]["solution_summary"]),
            "evidence_summary": normalize_text(raw_record["retrieval_index"]["evidence_summary"]),
        }
        for raw_record in records
    ]


def load_embedding_model(model_path: Path, device: str):
    tokenizer = ModelScopeAutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    model = ModelScopeAutoModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def create_chroma_client(index_dir: Path):
    return chromadb.PersistentClient(path=str(index_dir))


def create_collections(client) -> Dict[str, Any]:
    collections: Dict[str, Any] = {}
    for route_name in ROUTE_FIELDS:
        collections[route_name] = client.create_collection(
            name=route_name,
            metadata={"hnsw:space": "cosine"},
        )
    return collections


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


@torch.no_grad()
def encode_texts(
    tokenizer,
    model,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> List[List[float]]:
    inputs = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8,
    ).to(device)
    outputs = model(**inputs)
    embeddings = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype("float32").tolist()


def index_route(
    collection,
    route_name: str,
    entries: Sequence[Dict[str, Any]],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: str,
) -> None:
    route_field = ROUTE_FIELDS[route_name]
    total = len(entries)
    indexed = 0

    for batch in batched(entries, batch_size):
        documents = [entry[route_field] for entry in batch]
        embeddings = encode_texts(
            tokenizer=tokenizer,
            model=model,
            texts=documents,
            max_length=max_length,
            device=device,
        )
        collection.add(
            ids=[str(entry["idx"]) for entry in batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[
                {"mechanism_family": entry["mechanism_family"]}
                for entry in batch
            ],
        )
        indexed += len(batch)
        print(f"[INFO] Indexed {indexed}/{total} records into '{route_name}'.")


def rebuild_vector_store(
    example_jsonl_path: Path,
    index_dir: Path,
    embedding_model_path: Path,
    batch_size: int,
    max_length: int,
    device: str,
) -> None:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    raw_records = read_jsonl(example_jsonl_path)
    entries = prepare_entries(raw_records)

    print(f"[INFO] Rebuilding vector store from {example_jsonl_path}")
    print(f"[INFO] Loaded {len(raw_records)} source records.")
    tokenizer, model = load_embedding_model(embedding_model_path, device)
    client = create_chroma_client(index_dir)
    collections = create_collections(client)

    for route_name, collection in collections.items():
        index_route(
            collection=collection,
            route_name=route_name,
            entries=entries,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    print("[INFO] Final collection sizes:")
    for route_name, collection in collections.items():
        print(f"[INFO]   {route_name}: {collection.count()} records")

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def load_collections(chroma_db_path: Path) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=str(chroma_db_path))
    collections: Dict[str, Any] = {}
    for route_name in ROUTE_FIELDS:
        collections[route_name] = client.get_collection(route_name)
    return collections


def query_route(
    collection,
    query_embedding: Sequence[float],
    topk: int,
    mechanism_family: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [list(query_embedding)],
        "n_results": topk,
        "include": ["documents", "metadatas", "distances"],
    }
    if mechanism_family:
        query_kwargs["where"] = {"mechanism_family": mechanism_family}

    result = collection.query(**query_kwargs)
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    hits: List[Dict[str, Any]] = []
    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        example_idx = int(doc_id)
        distance_value = float(distance)
        hits.append(
            {
                "idx": example_idx,
                "document": normalize_text(document),
                "metadata": metadata or {},
                "distance": distance_value,
                "embedding_score": 1.0 - distance_value,
            }
        )
    return hits


def build_route_query(point: Dict[str, Any], route_name: str) -> str:
    if route_name == "mechanism":
        return normalize_text(point.get("retrieval_summary"))
    if route_name == "solution":
        return normalize_text(point.get("needed_example"))
    return normalize_text(point.get("evidence_hint"))


def build_reranker_query(point: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"retrieval_summary: {normalize_text(point.get('retrieval_summary'))}",
            f"needed_example: {normalize_text(point.get('needed_example'))}",
            f"evidence_hint: {normalize_text(point.get('evidence_hint'))}",
        ]
    )


def build_candidate_text_for_reranker(example_record: Dict[str, Any]) -> str:
    retrieval_index = example_record["retrieval_index"]
    return "\n".join(
        [
            f"mechanism_summary: {normalize_text(retrieval_index.get('mechanism_summary'))}",
            f"solution_summary: {normalize_text(retrieval_index.get('solution_summary'))}",
            f"evidence_summary: {normalize_text(retrieval_index.get('evidence_summary'))}",
        ]
    )


def load_reranker_model(model_path: Path, device: str):
    tokenizer = HfAutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        **model_kwargs,
    ).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def rerank_candidates(
    tokenizer,
    model,
    query: str,
    candidate_texts: Sequence[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> List[float]:
    scores: List[float] = []
    pairs = [[query, text] for text in candidate_texts]
    for batch in batched(pairs, batch_size):
        inputs = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)
        logits = model(**inputs, return_dict=True).logits.view(-1).float()
        scores.extend(logits.cpu().tolist())
    return scores
