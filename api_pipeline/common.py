import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_excluded_record_ids(path: Optional[Path]) -> set[int]:
    if path is None or not str(path) or not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    else:
        rows = payload.get("excluded_records", [])
    return {
        int(row["idx"] if isinstance(row, dict) else row)
        for row in rows
    }


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False)
        file.write("\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def select_records(
    records: Iterable[Dict[str, Any]],
    start_idx: int,
    end_idx: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for record in records:
        idx = int(record.get("idx", -1))
        if idx < start_idx:
            continue
        if end_idx >= 0 and idx > end_idx:
            continue
        selected.append(record)
    return selected


def load_processed_indices(output_path: Path) -> Set[int]:
    if not output_path.exists():
        return set()

    processed: Set[int] = set()
    for record in read_jsonl(output_path):
        idx = record.get("idx")
        if isinstance(idx, int):
            processed.add(idx)
    return processed


def derive_raw_input_path(raw_dir: Path, split: str) -> Path:
    return raw_dir / f"primevul_{split}_merged.jsonl"


def build_idx_map(records: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(record["idx"]): record for record in records}


def build_api_url(api_base: str) -> str:
    return api_base.rstrip("/")


def load_api_key(api_key: Optional[str], api_key_env: str) -> str:
    if api_key:
        return api_key
    value = os.environ.get(api_key_env, "").strip()
    if value:
        return value
    raise RuntimeError(
        f"Missing API key. Provide --api_key or set environment variable {api_key_env}."
    )


def build_client(api_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=api_url,
        api_key=api_key,
    )


def build_chat_extra_body(
    model_name: str,
    thinking_type: str,
    reasoning_effort: str,
    chat_template_family: str = "auto",
) -> Dict[str, Any]:
    resolved_family = chat_template_family
    if resolved_family == "auto":
        resolved_family = "default"
    extra_body: Dict[str, Any] = {"thinking": {"type": thinking_type}}
    if resolved_family == "template_kwargs":
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": thinking_type == "enabled",
            # Some local chat templates consume this value directly.
            "reasoning_effort": reasoning_effort,
        }
    return extra_body


def call_chat_completion(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: int,
    reasoning_effort: str,
    thinking_type: str,
    chat_template_family: str = "auto",
) -> str:
    content, _ = call_chat_completion_with_usage(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
        thinking_type=thinking_type,
        chat_template_family=chat_template_family,
    )
    return content


def call_chat_completion_with_usage(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: int,
    reasoning_effort: str,
    thinking_type: str,
    chat_template_family: str = "auto",
    response_format: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    extra_body = build_chat_extra_body(
        model_name=model_name,
        thinking_type=thinking_type,
        reasoning_effort=reasoning_effort,
        chat_template_family=chat_template_family,
    )
    request: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "timeout": timeout,
        "reasoning_effort": reasoning_effort,
        "extra_body": extra_body,
    }
    if response_format is not None:
        request["response_format"] = response_format
    completion = client.chat.completions.create(
        **request,
    )
    usage = completion.usage
    usage_dict = usage.model_dump() if usage is not None else {}
    return completion.choices[0].message.content or "", usage_dict


def call_chat_completion_with_retries(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep_seconds: float,
    reasoning_effort: str,
    thinking_type: str,
    chat_template_family: str = "auto",
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return call_chat_completion(
                client=client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                thinking_type=thinking_type,
                chat_template_family=chat_template_family,
            )
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def _json_object_candidates(text: str) -> List[str]:
    stripped = text.strip()
    candidates: List[str] = []
    if stripped:
        candidates.append(stripped)

    if "[FINAL_JSON]" in text:
        candidates.append(text.split("[FINAL_JSON]", 1)[-1].strip())

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())

    for source in list(candidates) + [text]:
        start = source.find("{")
        while start >= 0:
            depth = 0
            in_string = False
            escape = False
            for pos in range(start, len(source)):
                char = source[pos]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(source[start : pos + 1])
                        break
            start = source.find("{", start + 1)

    return candidates


def extract_json_object(text: str) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for candidate in _json_object_candidates(text):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = TypeError(f"Expected JSON object, got {type(parsed).__name__}")
    raise ValueError(f"Could not extract JSON object: {last_error}")


def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def render_messages_as_text(messages: List[Dict[str, str]]) -> str:
    rendered_chunks: List[str] = []
    for message in messages:
        rendered_chunks.append(f"<|{message['role']}|>\n{message['content']}")
    rendered_chunks.append("<|assistant|>\n")
    return "\n".join(rendered_chunks)


def count_message_tokens(tokenizer, messages: List[Dict[str, str]]) -> int:
    # Some locally saved tokenizers have an incompatible chat template that can
    # collapse an arbitrarily large prompt to only special tokens. Count the
    # rendered message text directly so length filtering cannot be bypassed.
    rendered_prompt = render_messages_as_text(messages)
    return len(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"])
