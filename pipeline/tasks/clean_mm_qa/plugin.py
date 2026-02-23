from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from pipeline.core.config import PipelineConfig
from pipeline.tasks.base import RequestSpec, TaskPlugin
from pipeline.tasks.clean_mm_qa.parser import parse_clean_mm_qa_response
from pipeline.tasks.clean_mm_qa.prompts import CLEAN_JUDGE_SYSTEM
from pipeline.tasks.clean_mm_qa.splitter import split_clean_dirty
from pipeline.types import ProcessedRecord, RawItem


class CleanMMQATask(TaskPlugin):
    name = "clean_mm_qa"

    def load_items(self, config: PipelineConfig) -> list[RawItem]:
        task_cfg = config.task_config
        input_jsonl = _cfg_str(task_cfg, "input_jsonl", required=True)
        max_records = _cfg_int(task_cfg, "max_records", default=0)
        image_root = _cfg_str(task_cfg, "image_root", default="")

        path = Path(input_jsonl)
        if not path.exists():
            raise FileNotFoundError(f"clean_mm_qa input file does not exist: {input_jsonl}")

        items: list[RawItem] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                expanded = _expand_to_qa_items(record, image_root=image_root, id_field=config.id_field)
                items.extend(expanded)
                if max_records and len(items) >= max_records:
                    items = items[:max_records]
                    break
        return items

    def build_request(self, item: RawItem, config: PipelineConfig) -> RequestSpec:
        task_cfg = config.task_config
        model = _cfg_str(task_cfg, "model", default="gpt-4o")
        detail = _cfg_str(task_cfg, "image_detail", default="low")

        question = str(item.get("question") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        if not question:
            return RequestSpec(
                payload={},
                skip_http=True,
                local_response={
                    "_local_verdict": {
                        "relevance": "unknown",
                        "necessity": "unknown",
                        "reason": "empty question",
                    }
                },
            )
        if not image_path or not os.path.isfile(image_path):
            return RequestSpec(
                payload={},
                skip_http=True,
                local_response={
                    "_local_verdict": {
                        "relevance": "unknown",
                        "necessity": "unknown",
                        "reason": f"image not found: {image_path}",
                    }
                },
            )

        encoded, mime = _encode_image(image_path)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": CLEAN_JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Question:\n{question}"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encoded}",
                                "detail": detail,
                            },
                        },
                    ],
                },
            ],
            "max_tokens": _cfg_int(task_cfg, "max_tokens", default=256),
            "temperature": _cfg_float(task_cfg, "temperature", default=0.0),
        }
        return RequestSpec(payload=payload)

    def parse_response(
        self,
        item: RawItem,
        llm_response: dict[str, Any],
        config: PipelineConfig,
    ) -> ProcessedRecord:
        return parse_clean_mm_qa_response(
            item=item,
            llm_response=llm_response,
            id_field=config.id_field,
        )

    def finalize_outputs(self, config: PipelineConfig) -> dict[str, Any] | None:
        task_cfg = config.task_config
        output_path = Path(config.output_jsonl)
        clean_path = task_cfg.get("clean_output_jsonl") or f"{output_path.with_suffix('')}_clean.jsonl"
        dirty_path = task_cfg.get("dirty_output_jsonl") or f"{output_path.with_suffix('')}_dirty.jsonl"
        return split_clean_dirty(
            judged_output_jsonl=config.output_jsonl,
            clean_output_jsonl=str(clean_path),
            dirty_output_jsonl=str(dirty_path),
        )


def _expand_to_qa_items(
    record: dict[str, Any],
    *,
    image_root: str,
    id_field: str,
) -> list[RawItem]:
    source_id = str(record.get("id") or record.get("video_id") or "")
    if not source_id:
        raw = json.dumps(record, sort_keys=True, ensure_ascii=False)
        source_id = f"line_{hashlib.md5(raw.encode('utf-8'), usedforsecurity=False).hexdigest()}"

    image_path = _resolve_image_path(record, image_root=image_root)
    qa_pairs = _extract_qa_pairs(record)
    items: list[RawItem] = []
    for qa_index, qa in enumerate(qa_pairs):
        item_id = f"{source_id}#qa{qa_index}"
        items.append(
            {
                id_field: item_id,
                "source_id": source_id,
                "qa_index": qa_index,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "image_path": image_path,
                "raw_record": record,
            }
        )
    return items


def _extract_qa_pairs(record: dict[str, Any]) -> list[dict[str, str | None]]:
    qas_raw = record.get("qas")
    if isinstance(qas_raw, list) and qas_raw:
        pairs: list[dict[str, str | None]] = []
        for qa in qas_raw:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question") or "").strip()
            answer_raw = qa.get("answer")
            answer = str(answer_raw).strip() if answer_raw is not None else None
            if question:
                pairs.append({"question": question, "answer": answer})
        if pairs:
            return pairs

    question = str(record.get("question") or "").strip()
    answer_raw = record.get("answer")
    answer = str(answer_raw).strip() if answer_raw is not None else None

    if not question:
        messages = record.get("messages")
        if isinstance(messages, list):
            question = _extract_question_from_messages(messages)
            answer = _extract_answer_from_messages(messages)

    if question:
        return [{"question": question, "answer": answer}]
    return [{"question": "", "answer": answer}]


def _extract_question_from_messages(messages: list[Any]) -> str:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        content = content.replace("<image>", "").strip()
        if content:
            return content
    return ""


def _extract_answer_from_messages(messages: list[Any]) -> str | None:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return None


def _resolve_image_path(record: dict[str, Any], *, image_root: str) -> str:
    candidate = ""
    if isinstance(record.get("image"), str):
        candidate = str(record["image"])
    elif isinstance(record.get("images"), list) and record["images"]:
        first = record["images"][0]
        if isinstance(first, str):
            candidate = first
    elif isinstance(record.get("images_source"), list) and record["images_source"]:
        first = record["images_source"][0]
        if isinstance(first, str):
            candidate = first

    if not candidate:
        return ""
    if image_root:
        return str(Path(image_root) / Path(candidate).name)
    return candidate


def _encode_image(path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return encoded, mime


def _cfg_str(task_cfg: dict[str, Any], key: str, *, default: str | None = None, required: bool = False) -> str:
    value = task_cfg.get(key, default)
    if value is None:
        if required:
            raise ValueError(f"task_config.{key} is required")
        return ""
    return str(value)


def _cfg_int(task_cfg: dict[str, Any], key: str, *, default: int) -> int:
    value = task_cfg.get(key, default)
    return int(value)


def _cfg_float(task_cfg: dict[str, Any], key: str, *, default: float) -> float:
    value = task_cfg.get(key, default)
    return float(value)
