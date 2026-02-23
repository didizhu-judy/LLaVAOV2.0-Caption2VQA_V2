from __future__ import annotations

from typing import Any

from pipeline.core.config import PipelineConfig
from pipeline.tasks.base import RequestSpec, TaskPlugin
from pipeline.tasks.caption_to_vqa.adapter import load_caption_items
from pipeline.tasks.caption_to_vqa.parser import parse_caption_to_vqa_response
from pipeline.tasks.caption_to_vqa.prompts import (
    BATCH_ALL_QA_SYSTEM,
    build_caption_to_vqa_user_prompt,
)
from pipeline.types import ProcessedRecord, RawItem


class CaptionToVQATask(TaskPlugin):
    name = "caption_to_vqa"

    def load_items(self, config: PipelineConfig) -> list[RawItem]:
        task_cfg = config.task_config
        input_jsonl = _cfg_str(task_cfg, "input_jsonl", required=True)
        max_records = _cfg_int(task_cfg, "max_records", default=0)
        default_duration = _cfg_float(task_cfg, "default_duration_sec", default=180.0)
        return load_caption_items(
            input_jsonl=input_jsonl,
            id_field=config.id_field,
            max_records=max_records,
            default_duration_sec=default_duration,
        )

    def build_request(self, item: RawItem, config: PipelineConfig) -> RequestSpec:
        task_cfg = config.task_config
        model = _cfg_str(task_cfg, "model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
        max_tokens = _cfg_int(task_cfg, "max_tokens", default=1024)
        temperature = _cfg_float(task_cfg, "temperature", default=0.3)
        max_understanding = _cfg_int(task_cfg, "max_understanding", default=5)
        max_segments = _cfg_int(task_cfg, "max_segments", default=15)

        segments = item.get("segments") or []
        segments_text = []
        for idx, segment in enumerate(segments[:max_segments], start=1):
            start = segment.get("start_sec")
            end = segment.get("end_sec")
            description = str(segment.get("description") or "")
            if end is None:
                segments_text.append(f"{idx}. [point ~{start}s] {description[:250]}")
            else:
                segments_text.append(f"{idx}. [range {start}-{end}s] {description[:250]}")

        structured_caption = str(item.get("structured_caption") or "")
        prompt = build_caption_to_vqa_user_prompt(
            video_id=str(item.get("video_id") or ""),
            duration_sec=float(item.get("duration_sec") or 0.0),
            segments_text="\n".join(segments_text),
            structured_caption=structured_caption,
            max_understanding=max_understanding,
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": BATCH_ALL_QA_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return RequestSpec(payload=payload)

    def parse_response(
        self,
        item: RawItem,
        llm_response: dict[str, Any],
        config: PipelineConfig,
    ) -> ProcessedRecord:
        return parse_caption_to_vqa_response(
            item=item,
            llm_response=llm_response,
            id_field=config.id_field,
        )


def _cfg_str(
    task_cfg: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str:
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
