from __future__ import annotations

import json
from typing import Any

from pipeline.types import ProcessedRecord


def parse_caption_to_vqa_response(
    *,
    item: dict[str, Any],
    llm_response: dict[str, Any],
    id_field: str,
) -> ProcessedRecord:
    out: ProcessedRecord = {
        id_field: str(item[id_field]),
        "video_id": str(item.get("video_id") or item[id_field]),
        "duration_sec": float(item.get("duration_sec", 0.0)),
        "segments": list(item.get("segments") or []),
        "temporal_grounding": [],
        "segment_qa": [],
        "understanding_qa": [],
    }
    images_source = item.get("images_source")
    if images_source:
        out["images_source"] = images_source

    content = _extract_content_text(llm_response)
    parsed = _parse_json_object(content)
    if parsed is None:
        out["vqa_error"] = f"Failed to parse JSON object from model output: {content[:300]}"
        return out

    out["temporal_grounding"] = _normalize_temporal_grounding(parsed.get("temporal_grounding"))
    out["segment_qa"] = _normalize_segment_qa(parsed.get("segment_qa"))
    out["understanding_qa"] = _normalize_understanding_qa(parsed.get("understanding_qa"))
    return out


def _extract_content_text(llm_response: dict[str, Any]) -> str:
    choices = llm_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = trimmed.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if "{" not in trimmed:
        return None
    try:
        start = trimmed.index("{")
        end = trimmed.rindex("}") + 1
        parsed = json.loads(trimmed[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _normalize_temporal_grounding(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        query = item.get("query") or item.get("question")
        answer = item.get("answer")
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        out.append(
            {
                "query": query,
                "answer": answer,
                "start_sec": _to_float_or_none(item.get("start_sec")),
                "end_sec": _to_float_or_none(item.get("end_sec")),
                "answer_type": "temporal_span",
            }
        )
    return out


def _normalize_segment_qa(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        query = item.get("query") or item.get("question")
        answer = item.get("answer")
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        out.append(
            {
                "query": query,
                "answer": answer,
                "start_sec": _to_float_or_none(item.get("start_sec")),
                "end_sec": _to_float_or_none(item.get("end_sec")),
                "answer_type": "caption",
            }
        )
    return out


def _normalize_understanding_qa(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if not isinstance(item, dict):
            continue
        query = item.get("query") or item.get("question")
        answer = item.get("answer")
        if not isinstance(query, str) or not isinstance(answer, str):
            continue
        out.append(
            {
                "query": query,
                "answer": answer,
                "category": str(item.get("category") or ""),
                "answer_type": "open_ended",
            }
        )
    return out


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
