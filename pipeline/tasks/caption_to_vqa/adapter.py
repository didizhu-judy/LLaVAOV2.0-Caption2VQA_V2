from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.types import RawItem

SECTION_HEADER_RE = re.compile(r"^###\s*\d+\.\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
RANGE_TIME_RE = re.compile(
    r"\*\*\s*(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*seconds?\s*\*\*\s*:\s*(.+?)(?=\n-\s*\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)
POINT_TIME_RE = re.compile(
    r"\*\*\s*At approximately\s*(\d+(?:\.\d+)?)\s*seconds?\s*\*\*\s*:\s*(.+?)(?=\n-\s*\*\*|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def load_caption_items(
    *,
    input_jsonl: str,
    id_field: str = "id",
    max_records: int = 0,
    default_duration_sec: float = 180.0,
) -> list[RawItem]:
    path = Path(input_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Caption input file does not exist: {input_jsonl}")

    items: list[RawItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = parse_caption_record(
                obj=obj,
                input_jsonl=input_jsonl,
                id_field=id_field,
                default_duration_sec=default_duration_sec,
            )
            if parsed is None:
                continue
            items.append(parsed)
            if max_records and len(items) >= max_records:
                break
    return items


def parse_caption_record(
    *,
    obj: dict[str, Any],
    input_jsonl: str,
    id_field: str,
    default_duration_sec: float,
) -> RawItem | None:
    video_id = _video_id_from_record(obj)
    if not video_id:
        return None

    caption_text = _extract_caption_text(obj)
    if not caption_text:
        return None

    duration_sec = _infer_duration_from_video_id(video_id)
    if duration_sec is None:
        duration_sec = _infer_duration_from_filename(input_jsonl, default=default_duration_sec)

    sections = _parse_caption_sections(caption_text)
    events = extract_events(
        motion_detail_description=sections.get("motion_detail_description", ""),
        highlight_moments=sections.get("highlight_moments", ""),
    )

    structured_caption = _build_structured_caption(sections)
    images_source = obj.get("images_source")
    if isinstance(images_source, list):
        images_source = [x for x in images_source if isinstance(x, str) and x.strip()]
    else:
        images_source = []

    item: RawItem = {
        id_field: video_id,
        "video_id": video_id,
        "duration_sec": float(duration_sec),
        "sections": sections,
        "structured_caption": structured_caption,
        "segments": events,
        "raw_record": obj,
        "images_source": images_source,
    }
    return item


def _extract_caption_text(obj: dict[str, Any]) -> str:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _parse_caption_sections(caption_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in caption_text.split("\n"):
        match = SECTION_HEADER_RE.match(line.strip())
        if match:
            if current_key and current_lines:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = _normalize_section_title(match.group(1))
            current_lines = []
            continue
        if current_key:
            current_lines.append(line)

    if current_key and current_lines:
        sections[current_key] = "\n".join(current_lines).strip()
    return sections


def _normalize_section_title(title: str) -> str | None:
    title_norm = title.strip().lower()
    if "context" in title_norm and "environment" in title_norm:
        return "context_and_environment"
    if "main subject" in title_norm:
        return "main_subject"
    if "actions" in title_norm and "interaction" in title_norm:
        return "actions_and_interactions"
    if "motion" in title_norm and "detail" in title_norm:
        return "motion_detail_description"
    if "background" in title_norm and "change" in title_norm:
        return "background_changes"
    if "highlight" in title_norm:
        return "highlight_moments"
    return None


def extract_events(
    *,
    motion_detail_description: str,
    highlight_moments: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for start, end, description in RANGE_TIME_RE.findall(motion_detail_description or ""):
        start_sec = float(start)
        end_sec = float(end)
        if end_sec <= start_sec:
            continue
        events.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "description": _clean_event_description(description),
                "source": "motion_detail_description",
                "type": "range",
            }
        )

    for point, description in POINT_TIME_RE.findall(highlight_moments or ""):
        point_sec = float(point)
        events.append(
            {
                "start_sec": point_sec,
                "end_sec": None,
                "description": _clean_event_description(description),
                "source": "highlight_moments",
                "type": "point",
            }
        )

    events.sort(key=lambda item: (float(item["start_sec"]), 1 if item["end_sec"] is None else 0))
    return events


def _clean_event_description(description: str) -> str:
    return " ".join(description.strip().split())


def _build_structured_caption(sections: dict[str, str]) -> str:
    ordered = [
        ("### 0. Context and Environment", "context_and_environment"),
        ("### 1. Main subject of the video", "main_subject"),
        ("### 2. Actions and Interactions", "actions_and_interactions"),
        ("### 3. Motion Detail Description", "motion_detail_description"),
        ("### 4. Background Changes", "background_changes"),
        ("### 5. Highlight Moments", "highlight_moments"),
    ]
    blocks: list[str] = []
    for header, key in ordered:
        value = sections.get(key)
        if value:
            blocks.append(f"{header}\n{value}")
    return "\n\n".join(blocks)


def _video_id_from_record(obj: dict[str, Any]) -> str:
    images_source = obj.get("images_source")
    if isinstance(images_source, list) and images_source:
        first = images_source[0]
        if isinstance(first, str) and first.strip():
            return Path(first.strip()).stem
    raw_id = obj.get("id")
    if isinstance(raw_id, str):
        return raw_id.strip()
    return ""


def _infer_duration_from_video_id(video_id: str) -> float | None:
    match = re.search(r"_d(\d+(?:\.\d+)?)(?:_|$)", video_id)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _infer_duration_from_filename(path: str, *, default: float) -> float:
    name = Path(path).name
    match = re.search(r"(\d+)s(?:_\d+w)?\.jsonl", name, re.IGNORECASE)
    if not match:
        return default
    try:
        return float(match.group(1))
    except ValueError:
        return default
