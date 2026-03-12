from __future__ import annotations

import glob
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator

from pipeline.tasks.benchmark_similarity.profiles import infer_example_profile


def default_molmo2_qa_sources(root: str = "/ov2/dataset_molmo2") -> list[dict[str, Any]]:
    return default_molmo2_candidate_sources(root)


def default_molmo2_candidate_sources(root: str = "/ov2/dataset_molmo2") -> list[dict[str, Any]]:
    base = Path(root)
    return [
        {
            "name": "molmo2_askmodelanything",
            "path": str(base / "Molmo2-AskModelAnything/data/HumanQA-00000-of-00001.parquet"),
            "question_field": "question",
            "answer_field": "answer",
            "id_field": "video_id",
            "metadata_fields": ["video_id"],
        },
        {
            "name": "molmo2_videocapqa",
            "path": str(base / "Molmo2-VideoCapQA/data/CapQA-00000-of-00001.parquet"),
            "question_field": "Question",
            "answer_field": "Answer",
            "context_fields": ["Category"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "Category"],
        },
        {
            "name": "molmo2_longcapqa",
            "path": str(base / "Molmo2-VideoCapQA/data/LongCapQA-00000-of-00001.parquet"),
            "explode_field": "qa_list",
            "question_field": "Question",
            "answer_field": "Answer",
            "context_fields": ["Category"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "Category"],
        },
        {
            "name": "molmo2_cap",
            "path": str(base / "Molmo2-Cap/data/*.parquet"),
            "answer_field": "video_frame_merged_caption",
            "text_mode": "answer_only",
            "include_context": True,
            "context_fields": ["video_transcript", "merged_caption"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "annotation_score"],
        },
        {
            "name": "molmo2_videosubtitleqa",
            "path": str(base / "Molmo2-VideoSubtitleQA/data/*.parquet"),
            "question_field": "Question",
            "answer_field": "Answer",
            "context_fields": ["Category"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "Category", "AlignmentType"],
        },
        {
            "name": "molmo2_videocounteval",
            "path": str(base / "Molmo2-VideoCountEval/data/*.parquet"),
            "question_field": "question",
            "answer_field": "label",
            "context_fields": ["category"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "category", "video_duration", "video_source"],
        },
        {
            "name": "molmo2_videopoint",
            "path": str(base / "Molmo2-VideoPoint/data/train-00000-of-00001.parquet"),
            "question_field": "question",
            "answer_field": "label",
            "context_fields": ["category"],
            "id_field": "video_id",
            "metadata_fields": ["video_id", "category", "video_duration", "video_source"],
        },
    ]


def load_examples_from_sources(
    sources: list[dict[str, Any]],
    *,
    role: str,
    id_field_name: str,
    text_mode: str,
    max_records: int = 0,
    include_context: bool = False,
    max_text_chars: int = 1200,
    dedupe_by_text: bool = False,
    skip_missing_paths: bool = True,
) -> list[dict[str, Any]]:
    if not sources:
        raise ValueError(f"{role}_sources is empty")

    examples: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for source_index, spec in enumerate(sources):
        source_name = str(spec.get("name") or f"{role}_{source_index}").strip()
        source_limit = int(spec.get("limit") or 0)
        loaded_for_source = 0

        for file_path in _expand_paths(spec.get("path")):
            path_obj = Path(file_path)
            if not path_obj.exists():
                message = f"{role} source path does not exist: {file_path}"
                if skip_missing_paths:
                    warnings.warn(message, stacklevel=2)
                    continue
                raise FileNotFoundError(message)

            for row_index, parent_record in enumerate(_iter_records(path_obj, spec)):
                explode_field = str(spec.get("explode_field") or "").strip()
                exploded_items = _resolve_exploded_items(parent_record, explode_field)

                for explode_index, exploded_record in exploded_items:
                    example = _build_example(
                        parent_record=parent_record,
                        exploded_record=exploded_record,
                        source_name=source_name,
                        source_path=str(path_obj),
                        row_index=row_index,
                        explode_index=explode_index,
                        spec=spec,
                        id_field_name=id_field_name,
                        text_mode=text_mode,
                        include_context=include_context,
                        max_text_chars=max_text_chars,
                    )
                    if example is None:
                        continue

                    example["profile"] = infer_example_profile(example, role=role, spec=spec)

                    if dedupe_by_text:
                        dedupe_key = str(example.get("similarity_text") or "").strip().lower()
                        if dedupe_key in seen_texts:
                            continue
                        seen_texts.add(dedupe_key)

                    examples.append(example)
                    loaded_for_source += 1

                    if source_limit and loaded_for_source >= source_limit:
                        break
                    if max_records and len(examples) >= max_records:
                        return examples

                if source_limit and loaded_for_source >= source_limit:
                    break
            if source_limit and loaded_for_source >= source_limit:
                break

    if not examples:
        raise ValueError(f"No examples loaded from {role}_sources")
    return examples


def _expand_paths(path_value: Any) -> list[str]:
    if isinstance(path_value, (list, tuple)):
        out: list[str] = []
        for item in path_value:
            out.extend(_expand_paths(item))
        return out

    path_str = str(path_value or "").strip()
    if not path_str:
        return []
    if any(ch in path_str for ch in "*?[]"):
        return sorted(glob.glob(path_str))
    return [path_str]


def _iter_records(path: Path, spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        yield from _iter_jsonl_records(path)
        return
    if suffix == ".json":
        yield from _iter_json_records(path, record_field=str(spec.get("record_field") or ""))
        return
    if suffix == ".parquet":
        yield from _iter_parquet_records(path, spec)
        return
    raise ValueError(f"Unsupported source file format: {path}")


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, dict):
                yield record


def _iter_json_records(path: Path, *, record_field: str) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: Iterable[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and record_field:
        nested = payload.get(record_field)
        if not isinstance(nested, list):
            raise ValueError(f"record_field '{record_field}' is not a list in {path}")
        records = nested
    elif isinstance(payload, dict):
        records = [payload]
    else:
        raise ValueError(f"Unsupported JSON payload in {path}: {type(payload).__name__}")

    for record in records:
        if isinstance(record, dict):
            yield record


def _iter_parquet_records(path: Path, spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    columns = _collect_parquet_columns(spec)
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns or None)
        for record in table.to_pylist():
            if isinstance(record, dict):
                yield record
        return
    except ImportError:
        pass

    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=columns or None)
        for record in frame.to_dict(orient="records"):
            if isinstance(record, dict):
                yield record
        return
    except ImportError as exc:
        raise ImportError(
            "Reading parquet requires either pyarrow or pandas to be installed."
        ) from exc


def _collect_parquet_columns(spec: dict[str, Any]) -> list[str]:
    explode_field = str(spec.get("explode_field") or "").strip()
    id_field = str(spec.get("id_field") or "").strip()
    raw_fields: list[Any] = [
        id_field,
        spec.get("question_field"),
        spec.get("answer_field"),
        explode_field,
        *(spec.get("context_fields") or []),
        *(spec.get("metadata_fields") or []),
    ]

    columns: set[str] = set()
    for field in raw_fields:
        field_str = str(field or "").strip()
        if not field_str:
            continue
        if explode_field and "." not in field_str and field_str not in {explode_field, id_field}:
            # In exploded parquet rows, question/answer/category usually live inside the exploded
            # list/struct column and are not valid top-level parquet columns.
            continue
        top_level = field_str.split(".", 1)[0]
        columns.add(top_level)
    return sorted(columns)


def _resolve_exploded_items(
    parent_record: dict[str, Any],
    explode_field: str,
) -> list[tuple[int | None, dict[str, Any] | None]]:
    if not explode_field:
        return [(None, None)]

    value = _get_by_path(parent_record, explode_field)
    if not isinstance(value, list):
        return [(None, None)]

    items: list[tuple[int | None, dict[str, Any] | None]] = []
    for idx, item in enumerate(value):
        if isinstance(item, dict):
            items.append((idx, item))
    return items or [(None, None)]


def _build_example(
    *,
    parent_record: dict[str, Any],
    exploded_record: dict[str, Any] | None,
    source_name: str,
    source_path: str,
    row_index: int,
    explode_index: int | None,
    spec: dict[str, Any],
    id_field_name: str,
    text_mode: str,
    include_context: bool,
    max_text_chars: int,
) -> dict[str, Any] | None:
    question = _extract_text_field(
        spec=spec,
        field_key="question_field",
        parent_record=parent_record,
        exploded_record=exploded_record,
    )
    answer = _extract_text_field(
        spec=spec,
        field_key="answer_field",
        parent_record=parent_record,
        exploded_record=exploded_record,
    )
    context_fields = spec.get("context_fields") or []
    context_parts = [
        _extract_text_path(field, parent_record=parent_record, exploded_record=exploded_record)
        for field in context_fields
    ]
    context = "\n".join(part for part in context_parts if part)

    resolved_text_mode = str(spec.get("text_mode") or text_mode)
    resolved_include_context = include_context or bool(spec.get("include_context"))
    similarity_text = build_similarity_text(
        question=question,
        answer=answer,
        context=context,
        text_mode=resolved_text_mode,
        include_context=resolved_include_context,
        max_text_chars=max_text_chars,
    )
    if not similarity_text:
        return None

    source_record_id = _extract_source_record_id(
        spec=spec,
        parent_record=parent_record,
        exploded_record=exploded_record,
        row_index=row_index,
        explode_index=explode_index,
    )
    item_id = _make_item_id(
        source_name=source_name,
        source_path=source_path,
        source_record_id=source_record_id,
        row_index=row_index,
        explode_index=explode_index,
    )

    metadata_fields = list(spec.get("metadata_fields") or [])
    metadata: dict[str, Any] = {}
    for field in metadata_fields:
        field_name = str(field or "").strip()
        if not field_name:
            continue
        value = _extract_value_path(field_name, parent_record=parent_record, exploded_record=exploded_record)
        if value is not None:
            metadata[field_name] = value

    return {
        id_field_name: item_id,
        "source_name": source_name,
        "source_path": source_path,
        "source_record_id": source_record_id,
        "question": question,
        "answer": answer,
        "context": context,
        "similarity_text": similarity_text,
        "metadata": metadata,
    }


def build_similarity_text(
    *,
    question: str,
    answer: str,
    context: str,
    text_mode: str,
    include_context: bool,
    max_text_chars: int,
) -> str:
    mode = str(text_mode or "qa_concat").strip().lower()
    parts: list[str] = []

    if mode == "question_only":
        if question:
            parts.append(f"Question: {question}")
    elif mode == "answer_only":
        if answer:
            parts.append(f"Answer: {answer}")
    else:
        if question:
            parts.append(f"Question: {question}")
        if answer:
            parts.append(f"Answer: {answer}")

    if include_context and context:
        parts.append(f"Context: {context}")

    text = "\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        return ""
    if max_text_chars > 0 and len(text) > max_text_chars:
        return text[: max_text_chars - 12].rstrip() + " [truncated]"
    return text


def _extract_source_record_id(
    *,
    spec: dict[str, Any],
    parent_record: dict[str, Any],
    exploded_record: dict[str, Any] | None,
    row_index: int,
    explode_index: int | None,
) -> str:
    field_name = str(spec.get("id_field") or "").strip()
    raw_value = _extract_value_path(
        field_name,
        parent_record=parent_record,
        exploded_record=exploded_record,
    ) if field_name else None

    if raw_value is not None and raw_value != "":
        base = _stringify_value(raw_value)
    else:
        digest = hashlib.md5(
            json.dumps(parent_record, sort_keys=True, ensure_ascii=False).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:10]
        base = f"row{row_index}_{digest}"

    if explode_index is not None:
        return f"{base}#e{explode_index}"
    return base


def _make_item_id(
    *,
    source_name: str,
    source_path: str,
    source_record_id: str,
    row_index: int,
    explode_index: int | None,
) -> str:
    path_stem = Path(source_path).stem
    suffix = f":r{row_index}"
    if explode_index is not None:
        suffix += f"e{explode_index}"
    return f"{source_name}:{path_stem}:{source_record_id}{suffix}"


def _extract_text_field(
    *,
    spec: dict[str, Any],
    field_key: str,
    parent_record: dict[str, Any],
    exploded_record: dict[str, Any] | None,
) -> str:
    field_name = str(spec.get(field_key) or "").strip()
    if field_name:
        return _extract_text_path(field_name, parent_record=parent_record, exploded_record=exploded_record)

    if field_key == "question_field":
        for candidate in ("question", "Question", "query", "prompt"):
            text = _extract_text_path(candidate, parent_record=parent_record, exploded_record=exploded_record)
            if text:
                return text
    if field_key == "answer_field":
        for candidate in ("answer", "Answer", "correct_answer", "gt_answer", "label"):
            text = _extract_text_path(candidate, parent_record=parent_record, exploded_record=exploded_record)
            if text:
                return text
    return ""


def _extract_text_path(
    path: str,
    *,
    parent_record: dict[str, Any],
    exploded_record: dict[str, Any] | None,
) -> str:
    value = _extract_value_path(path, parent_record=parent_record, exploded_record=exploded_record)
    return _stringify_value(value)


def _extract_value_path(
    path: str,
    *,
    parent_record: dict[str, Any],
    exploded_record: dict[str, Any] | None,
) -> Any:
    if exploded_record is not None:
        value = _get_by_path(exploded_record, path)
        if value is not None:
            return value
    return _get_by_path(parent_record, path)


def _get_by_path(record: dict[str, Any], path: str) -> Any:
    if not path:
        return None
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if "text" in value:
            return _stringify_value(value.get("text"))
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        if not value:
            return ""
        text_like = [_stringify_value(item) for item in value[:20]]
        text_like = [item for item in text_like if item]
        return " | ".join(text_like)
    return str(value)
