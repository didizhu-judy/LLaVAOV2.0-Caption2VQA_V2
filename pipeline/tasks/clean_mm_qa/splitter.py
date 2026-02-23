from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


def split_clean_dirty(
    *,
    judged_output_jsonl: str,
    clean_output_jsonl: str,
    dirty_output_jsonl: str,
) -> dict[str, Any]:
    output_path = Path(judged_output_jsonl)
    if not output_path.exists():
        return {
            "clean_output_jsonl": clean_output_jsonl,
            "dirty_output_jsonl": dirty_output_jsonl,
            "clean_count": 0,
            "dirty_count": 0,
        }

    clean_path = Path(clean_output_jsonl)
    dirty_path = Path(dirty_output_jsonl)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    dirty_path.parent.mkdir(parents=True, exist_ok=True)

    clean_count = 0
    dirty_count = 0

    with (
        output_path.open("rb") as source,
        clean_path.open("wb") as clean_handle,
        dirty_path.open("wb") as dirty_handle,
    ):
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            keep = bool(record.get("_clean_keep"))
            if keep:
                out = dict(record)
                out.pop("_clean_keep", None)
                out.pop("_clean_verdict", None)
                clean_handle.write(orjson.dumps(out))
                clean_handle.write(b"\n")
                clean_count += 1
            else:
                dirty_handle.write(orjson.dumps(record))
                dirty_handle.write(b"\n")
                dirty_count += 1

    return {
        "clean_output_jsonl": str(clean_path),
        "dirty_output_jsonl": str(dirty_path),
        "clean_count": clean_count,
        "dirty_count": dirty_count,
    }
