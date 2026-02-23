from __future__ import annotations

import json
from pathlib import Path

import orjson

from pipeline.core.config import PipelineConfig
from pipeline.tasks.clean_mm_qa import CleanMMQATask


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


def test_clean_task_load_build_parse_and_split(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")

    input_path = tmp_path / "input.jsonl"
    record = {
        "id": "sample-1",
        "question": "What object is visible in the image?",
        "answer": "A chair",
        "images": [str(image_path)],
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_path = tmp_path / "judged.jsonl"
    clean_path = tmp_path / "clean.jsonl"
    dirty_path = tmp_path / "dirty.jsonl"

    config = PipelineConfig(
        task_name="clean_mm_qa",
        output_jsonl=str(output_path),
        task_config={
            "input_jsonl": str(input_path),
            "clean_output_jsonl": str(clean_path),
            "dirty_output_jsonl": str(dirty_path),
        },
    )
    task = CleanMMQATask()
    items = task.load_items(config)
    assert len(items) == 1
    assert items[0]["id"] == "sample-1#qa0"

    req = task.build_request(items[0], config)
    assert req.skip_http is False
    assert "messages" in req.payload

    llm_response = {
        "choices": [
            {
                "message": {
                    "content": '{"relevance":"relevant","necessity":"necessary","reason":"Image is required"}'
                }
            }
        ]
    }
    out = task.parse_response(items[0], llm_response, config)
    assert out["_clean_keep"] is True

    with output_path.open("wb") as handle:
        handle.write(orjson.dumps(out))
        handle.write(b"\n")
        dirty_record = dict(out)
        dirty_record["id"] = "sample-2#qa0"
        dirty_record["_clean_keep"] = False
        dirty_record["_clean_verdict"] = {
            "relevance": "irrelevant",
            "necessity": "unnecessary",
            "reason": "question unrelated",
        }
        handle.write(orjson.dumps(dirty_record))
        handle.write(b"\n")

    stats = task.finalize_outputs(config)
    assert stats is not None
    assert stats["clean_count"] == 1
    assert stats["dirty_count"] == 1

    clean_records = _read_jsonl(clean_path)
    dirty_records = _read_jsonl(dirty_path)
    assert len(clean_records) == 1
    assert len(dirty_records) == 1
    assert "_clean_verdict" not in clean_records[0]
