from __future__ import annotations

import time
from pathlib import Path

import orjson
import ray

from pipeline.core.sink import ResultSink


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


def test_sink_deduplicate_and_flush(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    error_path = tmp_path / "errors.jsonl"

    sink = ResultSink.remote(
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        processed_ids={"0"},
        id_field="id",
        dump_every_n=100,
        dump_interval_sec=3600,
    )

    ray.get(
        sink.add_results.remote(
            [
                {"id": "1", "result": {"ok": True}},
                {"id": "1", "result": {"ok": True}},
                {"id": "0", "result": {"ok": True}},
                {"id": "2", "result": {"ok": True}},
            ]
        )
    )
    ray.get(
        sink.add_errors.remote(
            [
                {"id": "err-1", "error": "bad", "stage": "llm_request", "attempts": 1},
            ]
        )
    )
    ray.get(sink.flush.remote(force=True))

    results = _read_jsonl(output_path)
    errors = _read_jsonl(error_path)
    result_ids = {record["id"] for record in results}

    assert result_ids == {"1", "2"}
    assert len(errors) == 1
    assert errors[0]["id"] == "err-1"

    stats = ray.get(sink.finalize.remote())
    assert stats["accepted"] == 2
    assert stats["duplicates"] == 2
    assert stats["errors"] == 1
    assert stats["preexisting"] == 1


def test_sink_time_based_flush(tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    error_path = tmp_path / "errors.jsonl"

    sink = ResultSink.remote(
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        processed_ids=set(),
        id_field="id",
        dump_every_n=1000,
        dump_interval_sec=0.02,
    )
    ray.get(sink.add_results.remote([{"id": "1", "result": {"ok": True}}]))

    # First flush call should not flush because interval is not reached.
    first_flush = ray.get(sink.flush.remote(force=False))
    assert first_flush["flushed"] is False

    time.sleep(0.05)
    second_flush = ray.get(sink.flush.remote(force=False))
    assert second_flush["flushed"] is True
    assert output_path.exists()
