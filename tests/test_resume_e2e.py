from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import orjson

from pipeline.core.config import PipelineConfig
from pipeline.core.main import run_pipeline
from pipeline.tasks.base import RequestSpec


class _DummyTask:
    def __init__(self, items_ref: dict[str, list[dict[str, Any]]]) -> None:
        self._items_ref = items_ref

    def load_items(self, config):  # noqa: ANN001
        return list(self._items_ref["items"])

    def build_request(self, item, config):  # noqa: ANN001
        return RequestSpec(payload={"input": item})

    def parse_response(self, item, llm_response, config):  # noqa: ANN001
        return {
            config.id_field: str(item[config.id_field]),
            "result": {"attempt": llm_response.get("attempt")},
        }

    def on_error(self, item, error, *, stage, attempts, worker_id, config):  # noqa: ANN001
        return {
            config.id_field: str(item.get(config.id_field, "")),
            "error": str(error),
            "stage": stage,
            "attempts": attempts,
            "worker_id": worker_id,
        }

    def finalize_outputs(self, config):  # noqa: ANN001
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


def test_resume_skips_previous_results(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    error_path = tmp_path / "errors.jsonl"
    counters: dict[str, int] = {}
    current_items: dict[str, list[dict[str, object]]] = {
        "items": [{"id": str(i), "value": i} for i in range(3)]
    }

    plugin = _DummyTask(current_items)
    monkeypatch.setattr("pipeline.core.main.get_task_plugin", lambda _: plugin)

    async def fake_post(self, url, json=None, headers=None, timeout=None):  # noqa: ANN001
        del url, headers, timeout
        input_item = (json or {}).get("input", {})
        item_id = str(input_item.get("id", "unknown"))
        counters[item_id] = counters.get(item_id, 0) + 1
        request = httpx.Request("POST", "http://mock.local/llm")
        return httpx.Response(
            200,
            request=request,
            json={"ok": True, "id": item_id, "attempt": counters[item_id]},
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    config = PipelineConfig(
        task_name="caption_to_vqa",
        num_workers=2,
        fetch_batch_size=2,
        worker_async_concurrency=2,
        worker_queue_size=8,
        dump_interval_sec=0.02,
        dump_every_n=1,
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        resume=True,
        id_field="id",
        llm_url="http://mock.local/llm",
        endpoint_registry_file=str(tmp_path / "missing_endpoints.json"),
        endpoint_group="local_multi",
        llm_timeout_sec=5.0,
        llm_max_retries=1,
        llm_retry_backoff_sec=0.01,
        ray_local_mode=True,
    )

    first_summary = run_pipeline(config)
    assert first_summary["dispatcher"]["dispatched"] == 3
    assert first_summary["dispatcher"]["skipped"] == 0

    current_items["items"] = [{"id": str(i), "value": i} for i in range(5)]
    second_summary = run_pipeline(config)

    records = _read_jsonl(output_path)
    assert {record["id"] for record in records} == {"0", "1", "2", "3", "4"}
    assert second_summary["dispatcher"]["skipped"] == 3
    assert second_summary["dispatcher"]["dispatched"] == 2
    assert counters["3"] == 1
    assert counters["4"] == 1
