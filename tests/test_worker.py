from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import orjson
import ray

from pipeline.core.config import PipelineConfig
from pipeline.core.dispatcher import DataDispatcher
from pipeline.core.sink import ResultSink
from pipeline.core.worker import worker_main
from pipeline.tasks.base import RequestSpec


class _DummyTask:
    def load_items(self, config):  # noqa: ANN001
        return []

    def build_request(self, item, config):  # noqa: ANN001
        return RequestSpec(payload={"input": item})

    def parse_response(self, item, llm_response, config):  # noqa: ANN001
        return {
            config.id_field: str(item[config.id_field]),
            "result": {
                "attempt": llm_response.get("attempt"),
            },
        }

    def on_error(self, item, error, *, stage, attempts, worker_id, config):  # noqa: ANN001
        return {
            config.id_field: str(item.get(config.id_field, "")),
            "error": str(error),
            "stage": stage,
            "attempts": attempts,
            "worker_id": worker_id,
        }


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


def test_worker_retry_and_error_paths(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "results.jsonl"
    error_path = tmp_path / "errors.jsonl"
    counters: dict[str, int] = {}
    url_history: dict[str, list[str]] = {}

    async def fake_post(self, url, json=None, headers=None, timeout=None):  # noqa: ANN001
        del headers, timeout
        input_item = (json or {}).get("input", {})
        item_id = str(input_item.get("id", "unknown"))
        url_history.setdefault(item_id, []).append(str(url))
        counters[item_id] = counters.get(item_id, 0) + 1
        seen = counters[item_id]
        request = httpx.Request("POST", str(url))

        if item_id == "retry" and seen == 1:
            return httpx.Response(500, request=request, json={"error": "temporary"})
        if item_id == "bad":
            return httpx.Response(400, request=request, json={"error": "bad request"})
        return httpx.Response(200, request=request, json={"ok": True, "id": item_id, "attempt": seen})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr("pipeline.core.worker.get_task_plugin", lambda _: _DummyTask())

    dispatcher = DataDispatcher.remote(
        items=[{"id": "ok"}, {"id": "retry"}, {"id": "bad"}],
        processed_ids=set(),
        id_field="id",
        fetch_batch_size=8,
    )
    sink = ResultSink.remote(
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        processed_ids=set(),
        id_field="id",
        dump_every_n=2,
        dump_interval_sec=0.1,
    )
    config = PipelineConfig(
        task_name="caption_to_vqa",
        num_workers=1,
        fetch_batch_size=8,
        worker_async_concurrency=3,
        worker_queue_size=16,
        dump_interval_sec=1.0,
        dump_every_n=100,
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        resume=False,
        id_field="id",
        llm_urls=[
            "http://mock.local:10025/v1/chat/completions",
            "http://mock.local:10026/v1/chat/completions",
        ],
        endpoint_registry_file=str(tmp_path / "missing_endpoints.json"),
        endpoint_group="local_multi",
        llm_timeout_sec=5.0,
        llm_max_retries=2,
        llm_retry_backoff_sec=0.01,
        ray_local_mode=True,
    )
    config.endpoints = [
        {
            "name": "mock-0",
            "provider": "openai_compatible",
            "url": "http://mock.local:10025/v1/chat/completions",
            "auth_type": "none",
        },
        {
            "name": "mock-1",
            "provider": "openai_compatible",
            "url": "http://mock.local:10026/v1/chat/completions",
            "auth_type": "none",
        },
    ]

    summary = ray.get(
        worker_main.remote(
            worker_id=0,
            dispatcher_handle=dispatcher,
            sink_handle=sink,
            config_dict=config.to_dict(),
        )
    )
    sink_stats = ray.get(sink.finalize.remote())

    results = _read_jsonl(output_path)
    errors = _read_jsonl(error_path)
    result_ids = {record["id"] for record in results}

    assert summary["processed"] == 2
    assert summary["errors"] == 1
    assert sink_stats["accepted"] == 2
    assert sink_stats["errors"] == 1
    assert result_ids == {"ok", "retry"}
    assert errors[0]["id"] == "bad"
    assert counters["retry"] == 2
    assert counters["bad"] == 1
    assert url_history["retry"][0] != url_history["retry"][1]
