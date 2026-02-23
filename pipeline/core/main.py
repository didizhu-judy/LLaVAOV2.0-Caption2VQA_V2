from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import orjson
import ray

from pipeline.core.config import PipelineConfig
from pipeline.core.dispatcher import DataDispatcher
from pipeline.core.sink import ResultSink
from pipeline.core.worker import worker_main
from pipeline.providers.registry import resolve_endpoints_for_config
from pipeline.tasks import get_task_plugin, list_task_names


def load_processed_ids(path: str, id_field: str) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()

    processed_ids: set[str] = set()
    with output_path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue
            item_id = record.get(id_field)
            if item_id is None:
                continue
            processed_ids.add(str(item_id))
    return processed_ids


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    plugin = get_task_plugin(config.task_name)
    config.ensure_output_dirs()

    endpoints = resolve_endpoints_for_config(config)
    config.endpoints = [endpoint.to_dict() for endpoint in endpoints]

    items = plugin.load_items(config)
    processed_ids = load_processed_ids(config.output_jsonl, config.id_field) if config.resume else set()

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=config.ray_address,
            local_mode=config.ray_local_mode,
            ignore_reinit_error=True,
            include_dashboard=False,
        )
        started_ray = True

    try:
        dispatcher = DataDispatcher.remote(
            items=items,
            processed_ids=processed_ids,
            id_field=config.id_field,
            fetch_batch_size=config.fetch_batch_size,
        )
        sink = ResultSink.remote(
            output_jsonl=config.output_jsonl,
            error_jsonl=config.error_jsonl,
            processed_ids=processed_ids,
            id_field=config.id_field,
            dump_every_n=config.dump_every_n,
            dump_interval_sec=config.dump_interval_sec,
        )

        worker_refs = [
            worker_main.remote(
                worker_id=worker_id,
                dispatcher_handle=dispatcher,
                sink_handle=sink,
                config_dict=config.to_dict(),
            )
            for worker_id in range(config.num_workers)
        ]

        worker_summaries: list[dict[str, Any]] = []
        pending = worker_refs
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=config.dump_interval_sec,
            )
            ray.get(sink.flush.remote(force=False))
            if done:
                finished = ray.get(done[0])
                worker_summaries.append(finished)

        sink_stats = ray.get(sink.finalize.remote())
        dispatcher_stats = ray.get(dispatcher.stats.remote())
    finally:
        if started_ray:
            ray.shutdown()

    plugin_summary = plugin.finalize_outputs(config)
    return {
        "task_name": config.task_name,
        "input_total": len(items),
        "resume_loaded_ids": len(processed_ids),
        "endpoint_group": config.endpoint_group,
        "endpoint_count": len(endpoints),
        "dispatcher": dispatcher_stats,
        "sink": sink_stats,
        "workers": worker_summaries,
        "task_finalize": plugin_summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ray + asyncio LLM data processing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON or YAML config file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=f"Task name ({', '.join(list_task_names())}).",
    )
    parser.add_argument(
        "--task-config-json",
        type=str,
        default=None,
        help="Task-specific config as JSON object string.",
    )
    parser.add_argument(
        "--endpoint-group",
        type=str,
        default=None,
        help="Endpoint group name defined in endpoint registry file.",
    )
    parser.add_argument(
        "--endpoint-registry-file",
        type=str,
        default=None,
        help="Path to endpoint registry JSON file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig.from_file(args.config)
    if args.task:
        config.task_name = args.task
    if args.task_config_json:
        task_cfg = json.loads(args.task_config_json)
        if not isinstance(task_cfg, dict):
            raise ValueError("--task-config-json must decode to a JSON object")
        config.task_config = {**config.task_config, **task_cfg}
    if args.endpoint_group:
        config.endpoint_group = args.endpoint_group
    if args.endpoint_registry_file:
        config.endpoint_registry_file = args.endpoint_registry_file

    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
