from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
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


def _check_endpoints_reachable(endpoints: list[Any], timeout_sec: float = 2.0) -> int:
    """Quick reachability check; returns count of endpoints that respond (connection ok or any HTTP status)."""
    try:
        import urllib.request
        import urllib.error
        reachable = 0
        for ep in endpoints:
            url = getattr(ep, "url", None) or (ep.get("url") if isinstance(ep, dict) else None)
            if not url:
                continue
            try:
                req = urllib.request.Request(url, method="POST", data=b"{}", headers={"Content-Type": "application/json", "User-Agent": "pipeline-check/1.0"})
                urllib.request.urlopen(req, timeout=timeout_sec)
                reachable += 1
            except urllib.error.HTTPError:
                reachable += 1  # server responded with 4xx/5xx
            except OSError as e:
                if "Connection refused" in str(e) or "timed out" in str(e).lower() or "111" in str(e):
                    pass
                else:
                    reachable += 1
            except Exception:
                pass
        return reachable
    except Exception:
        return 0


def _count_dirty_in_jsonl(path: str) -> int:
    """Count records with _clean_keep is False (for initial_dirty_count on resume)."""
    p = Path(path)
    if not p.exists():
        return 0
    n = 0
    try:
        with p.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = orjson.loads(line)
                    if isinstance(rec, dict) and rec.get("_clean_keep") is False:
                        n += 1
                except orjson.JSONDecodeError:
                    continue
    except OSError:
        pass
    return n


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


def run_pipeline(config: PipelineConfig, *, shutdown_ray_after: bool = True) -> dict[str, Any]:
    plugin = get_task_plugin(config.task_name)
    config.ensure_output_dirs()

    endpoints = resolve_endpoints_for_config(config)
    config.endpoints = [endpoint.to_dict() for endpoint in endpoints]
    n_ep = len(endpoints)
    print(
        f"[pipeline] Resolved {n_ep} endpoint(s): {[e.name for e in endpoints]}",
        file=sys.stderr,
        flush=True,
    )
    reachable = _check_endpoints_reachable(endpoints)
    if reachable < n_ep and n_ep > 1:
        print(
            f"[pipeline] WARNING: Only {reachable}/{n_ep} endpoints reachable. Throughput may be limited. Start all instances (e.g. run_local_sglang.sh) or check ports.",
            file=sys.stderr,
            flush=True,
        )
    elif reachable == n_ep and n_ep > 1:
        print(f"[pipeline] All {n_ep} endpoints reachable.", file=sys.stderr, flush=True)

    items = plugin.load_items(config)
    if config.resume:
        out_path = config.output_jsonl
        processed_ids = None
        if hasattr(plugin, "get_processed_ids_for_resume") and callable(
            getattr(plugin, "get_processed_ids_for_resume")
        ):
            processed_ids = plugin.get_processed_ids_for_resume(config, items)
        if processed_ids is None:
            processed_ids = load_processed_ids(out_path, config.id_field)
    else:
        processed_ids = set()

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
        dirty_count_field = (
            config.task_config.get("sink_dirty_count_field")
            or ("_clean_keep" if config.task_name == "clean_mm_qa" else None)
        )
        initial_dirty = 0
        if dirty_count_field and processed_ids and Path(config.output_jsonl).exists():
            initial_dirty = _count_dirty_in_jsonl(config.output_jsonl)
        sink = ResultSink.remote(
            output_jsonl=config.output_jsonl,
            error_jsonl=config.error_jsonl,
            processed_ids=processed_ids,
            id_field=config.id_field,
            dump_every_n=config.dump_every_n,
            dump_interval_sec=config.dump_interval_sec,
            dirty_count_field=dirty_count_field,
            initial_dirty_count=initial_dirty,
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

        total_items = len(items)
        t0 = time.monotonic()
        last_printed = 0

        worker_summaries: list[dict[str, Any]] = []
        pending = worker_refs
        while pending:
            done, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=config.dump_interval_sec,
            )
            flush_result = ray.get(sink.flush.remote(force=False))
            if total_items > 0 and isinstance(flush_result, dict):
                written = flush_result.get("written_total", 0)
                elapsed = time.monotonic() - t0
                rps = written / elapsed if elapsed > 0 else 0.0
                if written != last_printed:
                    line = f"  [{written}/{total_items}] ({rps:.1f} rec/s)"
                    if "dirty_total" in flush_result:
                        line += f" dirty={flush_result['dirty_total']}"
                    print(line, flush=True, file=sys.stderr)
                    last_printed = written
            if done:
                finished = ray.get(done[0])
                worker_summaries.append(finished)

        sink_stats = ray.get(sink.finalize.remote())
        dispatcher_stats = ray.get(dispatcher.stats.remote())
    finally:
        if started_ray and shutdown_ray_after:
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


def _run_batch_mode(base_config: PipelineConfig, input_list_path: str) -> int:
    """Process multiple files in one process; Ray is inited once and shutdown at the end."""
    task_cfg = dict(base_config.task_config)
    judged_dir = os.environ.get("JUDGED_DIR") or task_cfg.get("judged_dir") or "output/openbee_judged"
    output_clean_dir = os.environ.get("OUTPUT_CLEAN_DIR") or task_cfg.get("output_clean_dir") or "openbee_clean"
    output_dirty_dir = os.environ.get("OUTPUT_DIRTY_DIR") or task_cfg.get("output_dirty_dir") or "openbee_dirty"
    judged_dir = Path(judged_dir)
    output_clean_dir = Path(output_clean_dir)
    output_dirty_dir = Path(output_dirty_dir)

    with Path(input_list_path).open("r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    total = len(paths)
    summaries: list[dict[str, Any]] = []
    for idx, path in enumerate(paths):
        base = Path(path).stem
        file_config_dict = asdict(base_config)
        file_config_dict["output_jsonl"] = str(judged_dir / f"{base}_judged.jsonl")
        file_config_dict["error_jsonl"] = str(judged_dir / f"{base}_clean_errors.jsonl")
        file_config_dict["task_config"] = {
            **task_cfg,
            "input_jsonl": path,
            "clean_output_jsonl": str(output_clean_dir / f"{base}_clean.jsonl"),
            "dirty_output_jsonl": str(output_dirty_dir / f"{base}_dirty.jsonl"),
        }
        file_config = PipelineConfig(**file_config_dict)
        print(f"[{idx + 1}/{total}] {base}", flush=True, file=sys.stderr)
        summary = run_pipeline(file_config, shutdown_ray_after=False)
        summaries.append(summary)
    if ray.is_initialized():
        ray.shutdown()
    print(json.dumps({"batch_files": total, "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


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

    input_list = os.environ.get("INPUT_LIST")
    if input_list and Path(input_list).exists():
        return _run_batch_mode(config, input_list)

    summary = run_pipeline(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
