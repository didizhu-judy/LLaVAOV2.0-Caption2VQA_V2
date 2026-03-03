from __future__ import annotations

import asyncio
from typing import Any

import httpx
import ray

from pipeline.core.config import PipelineConfig
from pipeline.core.routing import EndpointRouter
from pipeline.providers.base import EndpointConfig
from pipeline.providers.registry import get_provider
from pipeline.tasks import get_task_plugin
from pipeline.types import ErrorRecord, ProcessedRecord, RawItem, WorkerSummary

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@ray.remote
def worker_main(
    worker_id: int,
    dispatcher_handle: Any,
    sink_handle: Any,
    config_dict: dict[str, Any],
) -> WorkerSummary:
    config = PipelineConfig.from_mapping(config_dict, apply_env_overrides=False)
    return asyncio.run(_worker_loop(worker_id, dispatcher_handle, sink_handle, config))


async def _worker_loop(
    worker_id: int,
    dispatcher_handle: Any,
    sink_handle: Any,
    config: PipelineConfig,
) -> WorkerSummary:
    plugin = get_task_plugin(config.task_name)
    endpoints = _load_endpoints(config)
    router = EndpointRouter(
        endpoints=endpoints,
        route_strategy=config.route_strategy,
        route_failover=config.route_failover,
    )
    providers = {name: get_provider(name) for name in {ep.provider for ep in endpoints}}
    endpoint_sems = _build_endpoint_semaphores(endpoints, config.worker_async_concurrency)

    summary: WorkerSummary = {
        "worker_id": worker_id,
        "fetched_batches": 0,
        "processed": 0,
        "errors": 0,
    }

    default_timeout = httpx.Timeout(config.llm_timeout_sec)
    limits = httpx.Limits(
        max_connections=max(1, config.worker_async_concurrency * 2),
        max_keepalive_connections=max(1, config.worker_async_concurrency),
    )

    async with httpx.AsyncClient(timeout=default_timeout, limits=limits) as client:
        while True:
            batch: list[RawItem] | None = ray.get(
                dispatcher_handle.next_batch.remote(config.fetch_batch_size)
            )
            if batch is None:
                break

            summary["fetched_batches"] += 1
            results, errors = await _process_batch(
                worker_id=worker_id,
                batch=batch,
                config=config,
                client=client,
                plugin=plugin,
                router=router,
                providers=providers,
                endpoint_sems=endpoint_sems,
            )

            if results:
                ray.get(sink_handle.add_results.remote(results))
                summary["processed"] += len(results)
            if errors:
                ray.get(sink_handle.add_errors.remote(errors))
                summary["errors"] += len(errors)

    return summary


async def _process_batch(
    worker_id: int,
    batch: list[RawItem],
    config: PipelineConfig,
    client: httpx.AsyncClient,
    plugin: Any,
    router: EndpointRouter,
    providers: dict[str, Any],
    endpoint_sems: list[asyncio.Semaphore | None],
) -> tuple[list[ProcessedRecord], list[ErrorRecord]]:
    queue_size = max(config.worker_queue_size, config.worker_async_concurrency)
    queue: asyncio.Queue[RawItem | None] = asyncio.Queue(maxsize=queue_size)
    results: list[ProcessedRecord] = []
    errors: list[ErrorRecord] = []
    concurrency = max(1, config.worker_async_concurrency)

    async def consumer() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return

            result, error = await _process_item(
                worker_id=worker_id,
                item=item,
                config=config,
                client=client,
                plugin=plugin,
                router=router,
                providers=providers,
                endpoint_sems=endpoint_sems,
            )
            if result is not None:
                results.append(result)
            if error is not None:
                errors.append(error)
            queue.task_done()

    consumers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
    for item in batch:
        await queue.put(item)
    for _ in consumers:
        await queue.put(None)

    await queue.join()
    await asyncio.gather(*consumers)
    return results, errors


async def _process_item(
    worker_id: int,
    item: RawItem,
    config: PipelineConfig,
    client: httpx.AsyncClient,
    plugin: Any,
    router: EndpointRouter,
    providers: dict[str, Any],
    endpoint_sems: list[asyncio.Semaphore | None],
) -> tuple[ProcessedRecord | None, ErrorRecord | None]:
    item_id = _extract_item_id(item, config.id_field)
    if item_id is None:
        return None, plugin.on_error(
            item,
            ValueError(f"Missing id field: {config.id_field}"),
            stage="validate",
            attempts=0,
            worker_id=worker_id,
            config=config,
        )

    try:
        request_spec = await asyncio.to_thread(plugin.build_request, item, config)
    except Exception as exc:  # noqa: BLE001
        return None, plugin.on_error(
            item,
            exc,
            stage="build_request",
            attempts=1,
            worker_id=worker_id,
            config=config,
        )

    if request_spec.skip_http:
        try:
            processed = plugin.parse_response(item, request_spec.local_response or {}, config)
            processed[config.id_field] = str(processed.get(config.id_field, item_id))
            processed["worker_id"] = worker_id
            return processed, None
        except Exception as exc:  # noqa: BLE001
            return None, plugin.on_error(
                item,
                exc,
                stage="parse_response",
                attempts=1,
                worker_id=worker_id,
                config=config,
            )

    attempts = 0
    llm_response: dict[str, Any] | None = None
    current_request_spec = request_spec
    max_prefill_400_retries = 2  # 最多 2 次用更严截断重试

    while True:
        attempts += 1
        pick = router.pick(item_id=item_id, attempt=attempts)
        endpoint = pick.endpoint
        provider = providers[endpoint.provider]

        try:
            if current_request_spec.url_override:
                prepared_url = current_request_spec.url_override
                prepared_payload = current_request_spec.payload
                prepared_headers = {**config.llm_headers, **current_request_spec.headers}
                timeout_sec = None
            else:
                prepared = provider.prepare_request(
                    endpoint=endpoint,
                    payload=current_request_spec.payload,
                    headers={**config.llm_headers, **current_request_spec.headers},
                )
                prepared_url = prepared.url
                prepared_payload = prepared.payload
                prepared_headers = prepared.headers
                timeout_sec = prepared.timeout_sec

            sem = endpoint_sems[pick.index]
            router.mark_start(pick.index)
            try:
                if sem is not None:
                    async with sem:
                        response = await client.post(
                            prepared_url,
                            json=prepared_payload,
                            headers=prepared_headers,
                            timeout=timeout_sec,
                        )
                else:
                    response = await client.post(
                        prepared_url,
                        json=prepared_payload,
                        headers=prepared_headers,
                        timeout=timeout_sec,
                    )
            finally:
                router.mark_done(pick.index)

            if response.status_code in RETRYABLE_STATUS_CODES:
                raise httpx.HTTPStatusError(
                    f"Retryable HTTP status code: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if response.status_code == 400 and max_prefill_400_retries > 0:
                body = (response.text or "").lower()
                if "too long" in body or "multimodal prompt" in body or "token" in body or "origin_input_ids" in body:
                    try:
                        config_dict = config.to_dict()
                        config_copy = PipelineConfig.from_mapping(
                            config_dict, apply_env_overrides=False
                        )
                        config_copy.task_config = dict(config.task_config)
                        cur = config_copy.task_config.get("max_prefill_tokens") or 10240
                        # 第一次减半，第二次固定 2048
                        next_prefill = max(2048, cur // 2) if cur > 2048 else 2048
                        config_copy.task_config["max_prefill_tokens"] = next_prefill
                        current_request_spec = await asyncio.to_thread(plugin.build_request, item, config_copy)
                        max_prefill_400_retries -= 1
                        attempts -= 1
                        continue
                    except Exception:
                        pass
            if response.status_code != 200:
                if response.status_code == 400 and max_prefill_400_retries <= 0:
                    body_snippet = (response.text or "")[:600]
                    raise httpx.HTTPStatusError(
                        f"400 Bad Request: {body_snippet}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
            llm_response = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            if _can_retry(exc, attempts, config.llm_max_retries):
                await asyncio.sleep(config.llm_retry_backoff_sec * (2 ** (attempts - 1)))
                continue
            return None, plugin.on_error(
                item,
                exc,
                stage="llm_request",
                attempts=attempts,
                worker_id=worker_id,
                config=config,
            )

    try:
        processed = plugin.parse_response(item, llm_response or {}, config)
    except Exception as exc:  # noqa: BLE001
        return None, plugin.on_error(
            item,
            exc,
            stage="postprocess",
            attempts=attempts,
            worker_id=worker_id,
            config=config,
        )

    processed[config.id_field] = str(processed.get(config.id_field, item_id))
    processed["worker_id"] = worker_id
    return processed, None



def _load_endpoints(config: PipelineConfig) -> list[EndpointConfig]:
    if not config.endpoints:
        raise ValueError(
            "Runtime endpoints are not resolved. Ensure main loads endpoint registry first."
        )
    return [EndpointConfig.from_mapping(item) for item in config.endpoints]


def _build_endpoint_semaphores(
    endpoints: list[EndpointConfig],
    worker_async_concurrency: int,
) -> list[asyncio.Semaphore | None]:
    sems: list[asyncio.Semaphore | None] = []
    for endpoint in endpoints:
        if endpoint.max_concurrent > 0:
            limit = max(1, min(int(endpoint.max_concurrent), max(1, worker_async_concurrency)))
            sems.append(asyncio.Semaphore(limit))
        else:
            sems.append(None)
    return sems


def _can_retry(exc: Exception, attempts: int, max_retries: int) -> bool:
    if attempts > max_retries:
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _extract_item_id(item: RawItem, id_field: str) -> str | None:
    value = item.get(id_field)
    if value is None:
        return None
    return str(value)
