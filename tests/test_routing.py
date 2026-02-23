from __future__ import annotations

from pipeline.core.routing import EndpointRouter, pick_endpoint_index
from pipeline.providers.base import EndpointConfig


def _build_endpoints() -> list[EndpointConfig]:
    return [
        EndpointConfig(name="e1", provider="openai_compatible", url="u1", weight=1.0),
        EndpointConfig(name="e2", provider="openai_compatible", url="u2", weight=1.0),
        EndpointConfig(name="e3", provider="openai_compatible", url="u3", weight=2.0),
    ]


def test_stable_hash_route_is_deterministic() -> None:
    endpoints = _build_endpoints()
    first = pick_endpoint_index(item_id="sample-1", endpoints=endpoints, attempt=1)
    second = pick_endpoint_index(item_id="sample-1", endpoints=endpoints, attempt=1)
    assert first == second


def test_rotate_on_retry_changes_endpoint() -> None:
    endpoints = _build_endpoints()
    first = pick_endpoint_index(
        item_id="sample-2",
        endpoints=endpoints,
        attempt=1,
        route_failover="rotate_on_retry",
    )
    second = pick_endpoint_index(
        item_id="sample-2",
        endpoints=endpoints,
        attempt=2,
        route_failover="rotate_on_retry",
    )
    assert first != second


def test_same_endpoint_failover_keeps_endpoint() -> None:
    endpoints = _build_endpoints()
    first = pick_endpoint_index(
        item_id="sample-3",
        endpoints=endpoints,
        attempt=1,
        route_failover="same_endpoint",
    )
    second = pick_endpoint_index(
        item_id="sample-3",
        endpoints=endpoints,
        attempt=3,
        route_failover="same_endpoint",
    )
    assert first == second


def test_least_inflight_weighted_prefers_higher_weight_endpoint() -> None:
    endpoints = _build_endpoints()
    router = EndpointRouter(
        endpoints=endpoints,
        route_strategy="least_inflight_weighted",
        route_failover="same_endpoint",
    )

    # Overload low-weight endpoints and ensure weighted endpoint is selected.
    router.mark_start(0)
    router.mark_start(0)
    router.mark_start(1)

    pick = router.pick(item_id="sample-4", attempt=1)
    assert pick.index == 2
