from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pipeline.providers.base import EndpointConfig


@dataclass(slots=True)
class RoutePick:
    index: int
    endpoint: EndpointConfig


class EndpointRouter:
    def __init__(
        self,
        endpoints: list[EndpointConfig],
        route_strategy: str = "stable_hash",
        route_failover: str = "rotate_on_retry",
    ) -> None:
        if not endpoints:
            raise ValueError("endpoints must not be empty")
        self._endpoints = endpoints
        self._strategy = route_strategy.strip().lower()
        self._failover = route_failover.strip().lower()
        if self._strategy not in {"stable_hash", "least_inflight_weighted"}:
            raise ValueError(f"Unsupported route strategy: {route_strategy}")
        if self._failover not in {"rotate_on_retry", "same_endpoint"}:
            raise ValueError(f"Unsupported route failover: {route_failover}")
        self._inflight: list[int] = [0 for _ in endpoints]

    @property
    def endpoints(self) -> list[EndpointConfig]:
        return self._endpoints

    def pick(self, item_id: str, attempt: int = 1) -> RoutePick:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")

        if self._strategy == "stable_hash":
            index = self._pick_stable_hash(item_id=item_id, attempt=attempt)
        else:
            index = self._pick_least_inflight_weighted(item_id=item_id, attempt=attempt)

        return RoutePick(index=index, endpoint=self._endpoints[index])

    def mark_start(self, index: int) -> None:
        self._inflight[index] += 1

    def mark_done(self, index: int) -> None:
        if self._inflight[index] > 0:
            self._inflight[index] -= 1

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for idx, endpoint in enumerate(self._endpoints):
            out[endpoint.name] = {
                "inflight": self._inflight[idx],
                "weight": endpoint.weight,
            }
        return out

    def _pick_stable_hash(self, *, item_id: str, attempt: int) -> int:
        n = len(self._endpoints)
        base = _stable_hash_index(item_id, n)
        if self._failover == "same_endpoint":
            return base
        return (base + (attempt - 1)) % n

    def _pick_least_inflight_weighted(self, *, item_id: str, attempt: int) -> int:
        n = len(self._endpoints)
        start = _stable_hash_index(item_id, n)
        ordered_indices = sorted(
            range(n),
            key=lambda idx: (
                (self._inflight[idx] / _safe_weight(self._endpoints[idx].weight)),
                ((idx - start) % n),
            ),
        )
        if self._failover == "same_endpoint":
            return ordered_indices[0]
        return ordered_indices[(attempt - 1) % n]



def pick_endpoint_index(
    item_id: str,
    endpoints: list[EndpointConfig],
    attempt: int = 1,
    route_strategy: str = "stable_hash",
    route_failover: str = "rotate_on_retry",
) -> int:
    router = EndpointRouter(
        endpoints=endpoints,
        route_strategy=route_strategy,
        route_failover=route_failover,
    )
    return router.pick(item_id=item_id, attempt=attempt).index


def _safe_weight(weight: float) -> float:
    return weight if weight > 0 else 1.0


def _stable_hash_index(item_id: str, size: int) -> int:
    digest = hashlib.md5(item_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest, 16) % size
