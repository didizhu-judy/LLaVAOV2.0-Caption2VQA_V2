from __future__ import annotations

from typing import Optional

import ray

from pipeline.types import RawItem


@ray.remote
class DataDispatcher:
    def __init__(
        self,
        items: list[RawItem],
        processed_ids: set[str],
        id_field: str,
        fetch_batch_size: int,
    ) -> None:
        self._items = items
        self._processed_ids = processed_ids
        self._id_field = id_field
        self._fetch_batch_size = max(1, int(fetch_batch_size))
        self._cursor = 0
        self._total = len(items)
        self._skipped = 0
        self._dispatched = 0
        self._invalid = 0
        self._stop_early = False

    def stop_early(self) -> None:
        """Signal to stop dispatching; next_batch() will return None from now on."""
        self._stop_early = True

    def next_batch(self, request_size: Optional[int] = None) -> Optional[list[RawItem]]:
        if self._stop_early:
            return None
        desired = max(1, int(request_size or self._fetch_batch_size))
        batch: list[RawItem] = []

        while len(batch) < desired and self._cursor < self._total:
            item = self._items[self._cursor]
            self._cursor += 1

            item_id_raw = item.get(self._id_field)
            if item_id_raw is None:
                self._skipped += 1
                self._invalid += 1
                continue

            item_id = str(item_id_raw)
            if item_id in self._processed_ids:
                self._skipped += 1
                continue

            batch.append(item)
            self._dispatched += 1

        if not batch:
            return None
        return batch

    def stats(self) -> dict[str, int | bool]:
        return {
            "total": self._total,
            "cursor": self._cursor,
            "skipped": self._skipped,
            "invalid": self._invalid,
            "dispatched": self._dispatched,
            "exhausted": self._cursor >= self._total,
        }
