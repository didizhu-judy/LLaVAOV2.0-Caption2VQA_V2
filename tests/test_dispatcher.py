from __future__ import annotations

import ray

from pipeline.core.dispatcher import DataDispatcher


def test_dispatcher_skips_processed_items() -> None:
    items = [{"id": str(i), "value": i} for i in range(10)]
    dispatcher = DataDispatcher.remote(
        items=items,
        processed_ids={"1", "3", "8"},
        id_field="id",
        fetch_batch_size=3,
    )

    seen_ids: list[str] = []
    while True:
        batch = ray.get(dispatcher.next_batch.remote())
        if batch is None:
            break
        seen_ids.extend(str(item["id"]) for item in batch)

    assert len(seen_ids) == 7
    assert set(seen_ids) == {"0", "2", "4", "5", "6", "7", "9"}

    stats = ray.get(dispatcher.stats.remote())
    assert stats["total"] == 10
    assert stats["skipped"] == 3
    assert stats["dispatched"] == 7
    assert stats["exhausted"] is True
