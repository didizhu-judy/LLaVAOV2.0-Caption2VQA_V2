from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import orjson
import ray

from pipeline.types import ErrorRecord, ProcessedRecord


@ray.remote
class ResultSink:
    def __init__(
        self,
        output_jsonl: str,
        error_jsonl: str,
        processed_ids: set[str],
        id_field: str = "id",
        dump_every_n: int = 200,
        dump_interval_sec: float = 30.0,
        dirty_count_field: str | None = None,
        initial_dirty_count: int = 0,
    ) -> None:
        self._output_path = Path(output_jsonl)
        self._error_path = Path(error_jsonl)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._error_path.parent.mkdir(parents=True, exist_ok=True)

        self._id_field = id_field
        self._dump_every_n = max(1, int(dump_every_n))
        self._dump_interval_sec = max(0.0, float(dump_interval_sec))
        self._dirty_count_field = dirty_count_field

        self._buffer: list[ProcessedRecord] = []
        self._error_buffer: list[ErrorRecord] = []
        self._written_ids = {str(item_id) for item_id in processed_ids}
        self._preexisting = len(self._written_ids)

        self._accepted = 0
        self._duplicates = 0
        self._errors = 0
        self._invalid_results = 0
        self._dirty_count = int(initial_dirty_count)
        self._last_flush_ts = time.time()

    def add_results(self, results: list[ProcessedRecord]) -> dict[str, int]:
        for record in results:
            record_id_raw = record.get(self._id_field)
            if record_id_raw is None:
                self._invalid_results += 1
                continue

            record_id = str(record_id_raw)
            if record_id in self._written_ids:
                self._duplicates += 1
                continue

            record[self._id_field] = record_id
            self._written_ids.add(record_id)
            self._buffer.append(record)
            self._accepted += 1
            if self._dirty_count_field is not None and record.get(self._dirty_count_field) is False:
                self._dirty_count += 1

        if self._should_flush(force=False):
            self._flush_locked()

        return {
            "accepted": self._accepted,
            "duplicates": self._duplicates,
            "invalid_results": self._invalid_results,
        }

    def add_errors(self, errors: list[ErrorRecord]) -> dict[str, int]:
        for record in errors:
            record_id_raw = record.get(self._id_field)
            if record_id_raw is not None:
                record[self._id_field] = str(record_id_raw)
            self._error_buffer.append(record)
            self._errors += 1

        if self._should_flush(force=False):
            self._flush_locked()

        return {"errors": self._errors}

    def flush(self, force: bool = False) -> dict[str, Any]:
        flushed = False
        if self._should_flush(force=force):
            self._flush_locked()
            flushed = True
        return {"flushed": flushed, **self.stats()}

    def finalize(self) -> dict[str, Any]:
        if self._buffer or self._error_buffer:
            self._flush_locked()
        return self.stats()

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "preexisting": self._preexisting,
            "accepted": self._accepted,
            "duplicates": self._duplicates,
            "errors": self._errors,
            "invalid_results": self._invalid_results,
            "buffered_results": len(self._buffer),
            "buffered_errors": len(self._error_buffer),
            "written_total": len(self._written_ids),
        }
        if self._dirty_count_field is not None:
            out["dirty_total"] = self._dirty_count
        return out

    def _should_flush(self, force: bool) -> bool:
        if force:
            return True
        if len(self._buffer) >= self._dump_every_n:
            return True
        if len(self._error_buffer) >= self._dump_every_n:
            return True
        if not self._buffer and not self._error_buffer:
            return False
        return (time.time() - self._last_flush_ts) >= self._dump_interval_sec

    def _flush_locked(self) -> None:
        if self._buffer:
            _append_jsonl(self._output_path, self._buffer)
            self._buffer.clear()
        if self._error_buffer:
            _append_jsonl(self._error_path, self._error_buffer)
            self._error_buffer.clear()
        self._last_flush_ts = time.time()


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("ab") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
