from __future__ import annotations

from typing import Any, TypedDict


class RawItem(TypedDict, total=False):
    id: str


class ProcessedRecord(TypedDict, total=False):
    id: str
    result: dict[str, Any]
    worker_id: int


class ErrorRecord(TypedDict, total=False):
    id: str
    error: str
    error_type: str
    stage: str
    attempts: int
    worker_id: int
    item: dict[str, Any]


class WorkerSummary(TypedDict):
    worker_id: int
    fetched_batches: int
    processed: int
    errors: int
