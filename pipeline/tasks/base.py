from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pipeline.core.config import PipelineConfig
from pipeline.types import ErrorRecord, ProcessedRecord, RawItem


@dataclass(slots=True)
class RequestSpec:
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    url_override: str | None = None
    skip_http: bool = False
    local_response: dict[str, Any] | None = None
    # 第二次请求（与 payload 同 endpoint）：若 set，worker 先发 payload 再发 secondary_payload，将两次响应一并交给 parse_response
    secondary_payload: dict[str, Any] | None = None


class TaskPlugin(ABC):
    name: str = ""
    requires_endpoints: bool = True

    @abstractmethod
    def load_items(self, config: PipelineConfig) -> list[RawItem]:
        """Load all task items into memory."""

    @abstractmethod
    def build_request(self, item: RawItem, config: PipelineConfig) -> RequestSpec:
        """Build one LLM request from a raw item."""

    @abstractmethod
    def parse_response(
        self,
        item: RawItem,
        llm_response: dict[str, Any],
        config: PipelineConfig,
        *,
        secondary_llm_response: dict[str, Any] | None = None,
    ) -> ProcessedRecord:
        """Parse LLM response into final output schema. If secondary_llm_response is set, it is the second API response (e.g. two-phase judge)."""

    def on_error(
        self,
        item: RawItem,
        error: Exception,
        *,
        stage: str,
        attempts: int,
        worker_id: int,
        config: PipelineConfig,
    ) -> ErrorRecord:
        item_id = str(item.get(config.id_field, ""))
        return {
            config.id_field: item_id,
            "error": str(error),
            "error_type": error.__class__.__name__,
            "stage": stage,
            "attempts": attempts,
            "worker_id": worker_id,
            "item": dict(item),
        }

    def finalize_outputs(self, config: PipelineConfig) -> dict[str, Any] | None:
        """Optional post-run file processing."""
        return None
