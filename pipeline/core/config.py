from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class PipelineConfig:
    task_name: str = "caption_to_vqa"
    task_config: dict[str, Any] = field(default_factory=dict)

    num_workers: int = 4
    fetch_batch_size: int = 32
    worker_async_concurrency: int = 16
    worker_queue_size: int = 256

    dump_interval_sec: float = 30.0
    dump_every_n: int = 200
    output_jsonl: str = "outputs/results.jsonl"
    error_jsonl: str = "outputs/errors.jsonl"
    resume: bool = True
    id_field: str = "id"

    endpoint_registry_file: str = "scripts/env/endpoints.json"
    endpoint_group: str = "local_multi"

    # Transitional fallback fields, will be removed in next major version.
    llm_url: str = ""
    llm_urls: list[str] = field(default_factory=list)
    llm_headers: dict[str, str] = field(default_factory=dict)

    llm_timeout_sec: float = 60.0
    llm_max_retries: int = 3
    llm_retry_backoff_sec: float = 1.0

    route_strategy: str = "stable_hash"
    route_failover: str = "rotate_on_retry"

    # Runtime resolved endpoints (list[EndpointConfig.to_dict()])
    endpoints: list[dict[str, Any]] = field(default_factory=list)

    ray_address: str | None = None
    ray_local_mode: bool = False

    def __post_init__(self) -> None:
        self.task_name = str(self.task_name).strip() or "caption_to_vqa"
        self.llm_urls = _normalize_url_list(self.llm_urls)
        if not self.llm_urls and self.llm_url:
            self.llm_urls = [self.llm_url]
        if self.llm_urls and not self.llm_url:
            self.llm_url = self.llm_urls[0]

        self.route_strategy = str(self.route_strategy).strip().lower() or "stable_hash"
        self.route_failover = str(self.route_failover).strip().lower() or "rotate_on_retry"

        if self.route_strategy not in {"stable_hash", "least_inflight_weighted"}:
            raise ValueError(
                "route_strategy must be one of: stable_hash, least_inflight_weighted"
            )
        if self.route_failover not in {"rotate_on_retry", "same_endpoint"}:
            raise ValueError("route_failover must be one of: rotate_on_retry, same_endpoint")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None = None,
        *,
        apply_env_overrides: bool = True,
    ) -> "PipelineConfig":
        raw_data = dict(data or {})
        valid_field_names = {config_field.name for config_field in fields(cls)}
        filtered = {key: value for key, value in raw_data.items() if key in valid_field_names}
        config = cls(**filtered)
        if apply_env_overrides:
            config = config.with_env_overrides()
        return config

    @classmethod
    def from_file(cls, path: str | None) -> "PipelineConfig":
        if not path:
            return cls().with_env_overrides()

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")

        with path_obj.open("r", encoding="utf-8") as handle:
            if path_obj.suffix.lower() in {".yaml", ".yml"}:
                raw = yaml.safe_load(handle) or {}
            elif path_obj.suffix.lower() == ".json":
                raw = json.load(handle)
            else:
                raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError(f"Config file must decode to an object: {path}")
        return cls.from_mapping(raw, apply_env_overrides=True)

    def with_env_overrides(self, prefix: str = "PIPELINE_") -> "PipelineConfig":
        current = asdict(self)
        for key, old_value in current.items():
            env_name = f"{prefix}{key.upper()}"
            env_value = os.getenv(env_name)
            if env_value is None:
                continue
            current[key] = _parse_env_value(env_value, old_value)
        return PipelineConfig(**current)

    def ensure_output_dirs(self) -> None:
        Path(self.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(self.error_jsonl).parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_env_value(env_value: str, old_value: Any) -> Any:
    if isinstance(old_value, bool):
        return env_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(old_value, int) and not isinstance(old_value, bool):
        return int(env_value)
    if isinstance(old_value, float):
        return float(env_value)
    if isinstance(old_value, dict):
        loaded = json.loads(env_value)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object for env override, got: {env_value}")
        return loaded
    if isinstance(old_value, list):
        return _parse_list_env_value(env_value)
    if old_value is None:
        try:
            return json.loads(env_value)
        except json.JSONDecodeError:
            return env_value
    return env_value


def _parse_list_env_value(env_value: str) -> list[Any]:
    stripped = env_value.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        loaded = json.loads(stripped)
        if not isinstance(loaded, list):
            raise ValueError(f"Expected JSON array for list env override, got: {env_value}")
        return loaded

    return [part.strip() for part in stripped.split(",") if part.strip()]


def _normalize_url_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = _parse_list_env_value(value)
    elif isinstance(value, list):
        raw = value
    else:
        raw = list(value)

    normalized: list[str] = []
    for item in raw:
        item_str = str(item).strip()
        if item_str:
            normalized.append(item_str)
    return normalized
