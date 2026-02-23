from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EndpointConfig:
    name: str
    provider: str
    url: str
    model: str | None = None
    auth_type: str = "none"
    api_key: str | None = None
    api_key_env: str | None = None
    api_version: str | None = None
    deployment: str | None = None
    scope: str | None = None
    max_concurrent: int = 0
    weight: float = 1.0
    timeout_sec: float | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "EndpointConfig":
        return cls(
            name=str(data.get("name") or "").strip(),
            provider=str(data.get("provider") or "openai_compatible").strip(),
            url=str(data.get("url") or "").strip(),
            model=_to_opt_str(data.get("model")),
            auth_type=str(data.get("auth_type") or "none").strip(),
            api_key=_to_opt_str(data.get("api_key")),
            api_key_env=_to_opt_str(data.get("api_key_env")),
            api_version=_to_opt_str(data.get("api_version")),
            deployment=_to_opt_str(data.get("deployment")),
            scope=_to_opt_str(data.get("scope")),
            max_concurrent=int(data.get("max_concurrent") or 0),
            weight=float(data.get("weight") or 1.0),
            timeout_sec=_to_opt_float(data.get("timeout_sec")),
            extra_headers=_to_str_dict(data.get("extra_headers") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PreparedRequest:
    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    timeout_sec: float | None = None


class EndpointProvider(ABC):
    name: str

    @abstractmethod
    def prepare_request(
        self,
        endpoint: EndpointConfig,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> PreparedRequest:
        """Prepare final HTTP request details for this endpoint."""


def _to_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key, raw in value.items():
        key_text = str(key).strip()
        value_text = str(raw).strip()
        if key_text:
            out[key_text] = value_text
    return out
