from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipeline.providers.azure_openai import AzureOpenAIProvider
from pipeline.providers.base import EndpointConfig, EndpointProvider
from pipeline.providers.openai_compatible import OpenAICompatibleProvider

_PROVIDER_REGISTRY: dict[str, EndpointProvider] = {
    OpenAICompatibleProvider.name: OpenAICompatibleProvider(),
    AzureOpenAIProvider.name: AzureOpenAIProvider(),
}


def get_provider(provider_name: str) -> EndpointProvider:
    key = (provider_name or "").strip().lower()
    if key not in _PROVIDER_REGISTRY:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider '{provider_name}'. Supported providers: {supported}")
    return _PROVIDER_REGISTRY[key]


def list_providers() -> list[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def load_endpoint_group(path: str, group_name: str) -> list[EndpointConfig]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Endpoint registry file does not exist: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    groups = raw.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Endpoint registry missing 'groups': {path}")

    if group_name not in groups:
        supported = ", ".join(sorted(groups.keys()))
        raise ValueError(
            f"Endpoint group '{group_name}' not found in {path}. Supported groups: {supported}"
        )

    group_raw = groups[group_name]
    if isinstance(group_raw, dict):
        endpoints_raw = group_raw.get("endpoints", [])
    else:
        endpoints_raw = group_raw

    if not isinstance(endpoints_raw, list):
        raise ValueError(f"Endpoint group '{group_name}' must be a list or object with 'endpoints'")

    endpoints = [EndpointConfig.from_mapping(item) for item in endpoints_raw if isinstance(item, dict)]
    return _normalize_endpoints(endpoints)


def fallback_endpoints_from_urls(urls: list[str]) -> list[EndpointConfig]:
    endpoints: list[EndpointConfig] = []
    for idx, url in enumerate(urls):
        url_text = str(url).strip()
        if not url_text:
            continue
        endpoints.append(
            EndpointConfig(
                name=f"fallback-{idx}",
                provider="openai_compatible",
                url=url_text,
                auth_type="none",
            )
        )
    return endpoints


def resolve_endpoints_for_config(config: Any) -> list[EndpointConfig]:
    if getattr(config, "endpoints", None):
        parsed = [EndpointConfig.from_mapping(item) for item in config.endpoints]
        return _normalize_endpoints(parsed)

    endpoints: list[EndpointConfig] = []
    path = str(getattr(config, "endpoint_registry_file", "") or "").strip()
    group = str(getattr(config, "endpoint_group", "") or "").strip()

    if path and group:
        try:
            endpoints = load_endpoint_group(path, group)
        except FileNotFoundError:
            endpoints = []

    if not endpoints:
        urls = [str(url).strip() for url in (getattr(config, "llm_urls", []) or []) if str(url).strip()]
        llm_url = str(getattr(config, "llm_url", "") or "").strip()
        if not urls and llm_url:
            urls = [llm_url]
        endpoints = fallback_endpoints_from_urls(urls)

    if not endpoints:
        raise ValueError(
            "No endpoint resolved. Configure endpoint_registry_file + endpoint_group or llm_urls"
        )

    return _normalize_endpoints(endpoints)


def _normalize_endpoints(endpoints: list[EndpointConfig]) -> list[EndpointConfig]:
    out: list[EndpointConfig] = []
    for idx, endpoint in enumerate(endpoints):
        if not endpoint.name:
            endpoint.name = f"endpoint-{idx}"
        endpoint.provider = (endpoint.provider or "openai_compatible").strip().lower()
        if endpoint.provider not in _PROVIDER_REGISTRY:
            supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
            raise ValueError(
                f"Endpoint '{endpoint.name}' uses unknown provider '{endpoint.provider}'. "
                f"Supported providers: {supported}"
            )
        if endpoint.api_key_env and not endpoint.api_key:
            endpoint.api_key = os.getenv(endpoint.api_key_env)
        endpoint.auth_type = (endpoint.auth_type or "none").strip().lower()
        if endpoint.weight <= 0:
            endpoint.weight = 1.0
        if endpoint.max_concurrent < 0:
            endpoint.max_concurrent = 0
        out.append(endpoint)
    if not out:
        raise ValueError("Endpoint list is empty")
    return out
