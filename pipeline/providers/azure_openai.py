from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pipeline.providers.base import EndpointConfig, EndpointProvider, PreparedRequest

_DEFAULT_AZURE_API_VERSION = "2024-10-21"
_AZURE_TOKEN_PROVIDERS: dict[str, Any] = {}


class AzureOpenAIProvider(EndpointProvider):
    name = "azure_openai"

    def prepare_request(
        self,
        endpoint: EndpointConfig,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> PreparedRequest:
        if not endpoint.url:
            raise ValueError(f"Endpoint '{endpoint.name}' missing url")

        api_version = endpoint.api_version or _DEFAULT_AZURE_API_VERSION
        final_url = _build_azure_url(endpoint, api_version)

        final_payload = dict(payload)
        # Azure deployment path already determines target model.
        final_payload.pop("model", None)

        final_headers = dict(headers)
        final_headers.update(endpoint.extra_headers)

        auth_type = (endpoint.auth_type or "api_key").strip().lower()
        if auth_type == "api_key":
            if not endpoint.api_key:
                raise ValueError(
                    f"Endpoint '{endpoint.name}' auth_type=api_key but api_key is empty"
                )
            final_headers["api-key"] = endpoint.api_key
        elif auth_type == "azure_ad":
            scope = endpoint.scope or "https://cognitiveservices.azure.com/.default"
            token = _get_azure_ad_token(scope)
            final_headers["Authorization"] = f"Bearer {token}"
        else:
            raise ValueError(f"Unsupported azure auth_type: {endpoint.auth_type}")

        return PreparedRequest(
            url=final_url,
            payload=final_payload,
            headers=final_headers,
            timeout_sec=endpoint.timeout_sec,
        )


def _build_azure_url(endpoint: EndpointConfig, api_version: str) -> str:
    raw = endpoint.url.rstrip("/")
    if raw.endswith("/chat/completions"):
        return _ensure_api_version(raw, api_version)

    deployment = endpoint.deployment or endpoint.model
    if not deployment:
        raise ValueError(
            f"Endpoint '{endpoint.name}' must set deployment or model for azure_openai"
        )

    url = f"{raw}/openai/deployments/{deployment}/chat/completions"
    return _ensure_api_version(url, api_version)


def _ensure_api_version(url: str, api_version: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("api-version", api_version)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(query),
            parsed.fragment,
        )
    )


def _get_azure_ad_token(scope: str) -> str:
    provider = _AZURE_TOKEN_PROVIDERS.get(scope)
    if provider is None:
        try:
            from azure.identity import (  # type: ignore[import-not-found]
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "azure.identity is required for auth_type=azure_ad. "
                "Install with: pip install azure-identity"
            ) from exc
        provider = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
            scope,
        )
        _AZURE_TOKEN_PROVIDERS[scope] = provider
    return str(provider())
