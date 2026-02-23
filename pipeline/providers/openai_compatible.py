from __future__ import annotations

from typing import Any

from pipeline.providers.base import EndpointConfig, EndpointProvider, PreparedRequest


class OpenAICompatibleProvider(EndpointProvider):
    name = "openai_compatible"

    def prepare_request(
        self,
        endpoint: EndpointConfig,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> PreparedRequest:
        if not endpoint.url:
            raise ValueError(f"Endpoint '{endpoint.name}' missing url")

        final_payload = dict(payload)
        if endpoint.model and "model" not in final_payload:
            final_payload["model"] = endpoint.model

        final_headers = dict(headers)
        final_headers.update(endpoint.extra_headers)

        api_key = _resolve_api_key(endpoint)
        auth_type = (endpoint.auth_type or "none").strip().lower()
        if auth_type == "api_key":
            if not api_key:
                raise ValueError(
                    f"Endpoint '{endpoint.name}' auth_type=api_key but api_key is empty"
                )
            final_headers.setdefault("Authorization", f"Bearer {api_key}")
        elif api_key:
            final_headers.setdefault("Authorization", f"Bearer {api_key}")

        return PreparedRequest(
            url=endpoint.url,
            payload=final_payload,
            headers=final_headers,
            timeout_sec=endpoint.timeout_sec,
        )


def _resolve_api_key(endpoint: EndpointConfig) -> str | None:
    if endpoint.api_key:
        return endpoint.api_key
    return None
