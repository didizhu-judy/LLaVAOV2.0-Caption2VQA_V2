from __future__ import annotations

from pipeline.providers.azure_openai import AzureOpenAIProvider
from pipeline.providers.base import EndpointConfig
from pipeline.providers.openai_compatible import OpenAICompatibleProvider


def test_openai_compatible_provider_builds_auth_header() -> None:
    provider = OpenAICompatibleProvider()
    endpoint = EndpointConfig(
        name="openai",
        provider="openai_compatible",
        url="https://api.openai.com/v1/chat/completions",
        auth_type="api_key",
        api_key="k-test",
        model="gpt-4o-mini",
    )
    prepared = provider.prepare_request(endpoint, {"messages": []}, {})

    assert prepared.url == endpoint.url
    assert prepared.payload["model"] == "gpt-4o-mini"
    assert prepared.headers["Authorization"] == "Bearer k-test"


def test_azure_provider_api_key_mode_builds_deployment_url() -> None:
    provider = AzureOpenAIProvider()
    endpoint = EndpointConfig(
        name="azure",
        provider="azure_openai",
        url="https://example.openai.azure.com",
        auth_type="api_key",
        api_key="az-key",
        deployment="gpt-4o",
        api_version="2024-10-21",
    )
    prepared = provider.prepare_request(endpoint, {"model": "ignored", "messages": []}, {})

    assert "/openai/deployments/gpt-4o/chat/completions" in prepared.url
    assert "api-version=2024-10-21" in prepared.url
    assert "model" not in prepared.payload
    assert prepared.headers["api-key"] == "az-key"


def test_azure_provider_azure_ad_mode_uses_bearer_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "pipeline.providers.azure_openai._get_azure_ad_token",
        lambda scope: f"token-for:{scope}",
    )

    provider = AzureOpenAIProvider()
    endpoint = EndpointConfig(
        name="azure-ad",
        provider="azure_openai",
        url="https://trapi.research.microsoft.com/gcr/shared/openai/deployments/gpt-4o/chat/completions",
        auth_type="azure_ad",
        scope="api://trapi/.default",
    )
    prepared = provider.prepare_request(endpoint, {"messages": []}, {})

    assert prepared.headers["Authorization"] == "Bearer token-for:api://trapi/.default"
    assert "api-version=" in prepared.url
