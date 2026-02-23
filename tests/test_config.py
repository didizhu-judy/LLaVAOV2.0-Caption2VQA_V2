from __future__ import annotations

from pipeline.core.config import PipelineConfig


def test_config_llm_urls_from_env_csv(monkeypatch) -> None:
    monkeypatch.setenv(
        "PIPELINE_LLM_URLS",
        "http://127.0.0.1:10025/v1/chat/completions,http://127.0.0.1:10026/v1/chat/completions",
    )
    config = PipelineConfig().with_env_overrides()
    assert config.llm_urls == [
        "http://127.0.0.1:10025/v1/chat/completions",
        "http://127.0.0.1:10026/v1/chat/completions",
    ]


def test_config_llm_urls_from_env_json(monkeypatch) -> None:
    monkeypatch.setenv(
        "PIPELINE_LLM_URLS",
        '["http://127.0.0.1:10025/v1/chat/completions","http://127.0.0.1:10026/v1/chat/completions"]',
    )
    config = PipelineConfig().with_env_overrides()
    assert config.llm_urls == [
        "http://127.0.0.1:10025/v1/chat/completions",
        "http://127.0.0.1:10026/v1/chat/completions",
    ]
