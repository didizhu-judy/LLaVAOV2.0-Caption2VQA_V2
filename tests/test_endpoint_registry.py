from __future__ import annotations

import json
from pathlib import Path

from pipeline.core.config import PipelineConfig
from pipeline.providers.registry import load_endpoint_group, resolve_endpoints_for_config


def test_load_endpoint_group(tmp_path: Path) -> None:
    registry_path = tmp_path / "endpoints.json"
    registry_path.write_text(
        json.dumps(
            {
                "groups": {
                    "local_multi": {
                        "endpoints": [
                            {
                                "name": "ep-1",
                                "provider": "openai_compatible",
                                "url": "http://127.0.0.1:10025/v1/chat/completions",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    endpoints = load_endpoint_group(str(registry_path), "local_multi")
    assert len(endpoints) == 1
    assert endpoints[0].name == "ep-1"


def test_resolve_endpoints_fallback_to_llm_urls(tmp_path: Path) -> None:
    config = PipelineConfig(
        endpoint_registry_file=str(tmp_path / "missing.json"),
        endpoint_group="local_multi",
        llm_urls=[
            "http://127.0.0.1:10025/v1/chat/completions",
            "http://127.0.0.1:10026/v1/chat/completions",
        ],
    )

    endpoints = resolve_endpoints_for_config(config)
    assert len(endpoints) == 2
    assert endpoints[0].provider == "openai_compatible"
