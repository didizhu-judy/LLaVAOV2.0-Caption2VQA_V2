from __future__ import annotations

import json
from pathlib import Path

from pipeline.core.config import PipelineConfig
from pipeline.tasks.caption_to_vqa import CaptionToVQATask


def test_caption_task_load_build_parse(tmp_path: Path) -> None:
    input_path = tmp_path / "captions.jsonl"
    source = {
        "id": "Seed-1-8",
        "messages": [
            {"role": "user", "content": ""},
            {
                "role": "assistant",
                "content": """### 3. Motion Detail Description
- **0-10 seconds**: A person enters the room.
- **10-20 seconds**: The person sits on a chair.

### 5. Highlight Moments
- **At approximately 12 seconds**: The person waves at camera.
""",
            },
        ],
        "images_source": ["/tmp/video_a_d180.00.mp4"],
    }
    input_path.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")

    config = PipelineConfig(
        task_name="caption_to_vqa",
        task_config={"input_jsonl": str(input_path)},
        id_field="id",
    )
    task = CaptionToVQATask()
    items = task.load_items(config)
    assert len(items) == 1
    assert items[0]["id"] == "video_a_d180.00"
    assert len(items[0]["segments"]) >= 2

    req = task.build_request(items[0], config)
    assert "messages" in req.payload

    llm_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "temporal_grounding": [
                                {
                                    "query": "When does the person enter the room?",
                                    "answer": "0-10 seconds",
                                    "start_sec": 0,
                                    "end_sec": 10,
                                }
                            ],
                            "segment_qa": [
                                {
                                    "query": "What happens between 10 and 20 seconds?",
                                    "answer": "The person sits on a chair.",
                                    "start_sec": 10,
                                    "end_sec": 20,
                                }
                            ],
                            "understanding_qa": [
                                {
                                    "query": "What is the scene about?",
                                    "answer": "A person entering and sitting in a room.",
                                    "category": "Information Synopsis",
                                }
                            ],
                        }
                    )
                }
            }
        ]
    }
    out = task.parse_response(items[0], llm_response, config)
    assert out["video_id"] == "video_a_d180.00"
    assert len(out["temporal_grounding"]) == 1
    assert len(out["segment_qa"]) == 1
    assert len(out["understanding_qa"]) == 1
