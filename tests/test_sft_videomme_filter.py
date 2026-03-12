from __future__ import annotations

import json
from pathlib import Path

from pipeline.tasks.benchmark_similarity.sft_videomme_filter import run_filter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_run_filter_tags_and_filters_sft_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "Molmo2-videoforsft_long_60_180s.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "count-1",
                "images_source": ["/ov2/dataset_molmo2/Molmo2-VideoCountEval/youtube_vedio/v1.mp4"],
                "messages": [
                    {"role": "user", "content": "How many cups are on the table?"},
                    {"role": "assistant", "content": "3"},
                ],
            },
            {
                "id": "count-2",
                "images_source": ["/ov2/dataset_molmo2/Molmo2-VideoCountEval/youtube_vedio/v2.mp4"],
                "messages": [
                    {"role": "user", "content": "find the closest sign"},
                    {"role": "assistant", "content": "1"},
                ],
            },
            {
                "id": "subtitle-1",
                "images_source": ["/ov2/dataset_molmo2/Molmo2-VideoSubtitleQA/youtube_vedio/v3.mp4"],
                "messages": [
                    {"role": "user", "content": "What happens after the narrator says the final step?"},
                    {"role": "assistant", "content": "The crowd cheers."},
                ],
            },
            {
                "id": "ama-1",
                "images_source": ["/ov2/dataset_molmo2/Molmo2-AskModelAnything/youtube_vedio/v4.mp4"],
                "messages": [
                    {"role": "user", "content": "What color is the car?"},
                    {"role": "assistant", "content": "Red."},
                ],
            },
        ],
    )

    summary = run_filter(
        input_glob=str(input_path),
        output_dir=tmp_path / "out",
    )

    assert summary["total_rows"] == 4
    assert summary["kept_rows"] == 2
    assert summary["drop_reason_counts"]["count_eval_non_pure_counting"] == 1
    assert summary["drop_reason_counts"]["askmodelanything_general"] == 1
    assert summary["keep_reason_counts"]["count_eval_pure_counting"] == 1
    assert summary["keep_reason_counts"]["subtitleqa_subtitle_alignment"] == 1

    filtered_rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "filtered" / f"{input_path.stem}.filtered.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(filtered_rows) == 2
    assert {row["videomme_tags"]["skill_bucket"] for row in filtered_rows} == {"counting", "subtitle_alignment"}
    assert all(row["videomme_tags"]["duration_bucket"] == "medium" for row in filtered_rows)
