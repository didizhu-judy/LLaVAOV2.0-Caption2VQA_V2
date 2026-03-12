from __future__ import annotations

import json
from pathlib import Path

import orjson
import pytest

from pipeline.core.config import PipelineConfig
from pipeline.core.main import run_pipeline
from pipeline.tasks.benchmark_similarity import BenchmarkSimilarityTask


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(orjson.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")


def test_benchmark_similarity_task_profiled_selection_and_finalize(tmp_path: Path) -> None:
    videomme_path = tmp_path / "videomme.jsonl"
    _write_jsonl(
        videomme_path,
        [
            {
                "id": "videomme-count-1",
                "question": "How many players are visible in the basketball clip?",
                "answer": "three",
                "task_type": "Counting Problem",
                "domain": "Sports Competition",
                "sub_category": "Basketball",
                "duration": "short",
            }
        ],
    )

    lvbench_path = tmp_path / "lvbench.jsonl"
    _write_jsonl(
        lvbench_path,
        [
            {
                "id": "lvbench-sum-1",
                "question": "Summarize the important events across the whole documentary episode.",
                "answer": "A traveler explores several places and reflects on the journey.",
                "question_type": "summarization",
                "type": "documentary",
            }
        ],
    )

    longvideobench_path = tmp_path / "longvideobench.jsonl"
    _write_jsonl(
        longvideobench_path,
        [
            {
                "id": "lvb-rel-1",
                "question": "After the subtitle mentions the red train, what event happens next?",
                "answer": "The crowd starts cheering.",
                "question_category": "T3E",
                "topic_category": "KH-Knowledge-History",
                "level": "L2-Relation",
                "type": "global",
                "duration_group": 3600,
            }
        ],
    )

    ask_path = tmp_path / "ask.jsonl"
    _write_jsonl(
        ask_path,
        [
            {
                "video_id": "ask-1",
                "question": "How many players are dancing on the stage?",
                "answer": "three",
            }
        ],
    )

    count_path = tmp_path / "count.jsonl"
    _write_jsonl(
        count_path,
        [
            {
                "video_id": "count-1",
                "question": "How many basketball players are visible in the short sports clip?",
                "label": "three",
                "category": "action/event",
                "video_duration": 18.0,
                "video_source": "youtube",
            }
        ],
    )

    longcap_path = tmp_path / "longcap.jsonl"
    _write_jsonl(
        longcap_path,
        [
            {
                "video_id": "longcap-1",
                "qa_list": [
                    {
                        "Question": "Summarize the important events across the whole documentary episode.",
                        "Answer": "A traveler visits multiple cities and summarizes the trip.",
                        "Category": "event summary",
                    }
                ],
            }
        ],
    )

    subtitle_path = tmp_path / "subtitle.jsonl"
    _write_jsonl(
        subtitle_path,
        [
            {
                "video_id": "subtitle-1",
                "Question": "After the subtitle mentions the red train, what happens next?",
                "Answer": "The crowd starts cheering.",
                "Category": "temporal reasoning",
                "AlignmentType": "Temporal Sequence Bridging",
            }
        ],
    )

    capqa_path = tmp_path / "capqa.jsonl"
    _write_jsonl(
        capqa_path,
        [
            {
                "video_id": "capqa-1",
                "Question": "Where is the cat sitting in the frame?",
                "Answer": "On the sofa.",
                "Category": "object location",
            }
        ],
    )

    output_path = tmp_path / "results.jsonl"
    topk_path = tmp_path / "results_topk.jsonl"
    summary_path = tmp_path / "results_summary.json"

    config = PipelineConfig(
        task_name="benchmark_similarity",
        output_jsonl=str(output_path),
        task_config={
            "similarity_backend": "token_overlap",
            "ranking_score_field": "selection_score",
            "top_k_matches": 3,
            "export_top_k": 3,
            "topk_output_jsonl": str(topk_path),
            "summary_json": str(summary_path),
            "benchmark_weights": {
                "videomme": 1.0,
                "lvbench": 1.0,
                "longvideobench": 1.0,
            },
            "benchmark_sources": [
                {
                    "name": "videomme",
                    "path": str(videomme_path),
                    "id_field": "id",
                    "question_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": ["task_type", "domain", "sub_category", "duration"],
                },
                {
                    "name": "lvbench",
                    "path": str(lvbench_path),
                    "id_field": "id",
                    "question_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": ["question_type", "type"],
                },
                {
                    "name": "longvideobench",
                    "path": str(longvideobench_path),
                    "id_field": "id",
                    "question_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": ["question_category", "topic_category", "level", "type", "duration_group"],
                },
            ],
            "candidate_sources": [
                {
                    "name": "molmo2_askmodelanything",
                    "path": str(ask_path),
                    "id_field": "video_id",
                    "question_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": ["video_id"],
                },
                {
                    "name": "molmo2_videocounteval",
                    "path": str(count_path),
                    "id_field": "video_id",
                    "question_field": "question",
                    "answer_field": "label",
                    "context_fields": ["category"],
                    "metadata_fields": ["video_id", "category", "video_duration", "video_source"],
                },
                {
                    "name": "molmo2_longcapqa",
                    "path": str(longcap_path),
                    "id_field": "video_id",
                    "explode_field": "qa_list",
                    "question_field": "Question",
                    "answer_field": "Answer",
                    "context_fields": ["Category"],
                    "metadata_fields": ["video_id"],
                },
                {
                    "name": "molmo2_videosubtitleqa",
                    "path": str(subtitle_path),
                    "id_field": "video_id",
                    "question_field": "Question",
                    "answer_field": "Answer",
                    "context_fields": ["Category"],
                    "metadata_fields": ["video_id", "Category", "AlignmentType"],
                },
                {
                    "name": "molmo2_videocapqa",
                    "path": str(capqa_path),
                    "id_field": "video_id",
                    "question_field": "Question",
                    "answer_field": "Answer",
                    "context_fields": ["Category"],
                    "metadata_fields": ["video_id", "Category"],
                },
            ],
        },
    )

    task = BenchmarkSimilarityTask()
    items = task.load_items(config)
    assert len(items) == 5

    with output_path.open("wb") as handle:
        for item in items:
            response = task.build_request(item, config).local_response or {}
            record = task.parse_response(item, response, config)
            handle.write(orjson.dumps(record))
            handle.write(b"\n")

    summary = task.finalize_outputs(config)
    assert summary is not None
    assert topk_path.exists()
    top_records = _read_jsonl(topk_path)
    assert len(top_records) == 3

    top_families = {record["source_family"] for record in top_records}
    assert "count_eval" in top_families
    assert "longcapqa" in top_families
    assert "subtitleqa" in top_families

    recommended_mix = {item["source_family"]: item["count"] for item in summary["recommended_source_mix"]}
    assert recommended_mix["count_eval"] >= 1
    assert recommended_mix["longcapqa"] >= 1
    assert recommended_mix["subtitleqa"] >= 1


def test_run_pipeline_supports_local_only_task_without_endpoints(
    tmp_path: Path,
    ray_available: bool,
) -> None:
    if not ray_available:
        pytest.skip("Ray socket init is blocked in this environment")
    benchmark_path = tmp_path / "bench.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "id": "videomme-1",
                "question": "What color is the cat?",
                "answer": "black",
                "task_type": "Object Recognition",
                "domain": "Life Record",
                "duration": "short",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "video_id": "cand-1",
                "question": "What color is the kitten?",
                "answer": "black",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "pipeline_results.jsonl"
    error_path = tmp_path / "pipeline_errors.jsonl"
    config = PipelineConfig(
        task_name="benchmark_similarity",
        output_jsonl=str(output_path),
        error_jsonl=str(error_path),
        endpoint_registry_file=str(tmp_path / "missing_endpoints.json"),
        endpoint_group="missing_group",
        ray_local_mode=True,
        num_workers=1,
        fetch_batch_size=1,
        worker_async_concurrency=1,
        task_config={
            "similarity_backend": "token_overlap",
            "ranking_score_field": "selection_score",
            "benchmark_sources": [
                {
                    "name": "videomme",
                    "path": str(benchmark_path),
                    "id_field": "id",
                    "question_field": "question",
                    "answer_field": "answer",
                    "metadata_fields": ["task_type", "domain", "duration"],
                }
            ],
            "candidate_sources": [
                {
                    "name": "molmo2_sample",
                    "path": str(candidate_path),
                    "id_field": "video_id",
                    "question_field": "question",
                    "answer_field": "answer",
                }
            ],
            "export_top_k": 1,
        },
    )

    summary = run_pipeline(config)
    assert summary["endpoint_count"] == 0
    assert output_path.exists()
    results = _read_jsonl(output_path)
    assert len(results) == 1
    assert "selection_score" in results[0]
