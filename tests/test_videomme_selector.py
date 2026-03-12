from __future__ import annotations

import json
from pathlib import Path

from pipeline.tasks.benchmark_similarity.profiles import infer_example_profile
from pipeline.tasks.benchmark_similarity.videomme_analysis import aggregate_priority_buckets, load_videomme_samples
from pipeline.tasks.benchmark_similarity import videomme_selector


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")


def _videomme_row(
    *,
    doc_id: int,
    duration: str,
    task_type: str,
    score: float,
    question: str | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "target": "A",
        "filtered_resps": ["A" if score == 1.0 else "B"],
        "videomme_perception_score": {
            "question_id": f"{doc_id:03d}-1",
            "duration": duration,
            "category": "Knowledge",
            "sub_category": "Technology",
            "task_category": task_type,
            "pred_answer": "A" if score == 1.0 else "B",
            "answer": "A",
            "score": score,
            "videoID": f"video-{doc_id}",
        },
        "input": question or f"{task_type} question {doc_id}",
    }


def test_aggregate_priority_buckets_assigns_tiers(tmp_path: Path) -> None:
    current_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for index in range(13):
        duration = "long" if index % 2 else "medium"
        task_type = f"Task {index}"
        wrong_count = 13 - index
        for inner in range(20):
            score = 0.0 if inner < wrong_count else 1.0
            current_rows.append(_videomme_row(doc_id=(index * 100) + inner, duration=duration, task_type=task_type, score=score))
            baseline_score = 1.0 if inner < min(20, wrong_count + 3) else 0.0
            baseline_rows.append(
                _videomme_row(
                    doc_id=(index * 100) + inner,
                    duration=duration,
                    task_type=task_type,
                    score=baseline_score,
                )
            )

    current_path = tmp_path / "current.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    _write_jsonl(current_path, current_rows)
    _write_jsonl(baseline_path, baseline_rows)
    current = load_videomme_samples(current_path)
    baseline = load_videomme_samples(baseline_path)
    buckets = aggregate_priority_buckets(current, baseline)
    assert len(buckets) == 13
    assert all(bucket["difficulty_tier"] == "critical" for bucket in buckets[:5])
    assert all(bucket["difficulty_tier"] == "hard" for bucket in buckets[5:12])
    assert buckets[-1]["difficulty_tier"] in {"medium", "easy"}


def test_infer_example_profile_skill_bucket_mapping() -> None:
    capqa_profile = infer_example_profile(
        {
            "source_name": "molmo2_videocapqa",
            "question": "Which scene appears first?",
            "answer": "A beach.",
            "metadata": {"Category": "event sequence"},
        },
        role="candidate",
        spec=None,
    )
    assert capqa_profile["skill_bucket"] == "temporal_sequence"

    longcap_summary_profile = infer_example_profile(
        {
            "source_name": "molmo2_longcapqa",
            "question": "What is the main topic of the long video?",
            "answer": "A family road trip.",
            "metadata": {"Category": "video topic"},
        },
        role="candidate",
        spec=None,
    )
    assert longcap_summary_profile["skill_bucket"] == "summary"

    subtitle_profile = infer_example_profile(
        {
            "source_name": "molmo2_videosubtitleqa",
            "question": "What happens after the subtitle is spoken?",
            "answer": "The crowd cheers.",
            "metadata": {"Category": "object recognition", "AlignmentType": "Temporal Sequence Bridging"},
        },
        role="candidate",
        spec=None,
    )
    assert subtitle_profile["skill_bucket"] == "subtitle_alignment"

    subtitle_ocr_profile = infer_example_profile(
        {
            "source_name": "molmo2_videosubtitleqa",
            "question": "What text is shown on the sign?",
            "answer": "No Parking",
            "metadata": {"Category": "text recognition", "AlignmentType": "Temporal Sequence Bridging"},
        },
        role="candidate",
        spec=None,
    )
    assert subtitle_ocr_profile["skill_bucket"] == "ocr_text"

    cap_profile = infer_example_profile(
        {
            "source_name": "molmo2_cap",
            "question": "What happens between 0 and 10 seconds?",
            "answer": "A reporter speaks.",
            "metadata": {},
        },
        role="candidate",
        spec=None,
    )
    assert cap_profile["skill_bucket"] == "temporal_sequence"

    ask_profile = infer_example_profile(
        {
            "source_name": "molmo2_askmodelanything",
            "question": "How many cups are on the table?",
            "answer": "3",
            "metadata": {},
        },
        role="candidate",
        spec=None,
    )
    assert ask_profile["skill_bucket"] == "counting"

    count_profile = infer_example_profile(
        {
            "source_name": "molmo2_videocounteval",
            "question": "How many eggs are in the jar?",
            "answer": "7",
            "metadata": {"category": "object", "video_duration": 30.0},
        },
        role="candidate",
        spec=None,
    )
    assert count_profile["skill_bucket"] == "counting"
    assert count_profile["duration_hint"] == "short"

    excluded_count_profile = infer_example_profile(
        {
            "source_name": "molmo2_videocounteval",
            "question": "find the closest sign",
            "answer": "1",
            "metadata": {"category": "comparative reference", "video_duration": 30.0},
        },
        role="candidate",
        spec=None,
    )
    assert excluded_count_profile["skill_bucket"] == "general"


def test_videomme_selector_demo_outputs(tmp_path: Path, monkeypatch) -> None:
    current_path = tmp_path / "current.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"

    current_records = [
        _videomme_row(
            doc_id=1,
            duration="long",
            task_type="Temporal Reasoning",
            score=0.0,
            question="What is the correct order of the events in the video?",
        ),
        _videomme_row(
            doc_id=2,
            duration="long",
            task_type="Object Reasoning",
            score=0.0,
            question="Which tool shown early is used later in the video?",
        ),
        _videomme_row(
            doc_id=3,
            duration="long",
            task_type="Action Reasoning",
            score=0.0,
            question="Why does the person perform the action after the warning appears?",
        ),
        _videomme_row(
            doc_id=4,
            duration="medium",
            task_type="Counting Problem",
            score=0.0,
            question="How many times does the athlete jump in the video?",
        ),
    ]
    baseline_records = [
        _videomme_row(doc_id=1, duration="long", task_type="Temporal Reasoning", score=1.0),
        _videomme_row(doc_id=2, duration="long", task_type="Object Reasoning", score=1.0),
        _videomme_row(doc_id=3, duration="long", task_type="Action Reasoning", score=1.0),
        _videomme_row(doc_id=4, duration="medium", task_type="Counting Problem", score=1.0),
    ]
    _write_jsonl(current_path, current_records)
    _write_jsonl(baseline_path, baseline_records)

    capqa_path = tmp_path / "capqa.jsonl"
    subtitle_path = tmp_path / "subtitle.jsonl"
    longcap_path = tmp_path / "longcap.jsonl"
    count_path = tmp_path / "count.jsonl"
    ask_path = tmp_path / "ask.jsonl"
    cap_path = tmp_path / "cap.jsonl"

    _write_jsonl(
        capqa_path,
        [
            {
                "video_id": "capqa-1",
                "Question": "Which scene appears first in the tutorial?",
                "Answer": "The introduction screen.",
                "Category": "event sequence",
            },
            {
                "video_id": "capqa-2",
                "Question": "Which tool shown early is used later in the video?",
                "Answer": "A soldering iron.",
                "Category": "object reasoning",
            },
            {
                "video_id": "capqa-3",
                "Question": "Why does the person turn off the stove after stirring?",
                "Answer": "To avoid overcooking.",
                "Category": "action reasoning",
            },
        ],
    )
    _write_jsonl(
        subtitle_path,
        [
            {
                "video_id": "subtitle-1",
                "Question": "What happens after the narrator mentions the final step?",
                "Answer": "The crowd cheers.",
                "Category": "scene sequence",
                "AlignmentType": "Temporal Sequence Bridging",
            }
        ],
    )
    _write_jsonl(
        longcap_path,
        [
            {
                "video_id": "longcap-1",
                "qa_list": [
                    {
                        "Question": "Which scene appears first in the long travel vlog?",
                        "Answer": "The airport.",
                        "Category": "event sequence",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        count_path,
        [
            {
                "video_id": "count-1",
                "question": "How many times does the athlete jump in the video?",
                "label": "4",
                "category": "action/event",
                "video_duration": 120.0,
            },
            {
                "video_id": "count-2",
                "question": "find the closest sign",
                "label": "1",
                "category": "comparative reference",
                "video_duration": 30.0,
            },
        ],
    )
    _write_jsonl(
        ask_path,
        [
            {"video_id": "ask-1", "question": "What color is the car?", "answer": "red"},
            {"video_id": "ask-2", "question": "What happens after the door opens?", "answer": "The dog runs in."},
        ],
    )
    _write_jsonl(
        cap_path,
        [
            {
                "video_id": "cap-1",
                "question": "What happens between 0 and 10 seconds?",
                "answer": "The reporter introduces the topic.",
            }
        ],
    )

    monkeypatch.setattr(
        videomme_selector,
        "default_molmo2_candidate_sources",
        lambda _root: [
            {
                "name": "molmo2_videocapqa",
                "path": str(capqa_path),
                "id_field": "video_id",
                "question_field": "Question",
                "answer_field": "Answer",
                "metadata_fields": ["Category"],
            },
            {
                "name": "molmo2_videosubtitleqa",
                "path": str(subtitle_path),
                "id_field": "video_id",
                "question_field": "Question",
                "answer_field": "Answer",
                "metadata_fields": ["Category", "AlignmentType"],
            },
            {
                "name": "molmo2_longcapqa",
                "path": str(longcap_path),
                "id_field": "video_id",
                "explode_field": "qa_list",
                "question_field": "Question",
                "answer_field": "Answer",
                "metadata_fields": ["Category"],
            },
            {
                "name": "molmo2_videocounteval",
                "path": str(count_path),
                "id_field": "video_id",
                "question_field": "question",
                "answer_field": "label",
                "metadata_fields": ["category", "video_duration"],
            },
            {
                "name": "molmo2_askmodelanything",
                "path": str(ask_path),
                "id_field": "video_id",
                "question_field": "question",
                "answer_field": "answer",
                "metadata_fields": [],
            },
            {
                "name": "molmo2_cap",
                "path": str(cap_path),
                "id_field": "video_id",
                "question_field": "question",
                "answer_field": "answer",
                "metadata_fields": [],
            },
        ],
    )

    summary = videomme_selector.run_demo(
        current_results_path=current_path,
        baseline_results_path=baseline_path,
        output_dir=tmp_path / "demo",
        molmo2_root=str(tmp_path),
        per_source_limit=100,
        bucket_top_k=5,
        merged_top_k=20,
        similarity_backend="token_overlap",
    )

    assert summary["merged_top_count"] > 0
    assert Path(summary["bucket_sampling_plan_path"]).exists()
    merged_rows = [json.loads(line) for line in (tmp_path / "demo" / "merged_top200.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["profile"]["source_family"] != "count_eval" or row["metadata"].get("category") == "action/event" for row in merged_rows)
    assert all(not (row["profile"]["source_family"] == "askmodelanything" and row["profile"]["skill_bucket"] == "general") for row in merged_rows)

    temporal_rows = [
        json.loads(line)
        for line in (tmp_path / "demo" / "bucket_long_temporal_reasoning_top50.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    temporal_sources = {row["profile"]["source_family"] for row in temporal_rows}
    assert temporal_sources <= {"capqa", "subtitleqa", "longcapqa", "caption", "askmodelanything"}

    counting_rows = [
        json.loads(line)
        for line in (tmp_path / "demo" / "bucket_medium_counting_problem_top50.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert counting_rows
    assert all(row["profile"]["skill_bucket"] == "counting" for row in counting_rows)
