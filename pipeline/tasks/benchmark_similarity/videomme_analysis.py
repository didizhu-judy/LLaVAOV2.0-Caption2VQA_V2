from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_videomme_samples(path: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            meta = dict(row.get("videomme_perception_score") or {})
            question = str(row.get("input") or "")
            samples.append(
                {
                    "doc_id": row.get("doc_id"),
                    "question": question,
                    "target": str(row.get("target") or ""),
                    "pred": _pred_text(row.get("filtered_resps")),
                    "score": float(meta.get("score", 0.0)),
                    "question_id": str(meta.get("question_id") or ""),
                    "video_id": str(meta.get("videoID") or ""),
                    "duration": str(meta.get("duration") or "unknown"),
                    "category": str(meta.get("category") or ""),
                    "sub_category": str(meta.get("sub_category") or ""),
                    "task_type": str(meta.get("task_category") or ""),
                }
            )
    return samples


def aggregate_priority_buckets(
    current_samples: list[dict[str, Any]],
    baseline_samples: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    current_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in current_samples:
        key = (str(sample.get("duration") or "unknown"), str(sample.get("task_type") or ""))
        current_groups.setdefault(key, []).append(sample)

    baseline_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sample in baseline_samples or []:
        key = (str(sample.get("duration") or "unknown"), str(sample.get("task_type") or ""))
        baseline_groups.setdefault(key, []).append(sample)

    buckets: list[dict[str, Any]] = []
    for key, rows in current_groups.items():
        duration, task_type = key
        total_count = len(rows)
        wrong_count = sum(1 for row in rows if float(row.get("score") or 0.0) < 1.0)
        accuracy = _percent(sum(float(row.get("score") or 0.0) for row in rows), total_count)
        error_rate = round(100.0 - accuracy, 2)

        baseline_accuracy = 0.0
        baseline_gap_pp = 0.0
        if key in baseline_groups and baseline_groups[key]:
            baseline_rows = baseline_groups[key]
            baseline_accuracy = _percent(sum(float(row.get("score") or 0.0) for row in baseline_rows), len(baseline_rows))
            baseline_gap_pp = round(baseline_accuracy - accuracy, 2)

        priority_score = round(wrong_count * (1.0 + max(0.0, baseline_gap_pp) / 10.0), 4)
        buckets.append(
            {
                "bucket_key": f"{duration}|{task_type}",
                "duration": duration,
                "task_type": task_type,
                "total_count": total_count,
                "wrong_count": wrong_count,
                "accuracy": accuracy,
                "error_rate": error_rate,
                "baseline_accuracy": round(baseline_accuracy, 2),
                "baseline_gap_pp": baseline_gap_pp,
                "priority_score": priority_score,
            }
        )

    buckets.sort(
        key=lambda item: (
            float(item.get("priority_score") or 0.0),
            int(item.get("wrong_count") or 0),
            float(item.get("error_rate") or 0.0),
        ),
        reverse=True,
    )
    for index, bucket in enumerate(buckets, start=1):
        bucket["rank"] = index
        bucket["difficulty_tier"] = _difficulty_tier(index=index, error_rate=float(bucket["error_rate"]))
    return buckets


def build_priority_bucket_lookup(
    current_samples: list[dict[str, Any]],
    baseline_samples: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (bucket["duration"], bucket["task_type"]): bucket
        for bucket in aggregate_priority_buckets(current_samples, baseline_samples)
    }


def samples_for_bucket(
    samples: list[dict[str, Any]],
    *,
    duration: str,
    task_type: str,
    only_wrong: bool = True,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in samples
        if str(row.get("duration") or "") == duration and str(row.get("task_type") or "") == task_type
    ]
    if only_wrong:
        selected = [row for row in selected if float(row.get("score") or 0.0) < 1.0]
    return selected


def default_demo_bucket_keys() -> list[tuple[str, str]]:
    return [
        ("long", "Object Reasoning"),
        ("long", "Action Reasoning"),
        ("long", "Temporal Reasoning"),
        ("medium", "Counting Problem"),
        ("long", "Information Synopsis"),
        ("medium", "OCR Problems"),
    ]


def _difficulty_tier(*, index: int, error_rate: float) -> str:
    if index <= 5:
        return "critical"
    if index <= 12:
        return "hard"
    if error_rate >= 35.0:
        return "medium"
    return "easy"


def _percent(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((100.0 * numerator) / denominator, 2)


def _pred_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")
