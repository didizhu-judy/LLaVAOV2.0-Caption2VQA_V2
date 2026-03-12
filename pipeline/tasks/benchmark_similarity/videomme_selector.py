from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.tasks.benchmark_similarity.io import (
    build_similarity_text,
    default_molmo2_candidate_sources,
    load_examples_from_sources,
)
from pipeline.tasks.benchmark_similarity.profiles import infer_example_profile
from pipeline.tasks.benchmark_similarity.similarity import build_similarity_index
from pipeline.tasks.benchmark_similarity.videomme_analysis import (
    aggregate_priority_buckets,
    default_demo_bucket_keys,
    load_videomme_samples,
    samples_for_bucket,
)


_DEFAULT_CURRENT_RESULTS = (
    "/ov2/xiangan/lmms-eval-ov2/eval_log/llava_onevision2_chat/"
    "ax_instruct_4b_add180s_add10mins_4b_v2__iter_0004200_hf/20260311_154106_samples_videomme.jsonl"
)
_DEFAULT_BASELINE_RESULTS = (
    "/ov2/feilong/lmms-eval/eval_log/qwen3_vl/convert__Qwen3_vl_4B/20260308_115646_samples_videomme.jsonl"
)
_DEFAULT_OUTPUT_DIR = "output/videomme_selector_demo"

_SELECTOR_SOURCE_NAMES = {
    "molmo2_askmodelanything",
    "molmo2_videocapqa",
    "molmo2_longcapqa",
    "molmo2_cap",
    "molmo2_videosubtitleqa",
    "molmo2_videocounteval",
}

_BUCKET_SKILL_MAP: dict[str, set[str]] = {
    "Temporal Reasoning": {"temporal_sequence", "subtitle_alignment"},
    "Object Reasoning": {"object_reasoning", "subtitle_alignment"},
    "Action Reasoning": {"action_reasoning", "temporal_sequence", "subtitle_alignment"},
    "Counting Problem": {"counting"},
    "Information Synopsis": {"summary", "temporal_sequence", "subtitle_alignment"},
    "OCR Problems": {"ocr_text", "subtitle_alignment"},
}

_BUCKET_SKILL_PRIORITY: dict[str, dict[str, float]] = {
    "Temporal Reasoning": {
        "temporal_sequence": 1.0,
        "subtitle_alignment": 0.8,
    },
    "Object Reasoning": {
        "object_reasoning": 1.0,
        "subtitle_alignment": 0.75,
    },
    "Action Reasoning": {
        "action_reasoning": 1.0,
        "subtitle_alignment": 0.8,
        "temporal_sequence": 0.65,
    },
    "Counting Problem": {
        "counting": 1.0,
    },
    "Information Synopsis": {
        "summary": 1.0,
        "temporal_sequence": 0.75,
        "subtitle_alignment": 0.65,
    },
    "OCR Problems": {
        "ocr_text": 1.0,
        "subtitle_alignment": 0.7,
    },
}

_BUCKET_PRIMARY_SOURCES: dict[tuple[str, str], set[str]] = {
    ("long", "Temporal Reasoning"): {"longcapqa", "subtitleqa", "capqa"},
    ("long", "Object Reasoning"): {"longcapqa", "subtitleqa", "capqa"},
    ("long", "Action Reasoning"): {"longcapqa", "subtitleqa", "capqa"},
    ("medium", "Counting Problem"): {"count_eval"},
    ("long", "Information Synopsis"): {"longcapqa", "caption", "capqa"},
    ("medium", "OCR Problems"): {"subtitleqa", "capqa"},
}

_BUCKET_SECONDARY_SOURCES: dict[tuple[str, str], set[str]] = {
    ("long", "Temporal Reasoning"): {"caption", "askmodelanything"},
    ("long", "Object Reasoning"): {"caption", "askmodelanything"},
    ("long", "Action Reasoning"): {"caption", "askmodelanything"},
    ("medium", "Counting Problem"): {"capqa", "askmodelanything"},
    ("long", "Information Synopsis"): {"subtitleqa", "askmodelanything"},
    ("medium", "OCR Problems"): {"caption", "askmodelanything"},
}


def run_demo(
    *,
    current_results_path: str | Path = _DEFAULT_CURRENT_RESULTS,
    baseline_results_path: str | Path = _DEFAULT_BASELINE_RESULTS,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    molmo2_root: str = "/ov2/dataset_molmo2",
    per_source_limit: int = 5000,
    bucket_top_k: int = 50,
    merged_top_k: int = 200,
    similarity_backend: str = "token_overlap",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_rows = load_videomme_samples(current_results_path)
    baseline_rows = load_videomme_samples(baseline_results_path)
    priority_buckets = aggregate_priority_buckets(current_rows, baseline_rows)
    selected_bucket_keys = default_demo_bucket_keys()

    (output_path / "priority_buckets.json").write_text(
        json.dumps(
            {
                "current_results_path": str(current_results_path),
                "baseline_results_path": str(baseline_results_path),
                "selected_bucket_keys": [
                    {"duration": duration, "task_type": task_type} for duration, task_type in selected_bucket_keys
                ],
                "all_buckets": priority_buckets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    candidates = _load_demo_candidates(molmo2_root=molmo2_root, per_source_limit=per_source_limit)
    filtered_candidates = [item for item in candidates if _is_first_round_candidate(item)]

    label_summary = {
        "raw_count": len(candidates),
        "filtered_count": len(filtered_candidates),
        "raw_source_family_counts": _count_by(candidates, "source_family"),
        "raw_skill_bucket_counts": _count_by(candidates, "skill_bucket"),
        "raw_duration_hint_counts": _count_by(candidates, "duration_hint"),
        "filtered_source_family_counts": _count_by(filtered_candidates, "source_family"),
        "filtered_skill_bucket_counts": _count_by(filtered_candidates, "skill_bucket"),
        "filtered_duration_hint_counts": _count_by(filtered_candidates, "duration_hint"),
    }
    (output_path / "candidate_label_summary.json").write_text(
        json.dumps(label_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged_records: list[dict[str, Any]] = []
    bucket_mix: dict[str, Any] = {}
    for duration, task_type in selected_bucket_keys:
        bucket_examples = _make_bucket_examples(current_rows, duration=duration, task_type=task_type)
        if not bucket_examples:
            continue
        bucket_index = build_similarity_index(
            bucket_examples,
            requested_backend=similarity_backend,
            scoring_weights={"semantic": 1.0},
            benchmark_weights={bucket_examples[0]["source_name"]: 1.0},
            coverage_threshold=0.0,
        )
        selected = _select_for_bucket(
            candidates=filtered_candidates,
            bucket_index=bucket_index,
            duration=duration,
            task_type=task_type,
            top_k=bucket_top_k,
        )
        merged_records.extend(selected)
        bucket_key = _bucket_slug(duration, task_type)
        bucket_mix[bucket_key] = {
            "bucket": {"duration": duration, "task_type": task_type},
            "source_family_counts": _count_selected_by(selected, "source_family"),
            "skill_bucket_counts": _count_selected_by(selected, "skill_bucket"),
            "count": len(selected),
        }
        _write_jsonl(output_path / f"bucket_{bucket_key}_top50.jsonl", selected)

    merged_top = _merge_top_records(merged_records, top_k=merged_top_k)
    _write_jsonl(output_path / "merged_top200.jsonl", merged_top)

    sampling_plan = _build_sampling_plan(
        priority_buckets=priority_buckets,
        bucket_mix=bucket_mix,
        merged_top=merged_top,
    )
    (output_path / "bucket_sampling_plan.json").write_text(
        json.dumps(sampling_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "priority_buckets_path": str(output_path / "priority_buckets.json"),
        "candidate_label_summary_path": str(output_path / "candidate_label_summary.json"),
        "bucket_sampling_plan_path": str(output_path / "bucket_sampling_plan.json"),
        "bucket_mix": bucket_mix,
        "merged_top_count": len(merged_top),
        "merged_source_family_counts": _count_selected_by(merged_top, "source_family"),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_demo_candidates(*, molmo2_root: str, per_source_limit: int) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for spec in default_molmo2_candidate_sources(molmo2_root):
        if str(spec.get("name") or "") not in _SELECTOR_SOURCE_NAMES:
            continue
        updated = dict(spec)
        updated["limit"] = per_source_limit
        sources.append(updated)
    return load_examples_from_sources(
        sources,
        role="candidate",
        id_field_name="id",
        text_mode="qa_concat",
        max_records=0,
        include_context=False,
        max_text_chars=1200,
        dedupe_by_text=False,
        skip_missing_paths=True,
    )


def _make_bucket_examples(
    current_rows: list[dict[str, Any]],
    *,
    duration: str,
    task_type: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    source_name = f"videomme_{_bucket_slug(duration, task_type)}"
    for row in samples_for_bucket(current_rows, duration=duration, task_type=task_type, only_wrong=True):
        question = _normalize_videomme_question(str(row.get("question") or ""))
        example = {
            "id": str(row.get("question_id") or row.get("doc_id") or ""),
            "source_name": source_name,
            "source_record_id": str(row.get("question_id") or row.get("doc_id") or ""),
            "question": question,
            "answer": "",
            "context": "",
            "metadata": {
                "duration": duration,
                "task_type": task_type,
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
            },
        }
        example["similarity_text"] = build_similarity_text(
            question=question,
            answer="",
            context="",
            text_mode="question_only",
            include_context=False,
            max_text_chars=1200,
        )
        example["profile"] = infer_example_profile(example, role="benchmark", spec=None)
        examples.append(example)
    return examples


def _select_for_bucket(
    *,
    candidates: list[dict[str, Any]],
    bucket_index: Any,
    duration: str,
    task_type: str,
    top_k: int,
) -> list[dict[str, Any]]:
    allowed_skill_buckets = _BUCKET_SKILL_MAP.get(task_type, set())
    skill_priority = _BUCKET_SKILL_PRIORITY.get(task_type, {})
    primary_sources = _BUCKET_PRIMARY_SOURCES.get((duration, task_type), set())
    secondary_sources = _BUCKET_SECONDARY_SOURCES.get((duration, task_type), set())
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        profile = dict(candidate.get("profile") or {})
        skill_bucket = str(profile.get("skill_bucket") or "general")
        if skill_bucket not in allowed_skill_buckets:
            continue
        semantic_similarity = float(bucket_index.summarize(candidate, top_n=1).max_similarity)
        skill_bucket_match = float(skill_priority.get(skill_bucket, 0.0))
        duration_match = _duration_match(duration, str(profile.get("duration_hint") or "unknown"))
        selection_score = round((0.60 * semantic_similarity) + (0.25 * skill_bucket_match) + (0.15 * duration_match), 6)
        scored.append(
            {
                "id": candidate.get("id"),
                "source_name": candidate.get("source_name"),
                "source_record_id": candidate.get("source_record_id"),
                "question": candidate.get("question"),
                "answer": candidate.get("answer"),
                "metadata": candidate.get("metadata") or {},
                "profile": profile,
                "bucket_duration": duration,
                "bucket_task_type": task_type,
                "semantic_similarity": semantic_similarity,
                "skill_bucket_match": skill_bucket_match,
                "duration_match": duration_match,
                "source_priority": _source_priority(
                    source_family=str(profile.get("source_family") or ""),
                    primary_sources=primary_sources,
                    secondary_sources=secondary_sources,
                ),
                "selection_score": selection_score,
            }
        )

    scored.sort(key=lambda item: item["selection_score"], reverse=True)
    return _select_with_soft_source_cap(
        scored=scored,
        top_k=top_k,
        primary_sources=primary_sources,
        secondary_sources=secondary_sources,
    )


def _merge_top_records(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("id") or "")
        previous = deduped.get(key)
        if previous is None or float(record.get("selection_score") or 0.0) > float(previous.get("selection_score") or 0.0):
            deduped[key] = record

    ordered = sorted(deduped.values(), key=lambda item: float(item.get("selection_score") or 0.0), reverse=True)
    max_count_eval = max(1, int(top_k * 0.10))
    count_eval_count = 0
    selected: list[dict[str, Any]] = []
    for record in ordered:
        source_family = str((record.get("profile") or {}).get("source_family") or "")
        if source_family == "count_eval":
            if count_eval_count >= max_count_eval:
                continue
            count_eval_count += 1
        selected.append(record)
        if len(selected) >= top_k:
            break
    return selected


def _build_sampling_plan(
    *,
    priority_buckets: list[dict[str, Any]],
    bucket_mix: dict[str, Any],
    merged_top: list[dict[str, Any]],
) -> dict[str, Any]:
    priority_lookup = {
        _bucket_slug(str(bucket.get("duration") or ""), str(bucket.get("task_type") or "")): bucket
        for bucket in priority_buckets
    }

    selected_bucket_keys = [key for key in bucket_mix.keys() if key in priority_lookup]
    total_priority = sum(float(priority_lookup[key].get("priority_score") or 0.0) for key in selected_bucket_keys)

    bucket_plan: list[dict[str, Any]] = []
    for bucket_key in selected_bucket_keys:
        mix = dict(bucket_mix.get(bucket_key) or {})
        priority = dict(priority_lookup.get(bucket_key) or {})
        source_counts = dict(mix.get("source_family_counts") or {})
        skill_counts = dict(mix.get("skill_bucket_counts") or {})
        selected_count = int(mix.get("count") or 0)
        priority_score = float(priority.get("priority_score") or 0.0)
        bucket_weight = 0.0 if total_priority <= 0 else round(priority_score / total_priority, 4)

        bucket_plan.append(
            {
                "bucket_key": bucket_key,
                "duration": priority.get("duration"),
                "task_type": priority.get("task_type"),
                "difficulty_tier": priority.get("difficulty_tier"),
                "priority_score": priority_score,
                "bucket_weight": bucket_weight,
                "selected_count": selected_count,
                "source_share": _to_share(source_counts, selected_count),
                "skill_share": _to_share(skill_counts, selected_count),
                "output_file": f"bucket_{bucket_key}_top50.jsonl",
            }
        )

    bucket_plan.sort(key=lambda item: float(item.get("priority_score") or 0.0), reverse=True)

    return {
        "description": "Relative bucket weights derived from VideoMME priority_score; use as sampling quotas, not absolute counts.",
        "bucket_plan": bucket_plan,
        "merged_top_count": len(merged_top),
        "merged_source_share": _to_share(_count_selected_by(merged_top, "source_family"), len(merged_top)),
        "merged_skill_share": _to_share(_count_selected_by(merged_top, "skill_bucket"), len(merged_top)),
    }


def _is_first_round_candidate(candidate: dict[str, Any]) -> bool:
    profile = dict(candidate.get("profile") or {})
    source_family = str(profile.get("source_family") or "")
    skill_bucket = str(profile.get("skill_bucket") or "general")
    metadata = dict(candidate.get("metadata") or {})
    category = _normalize_text(str(metadata.get("category") or ""))

    if source_family == "capeval":
        return False
    if source_family == "count_eval":
        return category in {"object", "action/event", "animal"}
    if source_family == "askmodelanything" and skill_bucket == "general":
        return False
    return True


def _duration_match(bucket_duration: str, candidate_duration_hint: str) -> float:
    if candidate_duration_hint == "unknown":
        return 0.5
    if bucket_duration == "long":
        return 1.0 if candidate_duration_hint == "long" else 0.0
    if bucket_duration == "medium":
        return 1.0 if candidate_duration_hint in {"medium", "long"} else 0.0
    if bucket_duration == "short":
        return 1.0 if candidate_duration_hint == "short" else 0.0
    return 0.0


def _source_priority(
    *,
    source_family: str,
    primary_sources: set[str],
    secondary_sources: set[str],
) -> int:
    if source_family in primary_sources:
        return 2
    if source_family in secondary_sources:
        return 1
    return 0


def _select_with_soft_source_cap(
    *,
    scored: list[dict[str, Any]],
    top_k: int,
    primary_sources: set[str],
    secondary_sources: set[str],
) -> list[dict[str, Any]]:
    source_cap = max(1, int(top_k * 0.35))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()

    primary = [item for item in scored if _record_source_family(item) in primary_sources]
    secondary = [item for item in scored if _record_source_family(item) in secondary_sources]
    tertiary = [
        item
        for item in scored
        if _record_source_family(item) not in primary_sources and _record_source_family(item) not in secondary_sources
    ]

    for phase_records, enforce_cap in (
        (primary, True),
        (primary, False),
        (secondary, True),
        (secondary, False),
        (tertiary, True),
        (tertiary, False),
    ):
        for record in phase_records:
            record_id = str(record.get("id") or "")
            if record_id in selected_ids:
                continue
            source_family = _record_source_family(record)
            if enforce_cap and source_counts[source_family] >= source_cap:
                continue
            selected.append(record)
            selected_ids.add(record_id)
            source_counts[source_family] += 1
            if len(selected) >= top_k:
                return selected

    return selected


def _record_source_family(record: dict[str, Any]) -> str:
    return str((record.get("profile") or {}).get("source_family") or "")


def _normalize_videomme_question(text: str) -> str:
    cleaned = text.replace(
        "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
        "",
    )
    cleaned = cleaned.replace(
        "Answer with the option's letter from the given choices directly.",
        "",
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _bucket_slug(duration: str, task_type: str) -> str:
    slug = f"{duration}_{task_type}".strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _count_by(candidates: list[dict[str, Any]], profile_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for candidate in candidates:
        profile = dict(candidate.get("profile") or {})
        counts[str(profile.get(profile_key) or "")] += 1
    return dict(sorted(counts.items()))


def _count_selected_by(records: list[dict[str, Any]], profile_key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        profile = dict(record.get("profile") or {})
        counts[str(profile.get(profile_key) or "")] += 1
    return dict(sorted(counts.items()))


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _to_share(counts: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {
        key: round((value / total), 4)
        for key, value in sorted(counts.items())
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a VideoMME-oriented Molmo2 selection demo.")
    parser.add_argument("--current-results", default=_DEFAULT_CURRENT_RESULTS)
    parser.add_argument("--baseline-results", default=_DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--molmo2-root", default="/ov2/dataset_molmo2")
    parser.add_argument("--per-source-limit", type=int, default=5000)
    parser.add_argument("--bucket-top-k", type=int, default=50)
    parser.add_argument("--merged-top-k", type=int, default=200)
    parser.add_argument("--similarity-backend", default="token_overlap")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_demo(
        current_results_path=args.current_results,
        baseline_results_path=args.baseline_results,
        output_dir=args.output_dir,
        molmo2_root=args.molmo2_root,
        per_source_limit=args.per_source_limit,
        bucket_top_k=args.bucket_top_k,
        merged_top_k=args.merged_top_k,
        similarity_backend=args.similarity_backend,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
