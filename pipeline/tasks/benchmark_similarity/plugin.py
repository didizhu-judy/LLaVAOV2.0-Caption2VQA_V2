from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import orjson

from pipeline.core.config import PipelineConfig
from pipeline.tasks.base import RequestSpec, TaskPlugin
from pipeline.tasks.benchmark_similarity.io import (
    default_molmo2_qa_sources,
    load_examples_from_sources,
)
from pipeline.tasks.benchmark_similarity.similarity import build_similarity_index
from pipeline.types import ProcessedRecord, RawItem


class BenchmarkSimilarityTask(TaskPlugin):
    name = "benchmark_similarity"
    requires_endpoints = False

    def __init__(self) -> None:
        self._runtime_lock = threading.Lock()
        self._runtime_key: str | None = None
        self._benchmark_examples: list[dict[str, Any]] = []
        self._similarity_index: Any = None

    def load_items(self, config: PipelineConfig) -> list[RawItem]:
        task_cfg = config.task_config
        benchmark_sources = _benchmark_sources(task_cfg)
        load_examples_from_sources(
            benchmark_sources,
            role="benchmark",
            id_field_name=config.id_field,
            text_mode=_cfg_str(task_cfg, "text_mode", default="qa_concat"),
            max_records=_cfg_int(task_cfg, "max_benchmark_records", default=0),
            include_context=_cfg_bool(task_cfg, "include_benchmark_context", default=True),
            max_text_chars=_cfg_int(task_cfg, "max_text_chars", default=1200),
            dedupe_by_text=_cfg_bool(task_cfg, "dedupe_benchmark_by_text", default=False),
            skip_missing_paths=_cfg_bool(task_cfg, "skip_missing_paths", default=True),
        )

        candidate_sources = _candidate_sources(task_cfg)
        return load_examples_from_sources(
            candidate_sources,
            role="candidate",
            id_field_name=config.id_field,
            text_mode=_cfg_str(task_cfg, "text_mode", default="qa_concat"),
            max_records=_cfg_int(task_cfg, "max_records", default=0),
            include_context=_cfg_bool(task_cfg, "include_candidate_context", default=False),
            max_text_chars=_cfg_int(task_cfg, "max_text_chars", default=1200),
            dedupe_by_text=_cfg_bool(task_cfg, "dedupe_candidate_by_text", default=False),
            skip_missing_paths=_cfg_bool(task_cfg, "skip_missing_paths", default=True),
        )

    def build_request(self, item: RawItem, config: PipelineConfig) -> RequestSpec:
        self._ensure_runtime(config)
        if self._similarity_index is None:
            raise RuntimeError("Similarity index is not initialized")

        summary = self._similarity_index.summarize(
            item,
            top_n=_cfg_int(config.task_config, "top_k_matches", default=3),
        )
        return RequestSpec(
            payload={},
            skip_http=True,
            local_response={
                "_similarity": {
                    "backend": summary.backend,
                    "top_matches": summary.top_matches,
                    "benchmark_source_scores": summary.benchmark_source_scores,
                    "benchmark_source_semantic_scores": summary.benchmark_source_semantic_scores,
                    "max_similarity": summary.max_similarity,
                    "max_adjusted_similarity": summary.max_adjusted_similarity,
                    "mean_topk_similarity": summary.mean_topk_similarity,
                    "mean_topk_adjusted_similarity": summary.mean_topk_adjusted_similarity,
                    "balanced_score": summary.balanced_score,
                    "coverage_score": summary.coverage_score,
                    "selection_score": summary.selection_score,
                    "best_benchmark_source": summary.best_benchmark_source,
                }
            },
        )

    def parse_response(
        self,
        item: RawItem,
        llm_response: dict[str, Any],
        config: PipelineConfig,
        *,
        secondary_llm_response: dict[str, Any] | None = None,
    ) -> ProcessedRecord:
        payload = dict(llm_response.get("_similarity") or {})
        ranking_field = _cfg_str(config.task_config, "ranking_score_field", default="selection_score")
        ranking_score = float(payload.get(ranking_field) or 0.0)
        profile = dict(item.get("profile") or {})

        out: ProcessedRecord = {
            config.id_field: str(item.get(config.id_field) or ""),
            "source_name": item.get("source_name"),
            "source_path": item.get("source_path"),
            "source_record_id": item.get("source_record_id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "metadata": item.get("metadata") or {},
            "similarity_backend": payload.get("backend"),
            "ranking_score_field": ranking_field,
            "ranking_score": ranking_score,
            "source_family": profile.get("source_family"),
            "skill_bucket": profile.get("skill_bucket"),
            "duration_hint": profile.get("duration_hint"),
            "max_similarity": float(payload.get("max_similarity") or 0.0),
            "max_adjusted_similarity": float(payload.get("max_adjusted_similarity") or 0.0),
            "mean_topk_similarity": float(payload.get("mean_topk_similarity") or 0.0),
            "mean_topk_adjusted_similarity": float(payload.get("mean_topk_adjusted_similarity") or 0.0),
            "balanced_score": float(payload.get("balanced_score") or 0.0),
            "coverage_score": float(payload.get("coverage_score") or 0.0),
            "selection_score": float(payload.get("selection_score") or 0.0),
            "best_benchmark_source": payload.get("best_benchmark_source") or "",
            "benchmark_source_scores": payload.get("benchmark_source_scores") or {},
            "benchmark_source_semantic_scores": payload.get("benchmark_source_semantic_scores") or {},
            "top_matches": payload.get("top_matches") or [],
            "profile": profile,
        }
        if _cfg_bool(config.task_config, "save_similarity_text", default=False):
            out["similarity_text"] = item.get("similarity_text")
        if _cfg_bool(config.task_config, "save_context", default=False):
            out["context"] = item.get("context")
        return out

    def finalize_outputs(self, config: PipelineConfig) -> dict[str, Any] | None:
        output_path = Path(config.output_jsonl)
        if not output_path.exists():
            return None

        records = _load_jsonl(output_path)
        if not records:
            return None

        score_field = _cfg_str(config.task_config, "ranking_score_field", default="selection_score")
        sorted_records = sorted(
            records,
            key=lambda item: float(item.get(score_field) or 0.0),
            reverse=True,
        )

        top_k = _cfg_int(config.task_config, "export_top_k", default=1000)
        topk_path = Path(
            _cfg_str(
                config.task_config,
                "topk_output_jsonl",
                default=str(output_path.with_name(f"{output_path.stem}_topk.jsonl")),
            )
        )
        summary_path = Path(
            _cfg_str(
                config.task_config,
                "summary_json",
                default=str(output_path.with_name(f"{output_path.stem}_summary.json")),
            )
        )

        top_records = sorted_records[:top_k] if top_k > 0 else sorted_records
        _write_jsonl(topk_path, top_records)
        per_benchmark_top_k = _cfg_int(config.task_config, "per_benchmark_top_k", default=0)

        score_values = [float(item.get(score_field) or 0.0) for item in sorted_records]
        source_groups: dict[str, list[float]] = {}
        source_family_groups: dict[str, list[float]] = {}
        top1_benchmark_groups: dict[str, int] = {}
        benchmark_bucket_records: dict[str, list[dict[str, Any]]] = {}
        for record in sorted_records:
            source_name = str(record.get("source_name") or "")
            source_groups.setdefault(source_name, []).append(float(record.get(score_field) or 0.0))
            source_family = str(record.get("source_family") or source_name)
            source_family_groups.setdefault(source_family, []).append(float(record.get(score_field) or 0.0))
            top1_name = str(record.get("best_benchmark_source") or "")
            if top1_name:
                top1_benchmark_groups[top1_name] = top1_benchmark_groups.get(top1_name, 0) + 1
                benchmark_bucket_records.setdefault(top1_name, []).append(record)

        top_source_family_counts: dict[str, int] = {}
        for record in top_records:
            source_family = str(record.get("source_family") or record.get("source_name") or "")
            top_source_family_counts[source_family] = top_source_family_counts.get(source_family, 0) + 1

        benchmark_bucket_outputs: dict[str, str] = {}
        benchmark_bucket_mix: dict[str, list[dict[str, float | int | str]]] = {}
        benchmark_skill_mix: dict[str, list[dict[str, float | int | str]]] = {}
        for benchmark_source, bucket_records in sorted(benchmark_bucket_records.items()):
            family_counts: dict[str, int] = {}
            selected_records = bucket_records[:per_benchmark_top_k] if per_benchmark_top_k > 0 else bucket_records
            for record in selected_records:
                source_family = str(record.get("source_family") or record.get("source_name") or "")
                family_counts[source_family] = family_counts.get(source_family, 0) + 1
            benchmark_bucket_mix[benchmark_source] = [
                {
                    "source_family": source_family,
                    "count": count,
                    "fraction": float(count / len(selected_records)) if selected_records else 0.0,
                }
                for source_family, count in sorted(family_counts.items(), key=lambda item: item[1], reverse=True)
            ]
            if per_benchmark_top_k > 0:
                bucket_path = output_path.with_name(f"{output_path.stem}_{benchmark_source}_topk.jsonl")
                _write_jsonl(bucket_path, selected_records)
                benchmark_bucket_outputs[benchmark_source] = str(bucket_path)

        skill_bucket_records: dict[str, list[dict[str, Any]]] = {}
        for record in sorted_records:
            top_matches = record.get("top_matches") or []
            if not top_matches:
                continue
            top_match = top_matches[0]
            benchmark_source = str(top_match.get("benchmark_source") or record.get("best_benchmark_source") or "")
            benchmark_profile = dict(top_match.get("benchmark_profile") or {})
            primary_skill = _primary_skill_tag(benchmark_profile)
            if not benchmark_source or not primary_skill:
                continue
            bucket_key = f"{benchmark_source}:{primary_skill}"
            skill_bucket_records.setdefault(bucket_key, []).append(record)

        for bucket_key, bucket_records in sorted(skill_bucket_records.items()):
            family_counts: dict[str, int] = {}
            selected_records = bucket_records[:per_benchmark_top_k] if per_benchmark_top_k > 0 else bucket_records
            for record in selected_records:
                source_family = str(record.get("source_family") or record.get("source_name") or "")
                family_counts[source_family] = family_counts.get(source_family, 0) + 1
            benchmark_skill_mix[bucket_key] = [
                {
                    "source_family": source_family,
                    "count": count,
                    "fraction": float(count / len(selected_records)) if selected_records else 0.0,
                }
                for source_family, count in sorted(family_counts.items(), key=lambda item: item[1], reverse=True)
            ]

        summary = {
            "task_name": self.name,
            "ranking_score_field": score_field,
            "record_count": len(sorted_records),
            "topk_record_count": len(top_records),
            "similarity_backend": sorted_records[0].get("similarity_backend"),
            "topk_output_jsonl": str(topk_path),
            "summary_json": str(summary_path),
            "score_stats": _score_stats(score_values),
            "source_stats": [
                {
                    "source_name": source_name,
                    "count": len(values),
                    "score_stats": _score_stats(values),
                }
                for source_name, values in sorted(source_groups.items())
            ],
            "source_family_stats": [
                {
                    "source_family": source_family,
                    "count": len(values),
                    "score_stats": _score_stats(values),
                }
                for source_family, values in sorted(source_family_groups.items())
            ],
            "best_benchmark_source_counts": dict(sorted(top1_benchmark_groups.items())),
            "recommended_source_mix": [
                {
                    "source_family": source_family,
                    "count": count,
                    "fraction": float(count / len(top_records)) if top_records else 0.0,
                }
                for source_family, count in sorted(
                    top_source_family_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ],
            "recommended_source_mix_by_benchmark": benchmark_bucket_mix,
            "recommended_source_mix_by_benchmark_skill": benchmark_skill_mix,
            "benchmark_topk_outputs": benchmark_bucket_outputs,
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _ensure_runtime(self, config: PipelineConfig) -> None:
        runtime_key = _runtime_signature(config.task_config)
        if self._runtime_key == runtime_key and self._similarity_index is not None:
            return

        with self._runtime_lock:
            if self._runtime_key == runtime_key and self._similarity_index is not None:
                return

            task_cfg = config.task_config
            benchmark_examples = load_examples_from_sources(
                _benchmark_sources(task_cfg),
                role="benchmark",
                id_field_name=config.id_field,
                text_mode=_cfg_str(task_cfg, "text_mode", default="qa_concat"),
                max_records=_cfg_int(task_cfg, "max_benchmark_records", default=0),
                include_context=_cfg_bool(task_cfg, "include_benchmark_context", default=True),
                max_text_chars=_cfg_int(task_cfg, "max_text_chars", default=1200),
                dedupe_by_text=_cfg_bool(task_cfg, "dedupe_benchmark_by_text", default=False),
                skip_missing_paths=_cfg_bool(task_cfg, "skip_missing_paths", default=True),
            )
            self._benchmark_examples = benchmark_examples
            self._similarity_index = build_similarity_index(
                benchmark_examples,
                requested_backend=_cfg_str(task_cfg, "similarity_backend", default="auto"),
                embedding_model=_cfg_str(
                    task_cfg,
                    "embedding_model",
                    default="sentence-transformers/all-MiniLM-L6-v2",
                ),
                batch_size=_cfg_int(task_cfg, "embedding_batch_size", default=64),
                scoring_weights=_cfg_float_dict(task_cfg.get("scoring_weights")),
                benchmark_weights=_cfg_float_dict(task_cfg.get("benchmark_weights")),
                coverage_threshold=_cfg_float(task_cfg, "coverage_threshold", default=0.45),
            )
            self._runtime_key = runtime_key


def _benchmark_sources(task_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task_cfg.get("benchmark_sources")
    if not isinstance(raw, list) or not raw:
        raise ValueError("task_config.benchmark_sources must be a non-empty list")
    return [dict(item) for item in raw if isinstance(item, dict)]


def _candidate_sources(task_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = task_cfg.get("candidate_sources")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]

    if _cfg_bool(task_cfg, "use_default_molmo2_qa_sources", default=False):
        return default_molmo2_qa_sources(
            _cfg_str(task_cfg, "molmo2_root", default="/ov2/dataset_molmo2")
        )

    raise ValueError(
        "Provide task_config.candidate_sources or enable task_config.use_default_molmo2_qa_sources"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")


def _score_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered_values: list[float], q: float) -> float:
    if not ordered_values:
        return 0.0
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    q = min(1.0, max(0.0, float(q)))
    index = (len(ordered_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = index - lower
    return float(ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight)


def _primary_skill_tag(profile: dict[str, Any]) -> str:
    bucket = str(profile.get("skill_bucket") or "").strip()
    if bucket:
        return bucket
    skill_tags = [str(item) for item in (profile.get("skill_tags") or []) if str(item).strip()]
    if not skill_tags:
        return ""
    preferred_order = [
        "counting",
        "temporal_reasoning",
        "temporal_grounding",
        "summary",
        "dialogue",
        "ocr",
        "causal",
        "spatial_reasoning",
        "spatial_perception",
        "object_reasoning",
        "object_recognition",
        "action_reasoning",
        "action_recognition",
        "grounding",
        "tracking",
    ]
    for tag in preferred_order:
        if tag in skill_tags:
            return tag
    return skill_tags[0]


def _runtime_signature(task_cfg: dict[str, Any]) -> str:
    relevant = {
        "benchmark_sources": task_cfg.get("benchmark_sources"),
        "text_mode": task_cfg.get("text_mode"),
        "max_benchmark_records": task_cfg.get("max_benchmark_records"),
        "include_benchmark_context": task_cfg.get("include_benchmark_context"),
        "max_text_chars": task_cfg.get("max_text_chars"),
        "dedupe_benchmark_by_text": task_cfg.get("dedupe_benchmark_by_text"),
        "skip_missing_paths": task_cfg.get("skip_missing_paths"),
        "similarity_backend": task_cfg.get("similarity_backend"),
        "embedding_model": task_cfg.get("embedding_model"),
        "embedding_batch_size": task_cfg.get("embedding_batch_size"),
        "scoring_weights": task_cfg.get("scoring_weights"),
        "benchmark_weights": task_cfg.get("benchmark_weights"),
        "coverage_threshold": task_cfg.get("coverage_threshold"),
    }
    return json.dumps(relevant, sort_keys=True, ensure_ascii=False)


def _cfg_str(
    task_cfg: dict[str, Any],
    key: str,
    *,
    default: str | None = None,
) -> str:
    value = task_cfg.get(key, default)
    if value is None:
        return ""
    return str(value)


def _cfg_int(task_cfg: dict[str, Any], key: str, *, default: int) -> int:
    value = task_cfg.get(key, default)
    return int(value)


def _cfg_float(task_cfg: dict[str, Any], key: str, *, default: float) -> float:
    value = task_cfg.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cfg_bool(task_cfg: dict[str, Any], key: str, *, default: bool) -> bool:
    value = task_cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, item in value.items():
        try:
            out[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return out
