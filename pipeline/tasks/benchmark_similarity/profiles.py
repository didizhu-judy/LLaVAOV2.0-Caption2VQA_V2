from __future__ import annotations

import re
from typing import Any


_SPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


_DOMAIN_RULES: dict[str, tuple[str, ...]] = {
    "history": ("history", "humanity"),
    "sports": ("sports", "sport", "esports"),
    "basketball": ("basketball",),
    "football": ("football",),
    "documentary": ("documentary",),
    "news": ("news", "news report", "news programs"),
    "movie_tv": ("movie", "tv show", "television", "film", "tv"),
    "animation": ("animation", "cartoon"),
    "art": ("art", "literature", "stage play", "magic show", "variety show", "acrobatics"),
    "technology": ("technology", "computer science", "tech"),
    "science": ("astronomy", "biology", "medicine", "stem", "science"),
    "finance": ("finance", "commerce"),
    "law": ("law",),
    "travel": ("travel", "geography"),
    "food": ("food", "cooking"),
    "life": ("life", "daily life", "life record", "lifestyle", "vlog"),
    "animal": ("animal", "pet"),
    "multilingual": ("multilingual",),
}


_SKILL_RULES: dict[str, tuple[str, ...]] = {
    "counting": ("counting problem", "count", "how many", "number of", "quantity"),
    "temporal_reasoning": ("temporal reasoning", "before", "after", "earlier", "later", "first", "last"),
    "temporal_grounding": ("temporal grounding", "timestamp", "which moment", "what time"),
    "temporal_perception": ("temporal perception",),
    "spatial_reasoning": ("spatial reasoning",),
    "spatial_perception": ("spatial perception", "where is", "location"),
    "object_reasoning": ("object reasoning",),
    "object_recognition": ("object recognition", "object presence", "object properties"),
    "action_reasoning": ("action reasoning",),
    "action_recognition": ("action recognition", "event understanding", "event detection"),
    "summary": ("information synopsis", "summary", "summarization", "synopsis", "video topic", "event summary"),
    "grounding": ("grounding", "localization", "point", "pointing", "location"),
    "dialogue": ("dialogue", "subtitle", "conversation"),
    "ocr": ("ocr", "text recognition", "text location", "text count"),
    "causal": ("causal", "causality", "why"),
    "comparison": ("comparison", "compare"),
    "tracking": ("tracking", "trajectory"),
    "planning": ("planning", "prediction"),
    "scene": ("scene",),
    "event": ("event",),
}


_BENCHMARK_SOURCE_PRIORS: dict[str, dict[str, float]] = {
    "videomme": {
        "askmodelanything": 0.72,
        "caption": 0.70,
        "capqa": 0.90,
        "longcapqa": 0.92,
        "subtitleqa": 0.88,
        "count_eval": 0.84,
        "point_grounding": 0.72,
        "tracking": 0.60,
    },
    "lvbench": {
        "askmodelanything": 0.60,
        "caption": 0.88,
        "capqa": 0.80,
        "longcapqa": 0.98,
        "subtitleqa": 0.95,
        "count_eval": 0.55,
        "point_grounding": 0.55,
        "tracking": 0.48,
    },
    "longvideobench": {
        "askmodelanything": 0.55,
        "caption": 0.82,
        "capqa": 0.84,
        "longcapqa": 0.96,
        "subtitleqa": 1.00,
        "count_eval": 0.45,
        "point_grounding": 0.48,
        "tracking": 0.45,
    },
}


_SKILL_BUCKETS = {
    "temporal_sequence",
    "counting",
    "object_reasoning",
    "action_reasoning",
    "subtitle_alignment",
    "summary",
    "ocr_text",
    "general",
}

_VIDEOMME_TASK_TO_SKILL_BUCKET: dict[str, str] = {
    "temporal reasoning": "temporal_sequence",
    "counting problem": "counting",
    "object reasoning": "object_reasoning",
    "action reasoning": "action_reasoning",
    "ocr problems": "ocr_text",
    "information synopsis": "summary",
}

_TEMPORAL_SEQUENCE_CATEGORIES = {
    "scene sequence",
    "event sequence",
    "action sequence",
    "process description",
    "temporal reasoning",
    "temporal perception",
    "event causality",
    "causal reasoning",
}

_SUMMARY_CATEGORIES = {
    "event summary",
    "video topic",
}

_OBJECT_REASONING_CATEGORIES = {
    "object reasoning",
    "single object state change",
    "single object location change",
    "single object presence change",
}

_ACTION_REASONING_CATEGORIES = {
    "action reasoning",
    "action localization",
}

_COUNTING_CATEGORIES = {
    "action count",
    "moving count",
    "single object event count",
}

_SUBTITLE_ALIGNMENT_KEYWORDS = (
    "temporal sequence bridging",
    "forward alignment",
    "reverse alignment",
    "explanation grounding",
    "cross-modal reasoning",
    "visual explanation",
)

_TEXT_RECOGNITION_CATEGORIES = {
    "text recognition",
    "text location",
}

_COUNT_EVAL_ALLOWED_CATEGORIES = {
    "object",
    "action/event",
    "animal",
}


def infer_example_profile(
    example: dict[str, Any],
    *,
    role: str,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(example.get("metadata") or {})
    question = str(example.get("question") or "")
    answer = str(example.get("answer") or "")
    context = str(example.get("context") or "")
    source_name = _normalize_text(str(example.get("source_name") or ""))
    benchmark_name = _infer_benchmark_name(source_name, metadata) if role == "benchmark" else ""
    source_family = _infer_source_family(source_name)

    raw_texts = [question, answer, context, *_metadata_texts(metadata)]
    text_blob = " | ".join(text for text in raw_texts if text)

    duration_bucket = _infer_duration_bucket(
        role=role,
        source_name=source_name,
        source_family=source_family,
        benchmark_name=benchmark_name,
        metadata=metadata,
    )
    domain_tags = sorted(_infer_domain_tags(text_blob))
    skill_tags = sorted(_infer_skill_tags(text_blob, question=question, source_family=source_family))
    reasoning_tags = sorted(_infer_reasoning_tags(text_blob, skill_tags=skill_tags, metadata=metadata))
    modality_tags = sorted(_infer_modality_tags(text_blob, context=context, metadata=metadata, source_family=source_family))
    locality_tags = sorted(_infer_locality_tags(text_blob, source_family=source_family, metadata=metadata))
    answer_type = _infer_answer_type(answer)
    skill_bucket = _infer_skill_bucket(
        role=role,
        benchmark_name=benchmark_name,
        source_family=source_family,
        question=question,
        answer=answer,
        metadata=metadata,
    )
    duration_hint = _infer_duration_hint(
        role=role,
        source_family=source_family,
        duration_bucket=duration_bucket,
        metadata=metadata,
    )

    profile: dict[str, Any] = {
        "role": role,
        "benchmark_name": benchmark_name,
        "source_family": source_family,
        "duration_bucket": duration_bucket,
        "duration_hint": duration_hint,
        "skill_bucket": skill_bucket,
        "domain_tags": domain_tags,
        "skill_tags": skill_tags,
        "reasoning_tags": reasoning_tags,
        "modality_tags": modality_tags,
        "locality_tags": locality_tags,
        "answer_type": answer_type,
    }

    overrides = dict(spec.get("profile_overrides") or {}) if spec else {}
    if overrides:
        for key, value in overrides.items():
            if key not in profile:
                profile[key] = value
                continue
            if isinstance(profile[key], list):
                profile[key] = sorted({_normalize_text(str(item)) for item in _as_list(value) if str(item).strip()})
            elif value is not None:
                profile[key] = value
    return profile


def score_profile_match(
    benchmark_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> dict[str, float]:
    benchmark_skills = set(_as_list(benchmark_profile.get("skill_tags")))
    candidate_skills = set(_as_list(candidate_profile.get("skill_tags")))
    benchmark_domains = set(_as_list(benchmark_profile.get("domain_tags")))
    candidate_domains = set(_as_list(candidate_profile.get("domain_tags")))
    benchmark_reasoning = set(_as_list(benchmark_profile.get("reasoning_tags")))
    candidate_reasoning = set(_as_list(candidate_profile.get("reasoning_tags")))
    benchmark_modality = set(_as_list(benchmark_profile.get("modality_tags")))
    candidate_modality = set(_as_list(candidate_profile.get("modality_tags")))
    benchmark_locality = set(_as_list(benchmark_profile.get("locality_tags")))
    candidate_locality = set(_as_list(candidate_profile.get("locality_tags")))

    components = {
        "skill": _overlap_score(benchmark_skills, candidate_skills),
        "domain": _overlap_score(benchmark_domains, candidate_domains),
        "reasoning": _overlap_score(benchmark_reasoning, candidate_reasoning),
        "modality": _modality_score(benchmark_modality, candidate_modality),
        "locality": _locality_score(benchmark_locality, candidate_locality),
        "duration": _duration_score(
            str(benchmark_profile.get("duration_bucket") or "unknown"),
            str(candidate_profile.get("duration_bucket") or "unknown"),
        ),
        "answer_type": _answer_type_score(
            str(benchmark_profile.get("answer_type") or "unknown"),
            str(candidate_profile.get("answer_type") or "unknown"),
        ),
        "source_prior": _source_prior(benchmark_profile, candidate_profile),
    }
    return components


def _infer_benchmark_name(source_name: str, metadata: dict[str, Any]) -> str:
    source_text = f"{source_name} {' '.join(_metadata_texts(metadata))}".lower()
    if "videomme" in source_text:
        return "videomme"
    if "lvbench" in source_text and "longvideo" not in source_text:
        return "lvbench"
    if "longvideobench" in source_text or "lvb" in source_text:
        return "longvideobench"
    return ""


def _infer_source_family(source_name: str) -> str:
    source = source_name.lower()
    if "askmodelanything" in source:
        return "askmodelanything"
    if "longcapqa" in source:
        return "longcapqa"
    if "subtitle" in source:
        return "subtitleqa"
    if "count" in source:
        return "count_eval"
    if "point" in source and "track" not in source:
        return "point_grounding"
    if "track" in source:
        return "tracking"
    if "capeval" in source:
        return "capeval"
    if "capqa" in source or "videocapqa" in source:
        return "capqa"
    if "cap" in source:
        return "caption"
    return source or "unknown"


def _infer_duration_bucket(
    *,
    role: str,
    source_name: str,
    source_family: str,
    benchmark_name: str,
    metadata: dict[str, Any],
) -> str:
    for key in ("duration", "duration_group", "video_duration"):
        if key not in metadata:
            continue
        bucket = _duration_from_value(metadata.get(key))
        if bucket != "unknown":
            return bucket

    if role == "benchmark" and benchmark_name == "lvbench":
        return "very_long"
    if source_family == "longcapqa":
        return "long"
    if source_family in {"count_eval", "point_grounding", "tracking"}:
        return "short"
    if source_family in {"subtitleqa", "caption"}:
        return "medium"
    if "long" in source_name:
        return "long"
    return "unknown"


def _duration_from_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, str):
        text = _normalize_text(value)
        if text in {"short", "medium", "long", "very_long"}:
            return text
        if text.isdigit():
            return _duration_from_value(float(text))
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "unknown"

    if numeric in {15.0, 60.0, 600.0, 3600.0}:
        if numeric <= 15.0:
            return "short"
        if numeric <= 60.0:
            return "medium"
        if numeric <= 600.0:
            return "long"
        return "very_long"
    if numeric <= 60.0:
        return "short"
    if numeric <= 600.0:
        return "medium"
    if numeric <= 1800.0:
        return "long"
    return "very_long"


def _infer_domain_tags(text_blob: str) -> set[str]:
    text = _normalize_text(text_blob)
    tags = set()
    for tag, phrases in _DOMAIN_RULES.items():
        if any(phrase in text for phrase in phrases):
            tags.add(tag)
    return tags


def _infer_skill_tags(text_blob: str, *, question: str, source_family: str) -> set[str]:
    text = _normalize_text(text_blob)
    tags = set()
    for tag, phrases in _SKILL_RULES.items():
        if any(phrase in text for phrase in phrases):
            tags.add(tag)

    question_text = _normalize_text(question)
    if "how many" in question_text:
        tags.add("counting")
    if any(token in question_text for token in ("before", "after", "first", "last", "order")):
        tags.add("temporal_reasoning")
    if question_text.startswith("why ") or " why " in f" {question_text} ":
        tags.add("causal")
    if question_text.startswith("where ") or " where " in f" {question_text} ":
        tags.add("spatial_perception")
    if question_text.startswith("summarize") or " summary " in f" {question_text} ":
        tags.add("summary")

    source_defaults = {
        "caption": {"summary", "scene", "event"},
        "capqa": {"object_recognition", "action_recognition", "scene", "event"},
        "longcapqa": {"summary", "temporal_reasoning", "scene", "event"},
        "subtitleqa": {"dialogue", "temporal_reasoning", "causal"},
        "count_eval": {"counting", "grounding"},
        "point_grounding": {"grounding", "spatial_perception"},
        "tracking": {"tracking", "grounding"},
    }
    tags.update(source_defaults.get(source_family, set()))
    return _expand_skill_tags(tags)


def _expand_skill_tags(tags: set[str]) -> set[str]:
    expanded = set(tags)
    if any(tag.startswith("temporal") for tag in tags):
        expanded.add("temporal")
    if any(tag.startswith("spatial") for tag in tags):
        expanded.add("spatial")
    if any(tag.startswith("object") for tag in tags):
        expanded.add("object")
    if any(tag.startswith("action") for tag in tags):
        expanded.add("action")
    if "dialogue" in tags or "ocr" in tags:
        expanded.add("text")
    if "summary" in tags:
        expanded.add("global")
    return expanded


def _infer_reasoning_tags(text_blob: str, *, skill_tags: list[str], metadata: dict[str, Any]) -> set[str]:
    text = _normalize_text(text_blob)
    tags = set()
    if "relation" in text or "reasoning" in text:
        tags.add("reasoning")
    if "perception" in text:
        tags.add("perception")
    if "summary" in text or "synopsis" in text:
        tags.add("summary")
    if "grounding" in text:
        tags.add("grounding")
    if "l1-perception" in text:
        tags.add("perception")
    if "l2-relation" in text:
        tags.add("relation")
    if "reasoning" in skill_tags or any(tag.endswith("_reasoning") for tag in skill_tags):
        tags.add("reasoning")
    if "summary" in skill_tags:
        tags.add("summary")
    if "grounding" in skill_tags or "tracking" in skill_tags:
        tags.add("grounding")
    if "temporal_grounding" in skill_tags:
        tags.add("grounding")
    level = _normalize_text(str(metadata.get("level") or ""))
    if "perception" in level:
        tags.add("perception")
    if "relation" in level:
        tags.add("relation")
    return tags


def _infer_modality_tags(
    text_blob: str,
    *,
    context: str,
    metadata: dict[str, Any],
    source_family: str,
) -> set[str]:
    text = _normalize_text(text_blob)
    tags = {"visual"}
    if "subtitle" in text or "dialogue" in text or "transcript" in text:
        tags.add("subtitle")
    if "ocr" in text or "text recognition" in text or "text location" in text:
        tags.add("ocr")
    if source_family in {"subtitleqa"}:
        tags.update({"subtitle", "cross_modal"})
    if source_family == "caption" and context:
        tags.add("cross_modal")
    alignment_text = " ".join(_metadata_texts({"alignment": metadata.get("AlignmentType")}))
    if "alignment" in _normalize_text(alignment_text) or "cross-modal" in _normalize_text(alignment_text):
        tags.add("cross_modal")
    return tags


def _infer_locality_tags(text_blob: str, *, source_family: str, metadata: dict[str, Any]) -> set[str]:
    text = _normalize_text(text_blob)
    tags = set()
    if " local " in f" {text} ":
        tags.add("local")
    if " global " in f" {text} ":
        tags.add("global")
    if "temporal grounding" in text or "location" in text or "point" in text:
        tags.add("local")
    if "summary" in text or "synopsis" in text:
        tags.add("global")
    if source_family in {"longcapqa", "caption"}:
        tags.add("global")
        tags.add("multi_segment")
    if source_family in {"count_eval", "point_grounding", "tracking"}:
        tags.add("local")
    type_text = _normalize_text(str(metadata.get("type") or ""))
    if type_text in {"local", "global"}:
        tags.add(type_text)
    return tags


def _infer_answer_type(answer: str) -> str:
    text = _normalize_text(answer)
    if not text:
        return "unknown"
    if text in {"yes", "no"}:
        return "binary"
    if _NUMBER_RE.match(text):
        return "number"
    token_count = len(text.split())
    if token_count <= 3:
        return "short_phrase"
    return "sentence"


def _infer_skill_bucket(
    *,
    role: str,
    benchmark_name: str,
    source_family: str,
    question: str,
    answer: str,
    metadata: dict[str, Any],
) -> str:
    question_text = _normalize_text(question)
    metadata_category = _normalize_text(
        str(
            metadata.get("Category")
            or metadata.get("category")
            or metadata.get("task_type")
            or metadata.get("question_type")
            or ""
        )
    )
    alignment = _normalize_text(str(metadata.get("AlignmentType") or ""))

    if role == "benchmark" and benchmark_name == "videomme":
        mapped = _VIDEOMME_TASK_TO_SKILL_BUCKET.get(metadata_category)
        if mapped:
            return mapped

    if source_family in {"capqa", "longcapqa"}:
        if metadata_category in _SUMMARY_CATEGORIES:
            return "summary"
        if metadata_category in _TEMPORAL_SEQUENCE_CATEGORIES:
            return "temporal_sequence"
        if metadata_category in _OBJECT_REASONING_CATEGORIES:
            return "object_reasoning"
        if metadata_category in _ACTION_REASONING_CATEGORIES:
            return "action_reasoning"
        if metadata_category in _COUNTING_CATEGORIES:
            return "counting"
        return "general"

    if source_family == "subtitleqa":
        if metadata_category in _TEXT_RECOGNITION_CATEGORIES:
            return "ocr_text"
        if any(keyword in alignment for keyword in _SUBTITLE_ALIGNMENT_KEYWORDS):
            return "subtitle_alignment"
        if metadata_category in _TEMPORAL_SEQUENCE_CATEGORIES:
            return "temporal_sequence"
        if metadata_category == "object reasoning":
            return "object_reasoning"
        if metadata_category == "action reasoning":
            return "action_reasoning"
        return "general"

    if source_family == "caption":
        if _looks_temporal_sequence_question(question_text):
            return "temporal_sequence"
        return "summary"

    if source_family == "askmodelanything":
        if "how many" in question_text:
            return "counting"
        if _looks_temporal_sequence_question(question_text):
            return "temporal_sequence"
        if question_text.startswith("why ") or " why " in f" {question_text} " or "reason" in question_text:
            return "action_reasoning"
        return "general"

    if source_family == "count_eval":
        if metadata_category in _COUNT_EVAL_ALLOWED_CATEGORIES:
            return "counting"
        return "general"

    if benchmark_name == "videomme":
        mapped = _VIDEOMME_TASK_TO_SKILL_BUCKET.get(metadata_category)
        if mapped:
            return mapped

    if "how many" in question_text or _NUMBER_RE.match(_normalize_text(answer)):
        return "counting"
    if _looks_temporal_sequence_question(question_text):
        return "temporal_sequence"
    return "general"


def _infer_duration_hint(
    *,
    role: str,
    source_family: str,
    duration_bucket: str,
    metadata: dict[str, Any],
) -> str:
    if duration_bucket != "unknown":
        return duration_bucket

    if source_family == "longcapqa":
        return "long"

    if source_family == "count_eval":
        for key in ("video_duration", "duration", "video_length"):
            if key in metadata:
                bucket = _duration_from_value(metadata.get(key))
                if bucket != "unknown":
                    return bucket
        return "short"

    if role == "benchmark":
        return duration_bucket
    return "unknown"


def _looks_temporal_sequence_question(question_text: str) -> bool:
    return any(
        token in question_text
        for token in (
            "before",
            "after",
            "first",
            "next",
            "then",
            "later",
            "earlier",
            "finally",
            "order",
            "sequence",
            "at the beginning of",
            "at the end of",
            "between",
        )
    )


def _source_prior(benchmark_profile: dict[str, Any], candidate_profile: dict[str, Any]) -> float:
    benchmark_name = str(benchmark_profile.get("benchmark_name") or "")
    source_family = str(candidate_profile.get("source_family") or "unknown")
    base = _BENCHMARK_SOURCE_PRIORS.get(benchmark_name, {}).get(source_family, 0.55)

    benchmark_skills = set(_as_list(benchmark_profile.get("skill_tags")))
    benchmark_modality = set(_as_list(benchmark_profile.get("modality_tags")))
    benchmark_locality = set(_as_list(benchmark_profile.get("locality_tags")))

    if "counting" in benchmark_skills and source_family == "count_eval":
        base += 0.18
    if "subtitle" in benchmark_modality and source_family == "subtitleqa":
        base += 0.18
    if "summary" in benchmark_skills and source_family in {"longcapqa", "caption"}:
        base += 0.12
    if "global" in benchmark_locality and source_family == "longcapqa":
        base += 0.10
    if "grounding" in benchmark_skills and source_family in {"count_eval", "point_grounding", "tracking"}:
        base += 0.12

    return max(0.0, min(base, 1.0))


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap <= 0:
        return 0.0
    return float((2.0 * overlap) / (len(left) + len(right)))


def _modality_score(benchmark_tags: set[str], candidate_tags: set[str]) -> float:
    if not benchmark_tags:
        return 0.5
    if benchmark_tags <= candidate_tags:
        return 1.0
    overlap = benchmark_tags & candidate_tags
    if overlap:
        return float(len(overlap) / len(benchmark_tags))
    return 0.0


def _locality_score(benchmark_tags: set[str], candidate_tags: set[str]) -> float:
    if not benchmark_tags:
        return 0.5
    if "global" in benchmark_tags and {"global", "multi_segment"} & candidate_tags:
        return 1.0
    if "local" in benchmark_tags and "local" in candidate_tags:
        return 1.0
    return _overlap_score(benchmark_tags, candidate_tags)


def _duration_score(benchmark_bucket: str, candidate_bucket: str) -> float:
    if benchmark_bucket == "unknown" or candidate_bucket == "unknown":
        return 0.5
    if benchmark_bucket == candidate_bucket:
        return 1.0
    order = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
    if benchmark_bucket not in order or candidate_bucket not in order:
        return 0.0
    gap = abs(order[benchmark_bucket] - order[candidate_bucket])
    if gap == 1:
        return 0.65
    if gap == 2:
        return 0.25
    return 0.0


def _answer_type_score(benchmark_answer_type: str, candidate_answer_type: str) -> float:
    if benchmark_answer_type == "unknown" or candidate_answer_type == "unknown":
        return 0.5
    if benchmark_answer_type == candidate_answer_type:
        return 1.0
    if benchmark_answer_type in {"short_phrase", "sentence"} and candidate_answer_type in {
        "short_phrase",
        "sentence",
    }:
        return 0.75
    return 0.0


def _metadata_texts(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in metadata.items():
        if key in {"video_id", "id", "question_id"}:
            continue
        values.extend(_string_values(value))
    return values


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_string_values(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_string_values(item))
        return out
    return [str(value)]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip().lower())
