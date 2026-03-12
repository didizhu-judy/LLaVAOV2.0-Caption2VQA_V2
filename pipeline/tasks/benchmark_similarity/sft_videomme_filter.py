from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_DEFAULT_INPUT_GLOB = "/ov2/feilong/convert_json/Molmo2-videoforsft/*.jsonl"
_DEFAULT_OUTPUT_DIR = "output/videomme_sft_filter"

_RELEVANT_SKILLS = {
    "temporal_sequence",
    "counting",
    "object_reasoning",
    "action_reasoning",
    "subtitle_alignment",
    "summary",
    "ocr_text",
}

_SKILL_PRIORITY = [
    "ocr_text",
    "counting",
    "subtitle_alignment",
    "temporal_sequence",
    "object_reasoning",
    "action_reasoning",
    "summary",
    "general",
]

_SOURCE_NAME_MAP = {
    "askmodelanything": "molmo2_askmodelanything",
    "capqa": "molmo2_videocapqa",
    "longcapqa": "molmo2_longcapqa",
    "caption": "molmo2_cap",
    "subtitleqa": "molmo2_videosubtitleqa",
    "count_eval": "molmo2_videocounteval",
}


def run_filter(
    *,
    input_glob: str = _DEFAULT_INPUT_GLOB,
    output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    log_every: int = 100000,
) -> dict[str, Any]:
    input_paths = sorted(Path().glob(input_glob) if not input_glob.startswith("/") else Path("/").glob(input_glob.lstrip("/")))
    if not input_paths:
        raise FileNotFoundError(f"No jsonl matched: {input_glob}")

    out_dir = Path(output_dir)
    tags_dir = out_dir / "tags"
    filtered_dir = out_dir / "filtered"
    tags_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "filtered_merged.jsonl"

    overall = {
        "input_glob": input_glob,
        "file_count": len(input_paths),
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_rows": 0,
        "source_family_counts": Counter(),
        "duration_bucket_counts": Counter(),
        "skill_bucket_counts": Counter(),
        "keep_reason_counts": Counter(),
        "drop_reason_counts": Counter(),
        "target_bucket_counts": Counter(),
        "per_file": [],
    }

    with merged_path.open("w", encoding="utf-8") as merged_handle:
        for input_path in input_paths:
            file_summary = _process_file(
                input_path=input_path,
                tags_dir=tags_dir,
                filtered_dir=filtered_dir,
                merged_handle=merged_handle,
                log_every=log_every,
            )
            overall["total_rows"] += file_summary["total_rows"]
            overall["kept_rows"] += file_summary["kept_rows"]
            overall["dropped_rows"] += file_summary["dropped_rows"]
            overall["source_family_counts"].update(file_summary["source_family_counts"])
            overall["duration_bucket_counts"].update(file_summary["duration_bucket_counts"])
            overall["skill_bucket_counts"].update(file_summary["skill_bucket_counts"])
            overall["keep_reason_counts"].update(file_summary["keep_reason_counts"])
            overall["drop_reason_counts"].update(file_summary["drop_reason_counts"])
            overall["target_bucket_counts"].update(file_summary["target_bucket_counts"])
            overall["per_file"].append(file_summary)

    summary = {
        "input_glob": overall["input_glob"],
        "file_count": overall["file_count"],
        "total_rows": overall["total_rows"],
        "kept_rows": overall["kept_rows"],
        "dropped_rows": overall["dropped_rows"],
        "source_family_counts": dict(sorted(overall["source_family_counts"].items())),
        "duration_bucket_counts": dict(sorted(overall["duration_bucket_counts"].items())),
        "skill_bucket_counts": dict(sorted(overall["skill_bucket_counts"].items())),
        "keep_reason_counts": dict(sorted(overall["keep_reason_counts"].items())),
        "drop_reason_counts": dict(sorted(overall["drop_reason_counts"].items())),
        "target_bucket_counts": dict(sorted(overall["target_bucket_counts"].items())),
        "filtered_merged_path": str(merged_path),
        "per_file": overall["per_file"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _process_file(
    *,
    input_path: Path,
    tags_dir: Path,
    filtered_dir: Path,
    merged_handle: Any,
    log_every: int,
) -> dict[str, Any]:
    tags_path = tags_dir / f"{input_path.stem}.tags.jsonl"
    filtered_path = filtered_dir / f"{input_path.stem}.filtered.jsonl"

    summary = {
        "input_path": str(input_path),
        "tags_path": str(tags_path),
        "filtered_path": str(filtered_path),
        "total_rows": 0,
        "kept_rows": 0,
        "dropped_rows": 0,
        "source_family_counts": Counter(),
        "duration_bucket_counts": Counter(),
        "skill_bucket_counts": Counter(),
        "keep_reason_counts": Counter(),
        "drop_reason_counts": Counter(),
        "target_bucket_counts": Counter(),
    }

    duration_bucket = _infer_duration_bucket_from_filename(input_path.name)

    with input_path.open("r", encoding="utf-8") as in_handle, tags_path.open("w", encoding="utf-8") as tags_handle, filtered_path.open("w", encoding="utf-8") as filtered_handle:
        for line_number, line in enumerate(in_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            tags = _tag_row(row=row, input_name=input_path.name, duration_bucket=duration_bucket)

            summary["total_rows"] += 1
            summary["source_family_counts"][tags["source_family"]] += 1
            summary["duration_bucket_counts"][tags["duration_bucket"]] += 1
            summary["skill_bucket_counts"][tags["skill_bucket"]] += 1
            if tags["keep_for_videomme"]:
                summary["kept_rows"] += 1
                summary["keep_reason_counts"][tags["keep_reason"]] += 1
                for bucket in tags["target_buckets"]:
                    summary["target_bucket_counts"][bucket] += 1
                payload = dict(row)
                payload["videomme_tags"] = tags
                serialized = json.dumps(payload, ensure_ascii=False)
                filtered_handle.write(serialized + "\n")
                merged_handle.write(serialized + "\n")
            else:
                summary["dropped_rows"] += 1
                summary["drop_reason_counts"][tags["drop_reason"]] += 1

            tags_handle.write(
                json.dumps(
                    {
                        "line_number": line_number,
                        "id": row.get("id"),
                        "videomme_tags": tags,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if log_every > 0 and summary["total_rows"] % log_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "input_path": str(input_path),
                            "rows": summary["total_rows"],
                            "kept_rows": summary["kept_rows"],
                            "dropped_rows": summary["dropped_rows"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    return {
        "input_path": summary["input_path"],
        "tags_path": summary["tags_path"],
        "filtered_path": summary["filtered_path"],
        "total_rows": summary["total_rows"],
        "kept_rows": summary["kept_rows"],
        "dropped_rows": summary["dropped_rows"],
        "source_family_counts": dict(sorted(summary["source_family_counts"].items())),
        "duration_bucket_counts": dict(sorted(summary["duration_bucket_counts"].items())),
        "skill_bucket_counts": dict(sorted(summary["skill_bucket_counts"].items())),
        "keep_reason_counts": dict(sorted(summary["keep_reason_counts"].items())),
        "drop_reason_counts": dict(sorted(summary["drop_reason_counts"].items())),
        "target_bucket_counts": dict(sorted(summary["target_bucket_counts"].items())),
    }


def _tag_row(*, row: dict[str, Any], input_name: str, duration_bucket: str) -> dict[str, Any]:
    source_path = _first_image_source(row.get("images_source"))
    source_family = _infer_source_family(source_path)
    source_name = _SOURCE_NAME_MAP.get(source_family, "")
    user_questions = _extract_user_questions(row.get("messages") or [])
    question_skills = [_classify_question(question, source_family=source_family) for question in user_questions]
    skill_bucket = _choose_dominant_skill(question_skills, source_family=source_family)
    keep_for_videomme, keep_reason, drop_reason = _decide_keep(
        source_family=source_family,
        skill_bucket=skill_bucket,
        question_skills=question_skills,
    )
    target_buckets = _infer_target_buckets(
        duration_bucket=duration_bucket,
        skill_bucket=skill_bucket,
    ) if keep_for_videomme else []

    return {
        "input_name": input_name,
        "source_path": source_path,
        "source_family": source_family,
        "source_name": source_name,
        "duration_bucket": duration_bucket,
        "question_count": len(user_questions),
        "question_skill_counts": dict(sorted(Counter(question_skills).items())),
        "skill_bucket": skill_bucket,
        "keep_for_videomme": keep_for_videomme,
        "keep_reason": keep_reason,
        "drop_reason": drop_reason,
        "target_buckets": target_buckets,
    }


def _first_image_source(images_source: Any) -> str:
    if isinstance(images_source, list) and images_source:
        return str(images_source[0] or "")
    return str(images_source or "")


def _infer_source_family(source_path: str) -> str:
    if "LongCapQA" in source_path:
        return "longcapqa"
    if "Molmo2-AskModelAnything" in source_path:
        return "askmodelanything"
    if "Molmo2-VideoSubtitleQA" in source_path:
        return "subtitleqa"
    if "Molmo2-VideoCountEval" in source_path:
        return "count_eval"
    if "Molmo2-Cap" in source_path:
        return "caption"
    if "Molmo2-VideoCapQA" in source_path:
        return "capqa"
    return "unknown"


def _infer_duration_bucket_from_filename(name: str) -> str:
    lowered = name.lower()
    if "0_60s" in lowered:
        return "short"
    if "60_180s" in lowered:
        return "medium"
    if "180_600s" in lowered or "more_600s" in lowered:
        return "long"
    return "unknown"


def _extract_user_questions(messages: list[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for message in messages:
        if str(message.get("role") or "") != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            questions.append(content)
    return questions


def _classify_question(question: str, *, source_family: str) -> str:
    text = _normalize_text(question)
    if not text:
        return "general"

    if source_family == "count_eval":
        if _contains_any(text, ("how many", "number of", "count the", "count of")):
            return "counting"
        if text.startswith("find "):
            return "general"

    if _contains_any(
        text,
        (
            "what text",
            "what word",
            "what words",
            "what title",
            "what line of text",
            "what is written",
            "what does the sign say",
            "what does the text say",
            "what is the title",
            "what name is",
            "what appears right above",
            "what appears on the sign",
            "what is on the sign",
        ),
    ):
        return "ocr_text"

    if _contains_any(text, ("how many", "number of", "count the", "count of")):
        return "counting"

    if source_family == "subtitleqa" and _contains_any(
        text,
        (
            "narrator",
            "speaker",
            "dialogue",
            "subtitle",
            "subtitles",
            "says",
            "said",
            "mentioned",
            "speaks",
        ),
    ):
        return "subtitle_alignment"

    if _contains_any(
        text,
        (
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
            "at the beginning",
            "at the end",
            "what happens between",
            "what happens after",
            "what happens before",
            "in what order",
        ),
    ):
        return "temporal_sequence"

    if _contains_any(
        text,
        (
            "main topic",
            "main idea",
            "what is this video about",
            "what is the video about",
            "summarize",
            "summary",
            "overall",
        ),
    ):
        return "summary"

    if _contains_any(text, ("why", "reason")):
        return "action_reasoning"

    if _contains_any(
        text,
        (
            "which tool",
            "which object",
            "which item",
            "used later",
            "shown earlier is used later",
            "what object is used",
        ),
    ):
        return "object_reasoning"

    if source_family == "caption":
        return "summary"

    return "general"


def _choose_dominant_skill(question_skills: list[str], *, source_family: str) -> str:
    if not question_skills:
        return "summary" if source_family == "caption" else "general"

    counts = Counter(question_skills)
    best_skill = "general"
    best_count = -1
    for skill in _SKILL_PRIORITY:
        count = counts.get(skill, 0)
        if count > best_count:
            best_skill = skill
            best_count = count
    return best_skill


def _decide_keep(*, source_family: str, skill_bucket: str, question_skills: list[str]) -> tuple[bool, str, str]:
    if source_family not in _SOURCE_NAME_MAP:
        return False, "", "unknown_source"

    if source_family == "count_eval":
        if question_skills and all(skill == "counting" for skill in question_skills):
            return True, "count_eval_pure_counting", ""
        return False, "", "count_eval_non_pure_counting"

    if source_family == "askmodelanything":
        if skill_bucket == "general":
            return False, "", "askmodelanything_general"
        return True, "askmodelanything_targeted", ""

    if skill_bucket in _RELEVANT_SKILLS:
        return True, f"{source_family}_{skill_bucket}", ""

    return False, "", "general_skill"


def _infer_target_buckets(*, duration_bucket: str, skill_bucket: str) -> list[str]:
    buckets: list[str] = []
    if skill_bucket == "counting":
        if duration_bucket in {"short", "medium", "long"}:
            buckets.append(f"{duration_bucket}_counting_problem")
    if skill_bucket == "temporal_sequence":
        if duration_bucket in {"medium", "long"}:
            buckets.append(f"{duration_bucket}_temporal_reasoning")
    if skill_bucket == "object_reasoning":
        if duration_bucket in {"medium", "long"}:
            buckets.append(f"{duration_bucket}_object_reasoning")
    if skill_bucket == "action_reasoning":
        if duration_bucket in {"medium", "long"}:
            buckets.append(f"{duration_bucket}_action_reasoning")
    if skill_bucket == "summary":
        if duration_bucket == "long":
            buckets.append("long_information_synopsis")
    if skill_bucket == "ocr_text":
        if duration_bucket in {"medium", "long"}:
            buckets.append(f"{duration_bucket}_ocr_problems")
    if skill_bucket == "subtitle_alignment":
        if duration_bucket in {"medium", "long"}:
            buckets.append(f"{duration_bucket}_temporal_reasoning")
    return buckets


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag and filter Molmo2 SFT jsonl files for VideoMME-oriented selection.")
    parser.add_argument("--input-glob", default=_DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-every", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_filter(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
