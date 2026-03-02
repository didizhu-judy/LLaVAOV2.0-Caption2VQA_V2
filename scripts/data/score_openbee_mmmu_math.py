#!/usr/bin/env python3
"""
Score OpenBee jsonl files for MMMU + Math training value.

This script implements a weakly-supervised scoring pipeline:
- Build a candidate file set (default: union of math/mmmu flags from manifest CSV)
- Sample up to N user questions from each jsonl file
- Compute signal ratios (mcq/math/science/engineering/humanities/low_caption)
- Compute score_math / score_mmmu / combined_score
- Produce ranked outputs and a P0/P1 selection list

Usage:
  python scripts/data/score_openbee_mmmu_math.py
  python scripts/data/score_openbee_mmmu_math.py \
      --manifest-csv /tmp/openbee_rules_cmp/openbee_domain_manifest.csv \
      --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("/ov2/dataset_jsonl/openbee")
DEFAULT_CLEAN_DIR = Path("/ov2/dataset_jsonl/openbee_clean")
DEFAULT_MANIFEST_CSV = Path("output/openbee_domain_manifest.csv")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_MMMU_KEYWORD_BANK = Path("scripts/data/mmmu_keyword_bank.json")


MCQ_PATTERNS = [
    re.compile(r"\bchoices?\b", re.IGNORECASE),
    re.compile(r"\boptions?\b", re.IGNORECASE),
    re.compile(r"\(A\)|\(B\)|\(C\)|\(D\)", re.IGNORECASE),
    re.compile(r"\bA\s*[\.\)]\s*", re.IGNORECASE),
    re.compile(r"\bB\s*[\.\)]\s*", re.IGNORECASE),
    re.compile(r"请选择", re.IGNORECASE),
    re.compile(r"选项", re.IGNORECASE),
]

LOW_CAPTION_PATTERNS = [
    re.compile(r"what'?s the main subject", re.IGNORECASE),
    re.compile(r"describe (the|this) image", re.IGNORECASE),
    re.compile(r"\bcaption\b", re.IGNORECASE),
    re.compile(r"what do you see", re.IGNORECASE),
    re.compile(r"summarize the chart", re.IGNORECASE),
    re.compile(r"compare the quality", re.IGNORECASE),
    re.compile(r"please extract all text", re.IGNORECASE),
]

MATH_KEYWORDS = [
    "math",
    "algebra",
    "geometry",
    "geometric",
    "triangle",
    "angle",
    "equation",
    "solve",
    "calculate",
    "function",
    "integral",
    "derivative",
    "matrix",
    "probability",
    "ratio",
    "percentage",
    "area",
    "perimeter",
    "volume",
    "coordinate",
    "graph",
    "statistics",
    "方程",
    "几何",
    "函数",
    "求",
    "面积",
    "角",
    "概率",
    "代数",
    "数学",
]

SCIENCE_KEYWORDS = [
    "physics",
    "chemistry",
    "biology",
    "medicine",
    "medical",
    "health",
    "clinical",
    "lab",
    "experiment",
    "pharmacy",
    "anatomy",
]

ENGINEERING_KEYWORDS = [
    "engineering",
    "circuit",
    "electrical",
    "mechanical",
    "diagram",
    "schematic",
]

HUMANITIES_KEYWORDS = [
    "history",
    "literature",
    "music",
    "art",
    "philosophy",
    "economics",
    "business",
    "finance",
    "law",
    "sociology",
    "psychology",
    "geography",
]

REASONING_STYLE_KEYWORDS = [
    "which of the following",
    "according to the figure",
    "according to the table",
    "based on the graph",
    "determine",
    "calculate",
    "compute",
    "infer",
    "derive",
    "find the",
    "solve",
    "what is the value",
    "select the correct option",
    "answer with option letter",
    "根据图中信息",
    "请计算",
    "求",
]


@dataclass
class FileScore:
    file: str
    total_lines: int
    sample_n: int
    mcq_ratio: float
    math_ratio: float
    sci_ratio: float
    eng_ratio: float
    hum_ratio: float
    mmmu_topic_ratio: float
    low_caption_ratio: float
    avg_q_len: float
    score_math: float
    score_mmmu: float
    combined_score: float
    cleaned: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score OpenBee files for MMMU+Math value")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory with *.jsonl")
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR, help="Directory with *_clean.jsonl")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_MANIFEST_CSV,
        help="Manifest CSV (from classify_openbee_domains.py). If present, uses math/mmmu union as candidates.",
    )
    parser.add_argument("--use-all-files", action="store_true", help="Ignore manifest and score all *.jsonl files")
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=["other_STEM"],
        help="Exclude files whose stem starts with this prefix. Can be repeated.",
    )
    parser.add_argument(
        "--keep-clean-dirty-variants",
        action="store_true",
        help="By default files ending with _clean/_dirty are ignored; set this to keep them.",
    )
    parser.add_argument(
        "--always-include",
        action="append",
        default=["HME.jsonl", "MMTab.jsonl"],
        help="Always include these files in candidates. Can be repeated.",
    )
    parser.add_argument("--max-samples-per-file", type=int, default=400, help="Max sampled questions per file")
    parser.add_argument("--p0-size", type=int, default=20, help="Top-N files as P0")
    parser.add_argument("--p1-size", type=int, default=8, help="Next-N files as P1")
    parser.add_argument(
        "--math-weight",
        type=float,
        default=0.55,
        help="Weight of score_math in combined score. score_mmmu weight = 1 - math_weight",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--mmmu-keyword-bank",
        type=Path,
        default=DEFAULT_MMMU_KEYWORD_BANK,
        help="Optional JSON keyword bank for MMMU topic keywords",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="openbee_mmmu_math",
        help="Prefix for output files",
    )
    return parser.parse_args()


def as_abs(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path)


def load_mmmu_keyword_bank(keyword_bank_path: Path) -> dict[str, list[str]]:
    empty = {"subjects": [], "subfields": [], "question_cues": [], "visual_cues": []}
    if not keyword_bank_path.exists():
        return empty
    try:
        payload = json.loads(keyword_bank_path.read_text(encoding="utf-8"))
    except Exception:
        return empty

    out = dict(empty)
    for key in out.keys():
        values = payload.get(key, [])
        if not isinstance(values, list):
            continue
        normalized: list[str] = []
        for item in values:
            if not isinstance(item, str):
                continue
            text = item.strip().lower()
            if not text:
                continue
            normalized.append(text.replace("_", " "))
        out[key] = sorted(set(normalized))
    return out


def read_manifest_candidates(manifest_csv: Path) -> list[str]:
    candidates: list[str] = []
    with manifest_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_math = str(row.get("math_related", "")).strip().lower() == "true"
            is_mmmu = str(row.get("mmmu_related", "")).strip().lower() == "true"
            if is_math or is_mmmu:
                file_name = row.get("file", "").strip()
                if file_name:
                    candidates.append(file_name)
    return sorted(set(candidates))


def normalize_file_set(
    file_names: Iterable[str],
    input_dir: Path,
    exclude_prefixes: list[str],
    keep_clean_dirty_variants: bool,
) -> list[str]:
    out: list[str] = []
    for name in sorted(set(file_names)):
        fp = input_dir / name
        if not fp.exists():
            continue
        stem = fp.stem
        if any(stem.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if not keep_clean_dirty_variants and (stem.endswith("_clean") or stem.endswith("_dirty")):
            continue
        out.append(name)
    return out


def first_user_content(obj: dict) -> str:
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return " ".join(content.split())
        return ""
    return ""


def count_lines(path: Path) -> int:
    line_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            line_count += 1
    return line_count


def sample_user_questions(path: Path, max_samples: int) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(questions) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            question = first_user_content(obj)
            if question:
                questions.append(question)
    return questions


def hit_any_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def hit_any_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def score_file(
    file_name: str,
    input_dir: Path,
    clean_set: set[str],
    max_samples: int,
    math_weight: float,
    mmmu_keyword_bank: dict[str, list[str]],
) -> FileScore | None:
    path = input_dir / file_name
    total_lines = count_lines(path)
    questions = sample_user_questions(path, max_samples=max_samples)
    sample_n = len(questions)
    if sample_n == 0:
        return None

    mcq_hits = 0
    math_hits = 0
    sci_hits = 0
    eng_hits = 0
    hum_hits = 0
    mmmu_topic_hits = 0
    low_caption_hits = 0
    total_question_len = 0

    for question in questions:
        q_original = question
        q_lower = question.lower()
        total_question_len += len(q_original)
        is_mcq = hit_any_pattern(q_original, MCQ_PATTERNS)
        if is_mcq:
            mcq_hits += 1
        style_gate = is_mcq or hit_any_keyword(q_lower, REASONING_STYLE_KEYWORDS)
        if hit_any_keyword(q_lower, MATH_KEYWORDS):
            math_hits += 1
        if style_gate and hit_any_keyword(q_lower, SCIENCE_KEYWORDS):
            sci_hits += 1
        if style_gate and hit_any_keyword(q_lower, ENGINEERING_KEYWORDS):
            eng_hits += 1
        if style_gate and hit_any_keyword(q_lower, HUMANITIES_KEYWORDS):
            hum_hits += 1
        mmmu_subject_visual_keywords = (
            mmmu_keyword_bank.get("subjects", [])
            + mmmu_keyword_bank.get("subfields", [])
            + mmmu_keyword_bank.get("visual_cues", [])
        )
        mmmu_question_cues = mmmu_keyword_bank.get("question_cues", [])
        subject_or_visual_hit = bool(
            mmmu_subject_visual_keywords and hit_any_keyword(q_lower, mmmu_subject_visual_keywords)
        )
        question_style_hit = bool(mmmu_question_cues and hit_any_keyword(q_lower, mmmu_question_cues))
        if subject_or_visual_hit and (question_style_hit or is_mcq or style_gate):
            mmmu_topic_hits += 1
        if hit_any_pattern(q_original, LOW_CAPTION_PATTERNS):
            low_caption_hits += 1

    mcq_ratio = safe_ratio(mcq_hits, sample_n)
    math_ratio = safe_ratio(math_hits, sample_n)
    sci_ratio = safe_ratio(sci_hits, sample_n)
    eng_ratio = safe_ratio(eng_hits, sample_n)
    hum_ratio = safe_ratio(hum_hits, sample_n)
    mmmu_topic_ratio = safe_ratio(mmmu_topic_hits, sample_n)
    low_caption_ratio = safe_ratio(low_caption_hits, sample_n)
    avg_q_len = safe_ratio(total_question_len, sample_n)

    noise_penalty_term = 1.0 - min(low_caption_ratio, 0.5)
    score_math = 0.55 * math_ratio + 0.30 * mcq_ratio + 0.15 * noise_penalty_term
    # MMMU score combines subject/discipline cues + benchmark-style MCQ structure + noise penalty.
    score_mmmu = (
        0.30 * (sci_ratio + eng_ratio + hum_ratio)
        + 0.25 * mcq_ratio
        + 0.30 * mmmu_topic_ratio
        + 0.15 * noise_penalty_term
    )
    score_mmmu_weight = 1.0 - math_weight
    combined_score = math_weight * score_math + score_mmmu_weight * score_mmmu
    cleaned = file_name.replace(".jsonl", "_clean.jsonl") in clean_set

    return FileScore(
        file=file_name,
        total_lines=total_lines,
        sample_n=sample_n,
        mcq_ratio=mcq_ratio,
        math_ratio=math_ratio,
        sci_ratio=sci_ratio,
        eng_ratio=eng_ratio,
        hum_ratio=hum_ratio,
        mmmu_topic_ratio=mmmu_topic_ratio,
        low_caption_ratio=low_caption_ratio,
        avg_q_len=avg_q_len,
        score_math=score_math,
        score_mmmu=score_mmmu,
        combined_score=combined_score,
        cleaned=cleaned,
    )


def write_outputs(
    scores: list[FileScore],
    output_dir: Path,
    prefix: str,
    p0_size: int,
    p1_size: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_signal.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "file",
                "total_lines",
                "sample_n",
                "mcq_ratio",
                "math_ratio",
                "sci_ratio",
                "eng_ratio",
                "hum_ratio",
                "mmmu_topic_ratio",
                "low_caption_ratio",
                "avg_q_len",
                "score_math",
                "score_mmmu",
                "combined_score",
                "cleaned",
            ]
        )
        for s in scores:
            writer.writerow(
                [
                    s.file,
                    s.total_lines,
                    s.sample_n,
                    f"{s.mcq_ratio:.6f}",
                    f"{s.math_ratio:.6f}",
                    f"{s.sci_ratio:.6f}",
                    f"{s.eng_ratio:.6f}",
                    f"{s.hum_ratio:.6f}",
                    f"{s.mmmu_topic_ratio:.6f}",
                    f"{s.low_caption_ratio:.6f}",
                    f"{s.avg_q_len:.6f}",
                    f"{s.score_math:.6f}",
                    f"{s.score_mmmu:.6f}",
                    f"{s.combined_score:.6f}",
                    str(s.cleaned),
                ]
            )

    p0_files = [s.file for s in scores[:p0_size]]
    p1_files = [s.file for s in scores[p0_size : p0_size + p1_size]]
    selected_files = p0_files + p1_files
    excluded_files = [s.file for s in scores[p0_size + p1_size :]]
    not_cleaned_selected = [s.file for s in scores[: p0_size + p1_size] if not s.cleaned]

    selected_txt = output_dir / f"{prefix}_selected_files.txt"
    with selected_txt.open("w", encoding="utf-8") as f:
        for file_name in selected_files:
            f.write(file_name + "\n")

    summary_json = output_dir / f"{prefix}_selection.json"
    payload = {
        "config": {
            "input_dir": str(args.input_dir),
            "clean_dir": str(args.clean_dir),
            "manifest_csv": str(args.manifest_csv),
            "mmmu_keyword_bank": str(args.mmmu_keyword_bank),
            "use_all_files": bool(args.use_all_files),
            "exclude_prefix": args.exclude_prefix,
            "keep_clean_dirty_variants": bool(args.keep_clean_dirty_variants),
            "always_include": args.always_include,
            "max_samples_per_file": int(args.max_samples_per_file),
            "math_weight": float(args.math_weight),
            "p0_size": int(args.p0_size),
            "p1_size": int(args.p1_size),
        },
        "formula": {
            "score_math": "0.55*math_ratio + 0.30*mcq_ratio + 0.15*(1 - min(low_caption_ratio, 0.5))",
            "score_mmmu": "0.30*(sci_ratio+eng_ratio+hum_ratio) + 0.25*mcq_ratio + 0.30*mmmu_topic_ratio + 0.15*(1-min(low_caption_ratio,0.5))",
            "combined_score": "math_weight*score_math + (1-math_weight)*score_mmmu",
        },
        "counts": {
            "scored_files": len(scores),
            "selected_files": len(selected_files),
            "selected_not_cleaned": len(not_cleaned_selected),
        },
        "P0_core_must_use": p0_files,
        "P1_high_value_optional": p1_files,
        "selected_not_cleaned": not_cleaned_selected,
        "excluded_for_now": excluded_files,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {csv_path} ({len(scores)} rows)")
    print(f"Wrote {selected_txt} ({len(selected_files)} files)")
    print(f"Wrote {summary_json}")
    if not_cleaned_selected:
        print(f"Selected but not cleaned ({len(not_cleaned_selected)}):")
        for file_name in not_cleaned_selected:
            print(f"  - {file_name}")


def main() -> int:
    args = parse_args()

    input_dir = as_abs(args.input_dir)
    clean_dir = as_abs(args.clean_dir)
    manifest_csv = as_abs(args.manifest_csv)
    output_dir = as_abs(args.output_dir)
    mmmu_keyword_bank = as_abs(args.mmmu_keyword_bank)

    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 1
    if not clean_dir.exists():
        print(f"Clean dir not found: {clean_dir}", file=sys.stderr)
        return 1
    if not (0.0 <= args.math_weight <= 1.0):
        print("--math-weight must be between 0 and 1", file=sys.stderr)
        return 1

    if args.use_all_files:
        candidate_files = [p.name for p in sorted(input_dir.glob("*.jsonl"))]
    else:
        if not manifest_csv.exists():
            print(
                f"Manifest CSV not found: {manifest_csv}\n"
                "Run classify_openbee_domains.py first or pass --use-all-files.",
                file=sys.stderr,
            )
            return 1
        candidate_files = read_manifest_candidates(manifest_csv)

    candidate_files.extend(args.always_include or [])
    candidate_files = normalize_file_set(
        candidate_files,
        input_dir=input_dir,
        exclude_prefixes=args.exclude_prefix or [],
        keep_clean_dirty_variants=bool(args.keep_clean_dirty_variants),
    )

    clean_set = {p.name for p in clean_dir.glob("*.jsonl")}
    mmmu_keyword_bank_payload = load_mmmu_keyword_bank(mmmu_keyword_bank)
    scored: list[FileScore] = []
    for file_name in candidate_files:
        result = score_file(
            file_name=file_name,
            input_dir=input_dir,
            clean_set=clean_set,
            max_samples=args.max_samples_per_file,
            math_weight=args.math_weight,
            mmmu_keyword_bank=mmmu_keyword_bank_payload,
        )
        if result is not None:
            scored.append(result)

    if not scored:
        print("No files were scored (empty candidate set or unreadable data).", file=sys.stderr)
        return 1

    scored.sort(key=lambda item: (item.combined_score, item.total_lines), reverse=True)
    write_outputs(
        scores=scored,
        output_dir=output_dir,
        prefix=args.output_prefix,
        p0_size=args.p0_size,
        p1_size=args.p1_size,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
