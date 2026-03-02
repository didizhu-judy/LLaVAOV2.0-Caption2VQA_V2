#!/usr/bin/env python3
"""
Classify OpenBee jsonl files by domain and produce math-related / MMMU-related lists.

Reads: openbee_domain_rules.yaml, and optionally samples first N lines of generic-named
files for content-based domain hints. Writes: domain_manifest.csv, domain_to_files.json,
math_related_files.txt, mmmu_related_files.txt.

Usage:
  python scripts/data/classify_openbee_domains.py
  python scripts/data/classify_openbee_domains.py --input-dir /ov2/dataset_jsonl/openbee --output-dir scripts/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rules(rules_path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML required: pip install pyyaml")
    with open(rules_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_first_user_contents(jsonl_path: Path, max_lines: int) -> list[str]:
    contents = []
    try:
        with open(jsonl_path, "rb") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = obj.get("messages") or []
                for m in messages:
                    if m.get("role") == "user":
                        content = m.get("content")
                        if isinstance(content, str) and content.strip():
                            contents.append(content)
                        break
    except OSError:
        pass
    return contents


def infer_domain_from_content(contents: list[str], keyword_hints: dict[str, list]) -> list[str]:
    added = set()
    text = " ".join(contents).lower()
    for keyword, domains in (keyword_hints or {}).items():
        if keyword.lower() in text:
            for d in domains:
                added.add(d)
    return sorted(added)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify OpenBee jsonl by domain, output math/MMMU lists")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/ov2/dataset_jsonl/openbee"),
        help="Directory containing *.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/data"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("scripts/data/openbee_domain_rules.yaml"),
        help="Path to domain rules YAML",
    )
    parser.add_argument("--no-content-sample", action="store_true", help="Skip content sampling for generic files")
    args = parser.parse_args()

    rules_path = args.rules
    if not rules_path.is_absolute():
        rules_path = Path.cwd() / rules_path
    if not rules_path.exists():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        return 1

    rules = load_rules(rules_path)
    exact = rules.get("exact_file_domains") or {}
    mmmu_domains = set(rules.get("mmmu_domains") or [])
    math_related_basenames = set(rules.get("math_related_files") or [])
    content_sample_lines = int(rules.get("content_sample_lines") or 5)
    content_keyword_hints = rules.get("content_keyword_hints") or {}
    content_sample_prefixes = rules.get("content_sample_prefixes") or []

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 1

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No *.jsonl in {input_dir}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    domain_to_files: dict[str, list[str]] = {}

    for path in jsonl_files:
        basename = path.name
        base_stem = path.stem

        domains = exact.get(basename)
        if domains is None:
            domains = ["general"]

        if not args.no_content_sample and any(base_stem.startswith(p) for p in content_sample_prefixes):
            if domains == ["general"] or "general" in domains:
                contents = collect_first_user_contents(path, content_sample_lines)
                extra = infer_domain_from_content(contents, content_keyword_hints)
                if extra:
                    domains = list(set(domains) | set(extra))

        if isinstance(domains, str):
            domains = [domains]
        domains = list(domains)

        manifest.append({"file": basename, "domains": domains})

        for d in domains:
            domain_to_files.setdefault(d, []).append(basename)

        is_math = base_stem in math_related_basenames or "math" in [x.lower() for x in domains]
        is_mmmu = bool(mmmu_domains & set(d.lower() for d in domains))
        manifest[-1]["math_related"] = is_math
        manifest[-1]["mmmu_related"] = is_mmmu

    math_files = [m["file"] for m in manifest if m["math_related"]]
    mmmu_files = [m["file"] for m in manifest if m["mmmu_related"]]

    # domain_manifest.csv
    csv_path = output_dir / "openbee_domain_manifest.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file,domains,math_related,mmmu_related\n")
        for m in manifest:
            domains_str = ";".join(m["domains"])
            f.write(f"{m['file']},{domains_str},{m['math_related']},{m['mmmu_related']}\n")
    print(f"Wrote {csv_path} ({len(manifest)} rows)")

    # domain_manifest.json (full)
    json_manifest_path = output_dir / "openbee_domain_manifest.json"
    with open(json_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {json_manifest_path}")

    # domain_to_files.json
    domain_to_files_path = output_dir / "openbee_domain_to_files.json"
    with open(domain_to_files_path, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in sorted(domain_to_files.items())}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {domain_to_files_path}")

    # math_related_files.txt
    math_path = output_dir / "math_related_files.txt"
    with open(math_path, "w", encoding="utf-8") as f:
        for b in sorted(math_files):
            f.write(b + "\n")
    print(f"Wrote {math_path} ({len(math_files)} files)")

    # mmmu_related_files.txt
    mmmu_path = output_dir / "mmmu_related_files.txt"
    with open(mmmu_path, "w", encoding="utf-8") as f:
        for b in sorted(mmmu_files):
            f.write(b + "\n")
    print(f"Wrote {mmmu_path} ({len(mmmu_files)} files)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
