#!/usr/bin/env python3
"""
对已 judged 的 jsonl 仅重判「必要性」(necessity)，图+题不看答案，然后合并回原 verdict 并重新划分 clean/dirty。
用法（在项目根目录）:
  python scripts/tasks/rejudge_necessity_only.py \\
    --judged-dir output/openbee_judged_v2 \\
    --output-clean-dir /ov2/dataset_jsonl/openbee_clean_v2 \\
    --output-dirty-dir /ov2/dataset_jsonl/openbee_dirty_v2 \\
    --endpoint-registry runtime/endpoints.local.json \\
    --suffix _v2
可选: --max-records 100 试跑, --necessity-threshold 4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import re
import sys
from pathlib import Path

# 项目根
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
from pipeline.tasks.clean_mm_qa.prompts import CLEAN_JUDGE_NECESSITY_ONLY_SYSTEM
from pipeline.tasks.clean_mm_qa.splitter import split_clean_dirty


def _encode_image(path: str, max_longer_edge: int = 1536) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    if max_longer_edge > 0:
        try:
            from PIL import Image
            with open(path, "rb") as handle:
                img = Image.open(handle).convert("RGB")
            w, h = img.size
            if max(w, h) > max_longer_edge:
                ratio = max_longer_edge / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return encoded, "image/jpeg"
        except Exception:
            pass
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return encoded, mime


def _parse_necessity_from_response(content: str) -> tuple[float, str]:
    content = (content or "").strip()
    if not content or "{" not in content:
        return 1.0, "empty or no json"
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        obj = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        m = re.search(r'"necessity_score"\s*:\s*([1-5])', content)
        if m:
            return float(m.group(1)), ""
        return 1.0, "parse failed"
    if not isinstance(obj, dict):
        return 1.0, "not object"
    score = obj.get("necessity_score")
    try:
        s = max(1, min(5, float(score))) if score is not None else 1.0
    except (TypeError, ValueError):
        s = 1.0
    reason = str(obj.get("reason") or "")
    return s, reason


def _request_necessity_only(
    endpoint_url: str,
    model: str,
    question: str,
    image_b64: str,
    mime: str,
    timeout: int = 120,
) -> tuple[float, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CLEAN_JUDGE_NECESSITY_ONLY_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question:\n{question}"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}", "detail": "high"}},
                ],
            },
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    try:
        r = requests.post(endpoint_url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return 1.0, "no choices"
        content = (choices[0].get("message") or {}).get("content") or ""
        return _parse_necessity_from_response(content)
    except Exception as e:
        return 1.0, str(e)[:200]


def _relevance_ok(verdict: dict) -> bool:
    rel_score = verdict.get("relevance_score")
    if rel_score is not None:
        try:
            return float(rel_score) >= 4
        except (TypeError, ValueError):
            pass
    return str(verdict.get("relevance", "")).lower() == "relevant"


def process_file(
    judged_path: Path,
    revised_path: Path,
    endpoints: list[dict],
    necessity_threshold: float,
    max_records: int,
    max_image_longer_edge: int,
) -> tuple[int, int]:
    endpoints = [e for e in endpoints if e.get("url")]
    if not endpoints:
        raise ValueError("No endpoints with url")
    model = endpoints[0].get("model") or "Qwen/Qwen3-VL-32B-Instruct"
    records = []
    with open(judged_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_records and len(records) >= max_records:
                break

    n = 0
    for i, rec in enumerate(records):
        image_path = (rec.get("image_path") or "").strip()
        question = (str(rec.get("question") or "")).strip()
        verdict = rec.get("_clean_verdict") or {}
        if not question:
            rec["_clean_verdict"] = {**verdict, "necessity": "unknown", "necessity_score": None}
            rec["_clean_keep"] = False
            n += 1
            continue
        if not image_path or not Path(image_path).is_file():
            rec["_clean_verdict"] = {**verdict, "necessity": "unknown", "necessity_score": None}
            rec["_clean_keep"] = False
            n += 1
            continue
        try:
            encoded, mime = _encode_image(image_path, max_longer_edge)
        except Exception:
            rec["_clean_verdict"] = {**verdict, "necessity": "unknown", "necessity_score": None}
            rec["_clean_keep"] = False
            n += 1
            continue
        url = endpoints[n % len(endpoints)]["url"]
        nec_score, reason = _request_necessity_only(url, model, question, encoded, mime)
        verdict = dict(verdict)
        verdict["necessity_score"] = round(nec_score, 1)
        verdict["necessity"] = "necessary" if nec_score >= necessity_threshold else "unnecessary"
        if reason and "reason_necessity" not in verdict:
            verdict["reason_necessity"] = reason
        rec["_clean_verdict"] = verdict
        rec["_clean_keep"] = _relevance_ok(verdict) and nec_score >= necessity_threshold
        n += 1
        if n % 100 == 0:
            print(f"  [{n}/{len(records)}]", flush=True)

    revised_path.parent.mkdir(parents=True, exist_ok=True)
    with open(revised_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records), n


def main():
    ap = argparse.ArgumentParser(description="Rejudge necessity only (image+question, no answer) and re-split clean/dirty")
    ap.add_argument("--judged-dir", type=str, default="output/openbee_judged_v2", help="Judged jsonl directory")
    ap.add_argument("--output-clean-dir", type=str, default="/ov2/dataset_jsonl/openbee_clean_v2")
    ap.add_argument("--output-dirty-dir", type=str, default="/ov2/dataset_jsonl/openbee_dirty_v2")
    ap.add_argument("--endpoint-registry", type=str, default="runtime/endpoints.local.json")
    ap.add_argument("--endpoint-group", type=str, default="local_multi")
    ap.add_argument("--suffix", type=str, default="_v2", help="e.g. _v2 => stem_clean_v2.jsonl")
    ap.add_argument("--necessity-threshold", type=float, default=4.0)
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument("--max-image-longer-edge", type=int, default=1536)
    ap.add_argument("--dry-run", action="store_true", help="Only list files, do not call API")
    args = ap.parse_args()

    judged_dir = Path(args.judged_dir)
    if not judged_dir.is_dir():
        print(f"Judged dir not found: {judged_dir}", file=sys.stderr)
        sys.exit(1)

    with open(ROOT / args.endpoint_registry, "r", encoding="utf-8") as f:
        reg = json.load(f)
    groups = reg.get("groups") or {}
    endpoints = (groups.get(args.endpoint_group) or {}).get("endpoints") or []
    if not endpoints:
        print(f"No endpoints for group {args.endpoint_group}", file=sys.stderr)
        sys.exit(1)

    pattern = re.compile(r"^(.+)_judged(_v2)?\.jsonl$", re.I)
    files = sorted(judged_dir.glob("*_judged*.jsonl"))
    if not files:
        print(f"No *_judged*.jsonl in {judged_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for f in files:
            stem = f.stem
            for suf in ("_judged_v2", "_judged"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            print(f"  {f.name} -> stem={stem}")
        return

    clean_dir = Path(args.output_clean_dir)
    dirty_dir = Path(args.output_dirty_dir)
    for judged_path in files:
        # CMM-Math_judged_v2.jsonl -> stem = CMM-Math
        stem = judged_path.stem
        for suf in ("_judged_v2", "_judged"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        revised_path = judged_dir / f"{stem}_judged{args.suffix}_necessity_revised.jsonl"
        clean_path = clean_dir / f"{stem}_clean{args.suffix}.jsonl"
        dirty_path = dirty_dir / f"{stem}_dirty{args.suffix}.jsonl"
        print(f"Processing {judged_path.name} -> {revised_path.name}")
        total, done = process_file(
            judged_path,
            revised_path,
            endpoints,
            args.necessity_threshold,
            args.max_records,
            args.max_image_longer_edge,
        )
        print(f"  Written {total} records ({done} necessity re-judged) to {revised_path}")
        stats = split_clean_dirty(
            judged_output_jsonl=str(revised_path),
            clean_output_jsonl=str(clean_path),
            dirty_output_jsonl=str(dirty_path),
        )
        print(f"  Split: clean={stats['clean_count']}, dirty={stats['dirty_count']}")
    print("Done.")


if __name__ == "__main__":
    main()
