from __future__ import annotations

import json
import math
import re
from typing import Any

from pipeline.types import ProcessedRecord


def parse_clean_mm_qa_response(
    *,
    item: dict[str, Any],
    llm_response: dict[str, Any],
    id_field: str,
    task_config: dict[str, Any] | None = None,
    secondary_llm_response: dict[str, Any] | None = None,
) -> ProcessedRecord:
    # 两次 API：第一次 necessity（图+题），第二次 relevance（图+题+答案）
    if secondary_llm_response is not None:
        verdict = _merge_two_verdicts(llm_response, secondary_llm_response)
    else:
        verdict = extract_verdict(llm_response)
    gen_ppl = _ppl_from_logprobs(llm_response)
    if gen_ppl is not None:
        verdict["gen_ppl"] = gen_ppl
    rel_score = verdict.get("relevance_score")
    nec_score = verdict.get("necessity_score")
    if rel_score is not None and nec_score is not None:
        rel_thresh = 4
        nec_thresh = 4
        if task_config:
            rel_thresh = float(task_config.get("relevance_keep_threshold") or 4)
            nec_thresh = float(task_config.get("necessity_keep_threshold") or 4)
        keep = rel_score >= rel_thresh and nec_score >= nec_thresh
    else:
        relevance = verdict.get("relevance", "unknown")
        necessity = verdict.get("necessity", "unknown")
        keep = relevance == "relevant" and necessity == "necessary"

    # Optional: filter by judgment PPL when with_ppl and ppl_keep_threshold > 0
    if keep and task_config:
        threshold = task_config.get("ppl_keep_threshold")
        if threshold is not None and float(threshold) > 0 and gen_ppl is not None:
            if gen_ppl > float(threshold):
                keep = False

    base_record = dict(item.get("raw_record") or {})
    base_record[id_field] = str(item.get(id_field))
    base_record["question"] = item.get("question")
    if item.get("answer") is not None:
        base_record["answer"] = item.get("answer")
    base_record["image_path"] = item.get("image_path")
    base_record["_clean_verdict"] = verdict
    base_record["_clean_keep"] = keep
    base_record["_source_id"] = item.get("source_id")
    base_record["_qa_index"] = item.get("qa_index")
    return base_record


def _merge_two_verdicts(necessity_response: dict[str, Any], relevance_response: dict[str, Any]) -> dict[str, Any]:
    """Merge necessity-only (first) and relevance-only (second) API responses into one verdict."""
    nec_verdict = _extract_single_score_verdict(necessity_response, "necessity_score")
    rel_verdict = _extract_single_score_verdict(relevance_response, "relevance_score")
    nec_score = nec_verdict.get("necessity_score")
    rel_score = rel_verdict.get("relevance_score")
    if nec_score is None:
        nec_score = 1.0
    if rel_score is None:
        rel_score = 1.0
    try:
        nec_score = max(1, min(5, float(nec_score)))
    except (TypeError, ValueError):
        nec_score = 1.0
    try:
        rel_score = max(1, min(5, float(rel_score)))
    except (TypeError, ValueError):
        rel_score = 1.0
    reason_parts = []
    if nec_verdict.get("reason"):
        reason_parts.append(f"necessity: {nec_verdict['reason']}")
    if rel_verdict.get("reason"):
        reason_parts.append(f"relevance: {rel_verdict['reason']}")
    return _verdict_with_derived_labels(
        {"reason": "; ".join(reason_parts) if reason_parts else ""},
        rel_score,
        nec_score,
    )


def _extract_single_score_verdict(llm_response: dict[str, Any], score_key: str) -> dict[str, Any]:
    """Parse a single-score JSON from LLM response (necessity_score or relevance_score)."""
    content = _extract_content_text(llm_response)
    if not content or "{" not in content:
        return {}
    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = trimmed.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        start = trimmed.index("{")
        end = trimmed.rindex("}") + 1
        obj = json.loads(trimmed[start:end])
    except (ValueError, json.JSONDecodeError):
        m = re.search(rf'"{re.escape(score_key)}"\s*:\s*([1-5])', trimmed)
        if m:
            return {score_key: float(m.group(1)), "reason": ""}
        return {}
    if not isinstance(obj, dict):
        return {}
    score = obj.get(score_key)
    reason = str(obj.get("reason") or "")
    try:
        s = max(1, min(5, float(score))) if score is not None else None
    except (TypeError, ValueError):
        s = None
    out = {"reason": reason}
    if s is not None:
        out[score_key] = s
    return out


def extract_verdict(llm_response: dict[str, Any]) -> dict[str, Any]:
    if "_local_verdict" in llm_response:
        verdict = llm_response["_local_verdict"]
        if isinstance(verdict, dict):
            return verdict
        return {"relevance": "unknown", "necessity": "unknown", "reason": "invalid local verdict"}

    content = _extract_content_text(llm_response)
    if not content:
        return {"relevance": "unknown", "necessity": "unknown", "reason": "empty model output"}

    trimmed = content.strip()
    if trimmed.startswith("```"):
        trimmed = trimmed.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if "{" not in trimmed:
        return {"relevance": "unknown", "necessity": "unknown", "reason": "no json object in output"}
    try:
        start = trimmed.index("{")
        end = trimmed.rindex("}") + 1
        verdict = json.loads(trimmed[start:end])
    except (ValueError, json.JSONDecodeError):
        # Fallback: extract relevance/necessity from raw text (reason 里未转义引号或截断导致 JSON 失败)
        fallback = _extract_verdict_from_raw(trimmed)
        if fallback:
            fallback["reason"] = f"json parse failed (relevance/necessity from raw): {trimmed[:120]}..."
            return fallback
        return {"relevance": "unknown", "necessity": "unknown", "reason": f"json parse failed: {trimmed[:200]}"}
    if not isinstance(verdict, dict):
        return {"relevance": "unknown", "necessity": "unknown", "reason": "parsed output is not object"}
    # Normalize to scores and derived labels (supports both score and legacy label format)
    rel_score, nec_score = _score_from_verdict(verdict)
    return _verdict_with_derived_labels(verdict, rel_score, nec_score)


def _score_from_verdict(verdict: dict[str, Any]) -> tuple[float, float]:
    """Get (relevance_score, necessity_score) in 1-5; support both score and legacy label format."""
    rel = verdict.get("relevance_score")
    nec = verdict.get("necessity_score")
    if rel is not None and nec is not None:
        try:
            r = max(1, min(5, float(rel)))
            n = max(1, min(5, float(nec)))
            return (r, n)
        except (TypeError, ValueError):
            pass
    # Legacy: relevance/necessity strings -> 5 or 1
    r = 5 if (str(verdict.get("relevance", "")).lower() == "relevant") else 1
    n = 5 if (str(verdict.get("necessity", "")).lower() == "necessary") else 1
    return (r, n)


def _verdict_with_derived_labels(verdict: dict[str, Any], rel_score: float, nec_score: float) -> dict[str, Any]:
    """Ensure verdict has relevance_score, necessity_score, and derived relevance/necessity for compat."""
    out = dict(verdict)
    out["relevance_score"] = round(rel_score, 1)
    out["necessity_score"] = round(nec_score, 1)
    out["relevance"] = "relevant" if rel_score >= 4 else "irrelevant"
    out["necessity"] = "necessary" if nec_score >= 4 else "unnecessary"
    return out


def _extract_verdict_from_raw(raw: str) -> dict[str, Any] | None:
    """当 JSON 解析失败时，从原始文本用正则抽出 relevance_score/necessity_score 或 relevance/necessity。"""
    rel_m = re.search(r'"relevance_score"\s*:\s*([1-5])', raw)
    nec_m = re.search(r'"necessity_score"\s*:\s*([1-5])', raw)
    if rel_m and nec_m:
        return _verdict_with_derived_labels(
            {"reason": ""},
            float(rel_m.group(1)),
            float(nec_m.group(1)),
        )
    rel_match = re.search(r'"relevance"\s*:\s*"(relevant|irrelevant)"', raw, re.I)
    nec_match = re.search(r'"necessity"\s*:\s*"(necessary|unnecessary)"', raw, re.I)
    if rel_match and nec_match:
        r = 5 if rel_match.group(1).lower() == "relevant" else 1
        n = 5 if nec_match.group(1).lower() == "necessary" else 1
        return _verdict_with_derived_labels({"reason": ""}, r, n)
    return None


def _extract_content_text(llm_response: dict[str, Any]) -> str:
    choices = llm_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    return content.strip()


def _ppl_from_logprobs(llm_response: dict[str, Any]) -> float | None:
    """Compute perplexity from chat completion choice logprobs. Returns None if unavailable."""
    choices = llm_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    logprobs = first.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    logps = []
    for tok in content:
        if isinstance(tok, dict):
            lp = tok.get("logprob")
            if lp is not None:
                logps.append(float(lp))
    if not logps:
        return None
    mean_logp = sum(logps) / len(logps)
    return round(math.exp(-mean_logp), 4)
