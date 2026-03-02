from __future__ import annotations

import json
import math
from typing import Any

from pipeline.types import ProcessedRecord


def parse_clean_mm_qa_response(
    *,
    item: dict[str, Any],
    llm_response: dict[str, Any],
    id_field: str,
) -> ProcessedRecord:
    verdict = extract_verdict(llm_response)
    gen_ppl = _ppl_from_logprobs(llm_response)
    if gen_ppl is not None:
        verdict["gen_ppl"] = gen_ppl
    relevance = verdict.get("relevance", "unknown")
    necessity = verdict.get("necessity", "unknown")
    keep = relevance == "relevant" and necessity == "necessary"

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
        return {"relevance": "unknown", "necessity": "unknown", "reason": f"json parse failed: {trimmed[:200]}"}
    if not isinstance(verdict, dict):
        return {"relevance": "unknown", "necessity": "unknown", "reason": "parsed output is not object"}
    return verdict


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
