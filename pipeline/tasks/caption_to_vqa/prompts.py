from __future__ import annotations

BATCH_ALL_QA_SYSTEM = """You are a video QA data generator.
Given segments from one video, generate a JSON object with:
- temporal_grounding
- segment_qa
- understanding_qa

Rules:
1) One temporal_grounding and one segment_qa per segment.
2) understanding_qa should cover at least 3 categories:
   Counting Problem, Information Synopsis, Object Recognition, Action Reasoning, Object Reasoning.
3) Output only valid JSON object, no markdown fences."""


def build_caption_to_vqa_user_prompt(
    *,
    video_id: str,
    duration_sec: float,
    segments_text: str,
    structured_caption: str,
    max_understanding: int,
) -> str:
    return f"""Generate video QA for the following video.

Video ID: {video_id}
Duration: {duration_sec} seconds

Segments:
{segments_text}

Caption:
{structured_caption[:3000]}

Output JSON schema:
{{
  "temporal_grounding": [{{"query":"...","answer":"...","start_sec":0,"end_sec":10,"answer_type":"temporal_span"}}],
  "segment_qa": [{{"query":"...","answer":"...","start_sec":0,"end_sec":10,"answer_type":"caption"}}],
  "understanding_qa": [{{"query":"...","answer":"...","category":"Information Synopsis","answer_type":"open_ended"}}]
}}

Return exactly one JSON object.
Generate up to {max_understanding} items for understanding_qa."""
