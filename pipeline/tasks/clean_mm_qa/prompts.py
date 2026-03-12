from __future__ import annotations

# Aligned with original run_clean_stem_data.py JUDGE_SYSTEM (image + question only)
# Necessity must be based on whether the given question TEXT alone states the problem; avoid judging "unnecessary" by comparing to the problem read from the image.
# Output SCORES (1-5) instead of binary labels; downstream uses thresholds to decide clean/dirty.
CLEAN_JUDGE_SYSTEM = """You are a strict data quality judge for a multimodal STEM QA dataset.
Each sample has an image and a **question TEXT** (the exact string we give you below; it may include instructions and/or answer choices).

Evaluate TWO aspects and give each a **score from 1 to 5** (integer only):

1. **Relevance score (1-5)** – Does the image content match what the question is asking about? Judge only image-question alignment.
   • 5: Strongly aligned; the image clearly matches the question target.
   • 4: Mostly aligned; minor ambiguity.
   • 3: Partially aligned; weak or incomplete visual match.
   • 2: Largely mismatched; only small overlap.
   • 1: Clearly unrelated.

2. **Necessity score (1-5)** – How necessary is the image to understand or answer the question?
   **Decision rule (follow strictly):** Consider only the **question TEXT** (the exact string below). Ask: "If someone saw ONLY this text and had NO image, could they understand the full problem and answer it?"
   **Note:** Options (e.g. "A. ... B. ... C. ... D. ...") are only answer choices—they do NOT state the problem. The problem is the stem. If the text has only options and no stem, the image is necessary. Generic instructions like "solve this problem" do NOT state the problem.
   • 5: Image is clearly necessary (e.g. text empty, or only options, or refers to "the figure").
   • 4: Largely necessary.
   • 3: Partially necessary.
   • 2: Largely unnecessary.
   • 1: Question TEXT fully states the problem and does not refer to any figure; image unnecessary.

IMPORTANT: Your reply must be exactly one JSON object and nothing else. Do NOT output "Let's think step by step", any reasoning, or any text before or after the JSON. Start your response with { and end with }.
Format: {"relevance_score": <1-5>, "necessity_score": <1-5>, "reason": "brief one-sentence explanation"}"""

# Judge with image + question + answer. Relevance is image-question alignment; answer is auxiliary only.
CLEAN_JUDGE_WITH_ANSWER_SYSTEM = """You are a strict data quality judge for a multimodal QA dataset.
Each sample has an image, a **question TEXT** (the exact string we give you below), and a reference answer.

Evaluate TWO aspects and give each a **score from 1 to 5** (integer only). Note: **Relevance** is about image-question matching, and answer is only auxiliary; **Necessity** must NOT use the answer.

1. **Relevance score (1-5)** – Judge whether image and question match. You may read the answer only to detect topic mismatch, but DO NOT penalize relevance for answer reasoning/calculation mistakes if image-question are aligned.
   • 5: Image and question are strongly aligned.
   • 4: Mostly aligned; minor ambiguity.
   • 3: Partially aligned.
   • 2: Largely mismatched.
   • 1: Clearly unrelated.

2. **Necessity score (1-5)** – Consider **ONLY the question TEXT and the image**. Do NOT use the answer. Ask: "If someone saw ONLY the question text and had NO image, could they understand the full problem and answer it?"
   • 5: Image clearly necessary (e.g. question text empty, or only options, or refers to "the figure"/"图中").
   • 4: Largely necessary.
   • 3: Partially necessary.
   • 2: Largely unnecessary.
   • 1: Question TEXT by itself fully states the problem and does not refer to any figure; image unnecessary.

IMPORTANT: Your reply must be exactly one JSON object and nothing else. Do NOT output "Let's think step by step", any reasoning, or any text before or after the JSON. Start your response with { and end with }.
Format: {"relevance_score": <1-5>, "necessity_score": <1-5>, "reason": "brief one-sentence explanation"}"""

# 仅判必要性（图+题，不看答案）。用于对已 judged 数据重判 necessity 后合并，不重跑全量。
CLEAN_JUDGE_NECESSITY_ONLY_SYSTEM = """You are a data quality judge. You see an image and a **question TEXT** only (no answer).

Output **only the necessity score (1-5)** and a brief reason. Consider ONLY the question text and the image. Ask: "If someone saw ONLY the question text and had NO image, could they understand the full problem and answer it?"
• 5: Image clearly necessary (e.g. text empty, only options, or refers to "the figure").
• 4: Largely necessary.
• 3: Partially necessary.
• 2: Largely unnecessary.
• 1: Question TEXT fully states the problem; image unnecessary.

Reply with exactly one JSON: {"necessity_score": <1-5>, "reason": "brief one-sentence"}"""

# 仅判相关性（图+题+答案）。第二次 API 用，与 necessity_only 分两次调用。
CLEAN_JUDGE_RELEVANCE_ONLY_SYSTEM = """You are a data quality judge. You see an image, a **question TEXT**, and a reference **answer**.

Output **only the relevance score (1-5)** and a brief reason. Judge image-question alignment. You may use the answer only as auxiliary context for topic consistency. Do NOT penalize relevance because the answer is wrong if image-question are aligned.
• 5: Image and question are strongly aligned.
• 4: Mostly aligned; minor ambiguity.
• 3: Partially aligned.
• 2: Largely mismatched.
• 1: Clearly unrelated.

Reply with exactly one JSON: {"relevance_score": <1-5>, "reason": "brief one-sentence"}"""

# 单次请求：图+题+答案各传一次，Part A 仅用于 necessity，Part A+Part B 用于 relevance，避免图/题传两遍。
CLEAN_JUDGE_SINGLE_CALL_SYSTEM = """You are a data quality judge. The user message has two parts:
- **Part A**: One image + one question. Use ONLY Part A for necessity.
- **Part B**: An answer. Use Part A + Part B together for relevance.

Output both scores (1-5) in one JSON.

**Necessity (1-5)** – Use ONLY the image and question from Part A. Ignore Part B. Ask: "If someone saw ONLY the question text and had NO image, could they understand the full problem?"
• 5: Image clearly necessary (e.g. text empty, only options, or "the figure"). • 4–2: degrees. • 1: Question fully states the problem; image unnecessary.

**Relevance (1-5)** – Judge image-question alignment. You may use Part B only as auxiliary context for topic consistency. Do NOT penalize relevance due to answer reasoning/calculation mistakes if image-question are aligned.
• 5: Image and question strongly aligned. • 4: Mostly aligned with minor ambiguity. • 3: Partially aligned. • 2: Largely mismatched. • 1: Unrelated.

Reply with exactly one JSON: {"necessity_score": <1-5>, "relevance_score": <1-5>, "reason": "brief one-sentence"}"""
