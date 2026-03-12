from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pipeline.tasks.benchmark_similarity.profiles import score_profile_match


DEFAULT_SCORING_WEIGHTS: dict[str, float] = {
    "semantic": 0.42,
    "skill": 0.18,
    "reasoning": 0.08,
    "domain": 0.05,
    "duration": 0.06,
    "modality": 0.06,
    "locality": 0.03,
    "answer_type": 0.02,
    "source_prior": 0.10,
}


@dataclass(slots=True)
class ScoreSummary:
    backend: str
    top_matches: list[dict[str, Any]]
    benchmark_source_scores: dict[str, float]
    benchmark_source_semantic_scores: dict[str, float]
    max_similarity: float
    max_adjusted_similarity: float
    mean_topk_similarity: float
    mean_topk_adjusted_similarity: float
    balanced_score: float
    coverage_score: float
    selection_score: float
    best_benchmark_source: str


def build_similarity_index(
    benchmark_examples: list[dict[str, Any]],
    *,
    requested_backend: str = "auto",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    scoring_weights: dict[str, float] | None = None,
    benchmark_weights: dict[str, float] | None = None,
    coverage_threshold: float = 0.45,
) -> "_BaseSimilarityIndex":
    mode = str(requested_backend or "auto").strip().lower()
    if mode not in {"auto", "sentence_transformers", "tfidf", "token_overlap"}:
        raise ValueError(
            "similarity_backend must be one of: auto, sentence_transformers, tfidf, token_overlap"
        )

    errors: list[str] = []
    if mode in {"auto", "sentence_transformers"}:
        try:
            return _SentenceTransformerIndex(
                benchmark_examples=benchmark_examples,
                model_name=embedding_model,
                batch_size=batch_size,
                scoring_weights=scoring_weights,
                benchmark_weights=benchmark_weights,
                coverage_threshold=coverage_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            if mode != "auto":
                raise
            errors.append(f"sentence_transformers={exc}")

    if mode in {"auto", "tfidf"}:
        try:
            return _TfidfIndex(
                benchmark_examples=benchmark_examples,
                scoring_weights=scoring_weights,
                benchmark_weights=benchmark_weights,
                coverage_threshold=coverage_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            if mode != "auto":
                raise
            errors.append(f"tfidf={exc}")

    if mode in {"auto", "token_overlap"}:
        return _TokenCosineIndex(
            benchmark_examples=benchmark_examples,
            scoring_weights=scoring_weights,
            benchmark_weights=benchmark_weights,
            coverage_threshold=coverage_threshold,
        )

    raise RuntimeError(f"Unable to build similarity index. Fallback errors: {errors}")


class _BaseSimilarityIndex:
    backend_name = "base"

    def __init__(
        self,
        benchmark_examples: list[dict[str, Any]],
        *,
        scoring_weights: dict[str, float] | None = None,
        benchmark_weights: dict[str, float] | None = None,
        coverage_threshold: float = 0.45,
    ) -> None:
        self._benchmark_examples = benchmark_examples
        self._benchmark_sources = sorted(
            {str(item.get("source_name") or "") for item in benchmark_examples if item.get("source_name")}
        )
        self._benchmark_profiles = [dict(item.get("profile") or {}) for item in benchmark_examples]
        self._scoring_weights = dict(DEFAULT_SCORING_WEIGHTS)
        for key, value in (scoring_weights or {}).items():
            try:
                self._scoring_weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        self._benchmark_weights = {
            source_name: float((benchmark_weights or {}).get(source_name, 1.0))
            for source_name in self._benchmark_sources
        }
        self._coverage_threshold = max(0.0, min(float(coverage_threshold), 1.0))

    def summarize(self, candidate_example: dict[str, Any], *, top_n: int) -> ScoreSummary:
        text = str(candidate_example.get("similarity_text") or "")
        candidate_profile = dict(candidate_example.get("profile") or {})
        semantic_scores = [_normalize_semantic_score(score) for score in self._score_all(text)]

        scored_matches: list[tuple[int, float, float, dict[str, float]]] = []
        for index, semantic_score in enumerate(semantic_scores):
            profile_components = score_profile_match(self._benchmark_profiles[index], candidate_profile)
            adjusted_score = _blend_match_score(semantic_score, profile_components, self._scoring_weights)
            scored_matches.append((index, adjusted_score, semantic_score, profile_components))

        pairs = sorted(scored_matches, key=lambda item: item[1], reverse=True)
        top_pairs = pairs[: max(1, top_n)]

        source_scores = {name: 0.0 for name in self._benchmark_sources}
        source_semantic_scores = {name: 0.0 for name in self._benchmark_sources}
        for index, adjusted_score, semantic_score, _ in pairs:
            source_name = str(self._benchmark_examples[index].get("source_name") or "")
            if source_name and adjusted_score > source_scores.get(source_name, float("-inf")):
                source_scores[source_name] = float(adjusted_score)
            if source_name and semantic_score > source_semantic_scores.get(source_name, float("-inf")):
                source_semantic_scores[source_name] = float(semantic_score)

        top_matches: list[dict[str, Any]] = []
        for index, adjusted_score, semantic_score, profile_components in top_pairs:
            match = self._benchmark_examples[index]
            top_matches.append(
                {
                    "benchmark_id": match.get("source_record_id"),
                    "benchmark_source": match.get("source_name"),
                    "score": float(adjusted_score),
                    "adjusted_score": float(adjusted_score),
                    "semantic_score": float(semantic_score),
                    "profile_score": _profile_only_score(profile_components, self._scoring_weights),
                    "profile_components": profile_components,
                    "benchmark_profile": self._benchmark_profiles[index],
                    "question": match.get("question"),
                    "answer": match.get("answer"),
                }
            )

        max_similarity = float(max(semantic_scores) if semantic_scores else 0.0)
        max_adjusted_similarity = float(top_pairs[0][1]) if top_pairs else 0.0
        mean_topk_similarity = (
            float(sum(semantic_score for _, _, semantic_score, _ in top_pairs) / len(top_pairs))
            if top_pairs
            else 0.0
        )
        mean_topk_adjusted_similarity = (
            float(sum(adjusted_score for _, adjusted_score, _, _ in top_pairs) / len(top_pairs))
            if top_pairs
            else 0.0
        )
        balanced_score = (
            _weighted_average(source_scores, self._benchmark_weights) if source_scores else max_adjusted_similarity
        )
        coverage_score = _coverage_score(
            source_scores=source_scores,
            source_weights=self._benchmark_weights,
            threshold=self._coverage_threshold,
        )
        selection_score = float((0.85 * balanced_score) + (0.15 * coverage_score))
        best_benchmark_source = ""
        if source_scores:
            best_benchmark_source = max(source_scores.items(), key=lambda item: item[1])[0]

        return ScoreSummary(
            backend=self.backend_name,
            top_matches=top_matches,
            benchmark_source_scores=source_scores,
            benchmark_source_semantic_scores=source_semantic_scores,
            max_similarity=max_similarity,
            max_adjusted_similarity=max_adjusted_similarity,
            mean_topk_similarity=mean_topk_similarity,
            mean_topk_adjusted_similarity=mean_topk_adjusted_similarity,
            balanced_score=balanced_score,
            coverage_score=coverage_score,
            selection_score=selection_score,
            best_benchmark_source=best_benchmark_source,
        )

    def _score_all(self, text: str) -> list[float]:
        raise NotImplementedError


class _SentenceTransformerIndex(_BaseSimilarityIndex):
    backend_name = "sentence_transformers"

    def __init__(
        self,
        *,
        benchmark_examples: list[dict[str, Any]],
        model_name: str,
        batch_size: int,
        scoring_weights: dict[str, float] | None,
        benchmark_weights: dict[str, float] | None,
        coverage_threshold: float,
    ) -> None:
        super().__init__(
            benchmark_examples,
            scoring_weights=scoring_weights,
            benchmark_weights=benchmark_weights,
            coverage_threshold=coverage_threshold,
        )
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        texts = [str(item.get("similarity_text") or "") for item in benchmark_examples]
        self._matrix = self._model.encode(
            texts,
            batch_size=max(1, batch_size),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _score_all(self, text: str) -> list[float]:
        query = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        scores = self._matrix @ query
        return [float(score) for score in scores.tolist()]


class _TfidfIndex(_BaseSimilarityIndex):
    backend_name = "tfidf"

    def __init__(
        self,
        *,
        benchmark_examples: list[dict[str, Any]],
        scoring_weights: dict[str, float] | None,
        benchmark_weights: dict[str, float] | None,
        coverage_threshold: float,
    ) -> None:
        super().__init__(
            benchmark_examples,
            scoring_weights=scoring_weights,
            benchmark_weights=benchmark_weights,
            coverage_threshold=coverage_threshold,
        )
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        texts = [str(item.get("similarity_text") or "") for item in benchmark_examples]
        self._matrix = self._vectorizer.fit_transform(texts)

    def _score_all(self, text: str) -> list[float]:
        query = self._vectorizer.transform([text])
        scores = query @ self._matrix.T
        return [float(score) for score in scores.toarray().ravel().tolist()]


class _TokenCosineIndex(_BaseSimilarityIndex):
    backend_name = "token_overlap"

    def __init__(
        self,
        *,
        benchmark_examples: list[dict[str, Any]],
        scoring_weights: dict[str, float] | None,
        benchmark_weights: dict[str, float] | None,
        coverage_threshold: float,
    ) -> None:
        super().__init__(
            benchmark_examples,
            scoring_weights=scoring_weights,
            benchmark_weights=benchmark_weights,
            coverage_threshold=coverage_threshold,
        )
        self._bench_counters = [_token_counter(str(item.get("similarity_text") or "")) for item in benchmark_examples]
        self._bench_norms = [_counter_norm(counter) for counter in self._bench_counters]

    def _score_all(self, text: str) -> list[float]:
        query_counter = _token_counter(text)
        query_norm = _counter_norm(query_counter)
        if query_norm <= 0.0:
            return [0.0 for _ in self._bench_counters]

        scores: list[float] = []
        for counter, norm in zip(self._bench_counters, self._bench_norms, strict=False):
            if norm <= 0.0:
                scores.append(0.0)
                continue
            dot = 0.0
            for token, value in query_counter.items():
                dot += value * counter.get(token, 0.0)
            scores.append(float(dot / (query_norm * norm)))
        return scores


_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def _token_counter(text: str) -> Counter[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    return Counter(tokens)


def _counter_norm(counter: Counter[str]) -> float:
    if not counter:
        return 0.0
    return math.sqrt(sum(value * value for value in counter.values()))


def _normalize_semantic_score(score: float) -> float:
    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _blend_match_score(
    semantic_score: float,
    profile_components: dict[str, float],
    scoring_weights: dict[str, float],
) -> float:
    total = 0.0
    weight_sum = 0.0
    semantic_weight = max(0.0, float(scoring_weights.get("semantic", 0.0)))
    total += semantic_weight * semantic_score
    weight_sum += semantic_weight

    for key, value in profile_components.items():
        weight = max(0.0, float(scoring_weights.get(key, 0.0)))
        if weight <= 0.0:
            continue
        total += weight * max(0.0, min(float(value), 1.0))
        weight_sum += weight
    if weight_sum <= 0.0:
        return semantic_score
    return float(total / weight_sum)


def _profile_only_score(profile_components: dict[str, float], scoring_weights: dict[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for key, value in profile_components.items():
        weight = max(0.0, float(scoring_weights.get(key, 0.0)))
        if weight <= 0.0:
            continue
        total += weight * max(0.0, min(float(value), 1.0))
        weight_sum += weight
    if weight_sum <= 0.0:
        return 0.0
    return float(total / weight_sum)


def _weighted_average(values: dict[str, float], weights: dict[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key, value in values.items():
        weight = float(weights.get(key, 1.0))
        numerator += weight * float(value)
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _coverage_score(
    *,
    source_scores: dict[str, float],
    source_weights: dict[str, float],
    threshold: float,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for source_name, score in source_scores.items():
        weight = float(source_weights.get(source_name, 1.0))
        numerator += weight * (1.0 if float(score) >= threshold else 0.0)
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)
