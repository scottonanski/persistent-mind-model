#!/usr/bin/env python3
from __future__ import annotations
from typing import Optional, Tuple
import os

from pmm.struct_semantics import pos_tag, split_sentences, CentroidModel


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class CommitmentExtractor:
    def __init__(self):
        self.alpha = _get_env_float("PMM_SCORE_ALPHA", 0.3)
        self.beta = _get_env_float("PMM_SCORE_BETA", 0.7)
        self.commit_thresh = _get_env_float("PMM_COMMIT_SCORE_THRESH", 0.35)
        self.directive_thresh = _get_env_float("PMM_DIRECTIVE_SCORE_THRESH", 0.30)
        self.centroid = CentroidModel()
        # The caller should fit centroids from recent events' vectors only

    def _structure_score(self, text: str) -> float:
        # Semantic-only policy: no keyword or lexicon lists. Use only POS cues and minimal length.
        toks_tags = pos_tag(text)
        toks = [t for t in (text or "").strip().split() if t]
        token_count = len(toks)

        # POS-based signals when available
        first_is_verb = False
        has_first_person_pos = False
        has_modal = False
        if toks_tags:
            try:
                first_is_verb = bool(toks_tags and str(toks_tags[0][1]).startswith("VB"))
                has_first_person_pos = any(
                    (str(tag).startswith("PRP") or str(tag).startswith("PRP$")) for _, tag in toks_tags
                )
                has_modal = any(str(tag).startswith("MD") for _, tag in toks_tags)
            except Exception:
                first_is_verb = False
                has_first_person_pos = False
                has_modal = False

        # Structure-only scoring
        score = 0.0
        first_token = (toks[0].lower() if toks else "")
        # Strongly favor imperative (verb-first) structure
        if first_is_verb:
            score += 0.8
        # First-person pronoun provides only a small boost (not sufficient alone)
        if has_first_person_pos:
            score += 0.1
        # Minimal length requirement contributes enough to pass when vectors are neutral
        if token_count >= 3:
            score += 0.25
        # Penalize modal constructions (e.g., should/will) as they are often less actionable
        if has_modal:
            score -= 0.4
        # Slightly discourage pronoun-first constructions (e.g., "I commit ...")
        if first_token in {"i", "we"}:
            score -= 0.1
        if score < 0.0:
            score = 0.0
        return min(1.0, score)

    def _vector(self, text: str) -> list[float]:
        try:
            from pmm.semantic_analysis import get_semantic_analyzer

            analyzer = get_semantic_analyzer()
            return list(analyzer.embed(text))  # type: ignore
        except Exception:
            return []

    def fit_centroids(self, pos_vectors: list[list[float]], neg_vectors: list[list[float]]) -> None:
        self.centroid.fit(pos_vectors, neg_vectors)

    def score(self, text: str) -> float:
        s = self._structure_score(text)
        v = self._vector(text)
        c = self.centroid.score(v)
        # normalize centroid score from [-1,1] to [0,1]
        c_norm = (c + 1.0) / 2.0
        return self.alpha * s + self.beta * c_norm

    def extract_best_sentence(self, text: str) -> Optional[str]:
        candidates = split_sentences(text)
        if not candidates:
            return None
        best = (0.0, None)
        for s in candidates:
            sc = self.score(s)
            if sc > best[0]:
                best = (sc, s)
        return best[1]


def is_directive(text: str, extractor: CommitmentExtractor) -> bool:
    return extractor.score(text) >= extractor.directive_thresh
