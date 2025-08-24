# pmm/ngram_ban.py
from __future__ import annotations
from typing import Dict, List, Optional
import re
import random
import os

# Lazy import to avoid heavy deps
try:
    from .emergence import compute_emergence_scores
except Exception:  # pragma: no cover
    compute_emergence_scores = None  # type: ignore


class NGramBanSystem:
    """N-gram ban system for model-specific catchphrases."""

    def __init__(self):
        # Model-specific banned n-grams
        self.banned_ngrams = {
            "gemma": {
                "that's… extraordinary, scott",
                "i'm genuinely thrilled",
                "that's extraordinary scott",
                "fascinating scott",
                "remarkable insight scott",
                "that's… fascinating",
                "genuinely thrilled",
                "extraordinary development",
                "remarkable development",
                "circling back to a playful exchange",
                "committed to thoroughly testing the commitment system",
                "let's continue to explore this testing process",
                "i'm committed to thoroughly testing",
                "could you give me a single, concrete request",
                "something i can respond to directly",
                "this will allow me to demonstrate",
                "let's move beyond the circularity",
                "get to the core of the testing",
                "it's a calibration issue",
                "prioritizing acknowledging the testing",
            },
            "gpt": {
                "i appreciate you sharing",
                "thank you for bringing this",
                "that's a great point",
                "i'm happy to help",
                "that's an interesting perspective",
            },
            "claude": {
                "i'd be happy to",
                "that's a thoughtful question",
                "from my perspective",
                "i think it's important to consider",
                "that's a great observation",
            },
        }

        # Neutral replacement variants
        self.neutral_variants = [
            "Noted.",
            "Clear signal.",
            "Understood.",
            "Got it.",
            "Nice milestone.",
            "Acknowledged.",
            "Confirmed.",
            "Received.",
            "Processed.",
            "Logged.",
        ]

        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for model_family, ngrams in self.banned_ngrams.items():
            self._compiled_patterns[model_family] = [
                re.compile(re.escape(ngram), re.IGNORECASE) for ngram in ngrams
            ]

        # Core phrases to keep even in relaxed mode (small, stable subset)
        self.core_banned: Dict[str, List[str]] = {
            "gemma": [
                "that's extraordinary scott",
                "fascinating scott",
                "remarkable development",
            ],
            "gpt": [
                "i'm happy to help",
                "that's a great point",
            ],
            "claude": [
                "that's a thoughtful question",
                "from my perspective",
            ],
        }

    def _resolve_stage(self, stage: Optional[str]) -> Optional[str]:
        if stage:
            return stage
        try:
            hard = str(os.getenv("PMM_HARD_STAGE", "")).strip().upper()
            if hard in ("S0", "S1", "S2", "S3", "S4", "SS4"):
                if hard == "SS4":
                    return "SS4"
                return {
                    "S0": "S0: Substrate",
                    "S1": "S1: Resistance",
                    "S2": "S2: Adoption",
                    "S3": "S3: Self-Model",
                    "S4": "S4: Growth-Seeking",
                }[hard]
        except Exception:
            pass
        try:
            if compute_emergence_scores:
                scores = compute_emergence_scores()
                return scores.get("stage")
        except Exception:
            return None
        return None

    def _strictness(self, stage_label: Optional[str]) -> str:
        if not stage_label:
            return "normal"
        s = stage_label.lower()
        if s.startswith("s0") or s.startswith("s1"):
            return "strict"
        if s.startswith("s3") or s.startswith("s4") or s == "ss4":
            return "relaxed"
        return "normal"

    def get_model_family(self, model_name: str) -> str:
        """Extract model family from full model name."""
        model_lower = model_name.lower()
        if "gemma" in model_lower:
            return "gemma"
        elif "gpt" in model_lower or "openai" in model_lower:
            return "gpt"
        elif "claude" in model_lower:
            return "claude"
        else:
            return "unknown"

    def check_banned_ngrams(
        self, text: str, model_name: str, n: int = 4, stage: Optional[str] = None
    ) -> tuple[bool, List[str]]:
        """
        Check if text contains banned n-grams for the given model.

        Args:
            text: Text to check
            model_name: Name of the model
            n: N-gram size to check

        Returns:
            (has_banned_ngrams: bool, banned_phrases: List[str])
        """
        model_family = self.get_model_family(model_name)
        stage_label = self._resolve_stage(stage)
        mode = self._strictness(stage_label)

        if model_family not in self._compiled_patterns:
            return False, []

        banned_phrases = []

        # Check against model-specific banned patterns
        if mode == "relaxed":
            # Only enforce a minimal, core subset
            core_set = set([c.lower() for c in self.core_banned.get(model_family, [])])
            for pattern in self._compiled_patterns[model_family]:
                literal = pattern.pattern.replace("\\", "").lower()
                if literal in core_set and pattern.search(text):
                    banned_phrases.append(pattern.pattern.replace("\\", ""))
        else:
            for pattern in self._compiled_patterns[model_family]:
                if pattern.search(text):
                    banned_phrases.append(pattern.pattern.replace("\\", ""))

        # Also check n-gram extraction (skip in relaxed mode)
        if mode == "strict":
            # Try multiple sizes for stricter matching
            for size in (3, n, 5):
                banned_phrases.extend(
                    self._extract_banned_ngrams(text, model_family, size)
                )
        elif mode == "normal":
            banned_phrases.extend(self._extract_banned_ngrams(text, model_family, n))

        return len(banned_phrases) > 0, banned_phrases

    def _extract_banned_ngrams(self, text: str, model_family: str, n: int) -> List[str]:
        """Extract n-grams and check against banned list."""
        if model_family not in self.banned_ngrams:
            return []

        # Tokenize text
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        # Generate n-grams
        ngrams = {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

        # Check against banned list
        banned_found = []
        for banned_ngram in self.banned_ngrams[model_family]:
            if banned_ngram.lower() in ngrams:
                banned_found.append(banned_ngram)

        return banned_found

    def postprocess_style(
        self, text: str, model_name: str, stage: Optional[str] = None
    ) -> tuple[str, List[str]]:
        """
        Post-process text to remove banned n-grams and replace with neutral variants.

        Args:
            text: Text to process
            model_name: Name of the model

        Returns:
            (processed_text: str, replacements_made: List[str])
        """
        has_banned, banned_phrases = self.check_banned_ngrams(
            text, model_name, stage=stage
        )

        if not has_banned:
            return text, []

        processed_text = text
        replacements_made = []

        model_family = self.get_model_family(model_name)
        stage_label = self._resolve_stage(stage)
        mode = self._strictness(stage_label)

        # Replace banned patterns with neutral variants
        if model_family in self._compiled_patterns:
            if mode == "relaxed":
                core_set = set(
                    [c.lower() for c in self.core_banned.get(model_family, [])]
                )
                for pattern in self._compiled_patterns[model_family]:
                    literal = pattern.pattern.replace("\\", "").lower()
                    if literal in core_set and pattern.search(processed_text):
                        replacement = random.choice(self.neutral_variants)
                        old_text = processed_text
                        processed_text = pattern.sub(replacement, processed_text)
                        if old_text != processed_text:
                            replacements_made.append(
                                f"banned_ngram: {pattern.pattern} → {replacement}"
                            )
            else:
                for pattern in self._compiled_patterns[model_family]:
                    if pattern.search(processed_text):
                        replacement = random.choice(self.neutral_variants)
                        old_text = processed_text
                        processed_text = pattern.sub(replacement, processed_text)
                        if old_text != processed_text:
                            replacements_made.append(
                                f"banned_ngram: {pattern.pattern} → {replacement}"
                            )

        return processed_text, replacements_made

    def add_banned_ngram(self, model_family: str, ngram: str) -> None:
        """Add a new banned n-gram for a model family."""
        if model_family not in self.banned_ngrams:
            self.banned_ngrams[model_family] = set()

        self.banned_ngrams[model_family].add(ngram.lower())

        # Update compiled patterns
        if model_family not in self._compiled_patterns:
            self._compiled_patterns[model_family] = []

        self._compiled_patterns[model_family].append(
            re.compile(re.escape(ngram), re.IGNORECASE)
        )

    def get_stats(self, model_name: str) -> Dict:
        """Get statistics for banned n-grams."""
        model_family = self.get_model_family(model_name)

        return {
            "model_family": model_family,
            "banned_ngrams_count": len(self.banned_ngrams.get(model_family, [])),
            "banned_ngrams": list(self.banned_ngrams.get(model_family, [])),
            "neutral_variants_count": len(self.neutral_variants),
        }
