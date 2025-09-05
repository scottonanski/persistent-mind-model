# pmm/phrase_deduper.py
from __future__ import annotations
from typing import Dict, List
from collections import Counter
from dataclasses import dataclass


@dataclass
class ModelPhraseCache:
    """Cache of n-grams for a specific model to detect repetitive phrases."""

    model_name: str
    bigrams: Counter
    trigrams: Counter
    max_cache_size: int = 100

    def add_text(self, text: str) -> None:
        """Add text to the cache, extracting n-grams."""

        # Clean and tokenize (regex-free): keep alnum and spaces
        def _normalize(s: str) -> str:
            buf = []
            for ch in s.lower():
                if ch.isalnum() or ch.isspace():
                    buf.append(ch)
                else:
                    buf.append(" ")
            return "".join(buf)

        clean_text = _normalize(text)
        tokens = clean_text.split()

        # Extract bigrams and trigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            self.bigrams[bigram] += 1

        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            self.trigrams[trigram] += 1

        # Trim cache if too large
        if len(self.bigrams) > self.max_cache_size:
            # Keep only most frequent half
            keep_count = self.max_cache_size // 2
            self.bigrams = Counter(dict(self.bigrams.most_common(keep_count)))

        if len(self.trigrams) > self.max_cache_size:
            keep_count = self.max_cache_size // 2
            self.trigrams = Counter(dict(self.trigrams.most_common(keep_count)))

    def check_repetition(
        self, text: str, threshold: float = 0.35
    ) -> tuple[bool, List[str]]:
        """Check if text contains too many repeated n-grams."""

        # Normalize and tokenize without regex
        def _normalize(s: str) -> str:
            buf = []
            for ch in s.lower():
                if ch.isalnum() or ch.isspace():
                    buf.append(ch)
                else:
                    buf.append(" ")
            return "".join(buf)

        clean_text = _normalize(text)
        tokens = clean_text.split()

        repeated_phrases = []
        total_ngrams = 0
        repeated_ngrams = 0

        # Check bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            total_ngrams += 1
            if self.bigrams.get(bigram, 0) >= 3:  # Seen 3+ times
                repeated_ngrams += 1
                repeated_phrases.append(bigram)

        # Check trigrams
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            total_ngrams += 1
            if self.trigrams.get(trigram, 0) >= 2:  # Seen 2+ times
                repeated_ngrams += 1
                repeated_phrases.append(trigram)

        if total_ngrams == 0:
            return False, []

        repetition_ratio = repeated_ngrams / total_ngrams
        is_repetitive = repetition_ratio > threshold

        return is_repetitive, list(set(repeated_phrases))


class PhraseDeduper:
    """Model-aware phrase deduplication system."""

    def __init__(self):
        self.model_caches: Dict[str, ModelPhraseCache] = {}

        # Known problematic phrases by model family (regex-free).
        # Each entry maps to a list of detectors with a canonical key and an ordered phrase sequence.
        # The canonical key is used for suggestions matching.
        self.model_patterns = {
            "gemma": [
                {"key": "that's extraordinary", "seq": ["that's", "extraordinary"]},
                {"key": "fascinating scott", "seq": ["fascinating", "scott"]},
                {"key": "remarkable insight", "seq": ["remarkable", "insight"]},
                {"key": "intriguing perspective", "seq": ["intriguing", "perspective"]},
            ],
            "gpt": [
                {"key": "i appreciate you sharing", "seq": ["i appreciate", "sharing"]},
                {"key": "thank you for bringing", "seq": ["thank you", "bringing"]},
                {
                    "key": "that's an interesting point",
                    "seq": ["that's", "interesting", "point"],
                },
                {
                    "key": "i understand you're saying",
                    "seq": ["i understand", "you're", "saying"],
                },
            ],
            "claude": [
                {"key": "i'd be happy to", "seq": ["i'd be happy to"]},
                {
                    "key": "that's a thoughtful question",
                    "seq": ["that's", "thoughtful", "question"],
                },
                {
                    "key": "i think it's important to consider",
                    "seq": ["i think", "important", "consider"],
                },
                {"key": "from my perspective", "seq": ["from my perspective"]},
            ],
        }

        # Precompute lowercase canonical keys for suggestion matching
        for fam, items in self.model_patterns.items():
            for it in items:
                it["key_lc"] = it["key"].lower()

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

    def get_cache(self, model_name: str) -> ModelPhraseCache:
        """Get or create cache for model."""
        if model_name not in self.model_caches:
            self.model_caches[model_name] = ModelPhraseCache(
                model_name=model_name, bigrams=Counter(), trigrams=Counter()
            )
        return self.model_caches[model_name]

    def add_response(self, model_name: str, response_text: str) -> None:
        """Add a response to the model's phrase cache."""
        cache = self.get_cache(model_name)
        cache.add_text(response_text)

    def check_response(
        self, model_name: str, response_text: str
    ) -> tuple[bool, List[str], float]:
        """Check if response is too repetitive."""
        cache = self.get_cache(model_name)

        # Check n-gram repetition
        is_repetitive, repeated_phrases = cache.check_repetition(response_text)

        # Check known problematic patterns using ordered phrase detection
        model_family = self.get_model_family(model_name)
        pattern_matches = []

        def _contains_ordered(text_lc: str, seq: List[str]) -> bool:
            pos = 0
            for part in seq:
                idx = text_lc.find(part, pos)
                if idx == -1:
                    return False
                pos = idx + len(part)
            return True

        text_lc = response_text.lower()
        if model_family in self.model_patterns:
            for det in self.model_patterns[model_family]:
                if _contains_ordered(text_lc, det["seq"]):
                    pattern_matches.append(det["key_lc"])  # canonical phrase key

        # Calculate overall repetition score
        total_phrases = len(repeated_phrases) + len(pattern_matches)
        word_count = len(response_text.split())
        repetition_score = total_phrases / max(
            word_count / 10, 1
        )  # Normalize by response length

        all_issues = repeated_phrases + pattern_matches
        is_problematic = (
            is_repetitive or len(pattern_matches) > 0 or repetition_score > 0.3
        )

        return is_problematic, all_issues, repetition_score

    def suggest_rephrase(
        self, response_text: str, problematic_phrases: List[str]
    ) -> str:
        """Suggest alternative phrasings for repetitive content."""
        suggestions = {
            # Gemma-specific replacements
            "that's extraordinary": [
                "that's notable",
                "that's significant",
                "that's worth noting",
            ],
            "fascinating scott": [
                "interesting point",
                "worth considering",
                "notable observation",
            ],
            "remarkable insight": [
                "good point",
                "valuable observation",
                "useful perspective",
            ],
            # GPT-specific replacements
            "i appreciate you sharing": [
                "thanks for mentioning",
                "good to know",
                "noted",
            ],
            "thank you for bringing": ["you raise", "you mention", "you point out"],
            "that's an interesting point": [
                "good observation",
                "worth noting",
                "I see",
            ],
            # Claude-specific replacements
            "i'd be happy to": ["I can", "let me", "I'll"],
            "that's a thoughtful question": [
                "good question",
                "worth asking",
                "let me address",
            ],
            "from my perspective": ["I think", "it seems", "appears that"],
        }

        modified_text = response_text

        # Helper: case-insensitive replace for phrases
        def _ci_replace(haystack: str, needle: str, repl: str) -> str:
            if not needle:
                return haystack
            out = []
            i = 0
            n = len(needle)
            h_lc = haystack.lower()
            ndl_lc = needle.lower()
            while i <= len(haystack) - n:
                if h_lc[i : i + n] == ndl_lc:
                    out.append(repl)
                    i += n
                else:
                    out.append(haystack[i])
                    i += 1
            out.append(haystack[i:])
            return "".join(out)

        for phrase in problematic_phrases:
            # Find best replacement
            for pattern, replacements in suggestions.items():
                if pattern in phrase.lower():
                    import random

                    replacement = random.choice(replacements)
                    # Case-insensitive replace without regex
                    modified_text = _ci_replace(modified_text, phrase, replacement)
                    break

        return modified_text

    def get_stats(self, model_name: str) -> Dict:
        """Get statistics for a model's phrase usage."""
        if model_name not in self.model_caches:
            return {"bigrams": 0, "trigrams": 0, "top_bigrams": [], "top_trigrams": []}

        cache = self.model_caches[model_name]
        return {
            "bigrams": len(cache.bigrams),
            "trigrams": len(cache.trigrams),
            "top_bigrams": cache.bigrams.most_common(5),
            "top_trigrams": cache.trigrams.most_common(5),
        }
