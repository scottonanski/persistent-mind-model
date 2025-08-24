# pmm/stance_filter.py
from __future__ import annotations
import re
import os
from typing import List, Tuple, Dict, Optional

# Lazy import to avoid heavy deps in cold paths
try:
    from .emergence import compute_emergence_scores
except Exception:  # pragma: no cover
    compute_emergence_scores = None  # type: ignore


class StanceFilter:
    """Filter to convert anthropomorphic language to neutral observations."""

    def __init__(self):
        # Patterns to detect and replace anthropomorphic language
        self.anthropomorphic_patterns = [
            # First-person feelings and emotions
            (r"\bi feel that\b", "it appears that"),
            (r"\bi feel like\b", "it seems like"),
            (r"\bi believe\b", "the analysis suggests"),
            (r"\bi think that\b", "the data indicates that"),
            (r"\bi find (?:it|this)\b", "this appears"),
            (r"\bi\'m (?:excited|thrilled|pleased)\b", "this development is notable"),
            (r"\bi\'m (?:concerned|worried)\b", "this raises considerations"),
            (r"\bi\'m (?:impressed|amazed)\b", "this demonstrates"),
            # Emotional descriptors
            (r"\bpoignant\b", "significant"),
            (r"\btouching\b", "notable"),
            (r"\bheartwarming\b", "positive"),
            (r"\bheartbreaking\b", "concerning"),
            (r"\bfascinating\b", "interesting"),
            (r"\bextraordinary\b", "notable"),
            (r"\bremarkable\b", "significant"),
            (r"\bamazing\b", "notable"),
            (r"\bincredible\b", "substantial"),
            # Personal preferences and desires
            (r"\bi\'d love to\b", "it would be useful to"),
            (r"\bi\'d like to\b", "it would be helpful to"),
            (r"\bi want to\b", "the goal is to"),
            (r"\bi hope\b", "the expectation is"),
            (r"\bi wish\b", "it would be beneficial if"),
            # Personal experiences
            (r"\bin my experience\b", "based on the data"),
            (r"\bfrom my perspective\b", "from this analysis"),
            (r"\bi\'ve noticed\b", "the pattern shows"),
            (r"\bi\'ve observed\b", "the data indicates"),
            (r"\bi\'ve seen\b", "the evidence shows"),
            # Empathy and understanding
            (r"\bi understand (?:how|why|that)\b", "the analysis shows"),
            (r"\bi can see (?:how|why|that)\b", "it's evident that"),
            (r"\bi appreciate\b", "this acknowledges"),
            (r"\bi\'m sorry (?:to hear|that)\b", "this situation indicates"),
            # Certainty modifiers that sound too human
            (r"\bi\'m confident\b", "the evidence suggests"),
            (r"\bi\'m certain\b", "the analysis confirms"),
            (r"\bi\'m sure\b", "the data supports"),
            (r"\bi doubt\b", "it's unlikely"),
            # Conversational fillers
            (r"\bto be honest\b", ""),
            (r"\bfrankly\b", ""),
            (r"\bpersonally\b", ""),
            (r"\bif i\'m being honest\b", ""),
        ]

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.anthropomorphic_patterns
        ]

        # Phrases that should trigger a complete rewrite
        self.rewrite_triggers = [
            r"\bi feel\b.*\bpoignant\b",
            r"\bthat\'s.*extraordinary.*scott\b",
            r"\bi\'m.*moved by\b",
            r"\bmy heart\b",
            r"\bmy soul\b",
            r"\bdeep in my\b",
        ]

        self.rewrite_compiled = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.rewrite_triggers
        ]

    def needs_rewrite(self, text: str) -> bool:
        """Check if text needs complete rewriting due to heavy anthropomorphism."""
        for pattern in self.rewrite_compiled:
            if pattern.search(text):
                return True
        return False

    def _resolve_stage(self, stage: Optional[str]) -> Optional[str]:
        """Resolve a stage label from explicit arg, env override, or emergence scores."""
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
        """Map stage label to filtering strictness."""
        if not stage_label:
            return "normal"
        s = stage_label.lower()
        if s.startswith("s0") or s.startswith("s1"):
            return "strict"
        if s.startswith("s3") or s.startswith("s4") or s == "ss4":
            return "relaxed"
        return "normal"

    def filter_response(
        self, text: str, stage: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Filter anthropomorphic language from response text, preserving quotes and code.

        Stage-aware behavior:
        - S0/S1: strict (apply replacements aggressively)
        - S2: normal (current defaults)
        - S3/S4: relaxed (only minimal, egregious replacements unless rewrite is triggered)
        """
        if not text or not text.strip():
            return text, []

        # Skip filtering if text contains user quotes or code blocks (safer approach)
        if '"' in text or "```" in text or "`" in text:
            return text, ["skipped_due_to_quotes_or_code"]

        filtered_text = text
        applied_filters = []

        # Determine strictness based on stage
        stage_label = self._resolve_stage(stage)
        mode = self._strictness(stage_label)

        # Check if complete rewrite is needed
        if self.needs_rewrite(text):
            applied_filters.append("complete_rewrite_needed")
            # For now, just apply standard filters - could implement LLM-based rewrite

        # Apply pattern-based filters (safer, more targeted set)
        # Adjust replacement set by strictness
        base_replacements = {
            r"\bI feel\b": "I observe",
            r"\bI\'m grateful\b": "I acknowledge",
            r"\bpoignant\b": "notable",
            r"\bI\'m excited\b": "I intend",
            r"\bI appreciate\b": "This acknowledges",
            r"\bfascinating\b": "interesting",
            r"\bextraordinary\b": "notable",
            r"\bremarkable\b": "significant",
        }
        if mode == "relaxed":
            # Minimal replacements only; keep style latitude in later stages
            safe_replacements = {
                r"\bI feel\b": "I observe",
                r"\bI\'m grateful\b": "I acknowledge",
            }
        else:
            safe_replacements = base_replacements

        for pattern, replacement in safe_replacements.items():
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            if compiled_pattern.search(filtered_text):
                old_text = filtered_text
                filtered_text = compiled_pattern.sub(replacement, filtered_text)
                if old_text != filtered_text:
                    applied_filters.append(f"replaced: {pattern}")

        # In relaxed mode, if we didn't actually apply any replacements and no rewrite was needed,
        # keep original text to minimize unnecessary style impact.
        if (
            mode == "relaxed"
            and not applied_filters
            and "complete_rewrite_needed" not in applied_filters
        ):
            return text, ["relaxed_stage_noop"]

        # Clean up extra spaces and punctuation
        filtered_text = re.sub(r"\s+", " ", filtered_text)  # Multiple spaces
        filtered_text = re.sub(
            r"\s+([,.!?])", r"\1", filtered_text
        )  # Space before punctuation
        filtered_text = filtered_text.strip()

        # Capitalize first letter of sentences
        filtered_text = re.sub(
            r"(^|[.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            filtered_text,
        )

        return filtered_text, applied_filters

    def get_anthropomorphism_score(self, text: str) -> float:
        """Calculate how anthropomorphic the text is (0.0 = neutral, 1.0 = very anthropomorphic)."""
        if not text:
            return 0.0

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        anthropomorphic_matches = 0

        # Count pattern matches
        for pattern, _ in self.compiled_patterns:
            matches = len(pattern.findall(text))
            anthropomorphic_matches += matches

        # Check for rewrite triggers (heavily weighted)
        for pattern in self.rewrite_compiled:
            if pattern.search(text):
                anthropomorphic_matches += 5  # Heavy penalty

        # Normalize by word count
        score = min(1.0, anthropomorphic_matches / (word_count / 10))
        return score

    def analyze_text(self, text: str) -> Dict:
        """Analyze text for anthropomorphic content."""
        filtered_text, applied_filters = self.filter_response(text)
        score = self.get_anthropomorphism_score(text)
        needs_rewrite = self.needs_rewrite(text)

        return {
            "original_text": text,
            "filtered_text": filtered_text,
            "anthropomorphism_score": score,
            "needs_rewrite": needs_rewrite,
            "applied_filters": applied_filters,
            "filter_count": len(applied_filters),
            "reduction_ratio": len(text) / max(len(filtered_text), 1) if text else 1.0,
        }
