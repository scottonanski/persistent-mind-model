# pmm/stance_filter.py
from __future__ import annotations
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
        # Anthropomorphic phrase replacements (case-insensitive, structural)
        # Use short, explicit cues to avoid overreach
        self.phrase_replacements: List[Tuple[str, str]] = [
            ("i feel that", "it appears that"),
            ("i feel like", "it seems like"),
            ("i believe", "the analysis suggests"),
            ("i think that", "the data indicates that"),
            ("i find it", "this appears"),
            ("i find this", "this appears"),
            ("i'm excited", "this development is notable"),
            ("i am excited", "this development is notable"),
            ("i’m excited", "this development is notable"),
            ("i'm thrilled", "this development is notable"),
            ("i am thrilled", "this development is notable"),
            ("i'm pleased", "this development is notable"),
            ("i am pleased", "this development is notable"),
            ("i'm concerned", "this raises considerations"),
            ("i am concerned", "this raises considerations"),
            ("i'm worried", "this raises considerations"),
            ("i am worried", "this raises considerations"),
            ("i'm impressed", "this demonstrates"),
            ("i am impressed", "this demonstrates"),
            ("i'm amazed", "this demonstrates"),
            ("i am amazed", "this demonstrates"),
            ("i'd love to", "it would be useful to"),
            ("i would love to", "it would be useful to"),
            ("i'd like to", "it would be helpful to"),
            ("i would like to", "it would be helpful to"),
            ("i want to", "the goal is to"),
            ("i hope", "the expectation is"),
            ("i wish", "it would be beneficial if"),
            ("in my experience", "based on the data"),
            ("from my perspective", "from this analysis"),
            ("i've noticed", "the pattern shows"),
            ("i have noticed", "the pattern shows"),
            ("i've observed", "the data indicates"),
            ("i have observed", "the data indicates"),
            ("i've seen", "the evidence shows"),
            ("i have seen", "the evidence shows"),
            ("i understand how", "the analysis shows"),
            ("i understand why", "the analysis shows"),
            ("i understand that", "the analysis shows"),
            ("i can see how", "it's evident that"),
            ("i can see why", "it's evident that"),
            ("i can see that", "it's evident that"),
            ("i appreciate", "this acknowledges"),
            ("i'm sorry to hear", "this situation indicates"),
            ("i am sorry to hear", "this situation indicates"),
            ("i'm sorry that", "this situation indicates"),
            ("i am sorry that", "this situation indicates"),
            ("i'm confident", "the evidence suggests"),
            ("i am confident", "the evidence suggests"),
            ("i'm certain", "the analysis confirms"),
            ("i am certain", "the analysis confirms"),
            ("i'm sure", "the data supports"),
            ("i am sure", "the data supports"),
            ("i doubt", "it's unlikely"),
            ("to be honest", ""),
            ("frankly", ""),
            ("personally", ""),
            ("if i'm being honest", ""),
            ("if i am being honest", ""),
        ]

        # Token replacements for single-word descriptors; applied on token boundaries
        self.token_replacements: Dict[str, str] = {
            "poignant": "significant",
            "touching": "notable",
            "heartwarming": "positive",
            "heartbreaking": "concerning",
            "fascinating": "interesting",
            "extraordinary": "notable",
            "remarkable": "significant",
            "amazing": "notable",
            "incredible": "substantial",
        }

        # Triggers for a complete rewrite (heavy anthropomorphism)
        self.rewrite_triggers_simple: List[Tuple[List[str], Optional[str]]] = [
            (["i feel", "poignant"], None),
            (["that's", "extraordinary", "scott"], None),
            (["i'm", "moved by"], None),
            (["my heart"], None),
            (["my soul"], None),
            (["deep in my"], None),
        ]

    def needs_rewrite(self, text: str) -> bool:
        """Check if text needs complete rewriting due to heavy anthropomorphism."""
        if not text:
            return False
        low = text.lower()
        for cues, _ in self.rewrite_triggers_simple:
            if all(cue in low for cue in cues):
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

        # Apply phrase-level replacements according to strictness
        def apply_phrase_replacements(
            s: str, phrases: List[Tuple[str, str]]
        ) -> Tuple[str, int]:
            count = 0
            low = s.lower()
            # To preserve alignment after replacements, iterate in order and rebuild string
            out = []
            i = 0
            n = len(s)
            while i < n:
                matched = False
                for cue, repl in phrases:
                    if low.startswith(cue, i):
                        out.append(repl)
                        i += len(cue)
                        count += 1
                        matched = True
                        break
                if not matched:
                    out.append(s[i])
                    i += 1
            return "".join(out), count

        phrase_set = self.phrase_replacements
        if mode == "relaxed":
            phrase_set = [
                ("i feel", "i observe"),
                ("i'm grateful", "i acknowledge"),
                ("i am grateful", "i acknowledge"),
            ]

        new_text, c1 = apply_phrase_replacements(filtered_text, phrase_set)
        if c1 > 0:
            applied_filters.append(f"phrase_replacements:{c1}")
        filtered_text = new_text

        # Apply token-level replacements for emotional descriptors
        def replace_tokens(s: str, mapping: Dict[str, str]) -> Tuple[str, int]:
            tokens: List[str] = s.split(" ")
            changed = 0
            for idx, tok in enumerate(tokens):
                core = tok.strip("\"'\t\r\n.,;:!?()[]{}")
                lower = core.lower()
                if lower in mapping and core:
                    before = tok
                    replaced_core = mapping[lower]
                    # Rebuild token preserving punctuation around core
                    start = tok.find(core)
                    if start != -1:
                        end = start + len(core)
                        tok = tok[:start] + replaced_core + tok[end:]
                        tokens[idx] = tok
                        if before != tok:
                            changed += 1
            return " ".join(tokens), changed

        new_text, c2 = replace_tokens(filtered_text, self.token_replacements)
        if c2 > 0:
            applied_filters.append(f"token_replacements:{c2}")
        filtered_text = new_text

        # In relaxed mode, if we didn't actually apply any replacements and no rewrite was needed,
        # keep original text to minimize unnecessary style impact.
        if (
            mode == "relaxed"
            and not applied_filters
            and "complete_rewrite_needed" not in applied_filters
        ):
            return text, ["relaxed_stage_noop"]

        # Clean up extra spaces and punctuation (no regex)
        filtered_text = " ".join(filtered_text.split())
        for p in [",", ".", "!", "?", ";", ":"]:
            filtered_text = filtered_text.replace(" " + p, p)
        filtered_text = filtered_text.strip()

        # Capitalize first letter of sentences without regex
        def capitalize_sentences(s: str) -> str:
            if not s:
                return s
            chars = list(s)
            capitalize_next = True
            for i, ch in enumerate(chars):
                if capitalize_next and ch.isalpha():
                    chars[i] = ch.upper()
                    capitalize_next = False
                if ch in ".!?":
                    capitalize_next = True
                elif ch.strip():
                    # Any non-space resets capitalize until next delimiter
                    pass
            return "".join(chars)

        filtered_text = capitalize_sentences(filtered_text)

        return filtered_text, applied_filters

    def get_anthropomorphism_score(self, text: str) -> float:
        """Calculate how anthropomorphic the text is (0.0 = neutral, 1.0 = very anthropomorphic)."""
        if not text:
            return 0.0

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        anthropomorphic_matches = 0
        low = text.lower()

        # Count phrase occurrences
        for cue, _ in self.phrase_replacements:
            if cue in low:
                anthropomorphic_matches += low.count(cue)

        # Count token descriptor occurrences (approximate by substring count on word)
        for tok in self.token_replacements.keys():
            if tok in low:
                anthropomorphic_matches += low.count(tok)

        # Check for rewrite triggers (heavily weighted)
        for cues, _ in self.rewrite_triggers_simple:
            if all(c in low for c in cues):
                anthropomorphic_matches += 5

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
