# pmm/embodiment.py
"""Embodiment-aware bridges for PMM: one mind, multiple bodies."""

from __future__ import annotations
from typing import Protocol, Dict
import re
from .ngram_ban import NGramBanSystem


class RenderAdapter(Protocol):
    """Protocol for per-family embodiment adapters."""

    def render(self, canonical_text: str) -> str:
        """Render canonical text for this model family."""
        ...


class GPTAdapter:
    """Adapter for GPT family models (OpenAI)."""

    def __init__(self, ngram_ban: NGramBanSystem):
        self.ngram_ban = ngram_ban

    def render(self, text: str) -> str:
        """Render for GPT family: moderate hedging, clean structure."""
        # Apply moderate hedging/softening
        text = self._soften_hedges(text, target="moderate")

        # Apply family-specific n-gram bans
        text, _ = self.ngram_ban.postprocess_style(text, "gpt")

        return text

    def _soften_hedges(self, text: str, target: str) -> str:
        """Apply stance normalization."""
        if target == "moderate":
            # Add some hedging to overly direct statements
            text = re.sub(r"\bI will\b", "I plan to", text)
            text = re.sub(r"\bI must\b", "I should", text)
        return text


class GemmaAdapter:
    """Adapter for Gemma family models."""

    def __init__(self, ngram_ban: NGramBanSystem):
        self.ngram_ban = ngram_ban

    def render(self, text: str) -> str:
        """Render for Gemma family: shorter sentences, direct style."""
        # Shorten sentences for Gemma's style
        text = self._shorten_sentences(text, avg_len=16)

        # Apply family-specific n-gram bans
        text, _ = self.ngram_ban.postprocess_style(text, "gemma")

        return text

    def _shorten_sentences(self, text: str, avg_len: int) -> str:
        """Break long sentences for more direct communication."""
        sentences = text.split(". ")
        shortened = []

        for sentence in sentences:
            if len(sentence.split()) > avg_len:
                # Simple sentence breaking at conjunctions
                sentence = sentence.replace(", and ", ". ")
                sentence = sentence.replace(", but ", ". However, ")
            shortened.append(sentence)

        return ". ".join(shortened)


class LlamaAdapter:
    """Adapter for Llama family models."""

    def __init__(self, ngram_ban: NGramBanSystem):
        self.ngram_ban = ngram_ban

    def render(self, text: str) -> str:
        """Render for Llama family: balanced style similar to Gemma."""
        # Use similar approach to Gemma for now
        text = self._shorten_sentences(text, avg_len=18)

        # Apply family-specific n-gram bans
        text, _ = self.ngram_ban.postprocess_style(text, "llama")

        return text

    def _shorten_sentences(self, text: str, avg_len: int) -> str:
        """Break long sentences for more direct communication."""
        sentences = text.split(". ")
        shortened = []

        for sentence in sentences:
            if len(sentence.split()) > avg_len:
                sentence = sentence.replace(", and ", ". ")
                sentence = sentence.replace(", but ", ". However, ")
            shortened.append(sentence)

        return ". ".join(shortened)


class QwenAdapter:
    """Adapter for Qwen family models."""

    def __init__(self, ngram_ban: NGramBanSystem):
        self.ngram_ban = ngram_ban

    def render(self, text: str) -> str:
        """Render for Qwen family: placeholder using GPT-like style."""
        # Use GPT-like approach until tuned
        text = self._soften_hedges(text, target="moderate")

        # Apply family-specific n-gram bans
        text, _ = self.ngram_ban.postprocess_style(text, "qwen")

        return text

    def _soften_hedges(self, text: str, target: str) -> str:
        """Apply stance normalization."""
        if target == "moderate":
            text = re.sub(r"\bI will\b", "I plan to", text)
            text = re.sub(r"\bI must\b", "I should", text)
        return text


def extract_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    name_lower = model_name.lower()

    if "gpt" in name_lower or "chatgpt" in name_lower:
        return "gpt"
    elif "gemma" in name_lower:
        return "gemma"
    elif "llama" in name_lower:
        return "llama"
    elif "qwen" in name_lower:
        return "qwen"
    elif "claude" in name_lower:
        return "claude"
    elif "deepseek" in name_lower:
        return "deepseek"
    else:
        return "unknown"


def create_family_adapters(ngram_ban: NGramBanSystem) -> Dict[str, RenderAdapter]:
    """Create adapter instances for all model families."""
    return {
        "gpt": GPTAdapter(ngram_ban),
        "gemma": GemmaAdapter(ngram_ban),
        "llama": LlamaAdapter(ngram_ban),
        "qwen": QwenAdapter(ngram_ban),
        "claude": GPTAdapter(ngram_ban),  # placeholder
        "deepseek": LlamaAdapter(ngram_ban),  # placeholder
        "unknown": GPTAdapter(ngram_ban),  # safe default
    }


def get_adapter(family: str, adapters: Dict[str, RenderAdapter]) -> RenderAdapter:
    """Get adapter for model family with fallback."""
    return adapters.get(family, adapters["gpt"])
