"""Lazy construction of the configured LLM provider."""

from .base import LLMProvider
from .settings import LLMSettings, load_settings


def create_llm_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Create only the selected provider; never fall back to another vendor."""
    selected = settings or load_settings()
    if selected.provider == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(selected)
    if selected.provider == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(selected)
    raise ValueError(f"Unsupported LLM provider {selected.provider!r}.")
