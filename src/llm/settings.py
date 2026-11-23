"""Validated environment configuration for supported LLM providers."""

import math
import os
from dataclasses import dataclass, field
from typing import Mapping


SUPPORTED_PROVIDERS = frozenset({"gemini", "groq"})
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMConfigurationError(ValueError):
    """Raised when provider configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str = field(repr=False)
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


def _positive_float(name: str, raw: str, *, allow_zero: bool = False) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise LLMConfigurationError(f"{name} must be a number.") from error
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise LLMConfigurationError(f"{name} must be a finite {requirement} number.")
    return value


def _load_environment() -> Mapping[str, str]:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return os.environ
    load_dotenv()
    return os.environ


def load_settings(env: Mapping[str, str] | None = None) -> LLMSettings:
    """Load settings and validate only the selected provider's credentials."""
    values = _load_environment() if env is None else env

    provider = (values.get("LLM_PROVIDER") or "gemini").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER {provider!r}; choose one of: {supported}."
        )

    provider_model_key, api_key_name = (
        ("GEMINI_MODEL", "GEMINI_API_KEY")
        if provider == "gemini"
        else ("GROQ_MODEL", "GROQ_API_KEY")
    )
    global_model = (values.get("LLM_MODEL") or "").strip()
    provider_model = (values.get(provider_model_key) or "").strip()
    if provider == "groq" and not provider_model:
        provider_model = DEFAULT_GROQ_MODEL
    model = global_model or provider_model
    if not model:
        raise LLMConfigurationError(
            f"Missing model configuration: set LLM_MODEL or {provider_model_key}."
        )

    api_key = (values.get(api_key_name) or "").strip()
    if not api_key:
        raise LLMConfigurationError(
            f"Missing {api_key_name} for selected provider {provider!r}."
        )

    raw_temperature = (values.get("LLM_TEMPERATURE") or "").strip()
    temperature = (
        _positive_float("LLM_TEMPERATURE", raw_temperature, allow_zero=True)
        if raw_temperature
        else DEFAULT_TEMPERATURE
    )
    if temperature > 2.0:
        raise LLMConfigurationError("LLM_TEMPERATURE must not exceed 2.")

    raw_max_tokens = (values.get("LLM_MAX_OUTPUT_TOKENS") or "").strip()
    max_output_tokens = None
    if raw_max_tokens:
        try:
            max_output_tokens = int(raw_max_tokens)
        except ValueError as error:
            raise LLMConfigurationError(
                "LLM_MAX_OUTPUT_TOKENS must be a positive integer."
            ) from error
        if max_output_tokens <= 0:
            raise LLMConfigurationError(
                "LLM_MAX_OUTPUT_TOKENS must be a positive integer."
            )

    raw_timeout = (values.get("LLM_TIMEOUT_SECONDS") or "").strip()
    timeout_seconds = (
        _positive_float("LLM_TIMEOUT_SECONDS", raw_timeout)
        if raw_timeout
        else DEFAULT_TIMEOUT_SECONDS
    )

    return LLMSettings(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
    )
