"""Provider-neutral text generation types and errors."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


class LLMProviderError(RuntimeError):
    """Base class for sanitized provider and response failures."""


class LLMProviderTimeoutError(LLMProviderError):
    """The provider request timed out."""


class LLMProviderAuthenticationError(LLMProviderError):
    """The provider rejected the configured credentials."""


class LLMProviderRateLimitError(LLMProviderError):
    """The provider rate limit was reached."""


class LLMProviderCapabilityError(LLMProviderError):
    """The selected model or provider does not support a requested capability."""


class LLMProviderRequestError(LLMProviderError):
    """The provider rejected the request or one of its parameters."""


class LLMProviderResponseError(LLMProviderError):
    """The provider returned an unusable response."""


@dataclass(frozen=True)
class GenerationRequest:
    """One non-streaming text-generation request."""

    user_prompt: str
    system_instruction: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None
    json_mode: bool = False
    response_schema: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str) or not self.user_prompt.strip():
            raise ValueError("user_prompt must be a non-empty string.")
        if self.system_instruction is not None and not isinstance(self.system_instruction, str):
            raise TypeError("system_instruction must be a string or None.")
        if self.temperature is not None:
            if (
                isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(self.temperature)
                or not 0 <= self.temperature <= 2
            ):
                raise ValueError("temperature must be finite and between 0 and 2.")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None.")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive.")
        if not isinstance(self.json_mode, bool):
            raise TypeError("json_mode must be a boolean.")
        if self.response_schema is not None and not isinstance(self.response_schema, Mapping):
            raise TypeError("response_schema must be a mapping or None.")


@dataclass(frozen=True)
class GenerationResponse:
    """Normalized text and optional usage metadata from a provider."""

    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface implemented by supported provider adapters."""

    provider_name: str
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text and return provider-neutral response data."""
