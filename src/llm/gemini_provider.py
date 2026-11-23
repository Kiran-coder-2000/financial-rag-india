"""Google GenAI adapter for the provider-neutral generation interface."""

from typing import Any

from .base import (
    GenerationRequest,
    GenerationResponse,
    LLMProviderAuthenticationError,
    LLMProviderCapabilityError,
    LLMProviderError,
    LLMProviderRateLimitError,
    LLMProviderRequestError,
    LLMProviderResponseError,
    LLMProviderTimeoutError,
)
from .settings import LLMSettings
from . import validation


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, settings: LLMSettings, client: Any = None):
        self.model = settings.model
        self._settings = settings
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise LLMProviderError(
                    "Google GenAI SDK is unavailable; install the google-genai dependency."
                ) from error
            client = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(
                    timeout=round(settings.timeout_seconds * 1000)
                ),
            )
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        import httpx
        from google.genai import errors as genai_errors
        from google.genai import types

        if request.response_schema is not None:
            validation.validate_response_schema(request.response_schema)

        timeout = request.timeout_seconds or self._settings.timeout_seconds
        config_values: dict[str, Any] = {
            "temperature": (
                self._settings.temperature
                if request.temperature is None
                else request.temperature
            ),
            "http_options": types.HttpOptions(timeout=round(timeout * 1000)),
        }
        max_tokens = (
            self._settings.max_output_tokens
            if request.max_output_tokens is None
            else request.max_output_tokens
        )
        if max_tokens is not None:
            config_values["max_output_tokens"] = max_tokens
        if request.system_instruction is not None:
            config_values["system_instruction"] = request.system_instruction
        if request.json_mode or request.response_schema is not None:
            config_values["response_mime_type"] = "application/json"
        if request.response_schema is not None:
            config_values["response_json_schema"] = dict(request.response_schema)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=request.user_prompt,
                config=types.GenerateContentConfig(**config_values),
            )
        except (TimeoutError, ConnectionError) as error:
            raise LLMProviderTimeoutError("Gemini request timed out or lost its connection.") from None
        except genai_errors.APIError as error:
            code = getattr(error, "code", None)
            if code in {401, 403}:
                raise LLMProviderAuthenticationError(
                    "Gemini rejected the configured credentials."
                ) from None
            if code == 429:
                raise LLMProviderRateLimitError("Gemini rate limit was reached.") from None
            if (
                request.json_mode or request.response_schema is not None
            ) and _is_unsupported_structured_output_error(error):
                raise LLMProviderCapabilityError(
                    "The configured Gemini model does not support the requested structured-output capability."
                ) from None
            if code in {400, 404, 422}:
                raise LLMProviderRequestError(
                    f"Gemini rejected the request (status {code})."
                ) from None
            raise LLMProviderError(
                f"Gemini request failed (status {code if isinstance(code, int) else 'unknown'})."
            ) from None
        except httpx.TimeoutException:
            raise LLMProviderTimeoutError("Gemini request timed out.") from None
        except httpx.HTTPError:
            raise LLMProviderError("Gemini network request failed.") from None
        except genai_errors.UnknownApiResponseError:
            raise LLMProviderError("Gemini returned an unrecognized provider response.") from None

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderResponseError("Gemini returned no text content.")
        validation.validate_json_output(
            text,
            json_mode=request.json_mode,
            response_schema=request.response_schema,
        )

        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = str(getattr(reason, "name", reason)) if reason is not None else None
        return GenerationResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=_token_count(usage, "prompt_token_count"),
            output_tokens=_token_count(usage, "candidates_token_count"),
            total_tokens=_token_count(usage, "total_token_count"),
            finish_reason=finish_reason,
        )


def _token_count(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_unsupported_structured_output_error(error: Exception) -> bool:
    """Classify only provider messages that explicitly reject the capability."""
    message = getattr(error, "message", None)
    status = getattr(error, "status", None)
    searchable = " ".join(
        value.casefold()
        for value in (message, status)
        if isinstance(value, str)
    )
    capability_terms = (
        "structured output",
        "json schema",
        "json_schema",
        "response schema",
        "response_schema",
        "response_mime_type",
    )
    explicit_capability_phrases = (
        "not supported by this model",
        "not supported for this model",
        "not supported by the model",
        "model does not support",
        "does not support structured output",
        "structured output is not supported",
        "structured output is unsupported",
        "does not support json schema",
        "json schema is not supported by the model",
        "not implemented for this model",
    )
    return (
        "unimplemented" in searchable
        or (
            any(term in searchable for term in capability_terms)
            and any(term in searchable for term in explicit_capability_phrases)
        )
    )
