"""Groq adapter for the provider-neutral generation interface."""

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


class GroqProvider:
    provider_name = "groq"

    def __init__(self, settings: LLMSettings, client: Any = None):
        self.model = settings.model
        self._settings = settings
        if client is None:
            try:
                from groq import Groq
            except ImportError as error:
                raise LLMProviderError(
                    "Groq SDK is unavailable; install the groq dependency."
                ) from error
            client = Groq(
                api_key=settings.api_key,
                timeout=settings.timeout_seconds,
            )
        self._client = client

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        from groq import (
            APIError,
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            RateLimitError,
        )

        if request.response_schema is not None:
            validation.validate_response_schema(request.response_schema)

        timeout = request.timeout_seconds or self._settings.timeout_seconds
        messages = []
        if request.system_instruction is not None:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.user_prompt})
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self._settings.temperature
                if request.temperature is None
                else request.temperature
            ),
            "timeout": timeout,
        }
        max_tokens = (
            self._settings.max_output_tokens
            if request.max_output_tokens is None
            else request.max_output_tokens
        )
        if max_tokens is not None:
            arguments["max_completion_tokens"] = max_tokens
        if request.response_schema is not None:
            arguments["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": dict(request.response_schema),
                },
            }
        elif request.json_mode:
            arguments["response_format"] = {"type": "json_object"}

        try:
            completion = self._client.chat.completions.create(**arguments)
        except APITimeoutError:
            raise LLMProviderTimeoutError("Groq request timed out.") from None
        except TimeoutError:
            raise LLMProviderTimeoutError("Groq request timed out.") from None
        except AuthenticationError:
            raise LLMProviderAuthenticationError(
                "Groq rejected the configured credentials."
            ) from None
        except RateLimitError:
            raise LLMProviderRateLimitError("Groq rate limit was reached.") from None
        except BadRequestError as error:
            if request.json_mode or request.response_schema is not None:
                if _is_structured_output_capability_error(error):
                    raise LLMProviderCapabilityError(
                        "The configured Groq model does not support the requested structured-output capability."
                    ) from None
                raise LLMProviderRequestError(
                    "Groq rejected the structured-output request; check the schema and request parameters."
                ) from None
            raise LLMProviderError(
                "Groq rejected the request or configured model."
            ) from None
        except APIConnectionError:
            raise LLMProviderError("Could not connect to Groq.") from None
        except APIError as error:
            code = getattr(error, "status_code", None)
            raise LLMProviderError(
                f"Groq request failed (status {code if isinstance(code, int) else 'unknown'})."
            ) from None

        choices = getattr(completion, "choices", None)
        if not choices:
            raise LLMProviderResponseError("Groq returned no completion choices.")
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMProviderResponseError("Groq returned no text content.")
        validation.validate_json_output(
            text,
            json_mode=request.json_mode,
            response_schema=request.response_schema,
        )

        usage = getattr(completion, "usage", None)
        reason = getattr(choices[0], "finish_reason", None)
        return GenerationResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=_token_count(usage, "prompt_tokens"),
            output_tokens=_token_count(usage, "completion_tokens"),
            total_tokens=_token_count(usage, "total_tokens"),
            finish_reason=str(reason) if reason is not None else None,
        )


def _token_count(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None) if usage is not None else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_structured_output_capability_error(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    details = body.get("error", body)
    if not isinstance(details, dict):
        return False
    message = details.get("message")
    code = details.get("code")
    searchable = " ".join(
        value.casefold()
        for value in (message, code)
        if isinstance(value, str)
    )
    mentions_output_format = any(
        term in searchable
        for term in ("json_schema", "response_format", "structured output")
    )
    explicitly_unsupported = any(
        term in searchable
        for term in ("not supported", "unsupported", "does not support")
    )
    return mentions_output_format and explicitly_unsupported
