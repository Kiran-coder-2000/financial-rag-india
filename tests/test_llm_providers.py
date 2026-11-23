from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from src.llm.base import (
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
from src.llm.factory import create_llm_provider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.settings import LLMSettings
from src.llm import validation


def _settings(provider="gemini", model="test-model"):
    return LLMSettings(
        provider=provider,
        model=model,
        api_key="mock-api-key",
        temperature=0.25,
        max_output_tokens=128,
        timeout_seconds=12.0,
    )


def test_factory_selects_only_configured_provider(monkeypatch):
    gemini = Mock(return_value=object())
    groq = Mock(return_value=object())
    monkeypatch.setattr("src.llm.gemini_provider.GeminiProvider", gemini)
    monkeypatch.setattr("src.llm.groq_provider.GroqProvider", groq)

    result = create_llm_provider(_settings("gemini"))

    gemini.assert_called_once()
    groq.assert_not_called()
    assert result is gemini.return_value


@pytest.mark.parametrize(
    ("provider", "adapter"),
    [("gemini", GeminiProvider), ("groq", GroqProvider)],
)
def test_factory_returns_selected_adapter(provider, adapter):
    assert isinstance(create_llm_provider(_settings(provider)), adapter)


def test_provider_constructors_pass_only_selected_credentials_and_timeout(monkeypatch):
    from google import genai
    import groq

    gemini_client = Mock(return_value=object())
    groq_client = Mock(return_value=object())
    monkeypatch.setattr(genai, "Client", gemini_client)
    monkeypatch.setattr(groq, "Groq", groq_client)

    GeminiProvider(_settings())
    GroqProvider(_settings("groq"))

    gemini_client.assert_called_once()
    gemini_options = gemini_client.call_args.kwargs
    assert gemini_options["api_key"] == "mock-api-key"
    assert gemini_options["http_options"].timeout == 12_000
    groq_client.assert_called_once_with(api_key="mock-api-key", timeout=12.0)


def test_gemini_normalizes_text_usage_and_request_options():
    response = SimpleNamespace(
        text='{"ok": true}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            total_token_count=18,
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=response))
    )
    provider = GeminiProvider(_settings(), client=client)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = provider.generate(GenerationRequest(
        user_prompt="Return JSON.",
        system_instruction="Follow the schema.",
        timeout_seconds=4.5,
        json_mode=True,
        response_schema=schema,
    ))

    assert result == GenerationResponse(
        text='{"ok": true}',
        provider="gemini",
        model="test-model",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        finish_reason="STOP",
    )
    call = client.models.generate_content.call_args
    assert call.kwargs["contents"] == "Return JSON."
    config = call.kwargs["config"]
    assert config.system_instruction == "Follow the schema."
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema
    assert config.temperature == 0.25
    assert config.max_output_tokens == 128
    assert config.http_options.timeout == 4_500


def test_gemini_timeout_is_normalized():
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=Mock(side_effect=TimeoutError("secret details"))
        )
    )

    with pytest.raises(LLMProviderTimeoutError) as error:
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "secret details" not in str(error.value)


@pytest.mark.parametrize(
    ("exception", "expected_error"),
    [
        (httpx.ReadTimeout("sensitive timeout body"), LLMProviderTimeoutError),
        (httpx.ConnectError("sensitive network body"), LLMProviderError),
    ],
)
def test_gemini_http_errors_are_sanitized(exception, expected_error):
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=exception))
    )

    with pytest.raises(expected_error) as raised:
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert type(raised.value) is expected_error
    assert "sensitive" not in str(raised.value)


def test_gemini_unknown_provider_response_is_sanitized():
    from google.genai.errors import UnknownApiResponseError

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(
            side_effect=UnknownApiResponseError("sensitive response body")
        ))
    )

    with pytest.raises(LLMProviderError, match="unrecognized provider response") as raised:
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "sensitive response body" not in str(raised.value)


@pytest.mark.parametrize(
    ("code", "generation_request", "expected_error"),
    [
        (
            401,
            GenerationRequest(user_prompt="test"),
            LLMProviderAuthenticationError,
        ),
        (
            429,
            GenerationRequest(user_prompt="test"),
            LLMProviderRateLimitError,
        ),
        (
            400,
            GenerationRequest(user_prompt="test", json_mode=True),
            LLMProviderRequestError,
        ),
    ],
)
def test_gemini_api_errors_are_sanitized(code, generation_request, expected_error):
    from google.genai.errors import APIError

    error = APIError(code, {"error": {"message": "sensitive response content"}})
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=error))
    )

    with pytest.raises(expected_error) as raised:
        GeminiProvider(_settings(), client=client).generate(generation_request)
    assert "sensitive response content" not in str(raised.value)


def test_gemini_explicit_structured_output_capability_error():
    from google.genai.errors import APIError

    error = APIError(400, {
        "error": {
            "status": "INVALID_ARGUMENT",
            "message": "response_json_schema is not supported by this model",
        }
    })
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=error))
    )

    with pytest.raises(LLMProviderCapabilityError) as raised:
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "object"},
            )
        )
    assert "not supported" not in str(raised.value)
    assert "response_json_schema" not in str(raised.value)


@pytest.mark.parametrize(
    ("code", "details"),
    [
        (400, {"error": {"status": "INVALID_ARGUMENT", "message": "Invalid schema: required must be array"}}),
        (400, {"error": {"status": "INVALID_ARGUMENT", "message": "Invalid temperature parameter"}}),
        (404, {"error": {"status": "NOT_FOUND", "message": "model private-model was not found"}}),
        (400, {"error": {"status": "INVALID_ARGUMENT", "message": "Invalid request"}}),
    ],
)
def test_gemini_request_errors_are_not_misclassified_as_capability_errors(code, details):
    from google.genai.errors import APIError

    error = APIError(code, details)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=error))
    )

    with pytest.raises(LLMProviderRequestError) as raised:
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "object"},
            )
        )
    assert type(raised.value) is LLMProviderRequestError
    assert "private-model" not in str(raised.value)
    assert "Invalid schema" not in str(raised.value)


def test_gemini_unsupported_schema_keyword_is_not_capability_error():
    from google.genai.errors import APIError

    error = APIError(400, {
        "error": {
            "status": "INVALID_ARGUMENT",
            "message": "JSON schema has an unsupported keyword: dependentSchemas",
        }
    })
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(side_effect=error))
    )

    with pytest.raises(LLMProviderRequestError):
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "object"},
            )
        )


def test_gemini_invalid_schema_is_rejected_before_provider_request():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock())
    )

    with pytest.raises(LLMProviderRequestError):
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"properties": "must be an object"},
            )
        )
    client.models.generate_content.assert_not_called()


def test_gemini_missing_content_is_rejected():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text=None)))
    )

    with pytest.raises(LLMProviderResponseError, match="no text content"):
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(user_prompt="test")
        )


def test_gemini_malformed_json_content_is_rejected():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(
            text="{malformed",
        )))
    )

    with pytest.raises(LLMProviderResponseError, match="malformed JSON"):
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(user_prompt="test", json_mode=True)
        )


def test_gemini_valid_json_mode_without_schema():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(
            text='{"ok": true}',
        )))
    )

    result = GeminiProvider(_settings(), client=client).generate(
        GenerationRequest(user_prompt="test", json_mode=True)
    )

    assert result.text == '{"ok": true}'


def test_gemini_uses_shared_json_validation(monkeypatch):
    validate = Mock(wraps=validation.validate_json_output)
    monkeypatch.setattr(validation, "validate_json_output", validate)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(
            text='{"ok": true}',
        )))
    )

    GeminiProvider(_settings(), client=client).generate(
        GenerationRequest(user_prompt="test", json_mode=True)
    )

    validate.assert_called_once_with(
        '{"ok": true}',
        json_mode=True,
        response_schema=None,
    )


def test_gemini_invalid_schema_fails_before_provider_call():
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=Mock())
    )

    with pytest.raises(LLMProviderRequestError, match="not a valid JSON Schema"):
        GeminiProvider(_settings(), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "invalid-schema-type"},
            )
        )
    client.models.generate_content.assert_not_called()


def test_groq_normalizes_text_usage_and_json_schema():
    completion = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok": true}'),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    create = Mock(return_value=completion)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    provider = GroqProvider(_settings("groq", "openai/gpt-oss-20b"), client=client)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = provider.generate(GenerationRequest(
        user_prompt="Return JSON.",
        system_instruction="Follow the schema.",
        response_schema=schema,
    ))

    assert result == GenerationResponse(
        text='{"ok": true}',
        provider="groq",
        model="openai/gpt-oss-20b",
        input_tokens=5,
        output_tokens=3,
        total_tokens=8,
        finish_reason="stop",
    )
    kwargs = create.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "Follow the schema."},
        {"role": "user", "content": "Return JSON."},
    ]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "strict": True,
            "schema": schema,
        },
    }
    assert kwargs["max_completion_tokens"] == 128
    assert kwargs["timeout"] == 12.0


def test_groq_json_mode_without_schema_uses_json_object():
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="{}"),
            finish_reason="stop",
        )],
        usage=None,
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    GroqProvider(_settings("groq"), client=client).generate(
        GenerationRequest(user_prompt="test", json_mode=True)
    )

    assert create.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_groq_valid_json_mode_without_schema():
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok": true}'),
            finish_reason="stop",
        )],
        usage=None,
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = GroqProvider(_settings("groq"), client=client).generate(
        GenerationRequest(user_prompt="test", json_mode=True)
    )

    assert result.text == '{"ok": true}'


def test_groq_uses_shared_json_validation(monkeypatch):
    validate = Mock(wraps=validation.validate_json_output)
    monkeypatch.setattr(validation, "validate_json_output", validate)
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"ok": true}'),
            finish_reason="stop",
        )],
        usage=None,
    ))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    GroqProvider(_settings("groq"), client=client).generate(
        GenerationRequest(user_prompt="test", json_mode=True)
    )

    validate.assert_called_once_with(
        '{"ok": true}',
        json_mode=True,
        response_schema=None,
    )


def test_groq_invalid_schema_fails_before_provider_call():
    create = Mock()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderRequestError, match="not a valid JSON Schema"):
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "invalid-schema-type"},
            )
        )
    create.assert_not_called()


def test_groq_timeout_is_normalized():
    create = Mock(side_effect=TimeoutError("sensitive provider message"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderTimeoutError) as error:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "sensitive provider message" not in str(error.value)


def test_groq_malformed_json_content_is_rejected():
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="{malformed"),
                    finish_reason="stop",
                )],
                usage=None,
            )
        )))
    )

    with pytest.raises(LLMProviderResponseError, match="malformed JSON"):
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test", json_mode=True)
        )


def test_groq_rate_limit_is_normalized():
    from groq import RateLimitError

    error = RateLimitError.__new__(RateLimitError)
    Exception.__init__(error, "sensitive provider message")
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderRateLimitError) as raised:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "sensitive provider message" not in str(raised.value)


def test_groq_authentication_error_is_normalized():
    from groq import AuthenticationError

    error = AuthenticationError.__new__(AuthenticationError)
    Exception.__init__(error, "credential text")
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderAuthenticationError) as raised:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "credential text" not in str(raised.value)


def test_groq_response_validation_error_is_sanitized():
    from groq import APIResponseValidationError

    error = APIResponseValidationError.__new__(APIResponseValidationError)
    Exception.__init__(error, "sensitive response body")
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderError, match="Groq request failed") as raised:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test")
        )
    assert "sensitive response body" not in str(raised.value)


def test_groq_structured_output_capability_errors_are_explicit():
    from groq import BadRequestError

    error = BadRequestError.__new__(BadRequestError)
    Exception.__init__(error, "sensitive model capability details")
    error.body = {
        "error": {
            "code": "unsupported_value",
            "message": "response_format json_schema is not supported by this model",
        }
    }
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderCapabilityError) as raised:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(user_prompt="test", json_mode=True)
        )
    assert "sensitive model capability details" not in str(raised.value)


def test_groq_malformed_structured_output_request_is_not_a_capability_error():
    from groq import BadRequestError

    error = BadRequestError.__new__(BadRequestError)
    Exception.__init__(error, "sensitive invalid schema details")
    error.body = {
        "error": {
            "code": "json_validate_failed",
            "message": "Invalid schema: required must be an array",
        }
    }
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with pytest.raises(LLMProviderRequestError) as raised:
        GroqProvider(_settings("groq"), client=client).generate(
            GenerationRequest(
                user_prompt="test",
                response_schema={"type": "object", "required": "invalid"},
            )
        )
    assert not isinstance(raised.value, LLMProviderCapabilityError)
    assert "sensitive invalid schema details" not in str(raised.value)


def test_groq_missing_or_non_text_content_is_rejected():
    for content in (None, [{"type": "text", "text": "unsupported shape"}]):
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(
                return_value=SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=content),
                        finish_reason="stop",
                    )],
                    usage=None,
                )
            )))
        )
        with pytest.raises(LLMProviderResponseError):
            GroqProvider(_settings("groq"), client=client).generate(
                GenerationRequest(user_prompt="test")
            )
