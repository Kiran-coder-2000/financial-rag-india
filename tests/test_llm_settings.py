import pytest

from src.llm.settings import (
    DEFAULT_GROQ_MODEL,
    LLMConfigurationError,
    load_settings,
)


def test_gemini_configuration_and_provider_specific_model():
    settings = load_settings({
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "test-gemini-key",
        "GEMINI_MODEL": "gemini-test-model",
        "GROQ_API_KEY": "",
    })

    assert settings.provider == "gemini"
    assert settings.model == "gemini-test-model"
    assert settings.api_key == "test-gemini-key"


def test_groq_configuration_uses_default_model_and_only_selected_key():
    settings = load_settings({
        "LLM_PROVIDER": "groq",
        "GROQ_API_KEY": "test-groq-key",
        "GEMINI_API_KEY": "",
    })

    assert settings.provider == "groq"
    assert settings.model == DEFAULT_GROQ_MODEL
    assert settings.api_key == "test-groq-key"


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_global_model_overrides_provider_model(provider):
    env = {
        "LLM_PROVIDER": provider,
        "LLM_MODEL": "global-model",
        "GEMINI_MODEL": "gemini-specific",
        "GEMINI_API_KEY": "gemini-test-key",
        "GROQ_MODEL": "groq-specific",
        "GROQ_API_KEY": "groq-test-key",
    }

    assert load_settings(env).model == "global-model"


@pytest.mark.parametrize(
    ("provider", "selected_key"),
    [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")],
)
def test_missing_selected_provider_key_does_not_expose_secret(provider, selected_key):
    env = {
        "LLM_PROVIDER": provider,
        "GEMINI_MODEL": "gemini-model",
        "GROQ_MODEL": "groq-model",
        "GEMINI_API_KEY": "unused-secret",
        "GROQ_API_KEY": "unused-secret",
    }
    env[selected_key] = ""

    with pytest.raises(LLMConfigurationError) as error:
        load_settings(env)

    assert selected_key in str(error.value)
    assert "unused-secret" not in str(error.value)


def test_gemini_model_is_required_unless_global_override_is_set():
    with pytest.raises(LLMConfigurationError, match="GEMINI_MODEL"):
        load_settings({"GEMINI_API_KEY": "test-key"})

    settings = load_settings({
        "LLM_MODEL": "global-gemini-model",
        "GEMINI_API_KEY": "test-key",
    })
    assert settings.model == "global-gemini-model"


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_unused_provider_key_is_not_required(provider):
    env = {"LLM_PROVIDER": provider}
    env["GEMINI_MODEL"] = "gemini-model"
    env["GROQ_MODEL"] = "groq-model"
    env["GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"] = "selected-key"
    env["GROQ_API_KEY" if provider == "gemini" else "GEMINI_API_KEY"] = ""

    assert load_settings(env).provider == provider


def test_invalid_provider_is_rejected():
    with pytest.raises(LLMConfigurationError, match="Unsupported LLM_PROVIDER"):
        load_settings({"LLM_PROVIDER": "other"})


@pytest.mark.parametrize("temperature", ["nan", "inf", "-0.1", "2.1", "invalid"])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(LLMConfigurationError, match="LLM_TEMPERATURE"):
        load_settings({
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "LLM_TEMPERATURE": temperature,
        })


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "invalid"])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT_SECONDS"):
        load_settings({
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "LLM_TIMEOUT_SECONDS": timeout,
        })


@pytest.mark.parametrize("max_tokens", ["0", "-4", "1.5", "invalid"])
def test_invalid_max_output_tokens_is_rejected(max_tokens):
    with pytest.raises(LLMConfigurationError, match="LLM_MAX_OUTPUT_TOKENS"):
        load_settings({
            "GEMINI_API_KEY": "test-key",
            "GEMINI_MODEL": "test-model",
            "LLM_MAX_OUTPUT_TOKENS": max_tokens,
        })


def test_settings_repr_does_not_include_api_key():
    settings = load_settings({
        "GEMINI_API_KEY": "do-not-print-this",
        "GEMINI_MODEL": "test-model",
    })

    assert "do-not-print-this" not in repr(settings)
