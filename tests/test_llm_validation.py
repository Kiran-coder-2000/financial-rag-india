import pytest

from src.llm.base import LLMProviderRequestError, LLMProviderResponseError
from src.llm.validation import validate_json_output, validate_response_schema


def test_json_mode_validates_syntax_without_a_schema():
    assert validate_json_output('{"value": 7}', json_mode=True) == {"value": 7}


def test_json_mode_rejects_malformed_json_without_exposing_response():
    raw_response = '{"private": "do-not-include-this"'

    with pytest.raises(LLMProviderResponseError) as error:
        validate_json_output(raw_response, json_mode=True)

    assert "do-not-include-this" not in str(error.value)
    assert "malformed JSON" in str(error.value)


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
def test_json_mode_rejects_non_json_numeric_constants(text):
    with pytest.raises(LLMProviderResponseError, match="malformed JSON"):
        validate_json_output(text, json_mode=True)


def test_valid_json_matching_schema_is_returned():
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }

    assert validate_json_output(
        '{"count": 3}',
        response_schema=schema,
    ) == {"count": 3}


@pytest.mark.parametrize(
    "response",
    ['{"other": 1}', '{"count": "three"}', '{"count": 3, "other": true}'],
)
def test_valid_json_violating_schema_is_rejected_without_response_text(response):
    schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
        "additionalProperties": False,
    }

    with pytest.raises(LLMProviderResponseError) as error:
        validate_json_output(response, response_schema=schema)

    assert response not in str(error.value)
    assert "does not match" in str(error.value)


def test_schema_is_validated_before_generation():
    invalid_schema = {
        "type": "object",
        "properties": {"amount": {"type": "not-a-json-schema-type"}},
    }

    with pytest.raises(LLMProviderRequestError) as error:
        validate_response_schema(invalid_schema)

    assert "not-a-json-schema-type" not in str(error.value)


def test_invalid_schema_does_not_expose_schema_details():
    sensitive_schema = {
        "type": "object",
        "properties": {"private-property": {"type": "invalid-type"}},
    }

    with pytest.raises(LLMProviderRequestError) as error:
        validate_json_output("{}", response_schema=sensitive_schema)

    assert "private-property" not in str(error.value)
    assert "invalid-type" not in str(error.value)
