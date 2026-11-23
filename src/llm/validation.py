"""Provider-independent validation for generated JSON responses."""

import json
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .base import LLMProviderRequestError, LLMProviderResponseError


def validate_response_schema(response_schema: Mapping[str, Any]) -> None:
    """Reject invalid JSON Schemas without exposing schema contents."""
    if not isinstance(response_schema, Mapping):
        raise LLMProviderRequestError("response_schema must be a valid JSON Schema object.")
    try:
        Draft202012Validator.check_schema(dict(response_schema))
    except (SchemaError, TypeError, ValueError):
        raise LLMProviderRequestError("response_schema is not a valid JSON Schema.") from None


def validate_json_output(
    text: str,
    *,
    json_mode: bool = False,
    response_schema: Mapping[str, Any] | None = None,
) -> Any:
    """Parse JSON and optionally validate it against a JSON Schema."""
    if not json_mode and response_schema is None:
        return None

    try:
        instance = json.loads(
            text,
            parse_constant=_reject_non_json_constant,
        )
    except (TypeError, ValueError):
        raise LLMProviderResponseError(
            "Provider returned malformed JSON for a structured-output request."
        ) from None

    if response_schema is not None:
        validate_response_schema(response_schema)
        try:
            Draft202012Validator(dict(response_schema)).validate(instance)
        except ValidationError:
            raise LLMProviderResponseError(
                "Provider returned JSON that does not match the requested schema."
            ) from None
    return instance


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON value.")
