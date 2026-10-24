"""Validated schemas for generated chunk keyword metadata."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANNOTATION_VERSION = "1.0"


class KeywordSuggestion(BaseModel):
    """Strict model output; source metadata is attached by code, never by Gemini."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    keywords: list[str] = Field(min_length=1, max_length=15)
    description: str = Field(min_length=1, max_length=800)

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("keywords must not be empty strings")
        return values


class ChunkKeywordAnnotation(BaseModel):
    """A generated annotation plus exact provenance copied from its source chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source_text_sha256: str = Field(min_length=64, max_length=64)
    source_metadata: dict
    company: str | None = None
    document: str | None = None
    financial_year: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_type: str | None = None
    table_id: str | None = None
    table_heading: str | None = None
    keywords: list[str] = Field(default_factory=list, max_length=15)
    description: str | None = None
    annotation_version: str = ANNOTATION_VERSION
    annotation_provider: str = "gemini"
    annotation_model: str = Field(min_length=1)
    annotation_status: Literal["success", "failed", "skipped"]
    annotation_error: str | None = None

    @model_validator(mode="after")
    def validate_status_content(self):
        if self.annotation_status == "success":
            if not self.keywords or not self.description:
                raise ValueError("successful annotations require keywords and a description")
            if self.annotation_error is not None:
                raise ValueError("successful annotations cannot contain an error")
        elif self.keywords or self.description is not None:
            raise ValueError("failed or skipped annotations cannot contain generated content")
        return self