"""Validated schemas for generated chunk keyword metadata."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ANNOTATION_VERSION = "2.0"


class KeywordSuggestion(BaseModel):
    """Strict model output; source metadata is attached by code, never by Gemini."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    keywords: list[str] = Field(max_length=15)
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
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be whitespace-only")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.source_text_sha256):
            raise ValueError("source_text_sha256 must be exactly 64 hexadecimal characters")
        if any(not keyword.strip() for keyword in self.keywords):
            raise ValueError("keywords must not be empty strings")
        normalized_keywords = [re.sub(r"\s+", " ", keyword.strip()).casefold() for keyword in self.keywords]
        if len(set(normalized_keywords)) != len(normalized_keywords):
            raise ValueError("keywords must not contain duplicates")
        if self.annotation_status == "success":
            if not self.keywords or not self.description or not self.description.strip():
                raise ValueError("successful annotations require keywords and a description")
            if self.annotation_error is not None:
                raise ValueError("successful annotations cannot contain an error")
        elif self.annotation_status == "skipped":
            if (
                self.keywords
                or self.description is not None
                or self.annotation_error is not None
            ):
                raise ValueError("skipped annotations require no keywords, description, or error")
        else:
            if self.keywords or self.description is not None:
                raise ValueError("failed annotations cannot contain generated content")
            if self.annotation_error is None or not self.annotation_error.strip():
                raise ValueError("failed annotations require a non-empty error")
        return self