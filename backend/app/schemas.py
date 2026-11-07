from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FieldStatus = Literal["extracted", "unreadable", "missing", "not_applicable"]
LabelType = Literal["large", "small", "unknown"]
CanonicalField = Literal[
    "sample_id",
    "date_sampled",
    "machine",
    "component",
    "serial_number",
    "reference_lube",
]


class ExtractedField(BaseModel):
    value: str | None = None
    status: FieldStatus
    uncertain: bool = False
    reason: str | None = None


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_type: LabelType
    sample_id: ExtractedField
    date_sampled: ExtractedField
    date_received: ExtractedField
    machine: ExtractedField
    component: ExtractedField
    serial_number: ExtractedField
    reference_lube: ExtractedField
    warnings: list[str] = Field(default_factory=list)


class CompactExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_type: LabelType
    sample_id: str | None = None
    date_sampled: str | None = None
    machine: str | None = None
    component: str | None = None
    serial_number: str | None = None
    reference_lube: str | None = None
    uncertain_fields: list[CanonicalField] = Field(default_factory=list)
    unreadable_fields: list[CanonicalField] = Field(default_factory=list)

    @field_validator("label_type", mode="before")
    @classmethod
    def normalize_label_type(cls, value: str) -> str:
        normalized = str(value).lower()
        if normalized in {"small", "large", "unknown"}:
            return normalized
        return value


class AnalyzeResponse(BaseModel):
    success: bool
    model: str
    result: ExtractionResult | None = None
    raw_output: str | None = None
    diagnostic: dict[str, str | float] = Field(default_factory=dict)
    extraction_strategy: str | None = None
    transcription: str | None = None
    error: str | None = None


def user_supplied_date_field(receiving_date: str) -> ExtractedField:
    return ExtractedField(
        value=receiving_date,
        status="extracted",
        reason="Supplied by the user; not inferred by Gemini.",
    )
