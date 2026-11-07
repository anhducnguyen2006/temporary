from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FieldStatus = Literal["extracted", "unreadable", "missing", "not_applicable"]
LabelType = Literal["large", "small", "unknown"]


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
