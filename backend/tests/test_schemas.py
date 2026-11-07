import pytest
from pydantic import ValidationError

from app.schemas import ExtractionResult, user_supplied_date_field


def field(value=None, status="missing", uncertain=False):
    return {"value": value, "status": status, "uncertain": uncertain, "reason": None}


def valid(**overrides):
    result = {
        "label_type": "large",
        "sample_id": field("123779", "extracted"),
        "date_sampled": field("2026-01-02", "extracted"),
        "date_received": field("2026-01-03", "extracted"),
        "machine": field("53279869", "extracted"),
        "component": field("420 NDF", "extracted"),
        "serial_number": field(None, "not_applicable"),
        "reference_lube": field("Adjusting Side", "extracted"),
        "warnings": [],
    }
    return {**result, **overrides}


def test_large_mapping_contract():
    result = ExtractionResult.model_validate(valid())
    assert result.machine.value == "53279869"
    assert result.serial_number.status == "not_applicable"


def test_small_mapping_contract():
    result = ExtractionResult.model_validate(valid(
        label_type="small", machine=field("UP6509", "extracted"),
        component=field("4 GS", "extracted"), serial_number=field("SN-1", "extracted"),
        reference_lube=field(None, "not_applicable"),
    ))
    assert result.component.value == "4 GS"


def test_missing_values_are_allowed():
    assert ExtractionResult.model_validate(valid(sample_id=field(None, "unreadable"))).sample_id.value is None


def test_field_status_distinguishes_not_applicable_missing_and_unreadable():
    result = ExtractionResult.model_validate(valid(
        serial_number=field(None, "not_applicable"),
        component=field(None, "missing"),
        date_sampled=field(None, "unreadable"),
    ))
    assert result.serial_number.status == "not_applicable"
    assert result.component.status == "missing"
    assert result.date_sampled.status == "unreadable"


def test_invalid_structured_response_is_rejected():
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(valid(machine={"value": 123, "status": "extracted", "uncertain": False, "reason": None}))


def test_receiving_date_is_a_user_supplied_field():
    result = user_supplied_date_field("2026-11-07")
    assert result.value == "2026-11-07"
    assert "user" in (result.reason or "")
