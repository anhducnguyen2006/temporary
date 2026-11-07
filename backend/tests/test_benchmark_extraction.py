import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_extraction import expected_fields, field_value

from app.schemas import ExtractionResult


def test_benchmark_reads_canonical_objects_and_strings():
    result = ExtractionResult.model_validate({
        "label_type": "small",
        "sample_id": {"value": "001", "status": "extracted"},
        "date_sampled": {"value": None, "status": "missing"},
        "date_received": {"value": "2026-11-07", "status": "extracted"},
        "machine": {"value": "UP", "status": "extracted"},
        "component": {"value": "5 NGS", "status": "extracted"},
        "serial_number": {"value": "MY1207236", "status": "extracted"},
        "reference_lube": {"value": None, "status": "not_applicable"},
        "warnings": [],
    })
    assert field_value(result, "serial_number") == "MY1207236"
    assert field_value(result, "label_type") == "small"
    assert expected_fields({"serial_number": "My1207238", "verified_fields": ["serial_number"]}) == {
        "serial_number": "My1207238"
    }
