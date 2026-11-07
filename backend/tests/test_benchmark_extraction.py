import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_extraction import aggregate, expected_fields, field_value

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


def test_aggregate_reports_denominators_and_strategy_metrics():
    records = [
        {
            "api_succeeded": True,
            "pydantic_validation": True,
            "total_latency_seconds": 10.0,
            "field_accuracy": {"sample_id": True, "serial_number": False},
            "stages": [{"input_tokens": 100, "output_tokens": 20, "thought_tokens": 5}],
        },
        {
            "api_succeeded": False,
            "pydantic_validation": False,
            "total_latency_seconds": 20.0,
            "field_accuracy": None,
            "stages": [],
        },
    ]
    report = aggregate(records, 1.0, 2.0, 3.0)

    assert report["sample_count"] == 2
    assert report["provider_failure_rate"] == 0.5
    assert report["field_accuracy"]["serial_number"] == {
        "correct": 0, "denominator": 1, "accuracy": 0.0
    }
    assert report["sample_id_accuracy"]["denominator"] == 1
    assert report["latency_seconds"]["median"] == 10.0
    assert report["token_usage"]["totals"]["input_tokens"] == 100
    assert report["estimated_api_cost_usd"] == 0.000155
