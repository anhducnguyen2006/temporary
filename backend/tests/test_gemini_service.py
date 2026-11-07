from pathlib import Path
from types import SimpleNamespace

import pytest

from app import gemini_service
from app.schemas import CompactExtraction, ExtractionResult


def test_structured_video_validates_and_overrides_model_receiving_date(monkeypatch, tmp_path: Path):
    payload = {
        "label_type": "small",
        "sample_id": {"value": "00125", "status": "extracted", "uncertain": False, "reason": None},
        "date_sampled": {"value": "9/28/2026", "status": "extracted", "uncertain": True, "reason": "Handwriting"},
        "date_received": {"value": "invented", "status": "extracted", "uncertain": False, "reason": None},
        "machine": {"value": "UP 7246", "status": "extracted", "uncertain": False, "reason": None},
        "component": {"value": "5 NGS", "status": "extracted", "uncertain": False, "reason": None},
        "serial_number": {"value": "My1202236", "status": "extracted", "uncertain": False, "reason": None},
        "reference_lube": {"value": None, "status": "not_applicable", "uncertain": False, "reason": None},
        "warnings": [],
    }

    class FakeClient:
        files = SimpleNamespace(
            upload=lambda **_: SimpleNamespace(uri="file-uri", mime_type="video/quicktime", name="file-name", state=None),
            delete=lambda **_: None,
        )

        class interactions:
            @staticmethod
            def create(**kwargs):
                assert kwargs["model"] == gemini_service.MODEL_ID
                assert kwargs["input"][0]["type"] == "video"
                response_format = kwargs["response_format"]
                assert response_format["type"] == "text"
                assert response_format["mime_type"] == "application/json"
                assert response_format["schema"] == ExtractionResult.model_json_schema()
                assert "json_schema" not in response_format
                return SimpleNamespace(output_text=__import__("json").dumps(payload))

    monkeypatch.setattr(gemini_service, "make_client", lambda: FakeClient())
    video = tmp_path / "sample.mov"
    video.write_bytes(b"video")

    result, raw = gemini_service.extract_video(video, "2026-11-07")

    assert result.component.value == "5 NGS"
    assert result.date_received.value == "2026-11-07"
    assert '"value": "invented"' in raw
    assert result.reference_lube.status == "not_applicable"


def test_text_structured_extraction_uses_schema_and_validates(monkeypatch):
    payload = {
        "label_type": "small",
        "sample_id": {"value": "125767", "status": "extracted", "uncertain": False},
        "date_sampled": {"value": "9/28/2026", "status": "extracted", "uncertain": False},
        "date_received": {"value": "wrong", "status": "extracted", "uncertain": False},
        "machine": {"value": "UP 7246", "status": "extracted", "uncertain": False},
        "component": {"value": "5 NGS", "status": "extracted", "uncertain": False},
        "serial_number": {"value": "My1202236", "status": "extracted", "uncertain": False},
        "reference_lube": {"value": None, "status": "not_applicable", "uncertain": False},
        "warnings": [],
    }

    class FakeClient:
        class interactions:
            @staticmethod
            def create(**kwargs):
                assert kwargs["response_format"]["schema"] == ExtractionResult.model_json_schema()
                assert kwargs["timeout"] == gemini_service.GEMINI_REQUEST_TIMEOUT_SECONDS
                assert kwargs["generation_config"] == {"thinking_level": "minimal"}
                return SimpleNamespace(output_text=__import__("json").dumps(payload))

    monkeypatch.setattr(gemini_service, "make_client", lambda: FakeClient())
    result, _ = gemini_service.extract_text("Printed ID: 125767", "2026-11-07")

    assert result.sample_id.value == "125767"
    assert result.date_received.value == "2026-11-07"


def test_text_structured_extraction_accepts_configured_thinking_level(monkeypatch):
    class FakeClient:
        class interactions:
            @staticmethod
            def create(**kwargs):
                assert kwargs["generation_config"] == {"thinking_level": "low"}
                payload = {
                    "label_type": "small",
                    "sample_id": {"value": None, "status": "missing"},
                    "date_sampled": {"value": None, "status": "unreadable"},
                    "date_received": {"value": None, "status": "missing"},
                    "machine": {"value": None, "status": "missing"},
                    "component": {"value": None, "status": "missing"},
                    "serial_number": {"value": None, "status": "missing"},
                    "reference_lube": {"value": None, "status": "not_applicable"},
                    "warnings": [],
                }
                return SimpleNamespace(output_text=__import__("json").dumps(payload))

    monkeypatch.setattr(gemini_service, "make_client", lambda: FakeClient())
    result, _ = gemini_service.extract_text("No readable values.", "2026-11-07", "low")

    assert result.reference_lube.status == "not_applicable"


def test_two_step_strategy_orchestrates_shared_pipeline(monkeypatch, tmp_path: Path):
    calls = []

    def fake_two_step(path, receiving_date):
        calls.append((path, receiving_date))
        return "result", "transcription", "raw"

    monkeypatch.setattr(gemini_service, "extract_video_two_step", fake_two_step)
    result, raw, transcription = gemini_service.analyze_video(
        tmp_path / "sample.mov", "2026-11-07", "two_step"
    )

    assert (result, raw, transcription) == ("result", "raw", "transcription")
    assert calls == [(tmp_path / "sample.mov", "2026-11-07")]


def test_unknown_strategy_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="two_step"):
        gemini_service.analyze_video(tmp_path / "sample.mov", "2026-11-07", "unknown")


def test_single_pass_json_rejects_non_json_output(monkeypatch, tmp_path: Path):
    class FakeClient:
        files = SimpleNamespace(
            upload=lambda **_: SimpleNamespace(
                uri="file-uri", mime_type="video/quicktime", name="file-name", state=None
            ),
            delete=lambda **_: None,
        )

        class interactions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(output_text="not json")

    monkeypatch.setattr(gemini_service, "make_client", lambda: FakeClient())
    with pytest.raises(ValueError):
        gemini_service.extract_video_json(tmp_path / "sample.mov", "2026-11-07")


def test_compact_json_normalizes_small_label_and_preserves_leading_zero():
    compact = CompactExtraction.model_validate({
        "label_type": "small",
        "sample_id": "00125767",
        "date_sampled": "2026-09-28",
        "machine": "UP 7246",
        "component": "5 NGS",
        "serial_number": "My1207238",
        "reference_lube": None,
        "uncertain_fields": ["serial_number", "component"],
        "unreadable_fields": [],
    })
    result = gemini_service.normalize_compact_extraction(compact, "2026-11-07")

    assert result.sample_id.value == "00125767"
    assert result.serial_number.uncertain is True
    assert result.component.value == "5 NGS"
    assert result.reference_lube.status == "not_applicable"
    assert result.date_received.value == "2026-11-07"


def test_compact_json_normalizes_unreadable_and_missing_fields():
    compact = CompactExtraction.model_validate({
        "label_type": "small",
        "sample_id": None,
        "date_sampled": None,
        "machine": None,
        "component": None,
        "serial_number": None,
        "reference_lube": None,
        "uncertain_fields": [],
        "unreadable_fields": ["serial_number"],
    })
    result = gemini_service.normalize_compact_extraction(compact, "2026-11-07")

    assert result.serial_number.status == "unreadable"
    assert result.component.status == "missing"
    assert result.reference_lube.status == "not_applicable"


def test_compact_schema_rejects_unknown_keys_and_missing_label_type():
    with pytest.raises(ValueError):
        CompactExtraction.model_validate({"tracking_number": "123"})
    with pytest.raises(ValueError):
        CompactExtraction.model_validate({
            "sample_id": "123",
            "uncertain_fields": [],
            "unreadable_fields": [],
        })


def test_large_label_serial_number_is_not_applicable():
    compact = CompactExtraction.model_validate({
        "label_type": "large",
        "sample_id": "123",
        "date_sampled": "2026-09-28",
        "machine": "EQ-1",
        "component": "POINT-1",
        "serial_number": "should-not-survive",
        "reference_lube": "Lube A",
        "uncertain_fields": [],
        "unreadable_fields": [],
    })
    result = gemini_service.normalize_compact_extraction(compact, "2026-11-07")

    assert result.serial_number.value is None
    assert result.serial_number.status == "not_applicable"
    assert result.reference_lube.value == "Lube A"


def test_offline_replay_fenced_response_is_structurally_valid_but_serial_is_inaccurate():
    fixture = Path("examples/fixtures/single_pass_json_IMG_1880_response.txt").read_text()
    compact = gemini_service.parse_compact_json(fixture)
    result = gemini_service.normalize_compact_extraction(compact, "2026-11-07")

    assert result.date_sampled.value == "2026-09-28"
    assert result.component.value == "5 NGS"
    assert result.reference_lube.status == "not_applicable"
    assert result.serial_number.value == "MY1207236"
    assert result.serial_number.value != "My1207238"


def test_ambiguous_two_digit_date_is_preserved_for_review():
    assert gemini_service.normalize_sample_date("01-02-03") == "01-02-03"


def test_offline_single_pass_response_is_canonical_but_scores_serial_mismatch():
    raw = Path("examples/fixtures/single_pass_IMG_1880_response.json").read_text()
    result = ExtractionResult.model_validate_json(raw)
    result = gemini_service._apply_label_semantics(result)
    result.date_received = gemini_service.user_supplied_date_field("2026-11-07")

    assert result.serial_number.value == "MY1207236"
    assert result.component.value == "5 NGS"
    assert result.date_received.value == "2026-11-07"
    assert result.reference_lube.status == "not_applicable"
    assert result.serial_number.value != "My1207238"


@pytest.mark.parametrize("raw", [
    "prefix\n```json\n{}\n```\nsuffix",
    "```json\n{}\n```\n```json\n{}\n```",
    "```python\n{}\n```",
])
def test_compact_parser_rejects_prose_multiple_or_wrong_fences(raw):
    with pytest.raises((ValueError, TypeError)):
        gemini_service.parse_compact_json(raw)
