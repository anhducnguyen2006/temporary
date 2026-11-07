from pathlib import Path
from types import SimpleNamespace

from app import gemini_service
from app.schemas import ExtractionResult


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
                return SimpleNamespace(output_text=__import__("json").dumps(payload))

    monkeypatch.setattr(gemini_service, "make_client", lambda: FakeClient())
    result, _ = gemini_service.extract_text("Printed ID: 125767", "2026-11-07")

    assert result.sample_id.value == "125767"
    assert result.date_received.value == "2026-11-07"
