from fastapi.testclient import TestClient

from app import main
from app.schemas import ExtractionResult


def test_analyze_returns_validated_response_schema(monkeypatch):
    result = ExtractionResult.model_validate({
        "label_type": "large",
        "sample_id": {"value": "001", "status": "extracted", "uncertain": False},
        "date_sampled": {"value": None, "status": "unreadable", "uncertain": True},
        "date_received": {"value": "2026-11-07", "status": "extracted", "uncertain": False},
        "machine": {"value": None, "status": "missing", "uncertain": False},
        "component": {"value": None, "status": "missing", "uncertain": False},
        "serial_number": {"value": None, "status": "not_applicable", "uncertain": False},
        "reference_lube": {"value": None, "status": "missing", "uncertain": False},
        "warnings": [],
    })

    async def skip_video_validation(upload, destination):
        destination.write_bytes(b"video")

    monkeypatch.setattr(main, "save_and_validate_video", skip_video_validation)
    monkeypatch.setattr(
        main,
        "analyze_video",
        lambda path, receiving_date, strategy: (result, "{}", "transcription"),
    )

    response = TestClient(main.app).post(
        "/api/analyze",
        files={"video": ("sample.mov", b"video", "video/quicktime")},
        data={"receiving_date": "2026-11-07"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["result"]["sample_id"]["value"] == "001"
    assert body["transcription"] == "transcription"


def test_analyze_maps_provider_timeout_to_gateway_timeout(monkeypatch):
    async def skip_video_validation(upload, destination):
        destination.write_bytes(b"video")

    monkeypatch.setattr(main, "save_and_validate_video", skip_video_validation)
    monkeypatch.setattr(
        main,
        "analyze_video",
        lambda path, receiving_date, strategy: (_ for _ in ()).throw(
            TimeoutError("provider deadline reached")
        ),
    )

    response = TestClient(main.app).post(
        "/api/analyze",
        files={"video": ("sample.mov", b"video", "video/quicktime")},
        data={"receiving_date": "2026-11-07"},
    )

    assert response.status_code == 504
    assert response.json()["detail"] == "provider deadline reached"
