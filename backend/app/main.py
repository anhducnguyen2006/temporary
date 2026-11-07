import tempfile
import time
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .config import MODEL_ID
from .gemini_service import extract_video
from .schemas import AnalyzeResponse
from .video_utils import save_and_validate_video

app = FastAPI(title="MRG Gemini Prototype")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(video: UploadFile = File(...), receiving_date: date = Form(...)) -> AnalyzeResponse:  # noqa: B008
    started = time.monotonic()
    suffix = Path(video.filename or "").suffix.lower()
    with tempfile.TemporaryDirectory(prefix="mrg-") as directory:
        path = Path(directory) / f"input{suffix}"
        await save_and_validate_video(video, path)
        try:
            result, raw = extract_video(path, receiving_date.isoformat())
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(502, "Gemini returned invalid structured output.") from exc
        except Exception as exc:
            raise HTTPException(502, f"Gemini request failed: {exc}") from exc
    return AnalyzeResponse(
        success=True, model=MODEL_ID, result=result, raw_output=raw,
        diagnostic={"elapsed_seconds": round(time.monotonic() - started, 2)},
    )
