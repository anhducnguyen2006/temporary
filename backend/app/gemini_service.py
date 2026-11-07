import json
import time
from pathlib import Path
from typing import Any

from google import genai

from .config import MODEL_ID, gemini_api_key
from .schemas import ExtractionResult

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label_type": {"type": "string", "enum": ["large", "small", "unknown"]},
        **{
            name: {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "null"]},
                    "status": {"type": "string", "enum": ["extracted", "unreadable", "missing", "not_applicable"]},
                    "uncertain": {"type": "boolean"},
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["value", "status", "uncertain", "reason"],
                "additionalProperties": False,
            }
            for name in ("sample_id", "date_sampled", "date_received", "machine",
                         "component", "serial_number", "reference_lube")
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label_type", "sample_id", "date_sampled", "date_received", "machine",
                 "component", "serial_number", "reference_lube", "warnings"],
    "additionalProperties": False,
}


def make_client() -> genai.Client:
    return genai.Client(api_key=gemini_api_key())


def _text(interaction: Any) -> str:
    output = getattr(interaction, "output_text", None)
    if not output:
        raise RuntimeError("Gemini returned no output_text.")
    return output


def run_text(prompt: str = "Reply with exactly: MRG Gemini diagnostic OK.") -> str:
    return _text(make_client().interactions.create(model=MODEL_ID, input=prompt))


def run_image(path: Path) -> str:
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        return _text(client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "image", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": "Transcribe all visible label text exactly. Do not guess."},
            ],
        ))
    finally:
        client.files.delete(name=uploaded.name)


def run_video(path: Path, prompt: str | None = None) -> str:
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        while uploaded.state and getattr(uploaded.state, "name", "") == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError("Gemini Files API marked the uploaded video as FAILED.")
        return _text(client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt or "Describe and transcribe this video. Do not guess unreadable text."},
            ],
        ))
    finally:
        client.files.delete(name=uploaded.name)


def extract_video(path: Path, receiving_date: str) -> tuple[ExtractionResult, str]:
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        while uploaded.state and getattr(uploaded.state, "name", "") == "PROCESSING":
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state and getattr(uploaded.state, "name", "") == "FAILED":
            raise RuntimeError("Gemini Files API marked the uploaded video as FAILED.")
        prompt = f"""Extract the rotating sample-tube label across all video frames.
Receiving date supplied by the user: {receiving_date}. Use it exactly for date_received.
Determine large or small label. Map fields according to the MRG label conventions. Preserve
identifiers and leading zeros. Read circled small-label options. Never guess; use null and
the appropriate status for unreadable or intentionally absent values. Return only the schema."""
        interaction = client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            response_format={"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
        )
        raw = _text(interaction)
        result = ExtractionResult.model_validate(json.loads(raw))
        return result, raw
    finally:
        client.files.delete(name=uploaded.name)

