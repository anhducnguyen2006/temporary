import json
import time
from pathlib import Path
from typing import Any

from google import genai

from .config import (
    DATA_DIR,
    EXTRACTION_STRATEGY,
    GEMINI_REQUEST_TIMEOUT_SECONDS,
    GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
    MODEL_ID,
    gemini_api_key,
)
from .schemas import ExtractionResult, user_supplied_date_field


def make_client() -> genai.Client:
    return genai.Client(api_key=gemini_api_key())


def _text(interaction: Any) -> str:
    output = getattr(interaction, "output_text", None)
    if not output:
        raise RuntimeError("Gemini returned no output_text.")
    return output


def extraction_response_format() -> dict[str, Any]:
    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": ExtractionResult.model_json_schema(),
    }


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def _print_inference_diagnostics(interaction: Any, started: float, thinking_level: str) -> None:
    usage = getattr(interaction, "usage", None)
    print(
        "stage=structured-text diagnostics "
        f"request_seconds={time.monotonic() - started:.2f} "
        f"thinking_level={thinking_level} "
        f"input_tokens={_usage_value(usage, 'prompt_token_count', 'input_tokens')} "
        f"output_tokens={_usage_value(usage, 'response_token_count', 'output_tokens')} "
        f"thought_tokens={_usage_value(usage, 'thoughts_token_count', 'thought_tokens')} "
        f"status={getattr(interaction, 'status', 'unknown')}"
    )


def _apply_label_semantics(result: ExtractionResult) -> ExtractionResult:
    if result.label_type == "small":
        result.reference_lube = result.reference_lube.model_copy(
            update={
                "value": None,
                "status": "not_applicable",
                "uncertain": False,
                "reason": "Not applicable to Small labels.",
            }
        )
    elif result.label_type == "large":
        result.serial_number = result.serial_number.model_copy(
            update={
                "value": None,
                "status": "not_applicable",
                "uncertain": False,
                "reason": "Not applicable to Large labels.",
            }
        )
    return result


def extract_text(
    transcription: str,
    receiving_date: str,
    thinking_level: str = GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
) -> tuple[ExtractionResult, str]:
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise ValueError(
            "thinking_level must be one of: minimal, low, medium, high"
        )
    client = make_client()
    prompt = f"""Convert this prior video transcription into the MRG extraction schema.
Only use facts present in the transcription; never infer or invent missing values.
For Small labels, use RN for machine, SN for serial_number, set reference_lube to
not_applicable, and combine the visibly circled number and GS/NGS option as component.
For Large labels, use Equip ID for machine, Sample point for component, Ref Lube for
reference_lube, and set serial_number to not_applicable. Date Received must be exactly
{receiving_date} and is supplied by the user, not the transcription. Preserve identifiers
as strings, leading zeros, and uncertainty as a review signal rather than a probability.

Transcription:
{transcription}"""
    started = time.monotonic()
    print("stage=structured-text request_started")
    interaction = client.interactions.create(
        model=MODEL_ID,
        input=prompt,
        response_format=extraction_response_format(),
        generation_config={"thinking_level": thinking_level},
        timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
    )
    raw = _text(interaction)
    result = _apply_label_semantics(ExtractionResult.model_validate_json(raw))
    result.date_received = user_supplied_date_field(receiving_date)
    _print_inference_diagnostics(interaction, started, thinking_level)
    return result, raw


def extract_video_two_step(
    path: Path, receiving_date: str
) -> tuple[ExtractionResult, str, str]:
    transcription = run_video(
        path,
        "Transcribe the label across all frames. Explicitly describe which Small-label "
        "number and GS/NGS option are visibly circled. Do not guess.",
    )
    result, raw = extract_text(transcription, receiving_date)
    return result, transcription, raw


def analyze_video(
    path: Path, receiving_date: str, strategy: str = EXTRACTION_STRATEGY
) -> tuple[ExtractionResult, str, str | None]:
    if strategy == "two_step":
        result, transcription, raw = extract_video_two_step(path, receiving_date)
        return result, raw, transcription
    if strategy == "direct":
        result, raw = extract_video(path, receiving_date)
        return result, raw, None
    raise ValueError("GEMINI_EXTRACTION_STRATEGY must be 'two_step' or 'direct'.")


def inspect_extraction_schema() -> dict[str, Any]:
    schema = ExtractionResult.model_json_schema()
    return schema


def _wait_for_file(client: genai.Client, uploaded: Any) -> Any:
    deadline = time.monotonic() + GEMINI_REQUEST_TIMEOUT_SECONDS
    while uploaded.state and getattr(uploaded.state, "name", "") == "PROCESSING":
        if time.monotonic() >= deadline:
            raise TimeoutError("Gemini Files API video processing timed out.")
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state and getattr(uploaded.state, "name", "") == "FAILED":
        raise RuntimeError("Gemini Files API marked the uploaded video as FAILED.")
    return uploaded


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
        uploaded = _wait_for_file(client, uploaded)
        return _text(client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt or "Describe and transcribe this video. Do not guess unreadable text."},
            ],
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        ))
    finally:
        client.files.delete(name=uploaded.name)


def extract_video(path: Path, receiving_date: str) -> tuple[ExtractionResult, str]:
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        uploaded = _wait_for_file(client, uploaded)
        prompt = f"""Extract the rotating sample-tube label across all video frames.
Receiving date supplied by the user: {receiving_date}. Use it exactly for date_received.
Determine Large or Small. For Large, map barcode/printed number to sample_id, handwritten
Date to date_sampled, Equip ID to machine, Sample point to component, and Ref Lube to
reference_lube; serial_number is not_applicable. For Small, map barcode/printed number to
sample_id, handwritten Date to date_sampled, RN to machine, SN to serial_number, and combine
the physically circled number (1-6) with the physically circled GS or NGS option as
component, such as "5 NGS"; reference_lube is not_applicable. Preserve identifiers and
leading zeros. Never guess; use null and the appropriate status for unreadable or
intentionally absent values. Uncertainty is a review signal, not a probability. Return only
the schema."""
        interaction = client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            response_format=extraction_response_format(),
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
        raw = _text(interaction)
        result = _apply_label_semantics(ExtractionResult.model_validate_json(raw))
        result.date_received = user_supplied_date_field(receiving_date)
        return result, raw
    finally:
        client.files.delete(name=uploaded.name)


def run_background_video(path: Path, receiving_date: str) -> ExtractionResult:
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        uploaded = _wait_for_file(client, uploaded)
        prompt = f"""Extract the MRG sample label from all video frames. User-provided Date Received
is {receiving_date}; use it exactly. For Small labels, report the visibly circled number and
the visibly circled GS/NGS option and combine them as component. Never guess."""
        interaction = client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt},
            ],
            response_format=extraction_response_format(),
            background=True,
            timeout=30,
        )
        interaction_id = getattr(interaction, "id", None)
        if not interaction_id:
            raise RuntimeError("Background interaction returned no interaction ID.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record_path = DATA_DIR / "last-background-interaction.json"
        record_path.write_text(json.dumps({"interaction_id": interaction_id}), encoding="utf-8")
        deadline = time.monotonic() + GEMINI_REQUEST_TIMEOUT_SECONDS * 3
        interval = 3.0
        while True:
            current = client.interactions.get(id=interaction_id, timeout=30)
            status = str(getattr(current, "status", "")).lower()
            print(f"stage=background-poll interaction_id={interaction_id} status={status}")
            if status in {"completed", "incomplete"}:
                result = ExtractionResult.model_validate_json(_text(current))
                result.date_received = user_supplied_date_field(receiving_date)
                return result
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Background interaction ended with status={status}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Background interaction polling deadline reached; interaction_id={interaction_id}."
                )
            time.sleep(interval)
    finally:
        client.files.delete(name=uploaded.name)
