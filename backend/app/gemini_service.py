import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai

from .config import (
    DATA_DIR,
    EXTRACTION_STRATEGY,
    GEMINI_REQUEST_TIMEOUT_SECONDS,
    GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
    GEMINI_VIDEO_THINKING_LEVEL,
    MODEL_ID,
    gemini_api_key,
)
from .schemas import CompactExtraction, ExtractionResult, user_supplied_date_field


@dataclass
class InferenceMetrics:
    stage: str
    duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    status: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_seconds": round(self.duration_seconds, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thought_tokens": self.thought_tokens,
            "status": self.status,
        }


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


def _metrics(interaction: Any, started: float, stage: str) -> InferenceMetrics:
    usage = getattr(interaction, "usage", None)
    return InferenceMetrics(
        stage=stage,
        duration_seconds=time.monotonic() - started,
        input_tokens=_usage_value(usage, "prompt_token_count"),
        output_tokens=_usage_value(usage, "response_token_count"),
        thought_tokens=_usage_value(usage, "thoughts_token_count"),
        status=getattr(interaction, "status", None),
    )


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


class CompactExtractionError(ValueError):
    def __init__(
        self,
        stage: str,
        message: str,
        raw_output: str,
        metrics: InferenceMetrics | None = None,
    ):
        super().__init__(message)
        self.stage = stage
        self.raw_output = raw_output
        self.metrics = metrics


def parse_compact_json(raw_output: str) -> CompactExtraction:
    """Parse exactly one JSON object, optionally wrapped in one Markdown fence."""
    candidate = raw_output.strip()
    if candidate.startswith("```") or candidate.endswith("```"):
        match = re.fullmatch(r"```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```", candidate, re.DOTALL)
        if not match:
            raise ValueError("JSON response must contain one complete Markdown code fence.")
        candidate = match.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError("JSON response must be one object.")
    return CompactExtraction.model_validate(parsed)


def normalize_sample_date(value: str | None) -> str | None:
    if not value:
        return value
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2})", value)
    if match:
        month, day = int(match.group(1)), int(match.group(2))
        if month > 12 or day > 12:
            try:
                separator = "-" if "-" in value else "/"
                return datetime.strptime(value, f"%m{separator}%d{separator}%y").date().isoformat()
            except ValueError:
                return value
    return value


def normalize_compact_extraction(
    compact: CompactExtraction, receiving_date: str
) -> ExtractionResult:
    field_names = (
        "sample_id",
        "date_sampled",
        "machine",
        "component",
        "serial_number",
        "reference_lube",
    )
    fields: dict[str, Any] = {}
    for name in field_names:
        value = getattr(compact, name)
        if name == "date_sampled":
            value = normalize_sample_date(value)
        if value is not None and value != "":
            status = "extracted"
        elif name in compact.unreadable_fields:
            status = "unreadable"
        else:
            status = "missing"
        fields[name] = {
            "value": value,
            "status": status,
            "uncertain": name in compact.uncertain_fields,
            "reason": "Flagged for manual review by the model."
            if name in compact.uncertain_fields
            else None,
        }
    result = ExtractionResult.model_validate({
        "label_type": compact.label_type,
        **fields,
        "date_received": user_supplied_date_field(receiving_date).model_dump(),
        "warnings": (
            ["Label type is unknown; manual review required."]
            if compact.label_type == "unknown"
            else []
        ),
    })
    return _apply_label_semantics(result)


def extract_text(
    transcription: str,
    receiving_date: str,
    thinking_level: str = GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
) -> tuple[ExtractionResult, str]:
    result, raw, _ = extract_text_detailed(transcription, receiving_date, thinking_level)
    return result, raw


def extract_text_detailed(
    transcription: str,
    receiving_date: str,
    thinking_level: str = GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
) -> tuple[ExtractionResult, str, InferenceMetrics]:
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
    return result, raw, _metrics(interaction, started, "structured_text")


def extract_video_two_step(
    path: Path, receiving_date: str
) -> tuple[ExtractionResult, str, str]:
    result, transcription, raw, _ = extract_video_two_step_detailed(path, receiving_date)
    return result, transcription, raw


def extract_video_two_step_detailed(
    path: Path, receiving_date: str
) -> tuple[ExtractionResult, str, str, list[InferenceMetrics]]:
    transcription, transcription_metrics = run_video_detailed(
        path,
        "Transcribe the label across all frames. Explicitly describe which Small-label "
        "number and GS/NGS option are visibly circled. Do not guess.",
    )
    result, raw, structured_metrics = extract_text_detailed(transcription, receiving_date)
    return result, transcription, raw, [transcription_metrics, structured_metrics]


def analyze_video(
    path: Path, receiving_date: str, strategy: str = EXTRACTION_STRATEGY
) -> tuple[ExtractionResult, str, str | None]:
    if strategy == "two_step":
        result, transcription, raw = extract_video_two_step(path, receiving_date)
        return result, raw, transcription
    if strategy in {"direct", "single_pass"}:
        result, raw = extract_video(path, receiving_date)
        return result, raw, None
    if strategy == "single_pass_json":
        result, raw = extract_video_json(path, receiving_date)
        return result, raw, None
    raise ValueError(
        "GEMINI_EXTRACTION_STRATEGY must be 'two_step', 'direct', 'single_pass', "
        "or 'single_pass_json'."
    )


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


def run_video(
    path: Path,
    prompt: str | None = None,
    thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL,
) -> str:
    text, _ = run_video_detailed(path, prompt, thinking_level)
    return text


def run_video_detailed(
    path: Path, prompt: str | None = None, thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL
) -> tuple[str, InferenceMetrics]:
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise ValueError("thinking_level must be one of: minimal, low, medium, high")
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        uploaded = _wait_for_file(client, uploaded)
        started = time.monotonic()
        interaction = client.interactions.create(
            model=MODEL_ID,
            input=[
                {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                {"type": "text", "text": prompt or "Describe and transcribe this video. Do not guess unreadable text."},
            ],
            generation_config={"thinking_level": thinking_level},
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
        return _text(interaction), _metrics(interaction, started, "video_transcription")
    finally:
        client.files.delete(name=uploaded.name)


def extract_video(
    path: Path,
    receiving_date: str,
    thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL,
) -> tuple[ExtractionResult, str]:
    result, raw, _ = extract_video_detailed(path, receiving_date, thinking_level)
    return result, raw


def extract_video_detailed(
    path: Path,
    receiving_date: str,
    thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL,
) -> tuple[ExtractionResult, str, InferenceMetrics]:
    if thinking_level not in {"minimal", "low", "medium", "high"}:
        raise ValueError("thinking_level must be one of: minimal, low, medium, high")
    client = make_client()
    uploaded = client.files.upload(file=str(path))
    try:
        uploaded = _wait_for_file(client, uploaded)
        started = time.monotonic()
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
            generation_config={"thinking_level": thinking_level},
            timeout=GEMINI_REQUEST_TIMEOUT_SECONDS,
        )
        raw = _text(interaction)
        result = _apply_label_semantics(ExtractionResult.model_validate_json(raw))
        result.date_received = user_supplied_date_field(receiving_date)
        return result, raw, _metrics(interaction, started, "structured_video")
    finally:
        client.files.delete(name=uploaded.name)


def extract_video_json(
    path: Path,
    receiving_date: str,
    thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL,
) -> tuple[ExtractionResult, str]:
    result, raw, _ = extract_video_json_detailed(path, receiving_date, thinking_level)
    return result, raw


def extract_video_json_detailed(
    path: Path,
    receiving_date: str,
    thinking_level: str = GEMINI_VIDEO_THINKING_LEVEL,
) -> tuple[ExtractionResult, str, InferenceMetrics]:
    prompt = """Inspect every frame of this rotating MRG sample label video and return ONLY
one JSON object with exactly these keys:
label_type, sample_id, date_sampled, machine, component, serial_number, reference_lube,
uncertain_fields, unreadable_fields.
Values are strings or null. uncertain_fields and unreadable_fields are arrays containing
only canonical field names. Do not include date_received; it is supplied by the user.
For Small labels, use RN as machine, SN as serial_number, set reference_lube to null, and
combine the visibly circled number and visibly circled GS/NGS selection as component.
For Large labels, use Equip ID as machine, Sample point as component, Ref Lube as
reference_lube, and set serial_number to null. Preserve identifiers and leading zeros.
Do not guess unreadable handwriting; use null and list that field in unreadable_fields.
Do not rename fields, add keys, use Markdown code fences, or return commentary. Return raw
JSON only."""
    transcription, metrics = run_video_detailed(path, prompt, thinking_level)
    try:
        compact = parse_compact_json(transcription)
    except (TypeError, ValueError) as exc:
        stage = "json_parse" if str(exc).startswith(("Invalid JSON", "JSON response")) else "compact_schema"
        raise CompactExtractionError(stage, str(exc), transcription, metrics) from exc
    result = normalize_compact_extraction(compact, receiving_date)
    return result, transcription, metrics


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
