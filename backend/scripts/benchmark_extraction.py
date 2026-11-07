"""Run exactly one opt-in extraction benchmark and print a JSON record."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import (
    GEMINI_STRUCTURED_TEXT_THINKING_LEVEL,
    GEMINI_VIDEO_THINKING_LEVEL,
    MODEL_ID,
)
from app.gemini_service import (
    CompactExtractionError,
    extract_video_detailed,
    extract_video_json_detailed,
    extract_video_two_step_detailed,
)


def video_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def field_value(result: Any, field: str) -> Any:
    value = getattr(result, field)
    return value.value if hasattr(value, "value") else value


def accuracy(result: Any, expected: dict[str, Any]) -> dict[str, bool]:
    return {
        field: field_value(result, field) == expected.get(field)
        for field in expected
    }


def expected_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    verified = annotation.get("verified_fields")
    if verified is None:
        return {
            key: value
            for key, value in annotation.items()
            if key not in {"verified_fields", "label_type"}
        }
    return {field: annotation[field] for field in verified if field in annotation}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--strategy", choices=["two_step", "single_pass", "single_pass_json"], required=True)
    parser.add_argument("--receiving-date", required=True)
    parser.add_argument("--annotations", type=Path, default=Path("examples/benchmark_annotations.json"))
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotation = annotations.get(args.video.name)
    if annotation is None:
        annotation = next(
            (value for key, value in annotations.items() if key.lower() == args.video.name.lower()),
            {},
        )
    expected = expected_fields(annotation)
    started = time.monotonic()
    metrics = []
    transcription = None
    raw = None
    api_succeeded = False
    json_valid = None
    compact_schema_valid = None
    normalization_succeeded = None
    try:
        if args.strategy == "two_step":
            result, transcription, raw, metrics = extract_video_two_step_detailed(
                args.video, args.receiving_date
            )
        elif args.strategy == "single_pass":
            result, raw, metric = extract_video_detailed(
                args.video, args.receiving_date, GEMINI_VIDEO_THINKING_LEVEL
            )
            metrics = [metric]
        else:
            result, raw, metric = extract_video_json_detailed(
                args.video, args.receiving_date, GEMINI_VIDEO_THINKING_LEVEL
            )
            metrics = [metric]
        api_succeeded = True
        json_valid = True
        compact_schema_valid = None if args.strategy != "single_pass_json" else True
        canonical_schema_valid = True
        normalization_succeeded = True if args.strategy == "single_pass_json" else None
        record = {
            "video": str(args.video),
            "label_type": result.label_type,
            "strategy": args.strategy,
            "model": MODEL_ID,
            "thinking_level": (
                GEMINI_STRUCTURED_TEXT_THINKING_LEVEL if args.strategy == "two_step"
                else GEMINI_VIDEO_THINKING_LEVEL
            ),
            "video_duration_seconds": video_duration(args.video),
            "total_latency_seconds": round(time.monotonic() - started, 2),
            "stages": [metric.as_dict() for metric in metrics],
            "pydantic_validation": True,
            "api_succeeded": api_succeeded,
            "json_valid": json_valid,
            "compact_schema_valid": compact_schema_valid,
            "canonical_schema_valid": canonical_schema_valid,
            "normalization_succeeded": normalization_succeeded,
            "fields": {
                field: field_value(result, field)
                for field in (
                    "sample_id",
                    "date_sampled",
                    "date_received",
                    "machine",
                    "component",
                    "serial_number",
                    "reference_lube",
                )
            },
            "benchmark_scoring_succeeded": True,
            "field_accuracy": {},
            "failure_reason": None,
            "transcription": transcription,
            "raw_output": raw,
        }
        try:
            record["field_accuracy"] = accuracy(result, expected) if expected else {}
        except (AttributeError, KeyError, TypeError) as exc:
            record["benchmark_scoring_succeeded"] = False
            record["field_accuracy"] = None
            record["failure_reason"] = f"benchmark_scoring: {type(exc).__name__}: {exc}"
    except CompactExtractionError as exc:
        api_succeeded = True
        raw = exc.raw_output
        record = {
            "video": str(args.video),
            "strategy": args.strategy,
            "model": MODEL_ID,
            "total_latency_seconds": round(time.monotonic() - started, 2),
            "api_succeeded": True,
            "json_valid": exc.stage != "json_parse",
            "compact_schema_valid": False if exc.stage == "compact_schema" else None,
            "canonical_schema_valid": False,
            "normalization_succeeded": False if args.strategy == "single_pass_json" else None,
            "pydantic_validation": False,
            "benchmark_scoring_succeeded": None,
            "stages": [exc.metrics.as_dict()] if exc.metrics else [],
            "raw_output": raw,
            "failure_reason": f"{exc.stage}: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        record = {
            "video": str(args.video),
            "strategy": args.strategy,
            "model": MODEL_ID,
            "total_latency_seconds": round(time.monotonic() - started, 2),
            "pydantic_validation": False,
            "api_succeeded": api_succeeded,
            "json_valid": json_valid,
            "compact_schema_valid": compact_schema_valid,
            "canonical_schema_valid": None,
            "normalization_succeeded": normalization_succeeded,
            "raw_output": raw,
            "benchmark_scoring_succeeded": None,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(record, indent=2))
    return 0 if record["failure_reason"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
