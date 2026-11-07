"""Run controlled extraction benchmarks for one video or a directory of videos."""
import argparse
import json
import math
import statistics
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

RECORD_FIELDS = (
    "sample_id", "date_sampled", "date_received", "machine",
    "component", "serial_number", "reference_lube",
)


def video_duration(path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(completed.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def field_value(result: Any, field: str) -> Any:
    value = getattr(result, field)
    return value.value if hasattr(value, "value") else value


def expected_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    verified = annotation.get("verified_fields")
    if verified is None:
        return {
            key: value for key, value in annotation.items()
            if key not in {"verified_fields", "video_filename"}
        }
    return {field: annotation[field] for field in verified if field in annotation}


def load_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Annotation file must contain an object.")
        return data
    annotations: dict[str, dict[str, Any]] = {}
    for item in sorted(path.glob("*.json")):
        annotation = json.loads(item.read_text(encoding="utf-8"))
        if not isinstance(annotation, dict):
            raise TypeError(f"Annotation {item} must contain an object.")
        filename = annotation.get("video_filename", item.stem)
        annotations[str(filename)] = annotation
    return annotations


def find_annotation(annotations: dict[str, dict[str, Any]], video: Path) -> dict[str, Any]:
    return next(
        (value for key, value in annotations.items() if Path(key).name.lower() == video.name.lower()),
        {},
    )


def accuracy(result: Any, expected: dict[str, Any]) -> dict[str, bool]:
    return {field: field_value(result, field) == value for field, value in expected.items()}


def _failure_record(
    args: argparse.Namespace,
    started: float,
    raw: str | None,
    metrics: list[Any],
    api_succeeded: bool,
    json_valid: bool | None,
    compact_schema_valid: bool | None,
    normalization_succeeded: bool | None,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        "video": str(args.video),
        "strategy": args.strategy,
        "model": MODEL_ID,
        "total_latency_seconds": round(time.monotonic() - started, 2),
        "pydantic_validation": False,
        "api_succeeded": api_succeeded,
        "json_valid": json_valid,
        "compact_schema_valid": compact_schema_valid,
        "canonical_schema_valid": False,
        "normalization_succeeded": normalization_succeeded,
        "benchmark_scoring_succeeded": None,
        "stages": [metric.as_dict() for metric in metrics],
        "raw_output": raw,
        "field_accuracy": None,
        "failure_reason": failure_reason,
    }


def run_one(
    video: Path,
    strategy: str,
    receiving_date: str,
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    args = argparse.Namespace(video=video, strategy=strategy)
    expected = expected_fields(find_annotation(annotations, video))
    started = time.monotonic()
    metrics: list[Any] = []
    transcription = None
    raw = None
    try:
        if strategy == "two_step":
            result, transcription, raw, metrics = extract_video_two_step_detailed(video, receiving_date)
        elif strategy == "single_pass":
            result, raw, metric = extract_video_detailed(video, receiving_date, GEMINI_VIDEO_THINKING_LEVEL)
            metrics = [metric]
        else:
            result, raw, metric = extract_video_json_detailed(video, receiving_date, GEMINI_VIDEO_THINKING_LEVEL)
            metrics = [metric]
    except CompactExtractionError as exc:
        return _failure_record(
            args, started, exc.raw_output, [exc.metrics] if exc.metrics else [],
            True, exc.stage != "json_parse",
            False if exc.stage == "compact_schema" else None,
            False if strategy == "single_pass_json" else None,
            f"{exc.stage}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_record(
            args, started, raw, metrics, False, None, None, None,
            f"provider_or_extraction: {type(exc).__name__}: {exc}",
        )

    record: dict[str, Any] = {
        "video": str(video),
        "label_type": result.label_type,
        "strategy": strategy,
        "model": MODEL_ID,
        "thinking_level": (
            GEMINI_STRUCTURED_TEXT_THINKING_LEVEL
            if strategy == "two_step" else GEMINI_VIDEO_THINKING_LEVEL
        ),
        "video_duration_seconds": video_duration(video),
        "total_latency_seconds": round(time.monotonic() - started, 2),
        "stages": [metric.as_dict() for metric in metrics],
        "pydantic_validation": True,
        "api_succeeded": True,
        "json_valid": True,
        "compact_schema_valid": True if strategy == "single_pass_json" else None,
        "canonical_schema_valid": True,
        "normalization_succeeded": True if strategy == "single_pass_json" else None,
        "benchmark_scoring_succeeded": True,
        "fields": {
            field: field_value(result, field)
            for field in (
                "sample_id", "date_sampled", "date_received", "machine",
                "component", "serial_number", "reference_lube",
            )
        },
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
    return record


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return values[max(0, math.ceil(len(values) * 0.95) - 1)]


def aggregate(
    records: list[dict[str, Any]],
    input_price: float,
    output_price: float,
    thought_price: float,
) -> dict[str, Any]:
    successful = [record for record in records if record["api_succeeded"]]
    latencies = sorted(record["total_latency_seconds"] for record in successful)
    field_results: dict[str, list[bool]] = {}
    for record in records:
        for field, value in (record.get("field_accuracy") or {}).items():
            field_results.setdefault(field, []).append(value)
    field_accuracy = {
        field: {"correct": sum(values), "denominator": len(values),
                "accuracy": sum(values) / len(values)}
        for field, values in sorted(field_results.items())
    }
    complete_candidates = [
        record for record in records
        if record.get("field_accuracy")
        and all(field in record["field_accuracy"] for field in RECORD_FIELDS)
    ]
    complete_correct = sum(
        all(record["field_accuracy"].values()) for record in complete_candidates
    )
    token_totals = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    token_counts = {key: 0 for key in token_totals}
    for record in records:
        for stage in record.get("stages", []):
            for key in token_totals:
                value = stage.get(key)
                if isinstance(value, int):
                    token_totals[key] += value
                    token_counts[key] += 1
    cost = (
        token_totals["input_tokens"] * input_price
        + token_totals["output_tokens"] * output_price
        + token_totals["thought_tokens"] * thought_price
    ) / 1_000_000 if any(token_counts.values()) else None
    return {
        "sample_count": len(records),
        "api_success_count": len(successful),
        "provider_failure_rate": (
            (len(records) - len(successful)) / len(records) if records else None
        ),
        "pydantic_validation_rate": (
            sum(record["pydantic_validation"] for record in successful) / len(successful)
            if successful else None
        ),
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "p95": _p95(latencies),
            "denominator": len(latencies),
        },
        "field_accuracy": field_accuracy,
        "sample_id_accuracy": field_accuracy.get("sample_id"),
        "serial_number_accuracy": field_accuracy.get("serial_number"),
        "circled_component_accuracy": field_accuracy.get("component"),
        "complete_record_accuracy": {
            "correct": complete_correct,
            "denominator": len(complete_candidates),
            "accuracy": complete_correct / len(complete_candidates)
            if complete_candidates else None,
        },
        "token_usage": {"totals": token_totals, "observed_counts": token_counts},
        "estimated_api_cost_usd": cost,
        "pricing_usd_per_million_tokens": {
            "input": input_price, "output": output_price, "thought": thought_price,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="One video or a directory of videos.")
    parser.add_argument("--strategy", choices=["two_step", "single_pass", "single_pass_json"], required=True)
    parser.add_argument("--receiving-date", required=True)
    parser.add_argument("--annotations", type=Path, default=Path("examples/benchmark_annotations.json"))
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--thought-price-per-million", type=float, default=0.0)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations)
    videos = sorted(
        path for path in (args.video.iterdir() if args.video.is_dir() else [args.video])
        if path.is_file() and path.suffix.lower() in {".mov", ".mp4", ".m4v", ".webm"}
    )
    if not videos:
        parser.error("No supported video files found.")
    records = [run_one(video, args.strategy, args.receiving_date, annotations) for video in videos]
    output = {
        "strategy": args.strategy,
        "model": MODEL_ID,
        "records": records,
        "aggregate": aggregate(
            records,
            args.input_price_per_million,
            args.output_price_per_million,
            args.thought_price_per_million,
        ),
    }
    print(json.dumps(output, indent=2))
    return 0 if all(record["failure_reason"] is None for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
