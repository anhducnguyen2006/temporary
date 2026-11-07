"""Opt-in Gemini diagnostics. Each command performs exactly one inference request."""
import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import MODEL_ID, gemini_api_key
from app.gemini_service import (
    extract_text,
    extract_video,
    extract_video_two_step,
    inspect_extraction_schema,
    make_client,
    run_background_video,
    run_image,
    run_text,
    run_video,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["models", "text", "image", "video", "structured-text", "structured-video",
                 "two-step-video", "background-video", "schema"],
    )
    parser.add_argument("path", nargs="?")
    parser.add_argument("--receiving-date", default=None)
    args = parser.parse_args()
    started = time.monotonic()
    print(f"SDK version: {importlib.metadata.version('google-genai')}")
    print(f"Model ID: {MODEL_ID}")
    try:
        gemini_api_key()
        if args.command == "models":
            model = make_client().models.get(model=MODEL_ID)
            print(f"Model metadata: {model}")
        elif args.command == "text":
            print(f"Response: {run_text()}")
        elif args.command == "image":
            if not args.path:
                parser.error("image requires a path")
            print(f"Response: {run_image(Path(args.path))}")
        elif args.command == "video":
            if not args.path:
                parser.error("video requires a path")
            print(f"Response: {run_video(Path(args.path))}")
        elif args.command == "structured-text":
            receiving_date = args.receiving_date or parser.error(
                "structured-text requires --receiving-date"
            )
            transcription = """Printed ID: 125767
RN: UP 7246
Date: 9-28-26
SN: My1202236
Circled number: 5
Circled test type: NGS
Label type: Small"""
            result, raw = extract_text(transcription, receiving_date)
            print(f"Validated extraction: {result.model_dump_json(indent=2)}")
            print(f"Raw output: {raw}")
        elif args.command == "schema":
            print(f"Extraction schema: {inspect_extraction_schema()}")
        else:
            if not args.path or not args.receiving_date:
                parser.error(f"{args.command} requires a path and --receiving-date")
            if args.command == "structured-video":
                result, raw = extract_video(Path(args.path), args.receiving_date)
                print(f"Validated extraction: {result.model_dump_json(indent=2)}")
                print(f"Raw output: {raw}")
            elif args.command == "two-step-video":
                result, transcription, raw = extract_video_two_step(Path(args.path), args.receiving_date)
                print(f"Transcription: {transcription}")
                print(f"Validated extraction: {result.model_dump_json(indent=2)}")
                print(f"Raw output: {raw}")
            else:
                result = run_background_video(Path(args.path), args.receiving_date)
                print(f"Validated extraction: {result.model_dump_json(indent=2)}")
        print(f"Method: client.interactions.create; elapsed_seconds: {time.monotonic() - started:.2f}; success: true")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Method: client.interactions.create; elapsed_seconds: {time.monotonic() - started:.2f}; success: false")
        print(f"Failure: {type(exc).__name__}: {exc!s}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
