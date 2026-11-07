"""Opt-in Gemini diagnostics. Each command performs exactly one inference request."""
import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import MODEL_ID, gemini_api_key
from app.gemini_service import make_client, run_image, run_text, run_video


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["models", "text", "image", "video"])
    parser.add_argument("path", nargs="?")
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
        else:
            if not args.path:
                parser.error("video requires a path")
            print(f"Response: {run_video(Path(args.path))}")
        print(f"Method: client.interactions.create; elapsed_seconds: {time.monotonic() - started:.2f}; success: true")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Method: client.interactions.create; elapsed_seconds: {time.monotonic() - started:.2f}; success: false")
        print(f"Failure: {type(exc).__name__}: {exc!s}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
