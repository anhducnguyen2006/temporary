import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_VIDEO_SIZE_BYTES = int(os.getenv("MAX_VIDEO_SIZE_MB", "50")) * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = float(os.getenv("MAX_VIDEO_DURATION_SECONDS", "15"))
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured. Copy .env.example to .env and set it.")
    return key

