import asyncio
import json
import subprocess
from pathlib import Path

from fastapi import HTTPException, UploadFile

from .config import MAX_VIDEO_DURATION_SECONDS, MAX_VIDEO_SIZE_BYTES

ALLOWED_MIME_TYPES = {"video/mp4", "video/quicktime"}
ALLOWED_SUFFIXES = {".mp4", ".mov"}


async def save_and_validate_video(upload: UploadFile, destination: Path) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if upload.content_type not in ALLOWED_MIME_TYPES or suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Only MP4 and MOV video files are supported.")
    size = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_SIZE_BYTES:
                raise HTTPException(413, "Video exceeds the configured 50 MB limit.")
            target.write(chunk)
    try:
        probe = await asyncio.to_thread(
            subprocess.run,
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(destination)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise HTTPException(422, "Unable to inspect this video with ffprobe.") from exc
    if duration <= 0 or duration > MAX_VIDEO_DURATION_SECONDS:
        raise HTTPException(422, f"Video duration must be between 0 and {MAX_VIDEO_DURATION_SECONDS:g} seconds.")
