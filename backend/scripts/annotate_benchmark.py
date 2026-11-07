"""Open one local video and save manually verified benchmark annotation."""
import argparse
import json
import subprocess
from pathlib import Path

FIELDS = (
    "label_type", "sample_id", "date_sampled", "date_received",
    "machine", "component", "serial_number", "reference_lube",
)


def ask(field: str) -> tuple[str | None, str]:
    value = input(f"{field} (blank if not reviewed): ").strip()
    if not value:
        return None, "unverified"
    status = input("  status [verified/unreadable/not_applicable]: ").strip().lower()
    if status not in {"verified", "unreadable", "not_applicable"}:
        raise ValueError("Status must be verified, unreadable, or not_applicable.")
    return value, status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; use --force to replace it explicitly.")
    try:
        subprocess.run(["open", str(args.video)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(f"Could not open video: {exc}")
    print("Review the opened video. Blank fields are not saved as ground truth.")
    annotation = {"video_filename": args.video.name}
    verified: list[str] = []
    statuses: dict[str, str] = {}
    for field in FIELDS:
        value, status = ask(field)
        if status != "unverified":
            annotation[field] = value
            statuses[field] = status
        if status == "verified":
            verified.append(field)
    annotation["verified_fields"] = verified
    annotation["field_status"] = statuses
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(annotation, indent=2) + "\n", encoding="utf-8")
    print(f"Saved annotation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
