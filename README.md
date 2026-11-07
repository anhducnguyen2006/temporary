# MRG Sample Scanner — Gemini Prototype

Diagnostic-first proof of concept for extracting MRG sample label fields from short rotating-tube videos.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e 'backend[dev]'
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. The key is server-side only.

## Controlled diagnostics

Run one command at a time; these make real, non-retried Gemini requests:

```bash
cd backend
python scripts/test_gemini.py models
python scripts/test_gemini.py text
python scripts/test_gemini.py image path/to/label.jpg
python scripts/test_gemini.py video path/to/sample.mov
```

The current integration uses `client.interactions.create(...)`, Gemini Files API upload/polling, and a documented video content object. The default development model is `gemini-3.6-flash`; override it with `GEMINI_MODEL`. No live request is run automatically.

## Backend

```bash
uvicorn app.main:app --reload --app-dir backend
```

The API is `POST /api/analyze` with multipart fields `video` and `receiving_date`.
The backend defaults to `GEMINI_EXTRACTION_STRATEGY=two_step`; `direct` remains available
for experimentation without changing the working Gemini integration.

To run one explicit structured extraction against the verified sample:

```bash
cd backend
python scripts/test_gemini.py structured-video path/to/IMG_1880.mov --receiving-date 2026-11-07
```

To isolate structured-output behavior without sending video:

```bash
python scripts/test_gemini.py structured-text --receiving-date 2026-11-07
```

`structured-text` defaults to `GEMINI_STRUCTURED_TEXT_THINKING_LEVEL=minimal`; set it to
`low`, `medium`, or `high` in `.env` for comparison. The diagnostic reports request
duration, thinking level, token usage when exposed, and final interaction status.

The optional `two-step-video` diagnostic first performs the existing video transcription,
then sends only that transcription to the structured-output request. The opt-in
`background-video` diagnostic uses `background=True`, records its interaction ID in
`backend/data/last-background-interaction.json`, and polls it without resubmitting.

## One-case extraction benchmarks

The default `two_step` strategy remains the baseline. Optional single-pass diagnostics are
`single_pass` (structured video) and `single-pass-json` (video with strict JSON text).
Run exactly one paid benchmark at a time:

```bash
cd backend
../.venv/bin/python scripts/benchmark_extraction.py \
  path/to/IMG_1880.mov --strategy two_step --receiving-date 2026-11-07
```

Use `--strategy single_pass` or `--strategy single_pass_json` to compare one alternative.
Manually verified annotations are kept separately in `examples/benchmark_annotations.json`;
model output is never used as ground truth.

The compact `single_pass_json` path requires exactly the canonical JSON keys and normalizes
them into the existing nested `ExtractionResult` locally. Its benchmark distinguishes provider
success, JSON parsing, compact-schema validation, normalization, and field accuracy.

## Manual browser test

Start the backend, then start the frontend in a second terminal:

```bash
uvicorn app.main:app --reload --app-dir backend
cd frontend && npm run dev
```

Open the Vite URL, select `IMG_1880.mov`, choose a receiving date, and click **Analyze with
Gemini** once. Wait for `Completed`, verify the video, handwritten identifiers, circled
component, and editable fields, correct the serial number if needed, then download the CSV.

## Tests

```bash
pytest backend/tests
ruff check backend
```

## Limitations

This repository intentionally contains no fake production responses, database, queue, authentication, or automatic fallback model. Live Gemini availability and structured extraction require a configured key and an actual sample video.
