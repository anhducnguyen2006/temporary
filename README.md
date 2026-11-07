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

To run one explicit structured extraction against the verified sample:

```bash
cd backend
python scripts/test_gemini.py structured-video path/to/IMG_1880.mov --receiving-date 2026-11-07
```

To isolate structured-output behavior without sending video:

```bash
python scripts/test_gemini.py structured-text --receiving-date 2026-11-07
```

The optional `two-step-video` diagnostic first performs the existing video transcription,
then sends only that transcription to the structured-output request. The opt-in
`background-video` diagnostic uses `background=True`, records its interaction ID in
`backend/data/last-background-interaction.json`, and polls it without resubmitting.

## Tests

```bash
pytest backend/tests
ruff check backend
```

## Limitations

This repository intentionally contains no fake production responses, database, queue, authentication, or automatic fallback model. Live Gemini availability and structured extraction require a configured key and an actual sample video.
