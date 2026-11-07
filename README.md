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

The current integration uses `client.interactions.create(...)`, Gemini Files API upload/polling, and a documented video content object. No live request is run automatically.

## Backend

```bash
uvicorn app.main:app --reload --app-dir backend
```

The API is `POST /api/analyze` with multipart fields `video` and `receiving_date`.

## Tests

```bash
pytest backend/tests
ruff check backend
```

## Limitations

This repository intentionally contains no fake production responses, database, queue, authentication, or automatic fallback model. Live Gemini availability and structured extraction require a configured key and an actual sample video.

