# Development notes

- Keep Gemini calls in `backend/app/gemini_service.py` and use the Interactions API.
- Never add credentials, fake production inference, automatic model fallback, or retries to diagnostics.
- Keep user-provided receiving dates outside Gemini's inferred fields.
