# Dev Quickstart: Makefile Targets

Convenient aliases for running the PMM backend and Flutter UI.

## Prerequisites
- Python 3.11+
- Flutter SDK in PATH (for UI targets)

## Setup
```bash
make install
cp .env.example .env && $EDITOR .env
```

Key env vars (see `.env.example`):
- `OPENAI_API_KEY` (use OpenAI provider when set)
- `PMM_OPENAI_MODEL` (defaults to gpt-4o-mini)
- `PMM_CHAT_MODE` (`echo` to bypass LLM for wiring tests)
- `PMM_API_BASE` (optional: UI override via --dart-define)

## Backend
Start the FastAPI probe with auto-reload:
```bash
make pmm
```

Echo mode (no LLM):
```bash
PORT=8001 make pmm-echo
```

Notes:
- The app auto-loads `.env`.
- If port 8000 is busy, change it: `PORT=8001 make pmm`.

## Flutter UI
Run the UI and point it to the API:
```bash
# Web (Chrome)
make ui-web

# macOS desktop
make ui-macos

# Linux desktop
make ui-linux

# Diagnostics
make ui-doctor
```

Optional: override the API base used by the UI:
```bash
PMM_API_BASE=http://localhost:8001 make ui-web
```

## Troubleshooting
- “Address already in use”: another server is on 8000. Use `PORT=8001` or stop the old process.
- “flutter: command not found”: install Flutter and ensure `flutter` is in PATH.
- Target names use hyphens (e.g., `ui-macos`), not colons.
