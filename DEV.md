# Dev workflow: Run PMM + Flutter UI together

This repo contains:
- PMM FastAPI probe API at `pmm/api/probe.py`
- Flutter UI app in `ui/` (targets: web and macOS)

## Pre-reqs
- Python 3.11+, with venv created and deps installed:
  - `pip install -r requirements.txt`
- Flutter SDK installed and configured for web and macOS (on macOS run `flutter config --enable-macos-desktop`)
- First run of `flutter pub get` inside `ui/`

## Quick start (VS Code tasks)
Use VS Code's Run Task to start both services.

- Web UI + API:
  1) Run task: `dev:ui:web+api`
  - API: http://127.0.0.1:8000
  - UI:  http://localhost:5173 (default from `flutter run -d chrome --web-port 5173`)

- macOS UI + API:
  1) Run task: `dev:ui:macos+api`

CORS is enabled for development in the API.

## Manual start
- Start PMM API:
  - `python -m uvicorn pmm.api.probe:app --reload --host 127.0.0.1 --port 8000`
- Start UI (web):
  - `cd ui && flutter run -d chrome --web-port 5173`
- Start UI (macOS):
  - `cd ui && flutter run -d macos`

## UI configuration
In the UI Settings page:
- Set PMM Probe Base URL to `http://127.0.0.1:8000` or `http://localhost:8000`
- Save an API key if one is required (optional today)

The UI queries:
- `GET /health`, `GET /traits`, `GET /commitments`, `GET /events/recent`

`/events/recent` returns `{ items: [...] }`. The client handles that shape.

## Notes
- For production, restrict CORS `allow_origins` to specific domains.
- Consider a Makefile or a foreman-style Procfile if you prefer cli orchestration.
- Later, we can expose a real chat ingress; UI currently streams a placeholder response.
