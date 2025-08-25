# Getting Started

## Prerequisites
- Python 3.11+
- Flutter SDK (web + macOS targets enabled)

## Install
- Python deps: `pip install -r requirements.txt`
- Flutter deps: `cd ui && flutter pub get`

## Run dev (API + UI)
See `DEV.md` or VS Code tasks:
- `dev:ui:web+api`
- `dev:ui:macos+api`

Probe API runs at http://127.0.0.1:8000. UI defaults to that base URL but can be changed in Settings.
