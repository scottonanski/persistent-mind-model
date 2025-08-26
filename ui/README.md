# PMM UI (Flutter)

Minimal chat UI that talks to the PMM Probe API.

## Run

- macOS: from this folder
	- flutter run -d macos
- Web:
	- flutter run -d chrome

The app defaults to http://127.0.0.1:8000 for the API base. Tap the gear icon to change.

You can also set at compile time:

PMM_API_BASE=http://localhost:8000 flutter run -d chrome --dart-define=PMM_API_BASE=$PMM_API_BASE

## Test

flutter test
