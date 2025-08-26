# Persistent Mind Model (PMM) - Developer Makefile
# Usage: run `make help` to see available targets.

SHELL := /bin/zsh
.DEFAULT_GOAL := help

# --- Config -----------------------------------------------------------------
VENV ?= .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

BACKEND_APP := pmm.api.probe:app
HOST ?= 0.0.0.0
PORT ?= 8000

# Optional env passed through to backend; .env is auto-loaded by the app
PMM_CHAT_MODE ?=
OLLAMA_URL ?=
OPENAI_API_KEY ?=

# Optional API base override for the Flutter UI (if your UI reads it via dart-define)
PMM_API_BASE ?=

# --- Helpers ----------------------------------------------------------------
.PHONY: help venv install pmm pmm-echo ui-web ui-linux ui-macos ui-doctor clean

## help: Show this help message
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort

## venv: Create a local virtualenv at $(VENV)
venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

## install: Install Python dependencies into the venv
install: venv
	@$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -r requirements.txt

# --- Backend ---------------------------------------------------------------

## pmm: Start the PMM probe API with auto-reload on $(HOST):$(PORT)
pmm: install
	@echo "Starting PMM probe API on http://$(HOST):$(PORT)"
	@echo "(Tip: set OPENAI_API_KEY in your shell or .env to use OpenAI provider)"
	PMM_CHAT_MODE=$(PMM_CHAT_MODE) OLLAMA_URL=$(OLLAMA_URL) OPENAI_API_KEY=$(OPENAI_API_KEY) \
	$(UVICORN) $(BACKEND_APP) --reload --host $(HOST) --port $(PORT)

## pmm-echo: Start the PMM API in echo mode (no LLM needed)
pmm-echo: install
	@echo "Starting PMM probe API (echo mode) on http://$(HOST):$(PORT)"
	PMM_CHAT_MODE=echo $(UVICORN) $(BACKEND_APP) --reload --host $(HOST) --port $(PORT)

# --- Flutter UI ------------------------------------------------------------

FLUTTER := $(shell command -v flutter 2>/dev/null)

define require_flutter
	@if [ -z "$(FLUTTER)" ]; then \
		echo "Error: Flutter SDK not found in PATH. Install Flutter and ensure 'flutter' is available."; \
		exit 1; \
	fi
endef

## ui-web: Run the Flutter UI in Chrome (web)
ui-web:
	$(call require_flutter)
	@echo "Launching Flutter UI for web (Chrome)…"
	cd ui && flutter run -d chrome $${PMM_API_BASE:+--dart-define=PMM_API_BASE=$$PMM_API_BASE}

## ui-linux: Run the Flutter UI as a Linux desktop app
ui-linux:
	$(call require_flutter)
	@echo "Launching Flutter UI for Linux…"
	cd ui && flutter run -d linux $${PMM_API_BASE:+--dart-define=PMM_API_BASE=$$PMM_API_BASE}

## ui-macos: Run the Flutter UI as a macOS desktop app
ui-macos:
	$(call require_flutter)
	@echo "Launching Flutter UI for macOS…"
	cd ui && flutter run -d macos $${PMM_API_BASE:+--dart-define=PMM_API_BASE=$$PMM_API_BASE}

## ui-doctor: Run flutter doctor (diagnostics)
ui-doctor:
	$(call require_flutter)
	cd ui && flutter doctor -v

# --- Maintenance -----------------------------------------------------------

## clean: Remove the local virtualenv and Python caches
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
