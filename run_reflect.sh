#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/Documents/Projects/Business-Development/persistent-mind-model"
cd "$PROJECT_DIR"

# Load your venv
source .venv/bin/activate

# Load env vars from .env (your API key + model)
set -a
[ -f .env ] && source .env
set +a

# Make sure logs dir exists
mkdir -p logs

# Cadence-aware reflection (only runs if due)
python cli.py reflect-if-due >> logs/reflect.log 2>&1
