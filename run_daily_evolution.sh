#!/bin/bash

# PMM Daily Evolution Cron Script
# This script runs the daily agent evolution and logs output

# Set working directory
cd /home/scott/Documents/Projects/Business-Development/persistent-mind-model

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment and run duel.py
source .venv/bin/activate
python3 duel.py >> logs/daily_evolution.log 2>&1

# Log completion with timestamp
echo "$(date): Daily evolution completed" >> logs/cron_status.log
