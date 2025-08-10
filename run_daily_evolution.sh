#!/bin/bash

# PMM Daily Evolution Cron Script
# This script runs the daily agent evolution and logs output with enhanced error handling

# Set working directory
cd /home/scott/Documents/Projects/Business-Development/persistent-mind-model

# Create logs directory if it doesn't exist
mkdir -p logs

# Log start time
echo "$(date): Starting daily evolution..." >> logs/cron_status.log

# Activate virtual environment and run duel.py with retry logic
source .venv/bin/activate

# Run with up to 3 retries for API failures
for attempt in {1..3}; do
    echo "================" >> logs/daily_evolution.log
    echo "$(date): Evolution attempt $attempt/3" >> logs/daily_evolution.log
    
    if python3 duel.py >> logs/daily_evolution.log 2>&1; then
        echo "$(date): Daily evolution completed successfully on attempt $attempt" >> logs/cron_status.log
        exit 0
    else
        echo "$(date): Evolution attempt $attempt failed" >> logs/cron_status.log
        if [ $attempt -lt 3 ]; then
            echo "$(date): Waiting 60s before retry..." >> logs/cron_status.log
            sleep 60
        fi
    fi
done

# All attempts failed
echo "$(date): Daily evolution FAILED after 3 attempts" >> logs/cron_status.log
exit 1
