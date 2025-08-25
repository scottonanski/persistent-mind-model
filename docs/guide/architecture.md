# Architecture

## PMM Core
- Event store (SQLite, hash-chained)
- Reflection + emergence analysis
- Commitments + evidence
- Probe API: FastAPI for observability

## UI
- Flutter app
- Riverpod + Dio
- Hits Probe API for health, traits, commitments, events; chat ingress planned

## Data flow
- User input / external events → PMM logs → analysis → Probe API
- UI polls endpoints for status and historical insights
