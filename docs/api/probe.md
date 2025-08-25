# Probe API

Base URL: `http://localhost:8000`

## Endpoints

### GET /health
- Returns: `{ ok, db, events, last_kind }`

### GET /traits
- Big Five, HEXACO, drift metrics

### GET /events/recent
- Query: `limit`, `kind`
- Returns: `{ items: Event[] }`

### GET /commitments
- Query: `limit`, `status`, `fields`
- Returns: `{ items: Commitment[] }`

### GET /identity
- Returns current agent identity

### GET /emergence, /reflection/quality, /meta-cognition, /emergence/trends, /personality/adaptation
- Analysis endpoints used by dashboards and future UI views
