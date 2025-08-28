# Probe API

Base URL: `http://localhost:8000`

## Endpoints

### GET /endpoints
- Curated list of endpoints with short descriptions and examples (for discovery)

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
- Returns current agent identity and any active identity turn-scoped commitments
- Response:
  - `name`: current agent name (JSON model or latest identity event)
  - `id`: core identity id if present
  - `identity_commitments`: list of `{ policy, ttl_turns, remaining_turns, id }`

### GET /emergence, /reflection/quality, /meta-cognition, /emergence/trends, /personality/adaptation
- Analysis endpoints used by dashboards and future UI views
