# Changelog

All notable changes in this development pass are listed here. The summary focuses on user‑visible features, CLI help, and API improvements.

## 2025-08-28

- Turn‑Scoped Identity Commitments (TTL + auto‑enforcement)
  - `open_identity_commitment(policy, ttl_turns, note)` opens a turn‑scoped identity promise.
  - `tick_turn_scoped_identity_commitments(smm, reply_text)` decrements after each assistant reply, attaches minimal `evidence` at zero, then `commitment.close`.
  - `get_identity_turn_commitments(smm)` and `close_identity_turn_commitments(smm)` for viewing/closing.
  - Policy/TTL-aware dedupe to avoid near‑duplicate spam.

- Probe API (unified + discoverable)
  - Unified `GET /identity` returns `{ name, id, identity_commitments[] }`.
  - New `GET /endpoints` lists available probes with short descriptions.
  - Chat shim: `--@probe start | list | <path>` to start/query from chat.

- Discoverable Command Router (`--@`)
  - `--@help` global catalog with pasteable tips.
  - Identity: `--@identity list | open N | clear`.
  - Commitments: `--@commitments list | search X | close CID | clear`.
  - Events: `--@events list | recent N | kind K N | search X`.
  - Probe: `--@probe list | start | <path>`; directory shown inline.
  - Search: `--@find <text>` scans events and open commitments.
  - Tracking: `--@track list | on | off | status | legend | explain`.

- Real‑Time Telemetry (`[TRACK]`)
  - Two concise lines after replies: stage, identity/growth bands, close rate, S0 streak, reflection readiness + hints.
  - `--@track legend` (compact field guide) and `--@track explain` (plain‑English walkthrough).

- Friendlier Logs (colored, unified)
  - `[COMMIT+]` (green), `[REFLECT+]` (green), `[REFLECT~]` (yellow), `[API]` (cyan).
  - Noisy `DEBUG` prints moved behind `PMM_DEBUG` guard.

- Startup UX
  - Banner simplified; nudges to `--@help` and key examples.

Files of interest: `chat.py`, `pmm/commitments.py`, `pmm/api/probe.py`, `docs/api/probe.md`, `pmm/reflection.py`, `pmm/config/models.py`.

