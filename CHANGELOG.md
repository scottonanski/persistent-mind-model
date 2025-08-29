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

### Autonomy Extensions (Phase 1 scaffolding)

- Development Task Manager
  - `pmm/dev_tasks.py` with `DevTaskManager.open_task/update_task/close_task`.
  - Emits `task_created` / `task_progress` / `task_closed` rows to SQLite.
  - Optional JSON index mirrored to self‑model (`model.dev_tasks`).
  - CLI: `--@tasks open KIND TITLE` and `--@tasks close ID`.

- Behavior‑Based Evidence Engine
  - Hooked into `langchain_memory.save_context`: scans replies for `Done:` and synonyms.
  - Emits `evidence` events and auto‑closes mapped commitments when confidence ≥ threshold.

- Probe API additions
  - `GET /autonomy/status` — compact IAS/GAS/stage + open task count.
  - `GET /autonomy/tasks` — folded dev tasks (open/closed with progress).
  - `GET /autonomy/experiments` — scheduled and executed micro‑experiments.

- Micro‑Experiment Scheduler
  - `pmm/experiments.py` with `ExperimentManager.schedule()` and `run_due()`.
  - Records `experiment_scheduled` and `experiment_executed` with metrics snapshots.

- Policy Evolution
  - `pmm/policy/evolution.py` with a stagnation‑based tuner.
  - Emits `policy_adjusted` and mirrors evidence confidence threshold to env.

- Drift Watch + Self‑Heal
  - `pmm/drift_watch.py` monitors IAS/GAS/stage snapshot.
  - On drift, opens a short identity TTL or nudges reflection; logs `drift_detected` and `self_healing_initiated`.

- Autonomy Loop Integration
  - Hooks for drift watch, experiments runner, and policy evolution at the end of each tick.

