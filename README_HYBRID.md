# Persistent Mind Model — Hybrid Thought Representation (Experimental Branch)

This branch adds efficient thought representation to PMM:
- Summarization of stored thoughts
- Keyword extraction
- Optional semantic embeddings (placeholder bytes for now)
- Backward-compatible SQLite schema
- Environment toggles
- In-app transparency (startup banner + status command)

Status: Working, minimal, honest. No heavy dependencies required.

## What’s New

- `pmm/langchain_memory.py`
  - `enable_summary`, `enable_embeddings` flags
  - Heuristic `_summarize_and_extract()`:
    - Summary: first sentence (~<=160 chars)
    - Keywords: simple capitalized-token extraction (JSON list)
  - `save_context()` writes events with summary, keywords, full_text and (optionally) embedding
  - `load_memory_variables()` prefers summaries, handles legacy and new schemas, and surfaces keywords

- `pmm/self_model_manager.py`
  - `add_event(...)` accepts `tags`, `full_text`, `embedding`
  - Forwards summary, keywords, embedding to SQLite

- `pmm/storage/sqlite_store.py`
  - Backward compatible schema: nullable summary, keywords, embedding (BLOB)

- `chat.py`
  - Reads `.env` toggles
  - Startup banner shows Summarization/Embeddings ON/OFF
  - New `status` command with DB stats

## Requirements

- Python 3.10+
- OpenAI key (if using OpenAI models): `OPENAI_API_KEY`
- Optional local model (Ollama) if you prefer

Install:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Create a `.env` file in the project root (same folder as `chat.py`) or export vars in your shell:

```
# OpenAI (if using OpenAI models)
OPENAI_API_KEY=sk-...

# Feature toggles (default OFF if unset)
PMM_ENABLE_SUMMARY=true
PMM_ENABLE_EMBEDDINGS=false
```

Valid truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Restart the app after changing.

## Run

From the repo root:

```bash
python chat.py
# or select a model explicitly
python chat.py --model gpt-4o-mini
# or a local model if configured (Ollama)
python chat.py --model llama3.1
```

At startup you should see a banner like:

```
🧩 Thought Summarization: ON | 🔎 Semantic Embeddings: ON/OFF
```

## In-chat Commands

- `status` — shows summarization/embeddings state, DB file size, event counts
- `personality` — Big Five snapshot
- `memory` — recent cross-session memory context
- `models` — model selector
- `quit` — exit

## Verify It Works

- Summaries/keywords populated on new events (when summary is enabled):

```bash
sqlite3 pmm.db "SELECT id, substr(summary,1,60), keywords FROM events ORDER BY id DESC LIMIT 5;"
```

- Embeddings populated on new events (when embeddings are enabled):

```bash
sqlite3 pmm.db "SELECT id, length(embedding) FROM events WHERE embedding IS NOT NULL ORDER BY id DESC LIMIT 5;"
```

- Quick totals:

```bash
sqlite3 pmm.db "SELECT COUNT(*) FROM events WHERE summary IS NOT NULL;"
sqlite3 pmm.db "SELECT COUNT(*) FROM events WHERE embedding IS NOT NULL;"
```

## Notes

- Current keywords are simple capitalized-token heuristics; expect trivial words for short messages.
- Current embeddings are a placeholder (summary bytes) to exercise storage and toggles; swap to real vectors in a follow-up.