# I was just messing around and somehow ended up with this. I think this thing works?  🤔

# Persistent Mind Model (PMM)

[![Tests](https://img.shields.io/badge/tests-118%2F118%20passing-green)](https://github.com/scottonanski/persistent-mind-model)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/scottonanski/persistent-mind-model)
[![License](https://img.shields.io/badge/license-dual-orange)](https://github.com/scottonanski/persistent-mind-model)

PMM is a Python framework that gives an AI assistant a durable identity, memory, and behavior that persist across sessions, devices, and model backends. It logs interactions as structured events, tracks commitments, and applies controlled behavioral “drift” based on real usage.

Think of it as a persistent layer, or better yet a Ai "mind" that the user owns, running on top of any LLM. You can swap between local models or remote APIs without losing the mind’s state—PMM keeps the identity and memory intact while the underlying model can change, avoiding vendor lock‑in.

PMM isn’t just about recall—it’s about gradual identity formation. Each interaction is recorded, reflected on, and woven into a self‑model. Traits, habits, and commitments evolve slowly and deliberately, so behavior stays consistent while improving over time.

Model quality matters: very small models can introduce noise that degrades memory; more capable models generally improve fidelity. Regardless of substrate, the mind remains consistent.

By design, evolution is a slow burn—steady, evidence‑driven, and anchored in your interactions. It aims for stability over speed, so the assistant feels like a companion that grows with you rather than a tool that resets every session.

In simple terms: PMM is an AI that remembers, stays consistent, and gradually becomes itself.


**Side note:**
I'm currently working on a branch that allows the PMM to see its own code and analyze it. This is a work in progress and is not yet ready for release.
[self-code-analysis branch](https://github.com/scottonanski/persistent-mind-model/tree/self-code-analysis)

Here's how it works (sort of... Still working on it):

1. The PMM has a self-model that it uses to see snippet of itself (code).
2. The PMM uses this self-model to analyze itself (code).
3. The PMM uses this analysis to improve itself (code).

```markdown
> reflect... code SelfModelManager.add_commitment

Snipped function:
def add_commitment(...):
    ...

> How could this be improved?

PMM: This function appends commitments but doesn’t check for duplicates at the DB level.
I suggest adding a unique constraint or reinforcement event path.

```
---

Plain-English overview of key concepts:
- [🧠 PMM Glossary (Plain English)](PMM_Glossary.md)

Technical analysis (ChatGPT-5):
- [📊 Technical Analysis: PMM Architecture & Innovation](TECHNICAL_ANALYSIS.md)

- General analysis / Project State (ChatGPT-5):
  - [Persistent Mind Model (PMM) – Analysis](Persistent%20Mind%20Model%20(PMM)%20–%20Analysis.md)

- Potential Acquisition Analysis (PDF) — simulated report for curiosity/fun; not a real acquisition document.
  - [Simulated: Potential Acquisition Analysis (PDF)](Persistent%20Mind%20Model%20(PMM)%20–%20Potential%20Acquisition%20Analysis.pdf) 

---



## Why this matters

- Consistency across tools: PMM preserves identity and behavior across providers (OpenAI, Claude, Grok, Ollama) and across calls/sessions, reducing drift when switching models or scaling.
- Operational memory: Conversation events and extracted commitments survive restarts and redeploys, enabling reliable follow-ups, reminders, and long-horizon workflows.
- Accountability: Commitments are tracked with open/closed status and linked to events via a SHA-256 hash chain, providing an auditable trail.
- Personality control: Evidence-based Big Five drift lets you tune adaptation over time without losing intended voice or norms.
- Observability: Reflection and probe endpoints expose internal state (traits, commitments, emergence signals) for monitoring and debugging.
- Drop-in integration: The LangChain memory adapter adds persistence to existing chat systems without major changes.

## Core Features

- **Cross-session memory**: SQLite storage of conversation history and personality state
- **Personality modeling**: Big Five personality traits with evidence-based drift
- **Commitment tracking**: Automatic extraction and lifecycle management of AI commitments
- **Reflection system**: Event-driven self-analysis with configurable triggers
- **Hash-chain integrity**: SHA-256 linking of events for tamper detection
- **LangChain integration**: Drop-in compatibility with existing chatbot systems
- **FastAPI monitoring**: Real-time state inspection via REST endpoints

## Technical Architecture

**Memory System**: SQLite database stores conversation events, personality traits, commitments, and reflections. Each event includes metadata, timestamps, and SHA-256 hash chains for integrity verification.

**Personality Model**: Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, neuroticism) stored as floating-point values. Traits drift based on behavioral patterns and commitment completion rates.

**Commitment Lifecycle**: Natural language processing extracts commitments from AI responses using pattern matching. Evidence detection links completion signals to commitments via cryptographic hashes.

**Reflection System**: Configurable triggers initiate self-analysis based on event counts, time intervals, or commitment states. Reflections generate insights stored as structured events with provenance tracking.

## Quick Start

**Requirements:** Python 3.10+, OpenAI API key

```bash
git clone https://github.com/scottonanski/persistent-mind-model.git
cd persistent-mind-model
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
export OPENAI_API_KEY='sk-your-key-here'
```

**Start the main chat interface:**
```bash
python chat.py
```

**Optional: enable debugging and telemetry (off by default):**
```bash
# Show low-level request/response details
python chat.py --debug

# Print PMM telemetry snapshots (stages, IAS/GAS, cooldowns)
python chat.py --telemetry

# Enable both
python chat.py --debug --telemetry

# Or use environment variable for telemetry
PMM_TELEMETRY=true python chat.py
```

**In‑chat help system (type `--@help`):**

The chat now has a discoverable command router. Start with:

```
--@help
```

Common actions (pasteable examples):
- Identity mode: `--@identity list` • `--@identity open 3` • `--@identity clear`
- Probe API: `--@probe list` • `--@probe start` • `--@probe identity`
- Commitments: `--@commitments list` • `--@commitments search identity`
- Events: `--@events list` • `--@events kind response 5`
- Global search: `--@find something`
- Real‑time tracking: `--@track on` • `--@track legend` • `--@track explain`

You can still use simple words:
- `personality`, `memory`, `models`, `status`, `quit`


## Monitoring API

**Start the monitoring server:**
```bash
uvicorn pmm.api.probe:app --port 8000
# or from inside chat: --@probe start
```

**Endpoint discovery:**
- `GET /endpoints` - Curated directory with short descriptions and examples

**Common endpoints:**
- `GET /identity` - Current agent name + active identity turn‑scoped commitments
- `GET /commitments` - Commitment rows with open/closed status
- `GET /events/recent` - Recent events
- `GET /emergence` - Identity and growth metrics
- `GET /health` - Basic health snapshot

**Example response (`/emergence`):**
```json
{
  "IAS": 0.64,
  "GAS": 0.58,
  "stage": "S3: Self-Model",
  "events_analyzed": 12,
  "timestamp": "2025-08-15T19:45:00"
}
```

## Changelog

See `CHANGELOG.md` for a detailed list of recent improvements (identity TTL commitments, probe directory, in‑chat help, real‑time `[TRACK]` telemetry, and friendlier colored logs).

## File Structure

```
pmm/
├── emergence.py           # Identity and growth scoring
├── self_model_manager.py  # Core personality management
├── langchain_memory.py    # LangChain integration
├── commitments.py         # Commitment lifecycle tracking
├── api/probe.py          # FastAPI monitoring endpoints
└── storage/              # SQLite with hash-chain integrity

examples/
└── (removed - use chat.py instead)

tests/                    # Test suite (118 tests)
```

## Development

**Run tests:**
```bash
pytest  # 118/118 tests should pass
```

**Code quality:**
```bash
black .     # Format code
ruff check  # Lint code (zero violations)
```

## License

**Dual License Structure:**

### Non-Commercial License (Free)
- Personal projects and experimentation
- Academic research and educational purposes  
- Open source projects (non-commercial)
- Evaluation and testing

### Commercial License (Paid)
For commercial use, contact: **s.onanski@gmail.com**

**Attribution Required:**
- Copyright notice: "© 2025 Scott Onanski - Persistent Mind Model"
- Link to original repository
- License notice in derivative works

**Questions?** Contact s.onanski@gmail.com for licensing clarification.

**Copyright © 2025 Scott Onanski. All rights reserved.**
