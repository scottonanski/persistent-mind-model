# I was just messing around and somehow ended up with this. I think this thing works?  🤔

# Persistent Mind Model (PMM)

[![Tests](https://img.shields.io/badge/tests-79%2F79%20passing-green)](https://github.com/scottonanski/persistent-mind-model)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/scottonanski/persistent-mind-model)
[![License](https://img.shields.io/badge/license-dual-orange)](https://github.com/scottonanski/persistent-mind-model)

PMM is a Python system for maintaining persistent AI personality traits, memory, and behavioral patterns across conversations and models. It stores conversation events, tracks commitments, and applies personality drift based on interaction patterns.

Think of PMM as a personal, persistent AI assistant that maintains a consistent identity and behavior, adapting and extending itself over time across diverse operational contexts. Users can connect external API endpoints or local LLMs as substrate engines to feed persistence, while PMM develops a customized persona free from vendor lock-in.

In simple terms, it addresses key challenges in LLM applications, such as maintaining consistent personality, memory, and commitments across different models, sessions, and devices.

That said, lower-end LLMs—such as Gemma3:1b-qat—often struggle to keep pace with the "mind." Their outputs can contaminate the database by introducing hallucinations. Interestingly, switching to more capable models—such as gpt-4-nano—appears to restore memory fidelity.

And here’s the part worth stressing: the “evolution” of the mind is intentionally a slow burn 🔥. This is by design. The persistent mind-model reinforces steady, deliberate growth that evolves in sync with the user’s interactions, ensuring stability over speed.

---

Plain-English overview of key concepts:
- [🧠 PMM Glossary (Plain English)](PMM_Glossary.md)

Technical analysis (ChatGPT-5):
- [📊 Technical Analysis: PMM Architecture & Innovation](TECHNICAL_ANALYSIS.md)

- For a deeper architectural overview and roadmap-style notes, (A little outdated here and there, need to clean it up...):
   [Persistent Mind Model (PMM) Project Considerations — Moving Forward](Persistent%20Mind%20Model%20(PMM)%20Project%20Considerations%20-%20Moving%20Forward.md)

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

**Available commands in chat:**
- `personality` - View current personality traits
- `memory` - Show recent conversation history
- `models` - List available models and switch providers/models
- `status` - Show PMM status (feature toggles, DB size, event counts)
- `quit`/`exit`/`bye` - Exit and save state


## Monitoring API

**Start the monitoring server:**
```bash
uvicorn pmm.api.probe:app --port 8000
```

**Available endpoints:**
- `GET /health` - Database status and event count
- `GET /integrity` - Hash-chain verification  
- `GET /emergence` - Identity and growth scoring
- `GET /commitments` - Commitment tracking with status
- `GET /events/recent` - Recent conversation history
- `GET /traits` - Current personality traits

**Example response:**
```json
{
  "IAS": 0.64,
  "GAS": 0.58,
  "stage": "S3: Self-Model",
  "events_analyzed": 12,
  "timestamp": "2025-08-15T19:45:00"
}
```

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

tests/                    # Test suite (79 tests)
```

## Development

**Run tests:**
```bash
pytest  # 79/79 tests should pass
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
