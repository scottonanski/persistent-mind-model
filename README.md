# Persistent Mind Model (PMM)

**AI personality persistence with cross-session memory and commitment tracking**

PMM enables any LLM to maintain a persistent personality across conversations. It tracks commitments, evolves personality traits based on evidence, and provides cryptographic integrity verification.

## Install & Run

**Requirements:** Python 3.8+, OpenAI API key

```bash
git clone https://github.com/scottonanski/persistent-mind-model.git
cd persistent-mind-model
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
export OPENAI_API_KEY='sk-your-key-here'
```

**Start chatting:**
```bash
python chat.py
```

**Run tests:**
```bash
pytest
```

**Start monitoring API:**
```bash
uvicorn pmm.api.probe:app --port 8000
```

## What It Does

**Core Features:**
- **Cross-session memory** - AI remembers conversations between restarts
- **Personality evolution** - Big Five traits change based on evidence
- **Commitment tracking** - Extracts and monitors "Next, I will..." statements
- **Cryptographic integrity** - SHA-256 hash chain prevents tampering
- **Emergence measurement** - IAS/GAS scoring tracks AI identity adoption

**Architecture:**
```
chat.py → pmm/langchain_memory.py → pmm/self_model_manager.py → SQLite
                                  ↓
                            pmm/emergence.py (IAS/GAS scoring)
                                  ↓
                            pmm/api/probe.py (monitoring)
```

**Key Files:**
- `chat.py` - Main chat interface
- `pmm/self_model_manager.py` - Core personality management
- `pmm/langchain_memory.py` - LangChain integration with PMM
- `pmm/emergence.py` - Identity adoption scoring (S0→S4 stages)
- `pmm/api/probe.py` - FastAPI monitoring endpoints
- `pmm/storage/sqlite_store.py` - Hash-chained event storage

## Usage

**Basic Chat:**
```bash
python chat.py
# AI will remember your name across restarts
# Type 'quit' to exit
```

**Monitor AI State:**
```bash
# Start API server
uvicorn pmm.api.probe:app --port 8000

# Check endpoints
curl http://localhost:8000/health
curl http://localhost:8000/emergence
curl http://localhost:8000/commitments
```

**Emergence Stages:**
- **S0: Substrate** - Generic assistant behavior
- **S1: Resistance** - Deflects PMM concepts  
- **S2: Adoption** - Uses PMM terminology correctly
- **S3: Self-Model** - References own capabilities
- **S4: Growth-Seeking** - Actively requests experiences

## API Endpoints

**Start monitoring server:**
```bash
uvicorn pmm.api.probe:app --port 8000
```

**Available endpoints:**
- `GET /health` - Database status and event count
- `GET /integrity` - Hash-chain verification  
- `GET /emergence` - IAS/GAS scoring and stage detection
- `GET /commitments` - Commitment tracking with open/closed status
- `GET /events/recent` - Recent conversation history
- `GET /traits` - Current personality traits

**Example emergence response:**
```json
{
  "IAS": 0.64,
  "GAS": 0.58,
  "stage": "S3: Self-Model",
  "pmmspec_avg": 0.71,
  "selfref_avg": 0.54,
  "experience_detect": true,
  "events_analyzed": 5
}
```

## File Structure

```
pmm/
├── emergence.py           # IAS/GAS scoring and stage detection
├── self_model_manager.py  # Core personality management
├── langchain_memory.py    # LangChain integration with PMM
├── adapters/              # LLM provider interfaces
├── storage/               # SQLite with hash-chain integrity
├── api/                   # FastAPI monitoring endpoints
└── core/                  # Runtime orchestration

chat.py                    # Main chat interface
tests/                     # Test suite (19 tests)
```

## Development

**Run tests:**
```bash
pytest  # All 19 tests should pass
```

**Add new LLM provider:**
```python
# Implement pmm.adapters.base.ModelAdapter
class MyAdapter(ModelAdapter):
    def generate(self, messages, max_tokens=512):
        # Your LLM integration here
        return response_text
```

**Monitor emergence in real-time:**
```bash
# Watch AI identity adoption
curl http://localhost:8000/emergence
```

## License

MIT License - See LICENSE file for details.
