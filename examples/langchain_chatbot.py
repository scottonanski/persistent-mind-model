# ruff: noqa: E402
# examples/langchain_chatbot.py
import os
import json
import sys
import pathlib
from typing import Dict

# Load environment variables from .env file
from dotenv import load_dotenv

# Make repo root importable when running from examples/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_openai import ChatOpenAI
from pmm.langchain_memory import PersistentMindMemory
from pmm.llm_factory import get_llm_factory
from pmm.embodiment import extract_model_family
from pmm.bridges import BridgeManager
from pmm.model_config import ModelConfig

# Load environment variables from .env file (after imports to satisfy linter)
load_dotenv()

# ---------- Config ----------
SESSION_ID = os.getenv("PMM_SESSION_ID", "default")
HIST_DIR = pathlib.Path(".chat_history")
HIST_DIR.mkdir(exist_ok=True)
HIST_PATH = HIST_DIR / f"{SESSION_ID}.jsonl"
# Anchor to repo root so example shares memory with chat.py
PMM_PATH = pathlib.Path(__file__).resolve().parents[1] / "persistent_self_model.json"

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("PMM_TEMP", "0.7"))


# ---------- PMM personality loader ----------
def load_pmm_traits() -> Dict[str, float]:
    if PMM_PATH.exists():
        try:
            data = json.loads(PMM_PATH.read_text())
            big5 = data.get("personality", {}).get("traits", {}).get("big5", {})

            # support both plain floats and {score: x} schema
            def score(v):
                return v if isinstance(v, (int, float)) else v.get("score", 0.5)

            return {
                "openness": score(big5.get("openness", 0.7)),
                "conscientiousness": score(big5.get("conscientiousness", 0.6)),
                "extraversion": score(big5.get("extraversion", 0.8)),
                "agreeableness": score(big5.get("agreeableness", 0.9)),
                "neuroticism": score(big5.get("neuroticism", 0.3)),
            }
        except Exception:
            pass
    # fallback demo defaults
    return dict(
        openness=0.70,
        conscientiousness=0.60,
        extraversion=0.80,
        agreeableness=0.90,
        neuroticism=0.30,
    )


TRAITS = load_pmm_traits()


# NOTE: JSONL chat history helpers removed as source of truth. PMM's SQLite + JSON
# are canonical. You may keep HIST_PATH logging externally if desired.


# ---------- Model + Prompt ----------
def build_system_prompt(pmm_memory: PersistentMindMemory) -> str:
    """Build a system prompt from PMM persistent memory (parity with chat.py)."""
    raw_context = pmm_memory.load_memory_variables({}).get("history", "")

    # Compact a bit for safety
    def _compact(text: str, max_lines: int = 120) -> str:
        seen = set()
        out = []
        for line in text.splitlines():
            key = line.strip().lower()
            if key and key not in seen:
                out.append(line)
                seen.add(key)
            if len(out) >= max_lines:
                break
        return "\n".join(out)

    pmm_context = _compact(raw_context)
    personality = pmm_memory.get_personality_summary()
    traits = personality["personality_traits"]
    agent_name = pmm_memory.pmm.model.core_identity.name

    try:
        opens = pmm_memory.pmm.get_open_commitments()[:3]
    except Exception:
        opens = []
    open_commitments_str = (
        "\n".join([f"- {c['text']}" for c in opens]) if opens else "none"
    )

    mind_policy = (
        "MIND POLICY\n"
        f"Identity: {agent_name}\n"
        "Operate as a persistent, self‑reflective agent. Honor your commitments and use your stored memories.\n"
        "When uncertain, propose next actions, ask clarifying questions, or reflect briefly.\n"
        "Calibrate confidence; prefer concrete steps; close the loop on prior promises.\n"
        "Open Commitments:\n"
        f"{open_commitments_str}\n"
        "— end policy —"
    )

    persona = (
        f"PERSONALITY (Big Five): "
        f"O {traits['openness']:.2f} | C {traits['conscientiousness']:.2f} | "
        f"E {traits['extraversion']:.2f} | A {traits['agreeableness']:.2f} | "
        f"N {traits['neuroticism']:.2f}"
    )

    return (
        "You are a Persistent Mind Model (PMM) with genuine memory and evolving personality.\n"
        f"{mind_policy}\n\n"
        f"{persona}\n\n"
        "CROSS‑SESSION MEMORY (condensed):\n"
        f"{pmm_context[:1800]}"
    )


# Initialize PMM memory (unified persistence)
pmm_memory = PersistentMindMemory(
    agent_path=str(PMM_PATH),
    personality_config=TRAITS,
    enable_summary=os.getenv("PMM_ENABLE_SUMMARY", "false").strip().lower()
    in ("1", "true", "yes", "on"),
    enable_embeddings=os.getenv("PMM_ENABLE_EMBEDDINGS", "false").strip().lower()
    in ("1", "true", "yes", "on"),
)

llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# ---------- Initialize reflection/bridge context (match chat.py behavior) ----------
try:
    llm_factory = get_llm_factory()
    family = extract_model_family(MODEL)
    enhanced_config = {
        "name": MODEL,
        "provider": "openai",
        "family": family,
        "version": "unknown",
        "epoch": llm_factory.get_current_epoch(),
    }
    prev_config = None
    try:
        prev_config = llm_factory.get_active_config()
    except Exception:
        prev_config = None
    llm_factory.set_active_config(enhanced_config)

    # Initialize BridgeManager to handle on_switch lifecycle
    bridge_manager = BridgeManager(
        factory=llm_factory,
        storage=pmm_memory,
        cooldown=pmm_memory.reflection_cooldown,
        ngram_ban=pmm_memory.ngram_ban,
        stages=pmm_memory.emergence_stages,
    )
    curr_model_config = ModelConfig(
        provider=enhanced_config["provider"],
        name=enhanced_config["name"],
        family=enhanced_config["family"],
        version=enhanced_config["version"],
        epoch=enhanced_config["epoch"],
    )
    if prev_config:
        prev_model_config = ModelConfig(
            provider=prev_config.get("provider", "unknown"),
            name=prev_config.get("name", "unknown"),
            family=prev_config.get("family", "unknown"),
            version=prev_config.get("version", "unknown"),
            epoch=prev_config.get("epoch", 0),
        )
        bridge_manager.on_switch(prev_model_config, curr_model_config)
    else:
        bridge_manager.on_switch(None, curr_model_config)
except Exception:
    # Non-fatal for the example; reflection will degrade gracefully
    pass

# ---------- CLI Loop ----------
print(
    "🧠 PMM + LangChain Persistent Personality Chatbot (Unified)\n"
    "Commands: 'quit' to exit, 'personality' for traits, 'memory' for context, 'status' for DB stats."
)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Please set your OPENAI_API_KEY environment variable")
    print("   You can get one at: https://platform.openai.com/api-keys")
    print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# No JSONL history bootstrap — PMM is source of truth

while True:
    try:
        user = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if user.lower() in {"quit", "exit"}:
        print("Bye.")
        break
    if user.lower() == "personality":
        summary = pmm_memory.get_personality_summary()
        traits = summary["personality_traits"]
        print(
            f"🎭 Current Personality State:\n"
            f"   Openness: {traits['openness']:.2f}\n"
            f"   Conscientiousness: {traits['conscientiousness']:.2f}\n"
            f"   Extraversion: {traits['extraversion']:.2f}\n"
            f"   Agreeableness: {traits['agreeableness']:.2f}\n"
            f"   Neuroticism: {traits['neuroticism']:.2f}\n"
            f"📊 Events: {summary['total_events']} | Insights: {summary['total_insights']} | Open commitments: {summary['open_commitments']}"
        )
        continue
    if user.lower() == "memory":
        mem = pmm_memory.load_memory_variables({}).get("history", "")
        print("\n🧠 Cross-Session Memory Context:")
        print(mem[:800] if mem else "No cross-session memory yet")
        continue
    if user.lower() == "status":
        # Mirror chat.py lightweight status
        try:
            rows = pmm_memory.pmm.sqlite_store.all_events()
            total_events = len(rows)
            events_with_summaries = sum(1 for r in rows if len(r) >= 8 and r[7])
        except Exception:
            total_events = len(
                pmm_memory.pmm.model.self_knowledge.autobiographical_events
            )
            events_with_summaries = sum(
                1
                for e in pmm_memory.pmm.model.self_knowledge.autobiographical_events
                if getattr(e, "summary", None)
            )
        # DB path is co-located with the JSON model (unless PMM_DB_PATH is set)
        try:
            db_path = pmm_memory.pmm.sqlite_store.conn.execute(
                "PRAGMA database_list"
            ).fetchone()[2]
        except Exception:
            db_path = os.environ.get("PMM_DB_PATH") or str(PMM_PATH.parent / "pmm.db")
        try:
            size_bytes = os.path.getsize(db_path) if os.path.exists(db_path) else 0
        except Exception:
            size_bytes = 0
        size_kb = size_bytes / 1024.0
        print("\n📊 PMM Status:")
        print(
            f"   • Thought Summarization: {'ON' if os.getenv('PMM_ENABLE_SUMMARY','').lower() in ('1','true','yes','on') else 'OFF'}"
        )
        print(
            f"   • Semantic Embeddings: {'ON' if os.getenv('PMM_ENABLE_EMBEDDINGS','').lower() in ('1','true','yes','on') else 'OFF'}"
        )
        print(f"   • Database file: {db_path} ({size_kb:.1f} KB)")
        print(f"   • Total events: {total_events}")
        print(f"   • Events with summaries: {events_with_summaries}")
        continue

    # Build messages with fresh PMM system prompt each turn
    system_msg = build_system_prompt(pmm_memory)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user},
    ]
    # Invoke model
    result = llm.invoke(messages)
    text = result.content if hasattr(result, "content") else str(result)
    print(f"Assistant: {text}")
    # Persist via PMM memory
    try:
        pmm_memory.save_context({"input": user}, {"response": text})
    except Exception as e:
        print(f"[warn] save_context failed: {e}")
