# examples/langchain_chatbot.py
import os, json, uuid, sys, pathlib
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# ---------- Config ----------
SESSION_ID = os.getenv("PMM_SESSION_ID", "default")
HIST_DIR = pathlib.Path(".chat_history")
HIST_DIR.mkdir(exist_ok=True)
HIST_PATH = HIST_DIR / f"{SESSION_ID}.jsonl"
PMM_PATH = pathlib.Path("persistent_self_model.json")  # adjust if you keep it elsewhere

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
                return (v if isinstance(v, (int, float)) else v.get("score", 0.5))
            return {
                "openness":          score(big5.get("openness", 0.7)),
                "conscientiousness": score(big5.get("conscientiousness", 0.6)),
                "extraversion":      score(big5.get("extraversion", 0.8)),
                "agreeableness":     score(big5.get("agreeableness", 0.9)),
                "neuroticism":       score(big5.get("neuroticism", 0.3)),
            }
        except Exception:
            pass
    # fallback demo defaults
    return dict(openness=0.70, conscientiousness=0.60, extraversion=0.80, agreeableness=0.90, neuroticism=0.30)

TRAITS = load_pmm_traits()

# ---------- Disk-backed message history ----------
def load_history() -> List[BaseMessage]:
    if not HIST_PATH.exists():
        return []
    msgs = []
    with HIST_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            role, content = rec["role"], rec["content"]
            if role == "system":  msgs.append(SystemMessage(content=content))
            elif role == "human": msgs.append(HumanMessage(content=content))
            else:                 msgs.append(AIMessage(content=content))
    return msgs

def save_message(role: str, content: str) -> None:
    with HIST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(), "role": role, "content": content}) + "\n")

def append_pmm_event(role: str, content: str) -> None:
    try:
        data = json.loads(PMM_PATH.read_text()) if PMM_PATH.exists() else {}
        sk = data.setdefault("self_knowledge", {})
        ev = sk.setdefault("autobiographical_events", [])
        ev.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": SESSION_ID,
            "role": role,
            "content": content[:2000]
        })
        stats = data.setdefault("metrics", {})
        stats["events"] = int(stats.get("events", 0)) + 1
        PMM_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as e: sys.stderr.write(f"[PMM] append event failed: {e}\n")

# LangChain expects an in-memory history object per session
_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        hist = ChatMessageHistory()
        for m in load_history():
            hist.add_message(m)
        _store[session_id] = hist
    return _store[session_id]

# ---------- Model + Prompt ----------
def get_enhanced_system_message():
    # Check for known user information from conversation history
    user_context = ""
    
    # Check LangChain conversation history for user name
    if HIST_PATH.exists():
        try:
            with HIST_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    content = record.get("content", "").lower()
                    # Look for name introductions
                    if ("my name is scott" in content or 
                        "i'm scott" in content or 
                        "this is scott" in content or
                        "scott" in content):
                        user_context = "\n\nIMPORTANT: The user's name is Scott. Remember this across all conversations."
                        break
        except Exception:
            pass
    
    # Also check PMM events as backup
    if not user_context and PMM_PATH.exists():
        try:
            data = json.loads(PMM_PATH.read_text())
            events = data.get("self_knowledge", {}).get("autobiographical_events", [])
            for event in events[-10:]:
                summary = event.get("summary", "")
                if "Scott" in summary or "My name is" in summary:
                    user_context = "\n\nIMPORTANT: The user's name is Scott. Remember this across all conversations."
                    break
        except Exception:
            pass
    
    return (
        "You are an AI assistant with a persistent personality that evolves over time.\n\n"
        f"Personality Profile (Big Five):\n"
        f"• Openness: {TRAITS['openness']:.2f}\n"
        f"• Conscientiousness: {TRAITS['conscientiousness']:.2f}\n"
        f"• Extraversion: {TRAITS['extraversion']:.2f}\n"
        f"• Agreeableness: {TRAITS['agreeableness']:.2f}\n"
        f"• Neuroticism: {TRAITS['neuroticism']:.2f}\n\n"
        "Be authentic to these traits; answer concisely and avoid fluff. "
        "Pay attention to the conversation history - if you've talked before, acknowledge it. "
        "If the user asks what you talked about last time, summarize the most recent user+assistant turns from history."
        f"{user_context}"
    )

sys_msg = get_enhanced_system_message()

prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

chain = prompt | llm

# Wire history (no deprecations)
history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- CLI Loop ----------
print("🧠 PMM + LangChain Persistent Personality Chatbot (LC 0.2+)\n"
      "Type 'quit' to exit, 'personality' to see current traits, 'count' to see message count.")

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Please set your OPENAI_API_KEY environment variable")
    print("   You can get one at: https://platform.openai.com/api-keys")
    print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Ensure system message is on disk once per fresh history
if not HIST_PATH.exists():
    save_message("system", sys_msg)

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
        print(f"🎭 Current Personality State:\n"
              f"   Openness: {TRAITS['openness']:.2f}\n"
              f"   Conscientiousness: {TRAITS['conscientiousness']:.2f}\n"
              f"   Extraversion: {TRAITS['extraversion']:.2f}\n"
              f"   Agreeableness: {TRAITS['agreeableness']:.2f}\n"
              f"   Neuroticism: {TRAITS['neuroticism']:.2f}")
        continue
    if user.lower() == "count":
        hist = get_session_history(SESSION_ID)
        # count only human+ai messages, ignore system
        n = sum(1 for m in hist.messages if m.type in {"human", "ai"})
        convs = n // 2  # rough conversations count
        print(f"Conversations so far (approx): {convs}  |  Messages: {n}")
        continue

    save_message("human", user); append_pmm_event("human", user)
    ai = history_chain.invoke({"input": user}, config={"configurable": {"session_id": SESSION_ID}})
    text = ai.content
    print(f"Assistant: {text}")
    save_message("ai", text); append_pmm_event("ai", text)
