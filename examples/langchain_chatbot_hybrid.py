# examples/langchain_chatbot_hybrid.py
"""
Hybrid LangChain + PMM Integration: Best of Both Worlds

This combines:
- Modern LangChain APIs (ChatOpenAI + RunnableWithMessageHistory) 
- PMM cross-session persistence (remembers users across sessions)
- Real episodic memory (disk-backed conversation history)
- Zero deprecated APIs
"""

import os, json, sys, pathlib
from datetime import datetime, timezone
from typing import Dict, List, Any

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add PMM to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory

# Import PMM system for cross-session persistence
from pmm.langchain_memory import PersistentMindMemory

# ---------- Config ----------
SESSION_ID = os.getenv("PMM_SESSION_ID", "default")
HIST_DIR = pathlib.Path(".chat_history")
HIST_DIR.mkdir(exist_ok=True)
HIST_PATH = HIST_DIR / f"{SESSION_ID}.jsonl"

MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("PMM_TEMP", "0.7"))

# ---------- Initialize PMM System ----------
pmm_memory = PersistentMindMemory(
    agent_path="langchain_hybrid_agent.json",
    personality_config={
        "openness": 0.7,
        "conscientiousness": 0.6, 
        "extraversion": 0.8,
        "agreeableness": 0.9,
        "neuroticism": 0.3
    }
)

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

# LangChain expects an in-memory history object per session
_store: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _store:
        hist = ChatMessageHistory()
        for m in load_history():
            hist.add_message(m)
        _store[session_id] = hist
    return _store[session_id]

# ---------- Enhanced System Message with PMM Context ----------
def get_hybrid_system_message():
    # Get PMM personality context (includes cross-session memory)
    pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
    
    # Extract personality summary
    personality = pmm_memory.get_personality_summary()
    traits = personality["personality_traits"]
    
    return (
        "You are an AI assistant with a persistent personality that evolves over time.\n\n"
        f"Personality Profile (Big Five):\n"
        f"• Openness: {traits['openness']:.2f}\n"
        f"• Conscientiousness: {traits['conscientiousness']:.2f}\n"
        f"• Extraversion: {traits['extraversion']:.2f}\n"
        f"• Agreeableness: {traits['agreeableness']:.2f}\n"
        f"• Neuroticism: {traits['neuroticism']:.2f}\n\n"
        f"Cross-Session Memory Context:\n{pmm_context}\n\n"
        "Be authentic to your personality traits and remember information from previous sessions. "
        "If you recognize the user from past conversations, acknowledge it naturally. "
        "Pay attention to both the immediate conversation history and your cross-session memories."
    )

# ---------- Model + Prompt ----------
sys_msg = get_hybrid_system_message()

prompt = ChatPromptTemplate.from_messages([
    ("system", sys_msg),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
chain = prompt | llm

# Wire history (modern APIs)
history_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- CLI Loop ----------
print("\n" + "=" * 80)
print("🧠 PMM + LangChain Hybrid Chatbot")
print("   Modern APIs + Cross-Session Memory + Persistent Personality")
print("=" * 80)
print()

print("📋 Features:")
print("   • Modern LangChain APIs with zero deprecation warnings")
print("   • PMM persistent personality that evolves over time")
print("   • Cross-session memory (remembers you between conversations)")
print()

print("💬 Commands:")
print("   • Type your message to chat normally")
print("   • 'personality' - View current personality traits")
print("   • 'memory' - View cross-session memory context")
print("   • 'quit' or 'exit' - End conversation")
print()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY Required")
    print("   Get your key at: https://platform.openai.com/api-keys")
    print("   Set with: export OPENAI_API_KEY='your-key-here'")
    print()
    sys.exit(1)

# Show initial personality
personality = pmm_memory.get_personality_summary()
print("🎭 Current Personality State:")
print("   ┌─────────────────────────────────────┐")
for trait, score in personality["personality_traits"].items():
    print(f"   │ {trait.title():<15} : {score:>6.2f}     │")
print("   └─────────────────────────────────────┘")
print()

print("📊 Session Info:")
print(f"   • Total Events: {personality['total_events']}")
print(f"   • Active Commitments: {personality['open_commitments']}")
print(f"   • Model: {MODEL}")
print()

print("Ready to chat! 🚀")
print("-" * 40)
print()

# Ensure system message is on disk once per fresh history
if not HIST_PATH.exists():
    save_message("system", sys_msg)

while True:
    try:
        user = input("💬 You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        break

    if user.lower() in {"quit", "exit"}:
        print("👋 Goodbye!")
        break
        
    if user.lower() == "personality":
        personality = pmm_memory.get_personality_summary()
        print(f"\n🎭 Current Personality State:")
        print("   ┌─────────────────────────────────────┐")
        for trait, score in personality["personality_traits"].items():
            print(f"   │ {trait.title():<15} : {score:>6.2f}     │")
        print("   └─────────────────────────────────────┘")
        print()
        print("📊 Statistics:")
        print(f"   • Events: {personality['total_events']}")
        print(f"   • Insights: {personality['total_insights']}")
        print(f"   • Open Commitments: {personality['open_commitments']}")
        if personality["behavioral_patterns"]:
            print(f"   • Patterns: {list(personality['behavioral_patterns'].keys())}")
        print()
        continue
        
    if user.lower() == "memory":
        pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
        print(f"\n🧠 Cross-Session Memory Context:")
        print("   ┌" + "─" * 60 + "┐")
        if pmm_context:
            # Format memory context with proper line breaks
            context_lines = pmm_context[:500].split('\n')
            for line in context_lines[:8]:  # Show first 8 lines
                print(f"   │ {line[:58]:<58} │")
            if len(pmm_context) > 500:
                print(f"   │ {'... (truncated)':^58} │")
        else:
            print(f"   │ {'No cross-session memory yet':^58} │")
        print("   └" + "─" * 60 + "┘")
        print()
        continue

    if not user:
        continue

    # Save to both systems
    save_message("human", user)
    
    # Defer PMM persistence until after AI response to avoid duplicate entries
    # Get AI response through modern LangChain
    ai = history_chain.invoke({"input": user}, config={"configurable": {"session_id": SESSION_ID}})
    text = ai.content
    print(f"\n🤖 Assistant: {text}")
    print()
    
    # Save AI response to both systems
    save_message("ai", text)
    pmm_memory.save_context({"input": user}, {"response": text})

print("\n" + "=" * 60)
print("🎯 Session Complete!")
print("   Your AI assistant's personality has evolved through our conversation.")
print("   Cross-session memory and personality will be restored next time.")
print("=" * 60)
