#!/usr/bin/env python3
"""
Simple test to check SQLite memory loading without LangChain dependencies.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.storage.sqlite_store import SQLiteStore


def test_sqlite_memory():
    """Test if SQLite contains Scott's name and recent events."""
    print("🧪 Testing SQLite Memory Storage...")

    try:
        # Initialize SQLite store
        store = SQLiteStore("pmm.db")

        # Get recent events
        recent_events = store.recent_events(limit=20)
        print(f"✅ Found {len(recent_events)} recent events in SQLite")

        # Look for Scott's name
        scott_events = []
        for event in recent_events:
            # SQLite returns Row objects, access by column name
            event_id = event["id"]
            ts = event["ts"]
            kind = event["kind"]
            content = event["content"]
            if "scott" in content.lower():
                scott_events.append((event_id, ts, kind, content))

        print(f"\n🔍 Found {len(scott_events)} events mentioning 'Scott':")
        for event_id, ts, kind, content in scott_events[-5:]:  # Show last 5
            print(f"   [{event_id}] {ts} ({kind}): {content[:80]}...")

        # Test the memory loading logic manually
        print("\n🧠 Testing memory extraction logic...")
        conversation_history = []
        key_facts = []

        for event in reversed(recent_events):  # Reverse to get chronological order
            # SQLite returns Row objects, access by column name
            event_id = event["id"]
            ts = event["ts"]
            kind = event["kind"]
            content = event["content"]
            if kind in ["event", "response", "prompt"]:
                # Format for LLM context
                if "User said:" in content:
                    user_msg = content.replace("User said: ", "")
                    conversation_history.append(f"Human: {user_msg}")

                    # Extract key information automatically
                    if "my name is" in user_msg.lower() or "i am" in user_msg.lower():
                        key_facts.append(f"IMPORTANT: {user_msg}")
                        print(f"   [FOUND NAME] {user_msg}")

                elif "I responded:" in content:
                    ai_msg = content.replace("I responded: ", "")
                    conversation_history.append(f"Assistant: {ai_msg}")

                    # Extract commitments and identity info
                    if "next, i will" in ai_msg.lower() or "scott" in ai_msg.lower():
                        key_facts.append(f"COMMITMENT/IDENTITY: {ai_msg}")
                        print(f"   [FOUND IDENTITY] {ai_msg[:60]}...")

                elif kind == "event":
                    conversation_history.append(f"Context: {content}")

        print("\n📊 Summary:")
        print(f"   Key facts extracted: {len(key_facts)}")
        print(f"   Conversation items: {len(conversation_history)}")

        if key_facts:
            print("\n🔑 Key facts that should be remembered:")
            for fact in key_facts[-3:]:  # Show last 3
                print(f"   {fact}")
        else:
            print("\n❌ No key facts extracted - this is the problem!")

        if len(conversation_history) > 0:
            print("\n💬 Recent conversation context (last 5 items):")
            for item in conversation_history[-5:]:
                print(f"   {item[:80]}...")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_sqlite_memory()
