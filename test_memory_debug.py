#!/usr/bin/env python3
# ruff: noqa: E402
"""
Test memory loading with debug output to see exactly what's happening.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Mock LangChain dependencies to test our logic
class MockBaseChatMemory:
    def __init__(self):
        pass

    def load_memory_variables(self, inputs):
        return {}  # Return empty base variables

    def save_context(self, inputs, outputs):
        pass


# Monkey patch the import
sys.modules["langchain.memory.chat_memory"] = type(
    "module", (), {"BaseChatMemory": MockBaseChatMemory}
)()
sys.modules["langchain_core.pydantic_v1"] = type(
    "module", (), {"Field": lambda **kwargs: None}
)()

from pmm.langchain_memory import PersistentMindMemory
from pmm.self_model_manager import SelfModelManager


def test_memory_debug():
    """Test memory loading with full debug output."""
    print("🧪 Testing PMM Memory Loading with Debug Output...")

    try:
        # Initialize PMM memory system
        pmm = SelfModelManager("persistent_self_model.json")

        # Create memory instance with mock base class
        memory = PersistentMindMemory.__new__(PersistentMindMemory)
        memory.pmm = pmm
        memory.memory_key = "history"
        memory.input_key = "input"
        memory.output_key = "response"
        memory.personality_context = ""
        memory.commitment_context = ""

        print(
            f"✅ PMM initialized with {len(pmm.model.self_knowledge.autobiographical_events)} events"
        )

        # Test memory loading with debug output
        print("\n🔍 Testing memory variable loading with debug output...")
        memory_vars = memory.load_memory_variables({})

        print("\n📊 Final Results:")
        print(f"   Memory variables: {list(memory_vars.keys())}")

        if "history" in memory_vars:
            history = memory_vars["history"]
            print(f"   History length: {len(history)} characters")

            # Check for Scott's name
            if "scott" in history.lower():
                print("   ✅ SUCCESS: Scott's name found in memory!")
            else:
                print("   ❌ FAILURE: Scott's name NOT found")

            # Show a sample of the history
            print("\n📝 History sample (first 300 chars):")
            print(f"   {history[:300]}...")
        else:
            print("   ❌ No 'history' key in memory variables")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_memory_debug()
