#!/usr/bin/env python3
"""
Simple test to check if memory loading is working correctly.
This tests the PersistentMindMemory class without requiring LLM APIs.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pmm.langchain_memory import PersistentMindMemory
from pmm.self_model_manager import SelfModelManager


def test_memory_loading():
    """Test if memory loading correctly extracts historical context."""
    print("🧪 Testing PMM Memory Loading...")

    # Initialize PMM memory system
    try:
        pmm = SelfModelManager("persistent_self_model.json")
        memory = PersistentMindMemory(pmm=pmm)

        print(
            f"✅ PMM initialized with {len(pmm.model.self_knowledge.autobiographical_events)} events"
        )

        # Test memory loading
        print("\n🔍 Testing memory variable loading...")
        memory_vars = memory.load_memory_variables({})

        print(f"📊 Memory variables keys: {list(memory_vars.keys())}")

        if "history" in memory_vars:
            history_content = memory_vars["history"]
            print(f"📝 History content length: {len(history_content)} characters")

            # Check if Scott's name appears in the loaded context
            if "scott" in history_content.lower():
                print("✅ SUCCESS: Scott's name found in memory context!")
                print("\n📋 Relevant context snippet:")
                # Find and print lines containing Scott
                lines = history_content.split("\n")
                scott_lines = [line for line in lines if "scott" in line.lower()]
                for line in scott_lines[:3]:  # Show first 3 relevant lines
                    print(f"   {line}")
            else:
                print("❌ FAILURE: Scott's name NOT found in memory context")
                print("\n📋 Current context preview (first 500 chars):")
                print(history_content[:500])

        else:
            print("❌ No 'history' key found in memory variables")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_memory_loading()
