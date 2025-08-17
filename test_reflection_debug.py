#!/usr/bin/env python3
"""Debug script to isolate reflection failures."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add PMM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def test_openai_connection():
    """Test basic OpenAI API connection."""
    try:
        from pmm.adapters.openai_adapter import OpenAIAdapter

        print("🔍 Testing OpenAI connection...")
        adapter = OpenAIAdapter()

        # Simple test call
        response = adapter.chat(
            system="You are a helpful assistant.",
            user="Say 'Hello' in exactly one word.",
            max_tokens=10,
        )

        print(f"✅ OpenAI connection successful: '{response}'")
        return True

    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False


def test_reflection_system():
    """Test the reflection system in isolation."""
    try:
        from pmm.self_model_manager import SelfModelManager
        from pmm.adapters.openai_adapter import OpenAIAdapter
        from pmm.reflection import reflect_once

        print("🔍 Testing reflection system...")

        # Create minimal PMM manager
        mgr = SelfModelManager()
        adapter = OpenAIAdapter()

        # Add some basic context
        mgr.add_thought("Test thought for reflection context")

        print("🔍 Calling reflect_once...")
        insight = reflect_once(mgr, adapter)

        if insight:
            print(f"✅ Reflection successful: {insight.content[:100]}...")
            return True
        else:
            print("❌ Reflection returned None")
            return False

    except Exception as e:
        print(f"❌ Reflection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 PMM Reflection Debug Test")
    print("=" * 40)

    # Test 1: Basic OpenAI connection
    openai_ok = test_openai_connection()
    print()

    # Test 2: Reflection system (only if OpenAI works)
    if openai_ok:
        reflection_ok = test_reflection_system()
    else:
        print("⏭️  Skipping reflection test due to OpenAI connection failure")
        reflection_ok = False

    print()
    print("=" * 40)
    print("🏁 Debug Results:")
    print(f"   OpenAI Connection: {'✅ OK' if openai_ok else '❌ FAILED'}")
    print(f"   Reflection System: {'✅ OK' if reflection_ok else '❌ FAILED'}")
