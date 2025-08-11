#!/usr/bin/env python3
"""
Simple LangChain + PMM Integration Test

Tests the core functionality without interactive input.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pmm.langchain_memory import PersistentMindMemory

def run_langchain_integration() -> bool:
    """Test PMM LangChain integration without LLM calls."""
    print("🧠 Testing PMM + LangChain Integration")
    print("=" * 40)
    
    # Initialize PMM memory with custom personality
    memory = PersistentMindMemory(
        agent_path="test_langchain_agent.json",
        personality_config={
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.8,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        }
    )
    
    print("✅ PMM Memory initialized successfully")
    
    # Test personality summary
    personality = memory.get_personality_summary()
    print(f"✅ Agent ID: {personality['agent_id']}")
    print(f"✅ Agent Name: {personality['name']}")
    
    print("\n🎭 Personality Traits:")
    for trait, score in personality["personality_traits"].items():
        print(f"   {trait.title()}: {score:.2f}")
    
    print(f"\n📊 Stats:")
    print(f"   Events: {personality['total_events']}")
    print(f"   Insights: {personality['total_insights']}")
    print(f"   Commitments: {personality['open_commitments']}")
    
    # Test memory variables
    memory_vars = memory.load_memory_variables({})
    print(f"\n✅ Memory variables loaded: {list(memory_vars.keys())}")
    
    # Test save context (simulate conversation)
    test_inputs = {"input": "Hello, how are you?"}
    test_outputs = {"response": "I'm doing well! Next: I will help you with any questions you have."}
    
    try:
        memory.save_context(test_inputs, test_outputs)
        print("✅ Context saved successfully")
        
        # Check if commitment was extracted
        updated_personality = memory.get_personality_summary()
        print(f"✅ Commitments after interaction: {updated_personality['open_commitments']}")
        
    except Exception as e:
        print(f"❌ Error saving context: {e}")
        return False
    
    print("\n🎯 LangChain Integration Test: SUCCESS!")
    return True


def test_langchain_integration():
    ok = run_langchain_integration()
    assert ok is True

if __name__ == "__main__":
    success = run_langchain_integration()
    sys.exit(0 if success else 1)
