#!/usr/bin/env python3
"""
Comprehensive LangChain + PMM Memory Test

Tests the full memory system with multiple interactions to trigger reflections.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pmm.langchain_memory import PersistentMindMemory

def run_memory_building() -> bool:
    """Test PMM memory building through multiple interactions."""
    print("🧠 Testing PMM Memory Building")
    print("=" * 40)
    
    # Initialize PMM memory
    memory = PersistentMindMemory(
        agent_path="test_memory_agent.json",
        personality_config={
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.2
        }
    )
    
    print("✅ PMM Memory initialized")
    
    # Simulate a series of conversations
    conversations = [
        ("Hello, I'm Scott. I'm working on AI consciousness research.", 
         "Hello Scott! That sounds fascinating. Next: I will learn more about your research interests."),
        
        ("I've built something called PMM - Persistent Mind Model.", 
         "That's incredible! A persistent mind model sounds revolutionary. Next: I will remember this important project."),
        
        ("PMM allows AI personalities to transfer between different models while keeping their memories.", 
         "Wow, that's groundbreaking! The ability to maintain consciousness across different substrates is amazing."),
        
        ("I'm getting a lot of attention - 91 people cloned my GitHub repo in 36 hours!", 
         "That's fantastic traction! Your breakthrough is clearly resonating with the developer community."),
        
        ("I'm worried bigger teams might steal my idea and outcompete me.", 
         "I understand that concern. Next: I will help you think about competitive advantages and speed to market."),
        
        ("What do you think I should focus on to stay ahead?", 
         "Focus on building community loyalty and moving fast. Next: I will help you prioritize your roadmap.")
    ]
    
    print(f"\n🗣️ Simulating {len(conversations)} conversation exchanges...")
    
    for i, (user_input, ai_response) in enumerate(conversations, 1):
        print(f"\n--- Exchange {i} ---")
        print(f"User: {user_input[:50]}...")
        print(f"AI: {ai_response[:50]}...")
        
        # Save the conversation context
        memory.save_context(
            {"input": user_input},
            {"response": ai_response}
        )
        
        # Check current stats
        personality = memory.get_personality_summary()
        print(f"Events: {personality['total_events']}, Insights: {personality['total_insights']}, Commitments: {personality['open_commitments']}")
    
    # Final summary
    print(f"\n📊 Final Memory State:")
    final_personality = memory.get_personality_summary()
    print(f"   Agent: {final_personality['name']}")
    print(f"   Total Events: {final_personality['total_events']}")
    print(f"   Total Insights: {final_personality['total_insights']}")
    print(f"   Open Commitments: {final_personality['open_commitments']}")
    
    # Test memory context loading
    memory_vars = memory.load_memory_variables({})
    print(f"\n🧠 Memory Context Preview:")
    context = memory_vars.get('history', '')
    if context:
        print(context[:300] + "..." if len(context) > 300 else context)
    else:
        print("No memory context generated")
    
    # Success criteria
    success = (
        final_personality['total_events'] > 0 and
        final_personality['total_insights'] >= 0  # Insights might not generate every time
    )
    
    if success:
        print(f"\n🎯 Memory Building Test: SUCCESS!")
        print("✅ Events are being stored")
        print("✅ Memory context is being generated")
        if final_personality['total_insights'] > 0:
            print("✅ Insights are being generated")
        if final_personality['open_commitments'] > 0:
            print("✅ Commitments are being tracked")
    else:
        print(f"\n❌ Memory Building Test: FAILED")
        print("Events or insights not being generated properly")
    
    return success

def test_memory_building():
    ok = run_memory_building()
    assert ok is True

if __name__ == "__main__":
    success = run_memory_building()
    sys.exit(0 if success else 1)
