#!/usr/bin/env python3
"""
LangChain + PMM Integration Example: Persistent Personality Chatbot

This example demonstrates how to use PMM's LangChain wrapper to create
a chatbot with persistent personality that evolves over time.

Key Features Demonstrated:
- Persistent personality traits (Big Five)
- Automatic commitment extraction and tracking
- Personality evolution through conversations
- Model-agnostic consciousness (works with any LLM)

Usage:
    python examples/langchain_chatbot.py
"""

import os
import sys
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from pmm.langchain_memory import PersistentMindMemory


def create_personality_aware_prompt():
    """Create a prompt template that leverages PMM personality context."""
    template = """You are an AI assistant with a persistent personality that evolves over time.

{history}

Your responses should be consistent with your personality traits and any active commitments you've made. 
Be authentic to your personality while being helpful and engaging.

Human: {input}
Assistant:"""
    
    return PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )


def main():
    """Run the persistent personality chatbot demo."""
    print("🧠 PMM + LangChain Persistent Personality Chatbot")
    print("=" * 50)
    print("This chatbot has a persistent personality that evolves over time!")
    print("Type 'quit' to exit, 'personality' to see current traits.\n")
    
    # Initialize PMM memory with custom personality
    memory = PersistentMindMemory(
        agent_path="langchain_agent.json",
        personality_config={
            "openness": 0.7,        # Creative and curious
            "conscientiousness": 0.6, # Moderately organized
            "extraversion": 0.8,     # Outgoing and energetic
            "agreeableness": 0.9,    # Very cooperative
            "neuroticism": 0.3       # Calm and stable
        }
    )
    
    # Create LangChain conversation chain
    llm = OpenAI(temperature=0.7)
    prompt = create_personality_aware_prompt()
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    
    # Show initial personality
    print("🎭 Initial Personality:")
    personality = memory.get_personality_summary()
    for trait, score in personality["personality_traits"].items():
        print(f"   {trait.title()}: {score:.2f}")
    print()
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'personality':
                print("\n🎭 Current Personality State:")
                personality = memory.get_personality_summary()
                for trait, score in personality["personality_traits"].items():
                    print(f"   {trait.title()}: {score:.2f}")
                print(f"   Events: {personality['total_events']}")
                print(f"   Insights: {personality['total_insights']}")
                print(f"   Open Commitments: {personality['open_commitments']}")
                if personality["behavioral_patterns"]:
                    print(f"   Patterns: {personality['behavioral_patterns']}")
                print()
                continue
            elif user_input.lower() == 'reflect':
                print("\n🪞 Triggering reflection...")
                insight = memory.trigger_reflection()
                if insight:
                    print(f"Insight: {insight}")
                else:
                    print("No new insights generated.")
                print()
                continue
            
            if not user_input:
                continue
            
            # Get response from PMM-powered chain
            response = chain.predict(input=user_input)
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Show final personality state
    print("\n🎭 Final Personality State:")
    personality = memory.get_personality_summary()
    for trait, score in personality["personality_traits"].items():
        print(f"   {trait.title()}: {score:.2f}")
    print(f"\nTotal conversation events: {personality['total_events']}")
    print(f"Insights generated: {personality['total_insights']}")
    print(f"Active commitments: {personality['open_commitments']}")
    
    print("\n🎯 Your AI assistant's personality has evolved through our conversation!")
    print("The same personality will be restored in future conversations.")


if __name__ == "__main__":
    main()
