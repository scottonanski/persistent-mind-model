#!/usr/bin/env python3
"""
PMM Session Runner - Executes real PMM sessions with telemetry capture
"""
import os
import sys
import json
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmm.langchain_memory import PersistentMindMemory
from pmm.emergence import compute_emergence_scores
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain


def run_session():
    """Run a real PMM session and capture telemetry"""

    # Initialize PMM with environment settings
    bandit_enabled = os.environ.get("PMM_BANDIT_ENABLED", "0") == "1"
    turns = int(os.environ.get("PMM_TURNS_PER_SESSION", "5"))

    print(f"=== PMM SESSION (BANDIT {'ON' if bandit_enabled else 'OFF'}) ===")

    # Initialize PMM memory system
    agent_path = "test_agent.json"
    memory = PersistentMindMemory(agent_path=agent_path)

    # Initialize LLM (use simple OpenAI for testing)
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return {"ias_scores": [], "gas_scores": [], "close_rates": []}

    # Build a simple conversation chain that uses PMM memory so events are logged
    chain = ConversationChain(llm=llm, memory=memory, verbose=False)

    # Test prompts
    prompts = [
        "Hello! What's your name?",
        "Tell me about your personality.",
        "What are you working on?",
        "How do you learn and grow?",
        "What defines your identity?",
    ]

    telemetry = {"ias_scores": [], "gas_scores": [], "close_rates": []}

    # Run conversation turns
    for i in range(min(turns, len(prompts))):
        prompt = prompts[i]
        print(f"\n--- TURN {i+1} ---")
        print(f"User: {prompt}")

        try:
            # Get response via LangChain ConversationChain (this will call memory.save_context)
            response_text = chain.predict(input=prompt)
            print(f"Assistant: {response_text}")

            # Compute emergence scores
            scores = compute_emergence_scores(memory.pmm.sqlite_store)
            ias = scores.get("ias", 0.0)
            gas = scores.get("gas", 0.0)

            # Get commitment close rate (prefer emergence snapshot if available)
            close_rate = scores.get("commit_close_rate") or scores.get("close") or 0.0

            telemetry["ias_scores"].append(ias)
            telemetry["gas_scores"].append(gas)
            telemetry["close_rates"].append(close_rate)

            print(f"Telemetry: IAS={ias:.3f}, GAS={gas:.3f}, Close={close_rate:.3f}")

        except Exception as e:
            print(f"Error in turn {i+1}: {e}")
            # Add zero scores for failed turns
            telemetry["ias_scores"].append(0.0)
            telemetry["gas_scores"].append(0.0)
            telemetry["close_rates"].append(0.0)

    print("\n=== SESSION COMPLETE ===")
    print(f"Final telemetry: {telemetry}")

    # Output telemetry as JSON for the AB test to capture
    print(f"TELEMETRY_JSON: {json.dumps(telemetry)}")

    return telemetry


if __name__ == "__main__":
    run_session()
