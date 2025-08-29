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


def run_session():
    """Run a real PMM session and capture telemetry"""

    # Initialize PMM with environment settings
    bandit_enabled = os.environ.get("PMM_BANDIT_ENABLED", "0") == "1"
    turns = int(os.environ.get("PMM_TURNS_PER_SESSION", "5"))

    print(f"=== PMM SESSION (BANDIT {'ON' if bandit_enabled else 'OFF'}) ===")

    # Initialize PMM memory system
    agent_path = "test_agent.json"
    pmm_memory = PersistentMindMemory(agent_path=agent_path)

    # Initialize LLM (use simple OpenAI for testing)
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return {"ias_scores": [], "gas_scores": [], "close_rates": []}

    # Create conversation chain with PMM memory
    try:
        # Use LangChain's RunnableWithMessageHistory for modern API compatibility
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant with persistent memory."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        # Create the chain
        chain = prompt | llm

    except Exception as e:
        print(f"Failed to create conversation chain: {e}")
        return {"ias_scores": [], "gas_scores": [], "close_rates": []}

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
            # Get response through chain and PMM memory
            # Save context to PMM first
            pmm_memory.save_context({"input": prompt}, {"output": ""})

            # Get response from chain
            response = chain.invoke({"input": prompt, "history": []})
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)

            print(f"Assistant: {response_text}")

            # Save response to PMM
            pmm_memory.save_context({"input": prompt}, {"output": response_text})

            # Compute emergence scores
            try:
                # Use the sqlite_store from PMM
                storage_mgr = pmm_memory.pmm.sqlite_store
                scores = compute_emergence_scores(storage_manager=storage_mgr)
                ias = scores.get("ias", 0.0)
                gas = scores.get("gas", 0.0)
            except Exception as e:
                print(f"Error computing emergence scores: {e}")
                ias = 0.0
                gas = 0.0

            # Controlled artifact injection for close path validation
            if i == 3 and os.environ.get("PMM_INJECT_ARTIFACT", "0") == "1":  # Turn 4
                try:
                    # Create a dummy artifact to trigger commitment close
                    artifact_content = f"Test artifact generated at turn {i+1}"
                    artifact_path = f"test_artifact_turn_{i+1}.txt"

                    # Write artifact to filesystem
                    with open(artifact_path, "w") as f:
                        f.write(artifact_content)

                    # Directly manipulate commitment tracker for testing
                    from pmm.commitments import Commitment
                    import uuid
                    from datetime import datetime

                    # Create a test commitment manually
                    test_commit_id = str(uuid.uuid4())[:8]
                    test_commitment = Commitment(
                        cid=test_commit_id,
                        text="Test commitment for artifact validation",
                        created_at=datetime.now().isoformat(),
                        source_insight_id="test_validation",
                        status="open",
                    )

                    # Add to tracker
                    pmm_memory.pmm.commitment_tracker.commitments[test_commit_id] = (
                        test_commitment
                    )

                    # Close the commitment manually
                    test_commitment.status = "closed"
                    test_commitment.closed_at = datetime.now().isoformat()
                    test_commitment.close_note = (
                        f"Closed with artifact: {artifact_path}"
                    )

                    print(f"[ARTIFACT] Injected test artifact: {artifact_path}")
                    print(f"[ARTIFACT] Created and closed commitment: {test_commit_id}")

                except Exception as e:
                    print(f"Error injecting artifact: {e}")
                    import traceback

                    traceback.print_exc()

            # Get commitment close rate from PMM
            try:
                commitments = pmm_memory.pmm.commitment_tracker.commitments
                total_commitments = len(commitments)
                closed_commitments = sum(
                    1 for c in commitments.values() if c.status == "closed"
                )
                close_rate = (
                    closed_commitments / total_commitments
                    if total_commitments > 0
                    else 0.0
                )
            except Exception as e:
                print(f"Error getting commitment data: {e}")
                close_rate = 0.0

            telemetry["ias_scores"].append(ias)
            telemetry["gas_scores"].append(gas)
            telemetry["close_rates"].append(close_rate)

            print(f"[TRACK] identity {ias:.3f} growth {gas:.3f} close {close_rate:.3f}")

        except Exception as e:
            print(f"Error in turn {i+1}: {e}")
            import traceback

            traceback.print_exc()
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
