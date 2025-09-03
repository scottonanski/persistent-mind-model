#!/usr/bin/env python3
"""
PMM Session Runner - Executes real PMM sessions with telemetry capture
"""
import os
from datetime import datetime
import sys
import json
from pathlib import Path
import sqlite3

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

    # Initialize PMM memory system with a single canonical agent path
    AGENT_PATH = os.getenv("PMM_AGENT_PATH", ".data/pmm.json")
    # Optional CLI override: --agent <path>
    if "--agent" in sys.argv:
        try:
            AGENT_PATH = sys.argv[sys.argv.index("--agent") + 1]
        except Exception:
            pass
    print("AGENT_PATH:", AGENT_PATH)
    # Ensure parent directory exists to avoid silent fallback
    try:
        parent = os.path.dirname(AGENT_PATH) or "."
        os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    memory = PersistentMindMemory(agent_path=AGENT_PATH)

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
        "Please remember that my favorite color is blue. Can you do that for me?",
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

            # After turn 2, open one concrete, immediately satisfiable commitment
            if i == 1:  # After the 2nd turn (0-indexed)
                try:
                    cid = memory.pmm.add_commitment(
                        "I will produce a one-line PMM session summary now.",
                        source_insight_id="run_a_session",
                        due=None,
                    )
                    print("SYSTEM: opened commitment:", cid)
                except Exception as e:
                    print("SYSTEM: Failed to open commitment:", e)

            # After turn 3, simulate closing a commitment to test evidence mechanism
            if i == 2:  # After the 3rd turn (0-indexed)
                try:
                    tracker = memory.pmm.commitment_tracker  # CommitmentTracker instance
                    open_list = tracker.get_open_commitments()  # returns dicts with "hash" keys
                    if open_list:
                        commit_hash = open_list[-1]["hash"]  # close the most recent open one
                        ok = tracker.close_commitment_with_evidence(
                            commit_hash,
                            evidence_type="done",
                            description="auto evidence from run_a_session.py",
                            artifact="run_a_session.py",  # any non-empty artifact string is accepted
                        )
                        print(f"SYSTEM: close_with_evidence -> {ok}")
                        # Mirror to SQLite so analyzer sees closure
                        try:
                            smm = memory.pmm
                            # Append evidence event
                            evidence_content = {
                                "type": "done",
                                "summary": "Auto evidence from run_a_session.py",
                                "artifact": {"file": "run_a_session.py"},
                                "confidence": 0.9,
                            }
                            smm.sqlite_store.append_event(
                                kind="evidence",
                                content=json.dumps(evidence_content, ensure_ascii=False),
                                meta={"commit_ref": commit_hash, "subsystem": "harness"},
                            )
                            # Append commitment.close event
                            smm.sqlite_store.append_event(
                                kind="commitment.close",
                                content=json.dumps({"reason": "harness"}, ensure_ascii=False),
                                meta={"commit_ref": commit_hash, "subsystem": "harness"},
                            )
                        except Exception as ee:
                            print("SYSTEM: sqlite mirror failed:", ee)
                        print(tracker.get_commitment_metrics())
                except Exception as e:
                    print(f"SYSTEM: Failed to simulate commitment closure: {e}")

            # Debug: ensure we're analyzing the same store and snapshot (post-close)
            # Print the actual SQLite file path via PRAGMA database_list
            store_path = "?"
            try:
                conn = getattr(getattr(memory.pmm, "sqlite_store", object()), "conn", None)
                if conn:
                    rows = conn.execute("PRAGMA database_list").fetchall()
                    if rows:
                        store_path = rows[-1][2]
            except Exception:
                pass
            print("STORE PATH:", store_path)
            try:
                snapshot = memory.pmm.commitment_tracker.get_commitment_metrics()
                print("OPEN/CLOSED SNAPSHOT:", snapshot)
            except Exception as e:
                print("OPEN/CLOSED SNAPSHOT: <error>", e)

            # Compute emergence scores from the same sqlite store
            scores = compute_emergence_scores(storage_manager=memory.pmm.sqlite_store)
            print("ANALYZER CONTEXT:", scores.get("events_analyzed"), scores.get("stage"))
            print("ANALYZER DB:", store_path)
            ias = scores.get("ias", 0.0)
            gas = scores.get("gas", 0.0)

            # Get commitment close rate (prefer emergence snapshot if available)
            close_rate = scores.get("commit_close_rate") or scores.get("close") or 0.0

            # Assert alignment between tracker and analyzer close rate when available
            try:
                if snapshot and isinstance(snapshot, dict) and "close_rate" in snapshot:
                    assert abs((snapshot.get("close_rate") or 0.0) - (close_rate or 0.0)) < 1e-6, (
                        f"Close mismatch: tracker {snapshot.get('close_rate')} vs analyzer {close_rate}"
                    )
            except AssertionError as ae:
                print("ALIGNMENT WARNING:", ae)

            telemetry["ias_scores"].append(ias)
            telemetry["gas_scores"].append(gas)
            telemetry["close_rates"].append(close_rate)

            # Optional: show last 3 event kinds for sanity
            try:
                if store_path and store_path != "?":
                    con = sqlite3.connect(store_path)
                    cur = con.cursor()
                    cur.execute("SELECT kind FROM events ORDER BY id DESC LIMIT 3")
                    kinds = [r[0] for r in cur.fetchall()]
                    con.close()
                    print("LAST 3 EVENT KINDS:", kinds)
            except Exception as e:
                print("EVENT TAIL: <error>", e)

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
