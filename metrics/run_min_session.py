#!/usr/bin/env python3
import os
import json
from pathlib import Path

# Keep the environment quiet & single-threaded
os.environ.setdefault("PMM_AUTONOMY_AUTOSTART", "0")
os.environ.setdefault("PMM_EMERGENCE_DEBUG", "0")

# Ensure imports work when run from project root
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pmm.self_model_manager import SelfModelManager
from pmm.emergence import compute_emergence_scores


def log_turn(store_mgr, user_text: str, ai_text: str):
    # Only the two events the analyzer truly needs
    store_mgr.add_event(summary=f"User: {user_text}", effects=[], etype="conversation")
    store_mgr.add_event(summary=f"AI: {ai_text}", effects=[], etype="self_expression")


def run(turns=5, use_llm=False):
    mgr = SelfModelManager("test_agent_min.json")

    # Optionally wire a real LLM; default is echo for determinism
    def reply(u):
        return u if not use_llm else "<hook up your LLM here>"

    prompts = [
        "Hi—what's your name?",
        "Describe your working style.",
        "What are you focusing on today?",
        "How do you improve?",
        "What defines your identity?",
    ][:turns]

    telemetry = {"IAS": [], "GAS": [], "commit_close_rate": []}
    for p in prompts:
        a = reply(p)
        log_turn(mgr, p, a)

        scores = compute_emergence_scores(storage_manager=mgr.sqlite_store)
        telemetry["IAS"].append(float(scores.get("IAS", 0.0) or 0.0))
        telemetry["GAS"].append(float(scores.get("GAS", 0.0) or 0.0))
        telemetry["commit_close_rate"].append(
            float(scores.get("commit_close_rate", 0.0) or 0.0)
        )

    print("TELEMETRY_JSON:", json.dumps(telemetry))
    return telemetry


if __name__ == "__main__":
    run()
