#!/usr/bin/env python3
from dotenv import load_dotenv; load_dotenv()
"""
Two persistent AI agents talking to each other,
using the Persistent Mind Model's reflection loop.

Now enhanced to:
- inject a stimulus each round
- immediately apply event 'effects' to trait values
- clamp updates to configured drift bounds
"""

from pmm.self_model_manager import SelfModelManager
from pmm.reflection import reflect_once
from pmm.llm import OpenAIClient
import time
import random
from pathlib import Path
from inject_stimulus import EVENTS
from pmm.cadence_stimuli import CadenceStimulator

# ---- helpers to traverse dataclasses/dicts and apply effects ----
def _get(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    return obj[key]

def _set(obj, key, value):
    if hasattr(obj, key):
        setattr(obj, key, value)
    else:
        obj[key] = value

def apply_event_effects(mgr: SelfModelManager, effects):
    if not effects:
        return
    # bounds from drift_config if present
    try:
        b = _get(mgr.model.drift_config, "bounds")
        bmin = float(b["min"] if isinstance(b, dict) else _get(b, "min"))
        bmax = float(b["max"] if isinstance(b, dict) else _get(b, "max"))
    except Exception:
        bmin, bmax = 0.0, 1.0

    for eff in effects:
        path = eff["target"].split(".")
        delta = float(eff.get("delta", 0.0))
        conf = float(eff.get("confidence", eff.get("conf", 1.0)))
        step = delta * conf

        cur = mgr.model
        for seg in path[:-1]:
            cur = _get(cur, seg)
        leaf = path[-1]
        old = float(_get(cur, leaf))
        new = max(bmin, min(bmax, old + step))
        _set(cur, leaf, new)

    mgr.save_model()


# 1. Load/create two separate mind files
def load_agent(path, name):
    if not Path(path).exists():
        mgr = SelfModelManager(path)
        mgr.model.core_identity.name = name
        mgr.save_model()
    else:
        mgr = SelfModelManager(path)
    return mgr

agent_a = load_agent("mind_a.json", "Agent_A")
agent_b = load_agent("mind_b.json", "Agent_B")

# Initialize cadence stimulators
stimulator_a = CadenceStimulator()
stimulator_b = CadenceStimulator()

# Nudge drift aggressiveness slightly and document intent
try:
    agent_a.set_drift_params(max_delta_per_reflection=0.03, notes_append="Increase delta for exploration & calibration.")
    agent_b.set_drift_params(max_delta_per_reflection=0.03, notes_append="Increase delta for exploration & calibration.")
except Exception:
    pass

# 2. One turn for an agent: record heard msg, inject stimulus, apply effects, reflect
def agent_step(agent_mgr: SelfModelManager, other_name: str, incoming_text: str, stimulator: CadenceStimulator):
    if incoming_text:
        agent_mgr.add_event(
            summary=f"Heard from {other_name}: {incoming_text}",
            effects=[]
        )
    
    # Use cadence-based stimulus with pattern awareness
    patterns = agent_mgr.model.self_knowledge.behavioral_patterns
    stimulus = stimulator.get_pattern_triggered_stimulus(patterns)
    
    agent_mgr.add_event(
        summary=f"Stimulus: {stimulus['summary']}",
        effects=stimulus.get("effects", [])
    )
    apply_event_effects(agent_mgr, stimulus.get("effects", []))
    response = reflect_once(agent_mgr, OpenAIClient())
    # Log thought and update patterns based on the agent's utterance
    if response and response.content:
        agent_mgr.add_thought(response.content, trigger="duel_round")
        try:
            agent_mgr.update_patterns(response.content)
        except Exception:
            pass
    # Apply additional drift pass respecting drift_config
    try:
        agent_mgr.apply_drift_and_save()
    except Exception:
        pass
    agent_mgr.save_model()
    return response.content if response else "(no insight)"

# 3. Conversation loop
if __name__ == "__main__":
    rounds = 5  # number of back-and-forth exchanges
    last_message = "Hello there. How are you today?"

    print(f"Starting conversation with cadence stimuli for {rounds} rounds...\n")
    for i in range(rounds):
        # Agent A speaks to Agent B
        reply_a = agent_step(agent_a, "Agent_B", last_message, stimulator_a)
        print(f"\nA: {reply_a}")

        # Agent B speaks to Agent A
        reply_b = agent_step(agent_b, "Agent_A", reply_a, stimulator_b)
        print(f"\nB: {reply_b}")

        last_message = reply_b
        time.sleep(0.8)  # small delay so you can watch it happen

    # After conversation, show Big5 snapshots so drift is visible
    import json
    def _snapshot(path: str):
        try:
            with open(path) as f:
                m = json.load(f)
            b5 = m["personality"]["traits"]["big5"]
            print(path, {k: b5[k]["score"] for k in b5})
        except Exception as e:
            print(path, "- error:", type(e).__name__, str(e))

    print("\nBig5 snapshots:")
    for fn in ["mind_a.json", "mind_b.json"]:
        _snapshot(fn)
