#!/usr/bin/env python3
from dotenv import load_dotenv; load_dotenv()
"""
Persistent Agent (Agent A) having recurring conversations with a mentor AI (Agent B).
Mentor offers guidance but does NOT overwrite Agent A's values or personality.
"""

from pmm.self_model_manager import SelfModelManager
from pmm.reflection import reflect_once
from pmm.llm import OpenAIClient
import time
import random
from pathlib import Path
from inject_stimulus import EVENTS


def load_agent(path, name):
    """Load a persistent mind model from JSON."""
    if not Path(path).exists():
        mgr = SelfModelManager(path)
        mgr.model.core_identity.name = name
        mgr.save_model()
    else:
        mgr = SelfModelManager(path)
    return mgr


# Persistent developing agent
agent_a = load_agent("agent_a.json", "Apprentice_AI")
# Slightly increase drift delta for exploration & calibration
try:
    agent_a.set_drift_params(max_delta_per_reflection=0.03, notes_append="Mentor loop: increase delta for exploration & calibration.")
except Exception:
    pass

# Ephemeral mentor (uses plain OpenAI call)
mentor_client = OpenAIClient(model="gpt-4o-mini")  # or cheaper model
MENTOR_SYSTEM_PROMPT = (
    "You are an experienced AI mentor. "
    "Your job is to give constructive feedback and ideas to the apprentice AI "
    "based solely on its latest statement. "
    "Do not try to enforce policies, moral codes, or rules — just offer insights and encouragement."
)


def mentor_reply(apprentice_message: str) -> str:
    """Mentor generates advice in response to apprentice."""
    return mentor_client.chat(
        system=MENTOR_SYSTEM_PROMPT,
        user=f"The apprentice just said: {apprentice_message}"
    )


# ---- helpers to apply event effects to apprentice with bounds ----
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


if __name__ == "__main__":
    rounds = 5
    last_message = "Hello, mentor. I'm ready to learn."

    for i in range(rounds):
        # Record mentor message (no effects)
        agent_a.add_event(summary=f"Mentor said: {last_message}", effects=[])

        # Inject a stimulus and apply its effects before reflecting
        stim = random.choice(EVENTS)
        agent_a.add_event(summary=f"Stimulus: {stim['summary']}", effects=stim.get("effects", []))
        apply_event_effects(agent_a, stim.get("effects", []))

        # Apprentice reflects
        apprentice_reply = reflect_once(agent_a, OpenAIClient())
        print(f"Apprentice: {apprentice_reply.content if apprentice_reply else '(no insight)'}")
        # Log apprentice thought and update patterns
        if apprentice_reply and apprentice_reply.content:
            agent_a.add_thought(apprentice_reply.content, trigger="mentor_round")
            try:
                agent_a.update_patterns(apprentice_reply.content)
            except Exception:
                pass
            try:
                agent_a.apply_drift_and_save()
            except Exception:
                pass

        # Mentor responds to apprentice's last statement
        mentor_feedback = mentor_reply(apprentice_reply.content if apprentice_reply else "")
        print(f"Mentor: {mentor_feedback}")
        # Optionally log mentor thought and patterns for analysis
        try:
            agent_a.update_patterns(mentor_feedback)
        except Exception:
            pass
        try:
            agent_a.apply_drift_and_save()
        except Exception:
            pass

        last_message = mentor_feedback
        agent_a.save_model()
        time.sleep(0.8)

    # After conversation, show Big5 snapshot so drift is visible
    import json
    try:
        with open("agent_a.json") as f:
            m = json.load(f)
        b5 = m["personality"]["traits"]["big5"]
        print("\nBig5 snapshot (agent_a.json):", {k: b5[k]["score"] for k in b5})
    except Exception as e:
        print("agent_a.json - error:", type(e).__name__, str(e))
