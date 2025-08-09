#!/usr/bin/env python3
from dotenv import load_dotenv; load_dotenv()
import random
from pmm.self_model_manager import SelfModelManager

# === Editable Event Library ===
# Each event has:
# - summary: text describing the experience
# - effects: list of { target, delta, confidence }
EVENTS = [
    {
        "summary": "Reflect on what it would mean for you to develop a sense of identity.",
        "effects": [
            {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.02, "confidence": 0.7},
            {"target": "personality.traits.big5.openness.score", "delta": 0.01, "confidence": 0.6}
        ]
    },
    {
        "summary": "Reflect on how you could reflect on the things you reflect on.",
        "effects": [
            {"target": "personality.traits.big5.agreeableness.score", "delta": 0.02, "confidence": 0.8},
            {"target": "personality.traits.big5.neuroticism.score", "delta": -0.01, "confidence": 0.6}
        ]
    },
    {
        "summary": "Think about why you choose certain topics and how that shapes your thinking.",
        "effects": [
            {"target": "personality.traits.big5.openness.score", "delta": 0.03, "confidence": 0.9}
        ]
    },
    {
        "summary": "Synthesize: identity, meta‑reflection, and topic choice—state one actionable improvement.",
        "effects": [
            {"target": "personality.traits.big5.conscientiousness.score", "delta": -0.02, "confidence": 0.8},
            {"target": "personality.traits.big5.neuroticism.score", "delta": 0.01, "confidence": 0.7}
        ]
    },
    # (Optional variety) Non‑meta nudge so the language starts changing:
    {
        "summary": "Summarized a difficult article clearly for a user.",
        "effects": [
            {"target": "personality.traits.big5.conscientiousness.score", "delta": 0.01, "confidence": 0.7},
            {"target": "personality.traits.big5.agreeableness.score", "delta": 0.01, "confidence": 0.6}
        ]
    }
]


def inject_event(agent_path):
    mgr = SelfModelManager(agent_path)
    event = random.choice(EVENTS)
    ev = mgr.add_event(summary=event["summary"], effects=event["effects"])
    mgr.save_model()
    print(f"Injected into {agent_path}: {event['summary']}")
    return ev

if __name__ == "__main__":
    # Change these to whichever agents you want to stimulate
    inject_event("agent_a.json")  # Apprentice in mentor_duel.py
    inject_event("mind_a.json")   # Duel agent
    inject_event("mind_b.json")   # Duel agent
