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


# ---- simple helpers for run-time metrics ----
def _big5_snapshot(mgr: SelfModelManager):
    b5 = mgr.model.personality.traits.big5
    return {
        'openness': b5.openness.score,
        'conscientiousness': b5.conscientiousness.score,
        'extraversion': b5.extraversion.score,
        'agreeableness': b5.agreeableness.score,
        'neuroticism': b5.neuroticism.score,
    }


def _big5_delta(prev: dict, cur: dict):
    if not prev:
        return {k: 0.0 for k in cur}
    return {k: round(cur.get(k, 0) - prev.get(k, 0), 3) for k in cur}


def _ngram_set(text: str):
    words = (text or "").lower().split()
    grams = set()
    for n in (2, 3, 4):
        for i in range(len(words) - n + 1):
            gram = ' '.join(words[i:i+n])
            if all(len(w) > 2 for w in words[i:i+n]):
                grams.add(gram)
    return grams


def _overlap_pct(a: str, b: str) -> float:
    A = _ngram_set(a)
    B = _ngram_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = max(1, len(B))
    return round(100.0 * inter / denom, 1)


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
    # Capture pre-reflection state for metrics
    prev_insight_text = None
    if agent_mgr.model.self_knowledge.insights:
        prev_insight_text = agent_mgr.model.self_knowledge.insights[-1].content
    prev_b5 = _big5_snapshot(agent_mgr)

    # Add retry logic for LLM calls with exponential backoff
    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = reflect_once(agent_mgr, OpenAIClient())
            break
        except Exception as e:
            print(f"   ⚠️  LLM call failed (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"   ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ All LLM attempts failed, using fallback response")
                # Create fallback response based on current patterns
                fallback_content = f"I notice patterns in my behavior and will continue to reflect on my growth. Next: I will focus on {list(patterns.keys())[0] if patterns else 'self-improvement'}."
                from pmm.models import Insight
                response = Insight(
                    id=f"insight_{len(agent_mgr.model.self_knowledge.insights) + 1}",
                    content=fallback_content,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    references={'commitments': [], 'thought_ids': [], 'pattern_keys': []}
                )
    
    # Log thought and update patterns based on the agent's utterance
    if response and response.content:
        agent_mgr.add_thought(response.content, trigger="duel_round")
        try:
            agent_mgr.update_patterns(response.content)
        except Exception as e:
            print(f"   ⚠️  Pattern update failed: {type(e).__name__}: {str(e)}")
    
    # Apply additional drift pass respecting drift_config
    try:
        b5_before = _big5_snapshot(agent_mgr)
        _ = agent_mgr.apply_drift_and_save()
        b5_after = _big5_snapshot(agent_mgr)
        # Compute actual applied drift this call
        drift_applied = {k: round(b5_after[k] - b5_before[k], 6) for k in b5_before}
        if any(abs(v) >= 0.0001 for v in drift_applied.values()):
            drift_str = ', '.join([f"{k}:{v:+.3f}" for k, v in drift_applied.items() if abs(v) >= 0.0001])
            print(f"   🔄 Drift applied: {drift_str}")
            # Log evidence signals if available
            try:
                patterns = getattr(agent_mgr.model, 'self_knowledge', agent_mgr.model).get('behavioral_patterns') if isinstance(agent_mgr.model, dict) else agent_mgr.model.self_knowledge.behavioral_patterns
                exp_count = patterns.get('experimentation', 0)
                align_count = patterns.get('user_goal_alignment', 0)
                # Compute close rate directly from tracker
                ct = getattr(agent_mgr, 'commitment_tracker', None)
                if ct and getattr(ct, 'commitments', None):
                    total = len(ct.commitments)
                    closed = sum(1 for c in ct.commitments.values() if getattr(c, 'status', '') == 'closed')
                    close_rate = (closed / total) if total > 0 else 0.0
                else:
                    close_rate = 0.0
                exp_delta = max(0, exp_count - 3)
                align_delta = max(0, align_count - 2)
                close_rate_delta = max(0, close_rate - 0.3)
                signals = exp_delta + align_delta + close_rate_delta
                evidence_weight = min(1, signals / 3)
                print(f"   📊 Evidence: exp={exp_count}(+{exp_delta}), align={align_count}(+{align_delta}), close_rate={close_rate:.1%}(+{close_rate_delta:.2f}) → weight={evidence_weight:.2f}")
            except Exception as ex:
                print(f"   ⚠️  Evidence logging failed: {type(ex).__name__}")
        else:
            print(f"   ⚪ No significant drift this turn")
    except Exception as e:
        print(f"   ⚠️  Drift application failed: {type(e).__name__}: {str(e)}")
    
    agent_mgr.save_model()
    
    # Build per-turn info block
    cur_b5 = _big5_snapshot(agent_mgr)
    delta = _big5_delta(prev_b5, cur_b5)
    commitments = []
    if response and getattr(response, 'references', None):
        commitments = response.references.get('commitments', []) or []
    overlap = _overlap_pct(prev_insight_text or "", response.content if response else "")
    metrics = {
        'big5': cur_b5,
        'delta': delta,  # This shows immediate per-utterance delta (usually 0.0)
        'commitments': commitments,
        'overlap_pct_prev': overlap,
    }
    return (response.content if response else "(no insight)", metrics)

# 3. Conversation loop
if __name__ == "__main__":
    rounds = 5  # number of back-and-forth exchanges
    last_message = "Hello there. How are you today?"

    # Capture initial Big5 snapshots for session delta calculation
    start_b5_a = _big5_snapshot(agent_a)
    start_b5_b = _big5_snapshot(agent_b)
    
    print(f"Starting conversation with cadence stimuli for {rounds} rounds...\n")
    for i in range(rounds):
        # Agent A speaks to Agent B
        reply_a, info_a = agent_step(agent_a, "Agent_B", last_message, stimulator_a)
        print(f"\nA: {reply_a}")
        print(f"   · commitments: {info_a['commitments']}  · ΔBig5: {info_a['delta']}  · overlap(prev): {info_a['overlap_pct_prev']}%")

        # Agent B speaks to Agent A
        reply_b, info_b = agent_step(agent_b, "Agent_A", reply_a, stimulator_b)
        print(f"\nB: {reply_b}")
        print(f"   · commitments: {info_b['commitments']}  · ΔBig5: {info_b['delta']}  · overlap(prev): {info_b['overlap_pct_prev']}%")

        last_message = reply_b
        time.sleep(0.8)  # small delay so you can watch it happen

    # Apply final drift after conversation ends
    print("\nApplying final drift calculations...")
    for agent_name, agent_mgr in [("Agent A", agent_a), ("Agent B", agent_b)]:
        try:
            prev_traits = _big5_snapshot(agent_mgr)
            _ = agent_mgr.apply_drift_and_save()
            new_traits = _big5_snapshot(agent_mgr)
            delta_final = {k: round(new_traits[k] - prev_traits[k], 6) for k in prev_traits}
            if any(abs(v) >= 0.0001 for v in delta_final.values()):
                drift_str = ', '.join([f"{k}:{v:+.3f}" for k, v in delta_final.items() if abs(v) >= 0.0001])
                print(f"{agent_name} 🔄 Final drift: {drift_str}")
                # Show evidence
                try:
                    patterns = getattr(agent_mgr.model, 'self_knowledge', agent_mgr.model).get('behavioral_patterns') if isinstance(agent_mgr.model, dict) else agent_mgr.model.self_knowledge.behavioral_patterns
                    exp_count = patterns.get('experimentation', 0)
                    align_count = patterns.get('user_goal_alignment', 0)
                    ct = getattr(agent_mgr, 'commitment_tracker', None)
                    if ct and getattr(ct, 'commitments', None):
                        total = len(ct.commitments)
                        closed = sum(1 for c in ct.commitments.values() if getattr(c, 'status', '') == 'closed')
                        close_rate = (closed / total) if total > 0 else 0.0
                    else:
                        close_rate = 0.0
                    exp_delta = max(0, exp_count - 3)
                    align_delta = max(0, align_count - 2)
                    close_rate_delta = max(0, close_rate - 0.3)
                    signals = exp_delta + align_delta + close_rate_delta
                    evidence_weight = min(1, signals / 3)
                    print(f"{agent_name} 📊 Evidence: exp={exp_count}(+{exp_delta}), align={align_count}(+{align_delta}), close_rate={close_rate:.1%}(+{close_rate_delta:.2f}) → weight={evidence_weight:.2f}")
                except Exception as ex:
                    print(f"{agent_name} ⚠️ Evidence logging failed: {type(ex).__name__}")
            else:
                print(f"{agent_name} ⚪ No significant final drift")
        except Exception as e:
            print(f"{agent_name} ⚠️ Final drift failed: {type(e).__name__}: {str(e)}")

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

    # Show commitment summary after conversation
    print("\nCommitment Summary:")
    for agent_name, agent_mgr in [("Agent A", agent_a), ("Agent B", agent_b)]:
        try:
            ct = getattr(agent_mgr, 'commitment_tracker', None)
            if ct and getattr(ct, 'commitments', None):
                open_commitments = [cid for cid, c in ct.commitments.items() if getattr(c, 'status', '') == "open"]
                closed_commitments = [cid for cid, c in ct.commitments.items() if getattr(c, 'status', '') == "closed"]
                total = len(open_commitments) + len(closed_commitments)
                close_rate = (len(closed_commitments) / total) if total > 0 else 0.0
                print(f"{agent_name}: {len(open_commitments)} open, {len(closed_commitments)} closed, close_rate={close_rate:.1%}")
                if len(open_commitments) > 0:
                    print(f"  Open: {open_commitments[:5]}{'...' if len(open_commitments) > 5 else ''}")
            else:
                print(f"{agent_name}: No commitment tracker available")
        except Exception as e:
            print(f"{agent_name}: commitment summary error - {type(e).__name__}: {str(e)}")

    # Session-level Big5 delta summary
    try:
        print("\nSession ΔBig5:")
        for agent_name, agent_mgr, start_b5 in [("Agent A", agent_a, start_b5_a), ("Agent B", agent_b, start_b5_b)]:
            end_b5 = _big5_snapshot(agent_mgr)
            session_delta = {k: round(end_b5[k] - start_b5[k], 6) for k in start_b5}
            delta_str = ', '.join([f"{k}:{v:+.3f}" for k, v in session_delta.items() if abs(v) >= 0.0001])
            print(f"{agent_name}: {delta_str if delta_str else '(no visible change)'}")
    except Exception as e:
        print(f"⚠️  Session delta summary failed: {type(e).__name__}: {str(e)}")

    print("\nBig5 snapshots:")
    for fn in ["mind_a.json", "mind_b.json"]:
        _snapshot(fn)
