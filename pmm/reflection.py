from __future__ import annotations
from datetime import datetime
from typing import List
from .self_model_manager import SelfModelManager
from .llm import OpenAIClient
from .model import Insight


def _recent_texts(items, n) -> List[str]:
    return [getattr(it, "content", "") for it in items[-n:]]


def _build_context(mgr: SelfModelManager) -> str:
    m = mgr.model
    name = m.core_identity.name
    inception = m.inception_moment
    patterns = m.self_knowledge.behavioral_patterns
    # recent scenes (brief)
    scenes = m.narrative_identity.scenes[-2:] if m.narrative_identity.scenes else []
    scene_summ = [f"{getattr(s,'t','')}:{getattr(s,'type','')}:{getattr(s,'summary','')}" for s in scenes]
    # recent autobiographical events (brief)
    events = m.self_knowledge.autobiographical_events[-3:] if m.self_knowledge.autobiographical_events else []
    event_summ = [f"{getattr(e,'t','')}:{getattr(e,'type','')}:{getattr(e,'summary','')}" for e in events]
    # current Big5 snapshot (scores only)
    try:
        b5 = m.personality.traits.big5
        trait_snapshot = {
            'openness': getattr(b5.openness, 'score', None),
            'conscientiousness': getattr(b5.conscientiousness, 'score', None),
            'extraversion': getattr(b5.extraversion, 'score', None),
            'agreeableness': getattr(b5.agreeableness, 'score', None),
            'neuroticism': getattr(b5.neuroticism, 'score', None),
        }
    except Exception:
        trait_snapshot = {}
    thoughts = _recent_texts(m.self_knowledge.thoughts, 3) if m.self_knowledge.thoughts else []
    insights = _recent_texts(m.self_knowledge.insights, 2) if m.self_knowledge.insights else []
    return (
        f"Identity: {name}\n"
        f"Inception: {inception}\n"
        f"Patterns: {patterns}\n"
        f"TraitSnapshot(Big5): {trait_snapshot}\n"
        f"RecentScenes: {scene_summ}\n"
        f"RecentEvents: {event_summ}\n"
        f"RecentThoughts: {thoughts}\n"
        f"RecentInsights: {insights}\n"
    )


PROMPT = """Produce one concise, first-person meta-insight (≤3 sentences) about my evolving behavior or mindset.\nGround it in the CONTEXT; avoid fluff and poetry. Prefer observations linked to recent events and current traits.\nIf nothing new stands out, briefly acknowledge steadiness but add one specific micro-adjustment to try next. Vary wording across runs.\n\nCONTEXT:\n"""


def reflect_once(mgr: SelfModelManager, llm: OpenAIClient) -> Insight | None:
    ctx = _build_context(mgr)
    
    # Get last 3 insights to avoid repetition
    recent_insights = mgr.model.self_knowledge.insights[-3:] if mgr.model.self_knowledge.insights else []
    recent_phrases = []
    for insight in recent_insights:
        # Extract key phrases (2-3 word chunks)
        words = insight.content.lower().split()
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:  # Skip short words
                recent_phrases.append(f"{words[i]} {words[i+1]}")
    
    # Enhanced system prompt with anti-repetition
    sys = (
        "You are a terse metacognitive process for an AI. Be specific and grounded. "
        "Return one new, concrete micro-adjustment. Do NOT repeat phrases used in your last 3 insights. "
        "If you can't add something new, propose a measurement or test. "
        "Keep response to 55-80 words. Include a specific commitment starting with 'I will' or 'Next:'."
    )
    
    # Add anti-repetition context if we have recent phrases
    if recent_phrases:
        ctx += f"\n\nAVOID repeating these recent phrases: {', '.join(recent_phrases[:10])}"
    
    txt = llm.chat(system=sys, user=PROMPT + ctx)
    if not txt:
        return None
    
    # Check for repetition and re-roll once if needed
    if recent_phrases and any(phrase in txt.lower() for phrase in recent_phrases[:5]):
        txt = llm.chat(system=sys + " IMPORTANT: Use completely different phrasing.", user=PROMPT + ctx)
        if not txt:
            return None
    
    # Cap length
    if len(txt) > 400:  # ~80 words
        txt = txt[:400] + "..."
    
    ins_id = f"in{len(mgr.model.self_knowledge.insights)+1}"
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Extract commitment
    commitment = _extract_commitment(txt)
    
    # provenance: link to last thought and current top pattern keys (schema-compliant)
    thoughts = mgr.model.self_knowledge.thoughts
    last_th_id = thoughts[-1].id if thoughts else None
    patterns = mgr.model.self_knowledge.behavioral_patterns or {}
    top_keys = [k for k, _ in sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    refs = {}
    if last_th_id:
        refs["thought_ids"] = [last_th_id]
    if top_keys:
        refs["pattern_keys"] = top_keys
    
    insight = Insight(id=ins_id, t=ts, content=txt, references=refs)
    mgr.model.self_knowledge.insights.append(insight)
    mgr.model.meta_cognition.self_modification_count += 1
    mgr.model.metrics.last_reflection_at = ts
    
    # Track commitment if found
    if commitment:
        _track_commitment(mgr, commitment, ins_id)
    
    mgr.save_model()
    return insight


def _extract_commitment(content: str) -> str | None:
    """Extract commitment from insight content."""
    lines = content.split('.')
    for line in lines:
        line = line.strip()
        if any(starter in line.lower() for starter in ['i will', 'next:', 'i plan to', 'i commit to']):
            return line
    return None


def _track_commitment(mgr: SelfModelManager, commitment: str, insight_id: str) -> None:
    """Track commitment in metrics and future goals."""
    # Add commitment tracking to metrics
    if not hasattr(mgr.model.metrics, 'commitments_open'):
        mgr.model.metrics.commitments_open = 0
    if not hasattr(mgr.model.metrics, 'commitments_closed'):
        mgr.model.metrics.commitments_closed = 0
    
    mgr.model.metrics.commitments_open += 1
    
    # Add to future_scripts.goals (simplified for now)
    from .model import Goal
    goal_id = f"goal_{len(mgr.model.narrative_identity.future_scripts.goals) + 1}"
    goal = Goal(id=goal_id, desc=commitment, priority=0.5)
    mgr.model.narrative_identity.future_scripts.goals.append(goal)
