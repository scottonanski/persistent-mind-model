from __future__ import annotations
from datetime import datetime, UTC
from typing import List
from .self_model_manager import SelfModelManager, _log
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
    scene_summ = [
        f"{getattr(s,'t','')}:{getattr(s,'type','')}:{getattr(s,'summary','')}"
        for s in scenes
    ]
    # recent autobiographical events (brief)
    events = (
        m.self_knowledge.autobiographical_events[-3:]
        if m.self_knowledge.autobiographical_events
        else []
    )
    event_summ = [
        f"{getattr(e,'t','')}:{getattr(e,'type','')}:{getattr(e,'summary','')}"
        for e in events
    ]
    # current Big5 snapshot (scores only)
    try:
        b5 = m.personality.traits.big5
        trait_snapshot = {
            "openness": getattr(b5.openness, "score", None),
            "conscientiousness": getattr(b5.conscientiousness, "score", None),
            "extraversion": getattr(b5.extraversion, "score", None),
            "agreeableness": getattr(b5.agreeableness, "score", None),
            "neuroticism": getattr(b5.neuroticism, "score", None),
        }
    except Exception:
        trait_snapshot = {}
    thoughts = (
        _recent_texts(m.self_knowledge.thoughts, 3) if m.self_knowledge.thoughts else []
    )
    insights = (
        _recent_texts(m.self_knowledge.insights, 2) if m.self_knowledge.insights else []
    )
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

    # Enhanced n-gram cache from last 8 insights
    recent_insights = (
        mgr.model.self_knowledge.insights[-8:]
        if mgr.model.self_knowledge.insights
        else []
    )
    ngram_cache = set()

    for insight in recent_insights:
        words = insight.content.lower().split()
        # Extract 2-4 grams
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                if all(len(w) > 2 for w in words[i : i + n]):  # Skip short words
                    ngram_cache.add(gram)

    # Template jitter - rotate prompt stems
    prompt_templates = [
        "You are a terse metacognitive process. Observe your current state, then commit to one concrete micro-adjustment. Keep response to 55-80 words. Include 'I will' or 'Next:'.",
        "You are a reflective AI system. Identify a tension or limitation in your approach, then propose a specific experiment. Keep response to 55-80 words. Include 'I will' or 'Next:'.",
        "You are an introspective agent. Form a hypothesis about improving your effectiveness, then define a measurable test. Keep response to 55-80 words. Include 'I will' or 'Next:'.",
    ]

    # Select template based on insight count
    template_idx = len(mgr.model.self_knowledge.insights) % len(prompt_templates)
    sys = prompt_templates[template_idx]

    # Add anti-repetition context
    if ngram_cache:
        recent_phrases = list(ngram_cache)[:15]  # Limit context size
        ctx += f"\n\nAVOID repeating these recent phrases: {', '.join(recent_phrases)}"

    txt = llm.chat(system=sys, user=PROMPT + ctx)
    if not txt:
        return None

    # Check overlap ratio and re-roll if needed
    if ngram_cache:
        txt_words = txt.lower().split()
        txt_ngrams = set()
        for n in range(2, 5):
            for i in range(len(txt_words) - n + 1):
                gram = " ".join(txt_words[i : i + n])
                if all(len(w) > 2 for w in txt_words[i : i + n]):
                    txt_ngrams.add(gram)

        overlap_ratio = (
            len(txt_ngrams & ngram_cache) / len(txt_ngrams) if txt_ngrams else 0
        )

        if overlap_ratio > 0.35:  # GPT-5's threshold
            import os

            if os.getenv("PMM_DEBUG") == "1":
                print(
                    f"   🔄 High n-gram overlap detected ({overlap_ratio:.1%}), re-rolling with style constraint..."
                )
            # Re-roll with style constraint
            style_sys = (
                sys
                + " IMPORTANT: Use analogy or concrete example. Avoid abstract language."
            )
            try:
                txt = llm.chat(system=style_sys, user=PROMPT + ctx)
                if not txt:
                    if os.getenv("PMM_DEBUG") == "1":
                        print("   ⚠️  Re-roll failed, using original response")
                else:
                    if os.getenv("PMM_DEBUG") == "1":
                        print("   ✅ Re-roll successful, reduced repetition")
            except Exception as e:
                if os.getenv("PMM_DEBUG") == "1":
                    print(
                        f"   ⚠️  Re-roll failed ({type(e).__name__}), using original response"
                    )
                # Keep original txt if re-roll fails

    # Cap length
    if len(txt) > 400:  # ~80 words
        txt = txt[:400] + "..."

    ins_id = f"in{len(mgr.model.self_knowledge.insights)+1}"
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Extract commitment
    _commitment = _extract_commitment(txt)

    # provenance: link to last thought and current top pattern keys (schema-compliant)
    thoughts = mgr.model.self_knowledge.thoughts
    last_th_id = thoughts[-1].id if thoughts else None
    patterns = mgr.model.self_knowledge.behavioral_patterns or {}
    top_keys = [
        k for k, _ in sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]
    refs = {}
    if last_th_id:
        refs["thought_ids"] = [last_th_id]
    if top_keys:
        refs["pattern_keys"] = top_keys

    # Auto-close commitments based on reflection completion signals
    closed_cids = mgr.auto_close_commitments_from_reflection(txt)
    if closed_cids:
        _log(
            "commitment", f"Auto-closed {len(closed_cids)} commitments from reflection"
        )

    # Extract and track commitments with provenance
    commitment_text, _ = mgr.commitment_tracker.extract_commitment(txt)
    if commitment_text:
        # Add commitment to the manager's tracker
        cid = mgr.commitment_tracker.add_commitment(commitment_text, ins_id)
        refs["commitments"] = [cid]  # Store commitment ID for provenance
        _log("commitment", f"Added commitment {cid}: {commitment_text[:50]}...")
        # Sync to model for persistence
        mgr._sync_commitments_to_model()

    insight = Insight(id=ins_id, t=ts, content=txt, references=refs)
    mgr.model.self_knowledge.insights.append(insight)
    mgr.model.meta_cognition.self_modification_count += 1
    mgr.model.metrics.last_reflection_at = ts

    # Update behavioral patterns from reflection content
    mgr.update_patterns(txt)

    # Use add_insight to trigger commitment extraction
    mgr.save_model()
    return insight


def _extract_commitment(content: str) -> str | None:
    """Extract commitment from insight content."""
    lines = content.split(".")
    for line in lines:
        line = line.strip()
        if any(
            starter in line.lower()
            for starter in ["i will", "next:", "i plan to", "i commit to"]
        ):
            return line
    return None


def _track_commitment(mgr: SelfModelManager, commitment: str, insight_id: str) -> None:
    """Track commitment in metrics and future goals."""
    # Add commitment tracking to metrics
    if not hasattr(mgr.model.metrics, "commitments_open"):
        mgr.model.metrics.commitments_open = 0
    if not hasattr(mgr.model.metrics, "commitments_closed"):
        mgr.model.metrics.commitments_closed = 0

    mgr.model.metrics.commitments_open += 1

    # Add to future_scripts.goals (simplified for now)
    from .model import Goal

    goal_id = f"goal_{len(mgr.model.narrative_identity.future_scripts.goals) + 1}"
    goal = Goal(id=goal_id, desc=commitment, priority=0.5)
    mgr.model.narrative_identity.future_scripts.goals.append(goal)
