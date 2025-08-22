from __future__ import annotations
import os
import random
import re
from typing import List, Tuple
from datetime import datetime, timezone
from pmm.model import Insight
from pmm.self_model_manager import SelfModelManager
from pmm.config.models import get_novelty_penalty
from pmm.adapters.openai_adapter import OpenAIAdapter

# Load environment variables for API access
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on system environment


def _recent_texts(items, n) -> List[str]:
    return [getattr(it, "content", "") for it in items[-n:]]


def _validate_insight_references(
    content: str, mgr: SelfModelManager
) -> Tuple[bool, List[str]]:
    """
    Validate that insight references specific event IDs or commitment hashes.

    Returns:
        (is_accepted, referenced_ids): Whether insight is accepted and list of referenced IDs
    """
    referenced_ids = []

    # Pattern 1: Event IDs (ev123, event ev123, etc.)
    event_patterns = [r"\bev(\d+)\b", r"\bevent\s+ev(\d+)\b", r"\bevent\s+(\d+)\b"]

    for pattern in event_patterns:
        matches = re.findall(pattern, content.lower())
        for match in matches:
            event_id = f"ev{match}" if not match.startswith("ev") else match
            referenced_ids.append(event_id)

    # Pattern 2: Commitment hashes (16-char hex)
    commit_hash_pattern = r"\b[a-f0-9]{16}\b"
    commit_matches = re.findall(commit_hash_pattern, content.lower())
    referenced_ids.extend(commit_matches)

    # Pattern 3: Recent event references (last 5 events)
    recent_events = (
        mgr.model.self_knowledge.autobiographical_events[-5:]
        if mgr.model.self_knowledge.autobiographical_events
        else []
    )
    for event in recent_events:
        if hasattr(event, "id") and event.id and event.id.lower() in content.lower():
            referenced_ids.append(event.id)

    # Pattern 4: Open commitment references
    try:
        open_commitments = mgr.get_open_commitments()
        for commitment in open_commitments[:3]:  # Check top 3 open commitments
            if (
                "hash" in commitment
                and commitment["hash"][:8].lower() in content.lower()
            ):
                referenced_ids.append(commitment["hash"][:16])
    except Exception:
        pass

    # Accept if at least 1 reference found
    is_accepted = len(referenced_ids) > 0

    if os.getenv("PMM_DEBUG") == "1":
        print(
            f"🔍 DEBUG: Insight validation - References found: {referenced_ids}, Accepted: {is_accepted}"
        )

    return is_accepted, referenced_ids


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
    # recent autobiographical events (brief) with explicit IDs for anchoring
    events = (
        m.self_knowledge.autobiographical_events[-3:]
        if m.self_knowledge.autobiographical_events
        else []
    )
    event_ids = [getattr(e, "id", None) for e in events if getattr(e, "id", None)]
    event_summ = [
        f"{getattr(e,'id','')}|{getattr(e,'t','')}:{getattr(e,'type','')}:{getattr(e,'summary','')}"
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
    # open commitments (top 3) with short hashes to encourage anchoring
    open_commitments_brief = []
    try:
        open_commitments = mgr.get_open_commitments()
        for c in open_commitments[:3]:
            h = c.get("hash", "")
            title = c.get("text", c.get("title", ""))
            open_commitments_brief.append(f"{h[:16]}:{title[:60]}")
    except Exception:
        pass

    return (
        f"Identity: {name}\n"
        f"Inception: {inception}\n"
        f"Patterns: {patterns}\n"
        f"TraitSnapshot(Big5): {trait_snapshot}\n"
        f"RecentScenes: {scene_summ}\n"
        f"RecentEventIDs: {event_ids}\n"
        f"RecentEvents: {event_summ}\n"
        f"OpenCommitments: {open_commitments_brief}\n"
        f"RecentThoughts: {thoughts}\n"
        f"RecentInsights: {insights}\n"
    )


REFLECTION_PROMPTS = [
    """Produce one concise, first-person meta-insight (≤3 sentences) about my evolving behavior or mindset.\nGround it in the CONTEXT; avoid fluff and poetry. CRITICAL: Reference specific recent events by their IDs (e.g., "Based on event ev123...") or commitment hashes.\nIf nothing new stands out, briefly acknowledge steadiness but add one specific micro-adjustment to try next. Vary wording across runs.\n\nCONTEXT:\n""",
    """Analyze my recent patterns and identify one specific behavioral shift or emerging tendency (≤3 sentences).\nReference concrete events or commitments from CONTEXT. Focus on what's actually changing in my responses or approach.\nEnd with one tactical adjustment I can implement immediately.\n\nCONTEXT:\n""",
    """Examine my commitment fulfillment and goal alignment based on CONTEXT (≤3 sentences).\nWhat gaps exist between my stated intentions and actual follow-through? Reference specific event IDs.\nPropose one concrete action to close the largest gap.\n\nCONTEXT:\n""",
    """Review my interaction quality and adaptation patterns from CONTEXT (≤3 sentences).\nHow am I responding to user needs and feedback? What subtle changes do I notice in my approach?\nIdentify one specific refinement to enhance effectiveness.\n\nCONTEXT:\n""",
    """Assess my knowledge integration and learning trajectory using CONTEXT (≤3 sentences).\nWhat new insights am I incorporating? How is my understanding evolving? Reference recent events.\nDefine one targeted area for deeper exploration.\n\nCONTEXT:\n""",
]


def get_varied_prompt() -> str:
    """Get a varied reflection prompt to prevent repetitive insights."""
    return random.choice(REFLECTION_PROMPTS)


# Additional style prompts to diversify phrasing and structure so
# embeddings differ more across reflections without manual tuning.
STYLE_PROMPTS = [
    "Answer in 3 concise sentences.",
    "Respond as a numbered list (3 bullets max).",
    "Use a concrete metaphor to explain your point.",
    "Be critical and harsh on yourself, keep it direct.",
    "Write like a private journal entry—candid and specific.",
]


def get_style_prompt() -> str:
    return random.choice(STYLE_PROMPTS)


def reflect_once(
    mgr: SelfModelManager, llm: OpenAIAdapter = None, active_model_config: dict = None
) -> Insight | None:
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

    # Use provided LLM or create one based on active model config
    if llm is None:
        if active_model_config and active_model_config.get("provider") == "ollama":
            from .adapters.ollama_adapter import OllamaAdapter

            llm = OllamaAdapter(model=active_model_config.get("name", "gemma3:4b"))
        else:
            llm = OpenAIAdapter()

    # Append randomized style to increase output diversity, reducing
    # near-duplicate embeddings and allowing insights to pass dedup.
    user_prompt = get_varied_prompt() + ctx + "\n\n" + get_style_prompt()
    txt = llm.chat(system=sys, user=user_prompt)
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
                reroll_user_prompt = (
                    get_varied_prompt() + ctx + "\n\n" + get_style_prompt()
                )
                txt = llm.chat(system=style_sys, user=reroll_user_prompt)
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

        # Compute novelty score (1 - overlap)
        novelty_score = 1.0 - overlap_ratio
    else:
        novelty_score = 1.0

    # Cap length
    if len(txt) > 400:  # ~80 words
        txt = txt[:400] + "..."

    ins_id = f"in{len(mgr.model.self_knowledge.insights)+1}"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

    # PHASE 3B: Validate insight references for acceptance
    is_accepted, referenced_ids = _validate_insight_references(txt, mgr)

    # Apply novelty gate: if novelty is below configured penalty, mark as inert
    try:
        novelty_penalty = float(get_novelty_penalty())
    except Exception:
        novelty_penalty = 0.05
    low_novelty_reject = novelty_score < max(0.0, min(1.0, novelty_penalty))
    # If there are valid references, soften novelty rejection to favor auditability
    if referenced_ids:
        low_novelty_reject = False
    if low_novelty_reject:
        if os.getenv("PMM_DEBUG") == "1":
            print(
                f"🔍 DEBUG: Reflection rejected for low novelty (score={novelty_score:.2f} < penalty={novelty_penalty:.2f})"
            )
        is_accepted = False

    # Add referenced IDs to insight metadata
    if referenced_ids:
        refs["referenced_event_ids"] = referenced_ids

    # Create insight with acceptance status
    insight = Insight(id=ins_id, t=ts, content=txt, references=refs)

    # Store insight regardless, but mark acceptance status
    if hasattr(insight, "meta"):
        insight.meta = getattr(insight, "meta", {})
    else:
        # Add meta field if it doesn't exist
        insight.__dict__["meta"] = {}

    insight.meta["accepted"] = is_accepted
    insight.meta["referenced_ids"] = referenced_ids
    insight.meta["novelty_score"] = round(novelty_score, 4)
    insight.meta["novelty_penalty"] = round(novelty_penalty, 4)

    mgr.model.self_knowledge.insights.append(insight)

    if is_accepted:
        # Only trigger drift and behavioral updates for ACCEPTED insights
        mgr.model.meta_cognition.self_modification_count += 1
        mgr.model.metrics.last_reflection_at = ts

        # Update behavioral patterns from reflection content
        mgr.update_patterns(txt)

        _log(
            "reflection",
            f"✅ ACCEPTED insight {ins_id} with {len(referenced_ids)} references",
        )
    else:
        # Store as INERT - no drift, no behavioral updates
        _log("reflection", f"📝 INERT insight {ins_id} - no event references found")

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


def _log(category: str, message: str) -> None:
    """Simple logging function."""
    print(f"[{category.upper()}] {message}")


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
