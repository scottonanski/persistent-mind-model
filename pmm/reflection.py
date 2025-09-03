from __future__ import annotations
import os
import random
import re
from typing import List, Tuple
from datetime import datetime, timezone
from pmm.model import Insight
from pmm.self_model_manager import SelfModelManager
from pmm.meta_reflection import apply_ref_nudge
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

    # Soft-accept path: allow self-referential, PMM-anchored insights without strict IDs
    # Criteria: first-person language AND presence of PMM anchor terms
    if not is_accepted:
        try:
            has_self_ref = bool(
                re.search(r"\b(I|my|me|myself|mine)\b", content, re.IGNORECASE)
            )
        except Exception:
            has_self_ref = False
        pmm_anchors = [
            "commitment",
            "commitments",
            "memory",
            "identity",
            "drift",
            "emergence",
            "self-model",
            "pattern",
        ]
        try:
            has_anchor = any(a in content.lower() for a in pmm_anchors)
        except Exception:
            has_anchor = False

        if has_self_ref and has_anchor:
            # Mark as accepted but unverified; add a sentinel ref so downstream logic treats it as referenced
            referenced_ids.append("unverified:self_ref")
            is_accepted = True

    if os.getenv("PMM_DEBUG") == "1":
        print(
            f"🔍 DEBUG: Insight validation - References found: {referenced_ids}, Accepted: {is_accepted}"
        )

    return is_accepted, referenced_ids


def validate_insight_references(content: str, mgr: SelfModelManager) -> Tuple[bool, List[str]]:
    """
    Public wrapper exposing insight reference validation.

    Delegates to the internal `_validate_insight_references` while providing a
    stable import for tests and external callers.
    """
    return _validate_insight_references(content, mgr)


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
    """Produce one concise, first-person meta-insight (≤3 sentences) about my evolving behavior or mindset.\nGround it in the CONTEXT; avoid fluff and poetry. CRITICAL: Reference specific recent events by their IDs (e.g., "Based on event ev123...") or commitment hashes.\nInclude 2–3 concrete event references by ID from the last 20 events (e.g., "ev312", "ev315"); if none apply, pick the most relevant recent ones. Also mention one PMM anchor term (commitments, memory, drift, identity, emergence) once.\nIf nothing new stands out, briefly acknowledge steadiness but add one specific micro-adjustment to try next. Vary wording across runs.\nFormat: one short paragraph; then a single line "refs: ev###, ev###".\n\nCONTEXT:\n""",
    """Analyze my recent patterns and identify one specific behavioral shift or emerging tendency (≤3 sentences).\nReference concrete events or commitments from CONTEXT. Focus on what's actually changing in my responses or approach.\nInclude 2–3 concrete event references by ID from the last 20 events (e.g., "ev312", "ev315"); if none apply, pick the most relevant recent ones. Also mention one PMM anchor term (commitments, memory, drift, identity, emergence) once.\nEnd with one tactical adjustment I can implement immediately.\nFormat: one short paragraph; then a single line "refs: ev###, ev###".\n\nCONTEXT:\n""",
    """Examine my commitment fulfillment and goal alignment based on CONTEXT (≤3 sentences).\nWhat gaps exist between my stated intentions and actual follow-through? Reference specific event IDs.\nInclude 2–3 concrete event references by ID from the last 20 events (e.g., "ev312", "ev315"); if none apply, pick the most relevant recent ones. Also mention one PMM anchor term (commitments, memory, drift, identity, emergence) once.\nPropose one concrete action to close the largest gap.\nFormat: one short paragraph; then a single line "refs: ev###, ev###".\n\nCONTEXT:\n""",
    """Review my interaction quality and adaptation patterns from CONTEXT (≤3 sentences).\nHow am I responding to user needs and feedback? What subtle changes do I notice in my approach?\nInclude 2–3 concrete event references by ID from the last 20 events (e.g., "ev312", "ev315"); if none apply, pick the most relevant recent ones. Also mention one PMM anchor term (commitments, memory, drift, identity, emergence) once.\nIdentify one specific refinement to enhance effectiveness.\nFormat: one short paragraph; then a single line "refs: ev###, ev###".\n\nCONTEXT:\n""",
    """Assess my knowledge integration and learning trajectory using CONTEXT (≤3 sentences).\nWhat new insights am I incorporating? How is my understanding evolving? Reference recent events.\nInclude 2–3 concrete event references by ID from the last 20 events (e.g., "ev312", "ev315"); if none apply, pick the most relevant recent ones. Also mention one PMM anchor term (commitments, memory, drift, identity, emergence) once.\nDefine one targeted area for deeper exploration.\nFormat: one short paragraph; then a single line "refs: ev###, ev###".\n\nCONTEXT:\n""",
]

# Additional high-novelty prompts focused on identity evolution and evidence
EXTRA_REFLECTION_PROMPTS = [
    """Explore a concrete shift in identity expression from CONTEXT (≤3 sentences).\nIdentify one behavior you changed and one you will now adopt. Reference 2 event IDs or a commitment hash. Finish with one measurable action (timeframe, count, or percent).\n\nCONTEXT:\n""",
    """From CONTEXT, infer one stable principle and one evolving frontier (≤3 sentences).\nName the tradeoff you’ll experiment with next and why. Reference evIDs. Finish with a single measurable action in first person.\n\nCONTEXT:\n""",
    """Diagnose a recurring constraint in your reflections from CONTEXT (≤3 sentences).\nPropose a specific novelty tactic to break it. Reference evIDs; finish with one measurable action.\n\nCONTEXT:\n""",
    """Map one commitment to evidence from CONTEXT (≤3 sentences).\nState if closure is justified; if not, define one concrete evidence to seek. Reference commit hash or evID; finish with one measurable action.\n\nCONTEXT:\n""",
    """Extract an emerging pattern across events in CONTEXT (≤3 sentences).\nName its upside and risk; define a micro-adjustment. Reference evIDs; finish with one measurable action.\n\nCONTEXT:\n""",
]

REFLECTION_PROMPTS += EXTRA_REFLECTION_PROMPTS


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


# --- Simple epsilon-greedy bandit for prompt template selection ---
class _PromptBandit:
    def __init__(self, n: int, epsilon: float = 0.12):
        self.n = n
        self.epsilon = epsilon
        self.counts = [0] * n
        self.rewards = [0.0] * n

    def select(self) -> int:
        try:
            import random as _r

            if _r.random() < self.epsilon:
                return _r.randrange(self.n)
            # Exploit best average reward
            avgs = [
                self.rewards[i] / self.counts[i] if self.counts[i] else 0.0
                for i in range(self.n)
            ]
            return max(range(self.n), key=lambda i: avgs[i])
        except Exception:
            return 0

    def report(self, idx: int, reward: float) -> None:
        try:
            if idx < 0 or idx >= self.n:
                return
            # Optional: skip tiny updates to reduce churn
            try:
                curr_avg = (
                    (self.rewards[idx] / self.counts[idx]) if self.counts[idx] else None
                )
                if curr_avg is not None and abs(float(reward) - curr_avg) < 0.05:
                    return
            except Exception:
                pass
            self.counts[idx] += 1
            self.rewards[idx] += max(0.0, min(1.0, float(reward)))
        except Exception:
            pass


_bandit = _PromptBandit(n=3, epsilon=0.12)


def reflect_once(
    mgr: SelfModelManager, llm: OpenAIAdapter = None, active_model_config: dict = None
) -> Insight | None:
    # Restore bandit state from meta_cognition if present
    try:
        mc = mgr.model.meta_cognition
        if getattr(mc, "bandit_counts", None) and getattr(mc, "bandit_rewards", None):
            if (
                len(mc.bandit_counts) == _bandit.n
                and len(mc.bandit_rewards) == _bandit.n
            ):
                _bandit.counts = list(mc.bandit_counts)
                _bandit.rewards = list(mc.bandit_rewards)
    except Exception:
        pass

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

    # Template stems (primary bandit target)
    prompt_templates = [
        "You are a terse metacognitive process. Observe your current state, then commit to one concrete micro-adjustment. Keep response to 55-80 words. End with one measurable, first-person action (include a number, timeframe, or percent).",
        "You are a reflective AI system. Identify a tension or limitation in your approach, then propose a specific experiment. Keep response to 55-80 words. End with one measurable, first-person action (include a number, timeframe, or percent).",
        "You are an introspective agent. Form a hypothesis about improving your effectiveness, then define a measurable test. Keep response to 55-80 words. End with one measurable, first-person action (include a number, timeframe, or percent).",
    ]

    # Select template via epsilon-greedy bandit
    try:
        template_idx = _bandit.select()
    except Exception:
        template_idx = len(mgr.model.self_knowledge.insights) % len(prompt_templates)
    sys = prompt_templates[template_idx]
    template_id = "generic"
    if "introspective agent" in sys:
        template_id = "introspective"
    elif "reflective AI system" in sys:
        template_id = "principle_frontier"
    elif "terse metacognitive" in sys:
        template_id = "terse_micro"

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
    # Choose a secondary framing (non-bandit) to add variance
    varied_prompt, _refs_stub = get_varied_prompt(), None
    user_prompt = varied_prompt + ctx + "\n\n" + get_style_prompt()

    # Nudge prompt with explicit reference instructions and candidate IDs
    try:
        user_prompt = apply_ref_nudge(user_prompt, getattr(mgr, "sqlite_store", None))
    except Exception:
        pass
    txt = llm.chat(system=sys, user=user_prompt)
    # Clamp reflection length softly to ensure Next line fits
    try:
        MAX_REFLECTION_TOKENS = int(os.environ.get("PMM_REFLECTION_MAX_TOKENS", "160"))
    except Exception:
        MAX_REFLECTION_TOKENS = 160

    def _truncate_soft(s: str) -> str:
        parts = (s or "").splitlines()
        if len(parts) > 2:
            parts = parts[:2]
        s2 = "\n".join(parts)
        toks = s2.split()
        if len(toks) > MAX_REFLECTION_TOKENS:
            s2 = " ".join(toks[:MAX_REFLECTION_TOKENS])
        return s2

    txt = _truncate_soft(txt)
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
            import os as _os

            if _os.getenv("PMM_DEBUG") == "1":
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
                try:
                    reroll_user_prompt = apply_ref_nudge(
                        reroll_user_prompt, getattr(mgr, "sqlite_store", None)
                    )
                except Exception:
                    pass
                txt = llm.chat(system=style_sys, user=reroll_user_prompt)
                if not txt:
                    if _os.getenv("PMM_DEBUG") == "1":
                        print("   ⚠️  Re-roll failed, using original response")
                else:
                    if _os.getenv("PMM_DEBUG") == "1":
                        print("   ✅ Re-roll successful, reduced repetition")
            except Exception as e:
                if _os.getenv("PMM_DEBUG") == "1":
                    print(
                        f"   ⚠️  Re-roll failed ({type(e).__name__}), using original response"
                    )
            # Keep original txt if re-roll fails

        # Compute novelty score (1 - overlap)
        novelty_score = 1.0 - overlap_ratio
    else:
        novelty_score = 1.0

    # Contract enforcement: exactly one Next: line, actionable + measurable
    # Default OFF to allow autonomous, semantic commitments without the literal keyword
    enforce = str(os.environ.get("PMM_ENFORCE_NEXT_CONTRACT", "0")).lower() in (
        "1",
        "true",
        "yes",
    )
    contract_ok = True
    contract_reason = None
    contract_first = None
    contract_meta = {}

    def _is_next(line: str) -> bool:
        lower_line = line.strip().lower()
        return lower_line.startswith("next:") or lower_line.startswith("next,")

    lines = txt.splitlines()
    next_lines = [ln for ln in lines if _is_next(ln)]
    if enforce:
        if len(next_lines) == 0:
            contract_ok = False
            contract_reason = "no_next"
        elif len(next_lines) > 1:
            contract_ok = False
            contract_reason = "multi_next"
        else:
            nl_full = next_lines[0]
            nl = nl_full[5:].strip()
            # Build structural context: preceding paragraph + Next
            try:
                idx = lines.index(nl_full)
            except ValueError:
                idx = -1
            context_block = ""
            if idx > 0:
                start = idx - 1
                while start >= 0 and lines[start].strip() != "":
                    start -= 1
                start += 1
                context_block = "\n".join(lines[start:idx]).strip()
            combined_for_validation = (
                (context_block + "\n" + nl).strip() if context_block else nl
            )
            # Algorithmic actionability via structural validator
            # Use EnhancedCommitmentValidator confidence instead of verb lists.
            # 100% structural: rely solely on EnhancedCommitmentValidator
            from pmm.enhanced_commitment_validator import (
                EnhancedCommitmentValidator,
            )

            # Use built-in default threshold; env override optional but not required
            try:
                conf_thr = float(os.environ.get("PMM_NEXT_MIN_CONF", "0.3"))
            except Exception:
                conf_thr = 0.3

            _ecv = EnhancedCommitmentValidator()
            analysis = _ecv.validate_commitment(combined_for_validation, [])
            actionable = bool(analysis.is_valid and analysis.confidence >= conf_thr)
            import re

            # Strict measurability: require units or percent/threshold; reject bare numbers
            MEASURABLE = re.compile(
                r"(\b\d+(?:\.\d+)?\s*(minutes?|minute|hours?|days?|weeks?)\b|\b\d+\s*%|\bpercent\b|\bthreshold\b|\bcount\b)",
                re.IGNORECASE,
            )
            BARE_BAD = re.compile(
                r"^\s*\+?\d+(?:\.\d+)?\s*(minutes?|minute|hours?|days?|weeks?)?\s*$",
                re.IGNORECASE,
            )
            measurable = bool(MEASURABLE.search(nl)) and not bool(BARE_BAD.match(nl))
            if not actionable:
                contract_ok = False
                contract_reason = "non_actionable"
            elif not measurable:
                contract_ok = False
                contract_reason = "non_measurable"

            contract_first = "ok" if contract_ok else "inert"
            contract_meta = {
                "first_pass": contract_first,
                "first_reason": (None if contract_ok else contract_reason),
                "rerolled": False,
            }

            # One-reroll enforcement on contract fail
            reroll_enabled = str(
                os.environ.get("PMM_REFLECT_REROLL_ON_CONTRACT_FAIL", "1")
            ).lower() in (
                "1",
                "true",
                "yes",
            )
            if (
                enforce
                and reroll_enabled
                and not contract_ok
                and (contract_reason in {"no_next", "non_measurable", "non_actionable"})
            ):
                try:
                    HARD = (
                        "\n\nCONTRACT (MANDATORY): Produce exactly two lines.\n"
                        "Line 1: one concise reflection sentence (<=25 words).\n"
                        "Line 2: Next: I will <verb> <object> within <NUMBER> <minutes|hours|days> OR a percent target (e.g., 70%).\n"
                        "Rules: Output one Next: line only. The Next line must include 'I will' and a timeframe with units or a percent.\n"
                        "Do NOT write bare numbers like 'Next: 30.' or 'Next: +15 minutes.' Start line 2 with exactly 'Next: I will'."
                    )
                    BAD_TEMPLATES = {"principle_frontier"}
                    REROLL_SUFFIX_STRICT = (
                        "\n\nProduce exactly two lines:\n"
                        "Line 1: one concise reflection sentence (<=25 words).\n"
                        "Line 2: Next: I will <verb> <object> within <NUMBER> <minutes|hours|days>  (or with <NUMBER>% target).\n"
                        "Rules:\n• Output one Next: line only, starting with 'Next: I will'.\n"
                        "• Include a number + units or a percent target (e.g., 70%).\n"
                        "• Do not output bare numbers like 'Next: 30.' or 'Next: +15 minutes.'\n"
                        "Examples that PASS:\n- Next: I will summarize today’s 3 insights within 15 minutes\n- Next: I will draft 2 scene summaries within 30 minutes\n- Next: I will achieve ≥70% positive feedback this hour\n"
                    )
                    # Best-effort lower temperature
                    try:
                        fallback_temp = float(
                            os.environ.get("PMM_REFLECT_REROLL_TEMP", "0.1")
                        )
                    except Exception:
                        fallback_temp = 0.1
                    try:
                        old_temp = (
                            getattr(llm, "temperature")
                            if hasattr(llm, "temperature")
                            else None
                        )
                        if old_temp is not None:
                            setattr(llm, "temperature", fallback_temp)
                    except Exception:
                        old_temp = None

                    reroll_user_prompt = user_prompt + (
                        REROLL_SUFFIX_STRICT if template_id in BAD_TEMPLATES else HARD
                    )
                    txt2 = llm.chat(system=sys, user=reroll_user_prompt) or ""
                    txt2 = _truncate_soft(txt2)
                    try:
                        if old_temp is not None:
                            setattr(llm, "temperature", old_temp)
                    except Exception:
                        pass

                    # Validate reroll
                    lines2 = txt2.splitlines()
                    n2 = [ln for ln in lines2 if _is_next(ln)]
                    if len(n2) == 1:
                        nl2_full = n2[0]
                        nl2 = nl2_full[5:].strip()
                        try:
                            idx2 = lines2.index(nl2_full)
                        except ValueError:
                            idx2 = -1
                        ctx2 = ""
                        if idx2 > 0:
                            s2 = idx2 - 1
                            while s2 >= 0 and lines2[s2].strip() != "":
                                s2 -= 1
                            s2 += 1
                            ctx2 = "\n".join(lines2[s2:idx2]).strip()
                        comb2 = (ctx2 + "\n" + nl2).strip() if ctx2 else nl2
                        analysis2 = _ecv.validate_commitment(comb2)
                        actionable2 = bool(
                            analysis2.is_valid and analysis2.confidence >= conf_thr
                        )
                        measurable2 = bool(MEASURABLE.search(nl2)) and not bool(
                            BARE_BAD.match(nl2)
                        )
                        if actionable2 and measurable2:
                            txt = txt2
                            contract_ok = True
                            contract_reason = None
                    contract_meta["rerolled"] = True
                    contract_meta["final"] = "ok" if contract_ok else "inert"
                    contract_meta["final_reason"] = (
                        None if contract_ok else contract_reason
                    )
                except Exception:
                    contract_meta["rerolled"] = True
                    contract_meta["final"] = "ok" if contract_ok else "inert"
                    contract_meta["final_reason"] = (
                        None if contract_ok else contract_reason
                    )

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
    # Provisional closure hints (non-evidence) to support GAS boosting without permanent closure
    try:
        hinted = mgr.provisional_close_commitments_from_reflection(txt)
        if hinted:
            _log(
                "commitment",
                f"Provisional closure hints emitted for {len(hinted)} commitments",
            )
    except Exception:
        pass

    # Extract and track commitments with provenance from reflection text (already done above at line 658-664)

    # Scan recent conversation events for user messages and extract commitments
    try:
        if os.getenv("PMM_DEBUG") == "1":
            print("[DEBUG] Hook executed")
        recent_events = mgr.sqlite_store.recent_events(limit=10)
        if os.getenv("PMM_DEBUG") == "1":
            print(f"[DEBUG] Recent events count: {len(recent_events)}")
        for event in recent_events:
            if os.getenv("PMM_DEBUG") == "1":
                print(f"[DEBUG] Event kind: {event.get('kind')}, content preview: {event.get('content', '')[:100]}")
            if event.get("kind") == "event":
                content = event.get("content", "")
                if "User said:" in content:
                    user_message = content.split("User said:", 1)[1].strip()
                    if os.getenv("PMM_DEBUG") == "1":
                        print(f"[DEBUG] Checking user message: '{user_message}'")
                    user_commitment, _ = mgr.commitment_tracker.extract_commitment(user_message)
                    if user_commitment:
                        cid = mgr.add_commitment(user_commitment, ins_id)
                        if cid:
                            _log("commitment", f"Added commitment from user message {cid}: {user_commitment[:50]}...")
                    elif os.getenv("PMM_DEBUG") == "1":
                        print(f"[DEBUG] No commitment extracted from: '{user_message}'")
                elif os.getenv("PMM_DEBUG") == "1":
                    print(f"[DEBUG] No 'User said:' in content: '{content[:100]}'")
    except Exception as e:
        if os.getenv("PMM_DEBUG") == "1":
            print(f"[DEBUG] Error scanning events: {e}")
        pass
    is_accepted, referenced_ids = _validate_insight_references(txt, mgr)
    soft_accepted = any(str(r).startswith("unverified:") for r in referenced_ids)

    # Apply novelty gate: if novelty is below configured penalty, mark as inert
    novelty_penalty = 0.05
    try:
        novelty_penalty = float(get_novelty_penalty())
    except Exception:
        pass
    low_novelty_reject = novelty_score < max(0.0, min(1.0, novelty_penalty))
    # If there are valid references (including soft-accept sentinel), soften novelty rejection
    if referenced_ids:
        low_novelty_reject = False
    if low_novelty_reject:
        if os.getenv("PMM_DEBUG") == "1":
            print(
                f"🔍 DEBUG: Reflection rejected for low novelty (score={novelty_score:.2f} < penalty={novelty_penalty:.2f})"
            )
        is_accepted = False

    # Enforce Next: contract
    if enforce and not contract_ok:
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
    insight.meta["soft_accepted"] = bool(soft_accepted)
    insight.meta["referenced_ids"] = referenced_ids
    insight.meta["novelty_score"] = round(novelty_score, 4)
    insight.meta["novelty_penalty"] = round(novelty_penalty, 4)

    mgr.model.self_knowledge.insights.append(insight)

    # Bandit feedback: only for contract-passing accepted reflections
    if contract_ok and is_accepted:
        try:
            FIRST_PASS_BONUS = float(
                os.environ.get("PMM_BANDIT_FIRST_PASS_BONUS", "0.10")
            )
            reward = 0.7 * 1.0 + 0.3 * float(novelty_score)
            if contract_meta.get("first_pass") == "ok":
                reward += FIRST_PASS_BONUS
            _bandit.report(template_idx, reward)
            try:
                mgr.model.meta_cognition.bandit_counts = list(_bandit.counts)
                mgr.model.meta_cognition.bandit_rewards = list(_bandit.rewards)
                mgr.save_model()
            except Exception:
                pass
        except Exception:
            pass

    if is_accepted and contract_ok:
        # Only trigger drift and behavioral updates for ACCEPTED insights
        mgr.model.meta_cognition.self_modification_count += 1
        mgr.model.metrics.last_reflection_at = ts

        # Update behavioral patterns from reflection content
        mgr.update_patterns(txt)

        label = "SOFT-ACCEPTED" if soft_accepted else "ACCEPTED"
        _log(
            "reflection",
            f"✅ {label} insight {ins_id} with {len(referenced_ids)} references",
        )
    else:
        # Store as INERT - no drift, no behavioral updates
        reason = contract_reason or ("low_novelty" if low_novelty_reject else "no_refs")
        _log("reflection", f"📝 INERT insight {ins_id} - reason={reason}")

    # Use add_insight to trigger commitment extraction
    mgr.save_model()

    # Also append a 'reflection' event into SQLite for probes/analytics
    try:
        store = getattr(mgr, "sqlite_store", None)
        if store is not None:
            meta = {
                "source": "reflect_once",
                "accepted": bool(is_accepted and contract_ok),
                "soft_accepted": bool(soft_accepted),
                "referenced_ids": referenced_ids,
                "status": ("ok" if (is_accepted and contract_ok) else "inert"),
                "reason": (
                    None
                    if (is_accepted and contract_ok)
                    else (
                        contract_reason
                        or ("low_novelty" if low_novelty_reject else "no_refs")
                    )
                ),
                "contract": contract_meta,
            }
            store.append_event(kind="reflection", content=txt, meta=meta)
    except Exception:
        pass
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
    """Simple logging with friendly tags and ANSI color (always on)."""

    def _color(txt: str, code: str | None) -> str:
        return f"\033[{code}m{txt}\033[0m" if code else txt

    cat = (category or "").strip().lower()
    tag = f"[{cat.upper()}]"
    code = None
    if cat.startswith("commit"):
        tag = "[COMMIT+]" if "+" in message or "Added" in message else "[COMMIT]"
        code = "32"  # green
    elif cat.startswith("reflection"):
        # Accepted vs inert
        code = "36"  # cyan
        if "INERT" in message:
            tag = "[REFLECT~]"
            code = "33"  # yellow
        elif "ACCEPTED" in message or "SOFT-ACCEPTED" in message:
            tag = "[REFLECT+]"
            code = "32"  # green
        else:
            tag = "[REFLECT]"
    else:
        tag = f"[{cat.upper()}]"
        code = None

    print(f"{_color(tag, code)} {message}")


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
