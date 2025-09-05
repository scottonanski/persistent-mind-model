#!/usr/bin/env python3
"""
PMM Chat - Interactive interface for your Persistent Mind Model
Main entry point for chatting with your autonomous AI personality.
"""

import os
import sys
import argparse
import json

# Add current directory to path for PMM imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pmm.langchain_memory import PersistentMindMemory
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from pmm.config import (
    get_default_model,
    get_model_config,
    list_available_models,
    AVAILABLE_MODELS,
    get_ollama_models,
)
from pmm.emergence import compute_emergence_scores, EmergenceAnalyzer
from pmm.commitments import (
    tick_turn_scoped_identity_commitments,
    open_identity_commitment,
    get_identity_turn_commitments,
    close_identity_turn_commitments,
)
from pmm.reflection import reflect_once
from pmm.logging_config import pmm_tlog
from pmm.dev_tasks import DevTaskManager
from pmm.policy import bandit as pmm_bandit

# --- Structural helpers (regex-free) -----------------------------------------

def _looks_like_codey(text: str) -> bool:
    """Heuristic: does the text look like code or stack traces?"""
    if not text:
        return False
    t = text
    tl = t.lower()
    # Easy markers
    if "```" in t:
        return True
    if "traceback" in tl or "exception" in tl:
        return True
    # Common code punctuation
    if any(ch in t for ch in "{};="):
        return True
    # Stack trace-ish pattern like " at Foo(" without regex
    if " at " in t and "(" in t and ")" in t:
        return True
    return False


def _is_display_cid(token: str) -> bool:
    """Matches display ids like c12, c3 (formerly re.fullmatch(r"c\\d+", ...))."""
    if not token or len(token) < 2:
        return False
    if token[0].lower() != "c":
        return False
    return token[1:].isdigit()


def _is_hex_prefix(token: str) -> bool:
    """Matches 6-64 length hex strings (formerly regex [0-9a-fA-F]{6,64})."""
    if not token:
        return False
    n = len(token)
    if n < 6 or n > 64:
        return False
    hexdigits = set("0123456789abcdefABCDEF")
    return all(c in hexdigits for c in token)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PMM Chat - Interactive AI personality interface"
    )
    parser.add_argument("--model", help="Model name or number from the menu")
    parser.add_argument(
        "--agent",
        default=os.environ.get("PMM_AGENT_PATH", ".data/pmm.json"),
        help="Path to agent state or directory (defaults to PMM_AGENT_PATH or .data/pmm.json)",
    )
    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Force non-interactive mode; do not try to read from /dev/tty",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs (API call info, internal anchors)",
    )
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Enable PMM telemetry logs in chat output",
    )
    return parser.parse_args()


def show_model_selection(force_tty=True):
    """Show model selection interface and return selected model."""
    print("=== PMM Model Selection ===")
    print()

    # Show current default model at top
    default_model = get_default_model()
    default_config = get_model_config(default_model)
    default_cost_str = (
        f"${default_config.cost_per_1k_tokens:.4f}/1K"
        if default_config.cost_per_1k_tokens > 0
        else "Free (local)"
    )

    print(f"⭐ CURRENT DEFAULT: {default_model} ({default_config.provider})")
    print(f"   {default_config.description}")
    print(f"   Max tokens: {default_config.max_tokens:,} | Cost: {default_cost_str}")
    print()

    # Show all available models
    print("📋 Available Models:")
    available_models = list_available_models()
    for i, model_name in enumerate(available_models, 1):
        config = AVAILABLE_MODELS[model_name]
        cost_str = (
            f"${config.cost_per_1k_tokens:.4f}/1K"
            if config.cost_per_1k_tokens > 0
            else "Free (local)"
        )
        marker = "⭐" if model_name == default_model else f"{i:2d}."
        status = ""
        if config.provider == "ollama":
            # Quick check if Ollama model is available
            status = (
                " 🟢"
                if model_name in [m["name"] for m in get_ollama_models()]
                else " 🔴"
            )

        print(f"{marker} {model_name} ({config.provider}){status}")
        print(f"    {config.description}")
        print(f"    Max tokens: {config.max_tokens:,} | Cost: {cost_str}")
        print()

    print("💡 Select a model:")
    print("   • Press ENTER to use current default")
    print(
        "   • Type model number (1-{}) or exact model name".format(
            len(available_models)
        )
    )
    print("   • Type 'list' to see this menu again")
    print()

    # Handle piped input more gracefully
    if not sys.stdin.isatty():
        if not force_tty:
            print("🎯 Non-interactive mode detected, using default model")
            return default_model

        # Try to open /dev/tty for interactive selection even with piped stdin
        try:
            with open("/dev/tty", "r+") as tty:
                print(
                    "🎯 Piped input detected, but opening /dev/tty for model selection..."
                )
                while True:
                    tty.write("🎯 Your choice: ")
                    tty.flush()
                    choice = tty.readline().strip()

                    if not choice:
                        return default_model

                    if choice.lower() == "list":
                        tty.write("\n📋 Available Models (see above)\n")
                        tty.flush()
                        continue

                    # Try to parse as number
                    if choice.isdigit():
                        idx = int(choice)
                        if 1 <= idx <= len(available_models):
                            selected_model = available_models[idx - 1]
                            tty.write(f"✅ Selected model {idx}: {selected_model}\n")
                            tty.flush()
                            return selected_model
                        tty.write(
                            f"❌ Please enter a number between 1 and {len(available_models)}\n"
                        )
                        tty.flush()
                        continue

                    # Try exact model name
                    if choice in available_models:
                        tty.write(f"✅ Selected model by name: {choice}\n")
                        tty.flush()
                        return choice

                    tty.write(
                        f"❌ Unknown model '{choice}'. Type 'list' to see available models.\n"
                    )
                    tty.flush()

        except Exception as e:
            print(
                f"🎯 Non-interactive mode & no /dev/tty available ({e}); using default model"
            )
            return default_model

    while True:
        try:
            choice = input("🎯 Your choice: ").strip()

            if not choice:
                return default_model

            if choice.lower() == "list":
                show_model_selection()
                continue

            # Try to parse as number
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    return selected_model
                else:
                    print(
                        f"❌ Please enter a number between 1 and {len(available_models)}"
                    )
                    continue
            except ValueError:
                pass

            # Try exact model name
            if choice in available_models:
                return choice

            print(f"❌ Unknown model '{choice}'. Type 'list' to see available models.")

        except KeyboardInterrupt:
            print("\n👋 Exiting model selection...")
            return None


def main():
    """Interactive chat with PMM using working LangChain memory system."""
    load_dotenv()
    args = parse_args()

    print("🧠 PMM Chat - Your Persistent AI Mind")
    print("=====================================\n")

    # Feature toggles
    SUMMARY_ENABLED = os.getenv("PMM_ENABLE_SUMMARY", "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    EMBEDDINGS_ENABLED = os.getenv("PMM_ENABLE_EMBEDDINGS", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # Model selection
    if args.model:
        # Allow number or name from CLI
        available_models = list_available_models()
        chosen = None

        # Try as number first
        if args.model.isdigit():
            idx = int(args.model)
            if 1 <= idx <= len(available_models):
                chosen = available_models[idx - 1]
                print(f"✅ CLI selected model {idx}: {chosen}")

        # Try as exact name if number didn't work
        if not chosen and args.model in available_models:
            chosen = args.model
            print(f"✅ CLI selected model by name: {chosen}")

        if not chosen:
            print(f"❌ Invalid model '{args.model}', showing selection menu...")
            model_name = show_model_selection(force_tty=not args.noninteractive)
        else:
            model_name = chosen
    else:
        model_name = show_model_selection(force_tty=not args.noninteractive)

    if not model_name:
        return

    print(f"🔄 {model_name} selected... Loading model... Please wait...")
    print()

    # Initialize PMM with selected model
    model_config = get_model_config(model_name)

    # Only require OpenAI key when using an OpenAI provider
    if model_config.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            "❌ OPENAI_API_KEY not set for OpenAI provider. Set it or choose a local Ollama model."
        )
        return

    # Use explicit agent path (CLI or PMM_AGENT_PATH) so chat shares the same store as harness
    agent_path = getattr(args, "agent", None) or os.environ.get(
        "PMM_AGENT_PATH", ".data/pmm.json"
    )
    # Ensure parent directory exists to avoid silent fallback
    try:
        parent = os.path.dirname(agent_path) or "."
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    pmm_memory = PersistentMindMemory(
        agent_path=agent_path,
        personality_config={
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.8,
            "agreeableness": 0.9,
            "neuroticism": 0.3,
        },
        enable_summary=SUMMARY_ENABLED,
        enable_embeddings=EMBEDDINGS_ENABLED,
    )

    print(f"🤖 Using model: {model_name} ({model_config.description})")
    print(
        f"🧩 Thought Summarization: {'ON' if SUMMARY_ENABLED else 'OFF'} | 🔎 Semantic Embeddings: {'ON' if EMBEDDINGS_ENABLED else 'OFF'}"
    )

    # Show personality state
    personality = pmm_memory.get_personality_summary()
    print(f"📚 Loaded PMM with {personality['total_events']} events")
    print(
        f"🎭 Personality: O:{personality['personality_traits']['openness']:.2f} C:{personality['personality_traits']['conscientiousness']:.2f} E:{personality['personality_traits']['extraversion']:.2f} A:{personality['personality_traits']['agreeableness']:.2f} N:{personality['personality_traits']['neuroticism']:.2f}"
    )

    # Initialize dual LLM components for automatic deep mode
    if model_config.provider == "ollama":
        llm_normal = OllamaLLM(model=model_name, temperature=0.7)
        llm_deep = OllamaLLM(model=model_name, temperature=0.2)
    else:  # openai
        llm_normal = ChatOpenAI(model=model_name, temperature=0.7)
        llm_deep = ChatOpenAI(model=model_name, temperature=0.2)

    # Set active config in LLM factory for reflection system
    from pmm.llm_factory import get_llm_factory
    from pmm.embodiment import extract_model_family
    from pmm.bridges import BridgeManager
    from pmm.model_config import ModelConfig

    llm_factory = get_llm_factory()

    # Extract family and version from model name
    family = extract_model_family(model_name)
    version = "unknown"  # Could be extracted from model_config if available

    # Create enhanced config with family info
    enhanced_config = {
        "name": model_name,
        "provider": model_config.provider,
        "family": family,
        "version": version,
        "epoch": llm_factory.get_current_epoch(),
    }

    # Set active config
    prev_config = None
    try:
        prev_config = llm_factory.get_active_config()
    except Exception:
        pass

    llm_factory.set_active_config(enhanced_config)

    # Initialize bridge manager for embodiment-aware rendering
    bridge_manager = BridgeManager(
        factory=llm_factory,
        storage=pmm_memory,
        cooldown=pmm_memory.reflection_cooldown,
        ngram_ban=pmm_memory.ngram_ban,
        stages=pmm_memory.emergence_stages,
    )

    # Handle model switch
    if prev_config:
        prev_model_config = ModelConfig(
            provider=prev_config.get("provider", "unknown"),
            name=prev_config.get("name", "unknown"),
            family=prev_config.get("family", "unknown"),
            version=prev_config.get("version", "unknown"),
            epoch=prev_config.get("epoch", 0),
        )
        curr_model_config = ModelConfig(
            provider=enhanced_config["provider"],
            name=enhanced_config["name"],
            family=enhanced_config["family"],
            version=enhanced_config["version"],
            epoch=enhanced_config["epoch"],
        )
        bridge_manager.on_switch(prev_model_config, curr_model_config)
    else:
        # First initialization
        curr_model_config = ModelConfig(
            provider=enhanced_config["provider"],
            name=enhanced_config["name"],
            family=enhanced_config["family"],
            version=enhanced_config["version"],
            epoch=enhanced_config["epoch"],
        )
        bridge_manager.on_switch(None, curr_model_config)

    # Recovery state: identity nudge + S0 streak tracking
    s0_streak_threshold = 3
    try:
        env_thr = os.getenv("PMM_S0_STREAK_THRESHOLD")
        if env_thr is not None:
            s0_streak_threshold = max(1, int(env_thr))
    except Exception:
        pass

    s0_consecutive = 0
    identity_nudge_flag = False
    # Local turn tracking for identity anchor cooldown
    current_turn = 0
    last_anchor_turn = None

    # Top-level helpers for adaptive memory budgeting
    def _approx_tokens(text: str) -> int:
        return max(1, len(text) // 4)  # rough chars→tokens

    def _trim_to_tokens(text: str, budget_tokens: int) -> str:
        if _approx_tokens(text) <= budget_tokens:
            return text
        # Fast char-based trim with a small safety margin
        target_chars = max(0, (budget_tokens - 16) * 4)
        return text[:target_chars]

    # Auto-detection heuristic for deep reasoning mode
    def should_deep_mode(text: str, scores: dict) -> bool:
        """Detect when to use focused reasoning mode automatically."""
        t = (text or "").lower()
        long = len(text) > 350
        codey = _looks_like_codey(text)
        keywords = any(
            k in t
            for k in [
                "prove",
                "analyze",
                "diagnose",
                "precise",
                "step-by-step",
                "walk me through",
                "explain how",
                "technical",
            ]
        )
        # use emergence snapshot from this turn if available
        stage = (scores or {}).get("stage", "")
        ias = float((scores or {}).get("ias", 0.0) or 0.0)
        gas = float((scores or {}).get("gas", 0.0) or 0.0)
        low_emergence = (
            stage.startswith("S0") or stage.startswith("S1") or ias < 0.25 or gas < 0.50
        )
        return long or codey or keywords or low_emergence

    # Create enhanced system prompt with PMM context
    def get_pmm_system_prompt(
        identity_nudge: bool = False,
        last_user_text: str | None = None,
        memory_chars: int = 1800,
    ):
        # Always pull fresh memory right before each call
        raw_context = pmm_memory.load_memory_variables({}).get("history", "")

        # Normalize context to string: load_memory_variables may return a list of LangChain messages
        def _messages_to_text(msgs) -> str:
            try:
                out = []
                for m in msgs or []:
                    # LangChain BaseMessage typically has .type and .content
                    mtype = getattr(m, "type", None) or getattr(m, "role", "")
                    if mtype == "system":
                        role = "System"
                    elif mtype == "human":
                        role = "Human"
                    elif mtype == "ai" or mtype == "assistant":
                        role = "Assistant"
                    else:
                        role = str(mtype or "Message")
                    content = getattr(m, "content", str(m))
                    out.append(f"{role}: {content}")
                return "\n".join(out)
            except Exception:
                # Fallback best-effort
                return "\n".join([str(x) for x in (msgs or [])])

        if isinstance(raw_context, list):
            raw_context = _messages_to_text(raw_context)
        elif isinstance(raw_context, dict):
            raw_context = str(raw_context)

        # Compact context: dedupe lines, cap each block
        def _compact(text: str, max_lines: int = 120) -> str:
            seen = set()
            out = []
            for line in text.splitlines():
                key = line.strip().lower()
                if key and key not in seen:
                    out.append(line)
                    seen.add(key)
                if len(out) >= max_lines:
                    break
            return "\n".join(out)

        pmm_context = _compact(raw_context)

        # NEW: semantic pull for THIS turn (if embeddings enabled and we have input)
        semantic_bits = []
        try:
            if last_user_text and getattr(pmm_memory, "enable_embeddings", False):
                # uses the same analyzer you already have
                semantic_bits = (
                    pmm_memory._get_semantic_context(last_user_text, max_results=6)
                    or []
                )
        except Exception:
            semantic_bits = []

        # Lightweight loop-avoidance hint for the assistant
        loop_hint = ""
        try:
            # peek at last few user messages from PMM events
            recent = pmm_memory.pmm.sqlite_store.recent_events(limit=20)
            user_msgs = []
            for event_tuple in recent:
                # Handle variable tuple length safely
                if len(event_tuple) >= 4:
                    kind = event_tuple[2]  # event kind
                    content = event_tuple[3]  # event content
                    if kind == "conversation":
                        user_msgs.append(content)
            last_texts = " ".join(user_msgs[:6]).lower()
            if last_texts.count("slop code") >= 2:
                loop_hint = "\nAVOID TOPIC LOOPING: The user already clarified the 'slop code' story; do not rehash it unless explicitly asked."
        except Exception:
            pass

        personality = pmm_memory.get_personality_summary()
        traits = personality["personality_traits"]
        agent_name = pmm_memory.pmm.model.core_identity.name

        # Extract top signals from PMM for a compact policy header
        patterns = personality.get("behavioral_patterns", {}) or {}
        top_patterns = sorted(patterns.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_patterns_str = (
            ", ".join([f"{k}({v})" for k, v in top_patterns])
            if top_patterns
            else "none"
        )

        # Auto-nudge: if identity is LOW and no active identity identity-mode, open an adaptive identity TTL
        try:
            storage = (
                pmm_memory.pmm.sqlite_store if hasattr(pmm_memory.pmm, "sqlite_store") else None
            )
            scores = compute_emergence_scores(window=15, storage_manager=storage)
            ias_now = float(scores.get("ias", 0.0) or 0.0)
            # Only when clearly LOW and not already in identity mode
            if ias_now < 0.20:
                items_now = get_identity_turn_commitments(pmm_memory.pmm)
                if not items_now:
                    try:
                        # Adaptive TTL: deeper dips = longer TTL
                        try:
                            max_cap = int(os.getenv("PMM_IDENTITY_MAX_TTL", "30"))
                        except Exception:
                            max_cap = 30
                        ttl_adaptive = int((0.20 - ias_now) * 100)
                        if ttl_adaptive < 5:
                            ttl_adaptive = 5
                        if ttl_adaptive > max_cap:
                            ttl_adaptive = max_cap
                        h_auto = open_identity_commitment(
                            pmm_memory.pmm,
                            policy="express_core_principles",
                            ttl_turns=ttl_adaptive,
                            note="auto open due to low identity",
                        )
                        if h_auto:
                            print(
                                f"[PMM] auto identity: opened {ttl_adaptive}-turn identity mode (IAS={ias_now:.3f}, id={h_auto[:8]})"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

        # Pull up to three open commitments to actually steer behavior
        try:
            opens = pmm_memory.pmm.get_open_commitments()[:3]
        except Exception:
            opens = []
        open_commitments_str = (
            "\n".join([f"- {c['text']}" for c in opens]) if opens else "none"
        )

        # Mind Policy: neutral, concise, placed BEFORE the memory blob
        mind_policy_lines = [
            "MIND POLICY",
            f"Identity: {agent_name}",
            "Be direct and concise. If uncertain, state uncertainty briefly and answer your best.",
            "You may propose next steps and form commitments when helpful. If the user opts out, stop.",
            "Ask clarifying questions only when necessary to proceed.",
            "Evidence rule: In substantive answers, cite at least one recent event ID (e.g., ev123) or a commitment hash when relevant.",
            "Action rule: When you propose an action, make it measurable (e.g., include a timeframe, count, or percent). Avoid multiple competing actions in one reply.",
            "Variation rule: Vary verbs, quantities, and timeframes to avoid generating duplicate commitments; do not restate identical commitments across turns.",
            "Directive subject rule: Treat third-person (e.g., 'PMM will…', 'The assistant will…') or imperative lines (e.g., 'Complete X now') as your own commitments; register and honor them like first-person commitments.",
            "No background execution: Do not imply ongoing or background work between messages (e.g., 'I'll keep working after this message', 'running this in the background'). Only take actions within explicit turns and report results here.",
            f"Top Patterns (awareness only): {top_patterns_str}",
            "Open Commitments:",
            f"{open_commitments_str}",
        ]
        if identity_nudge:
            mind_policy_lines.append(
                "Identity Nudge: Speak in first-person as this PMM agent. Prefer 'I' statements, reference my open commitments explicitly, avoid generic assistant voice."
            )
        # If an identity turn commitment is active, require an explicit self-ascription early
        try:
            active_idents = get_identity_turn_commitments(pmm_memory.pmm)
            if any(
                (it or {}).get("remaining_turns", 0) > 0 for it in (active_idents or [])
            ):
                mind_policy_lines.append(
                    "Identity Mode: In the first sentence, include a brief explicit self-ascription (e.g., 'I am currently …' or 'My name is …'), then continue naturally. Only once."
                )
        except Exception:
            pass
        mind_policy_lines.append("— end policy —")
        mind_policy = "\n".join(mind_policy_lines)

        # Personality + cross‑session memory for rich context (truncated for prompt safety)
        persona = (
            f"PERSONALITY (Big Five): "
            f"O {traits['openness']:.2f} | C {traits['conscientiousness']:.2f} | "
            f"E {traits['extraversion']:.2f} | A {traits['agreeableness']:.2f} | "
            f"N {traits['neuroticism']:.2f}"
        )

        # --- ADAPTIVE BUDGETING ---
        # Reserve tokens for the model's reply + the live conversation turns
        cfg = get_model_config(model_name)
        max_ctx = max(
            4096, getattr(cfg, "max_tokens", 4096)
        )  # be conservative if unknown
        reserve_for_reply = 512
        reserve_for_header = 400  # mind policy + persona + scaffolding
        reserve_for_history = 900  # running chat thread (messages list)

        available_for_memory = max(
            0, max_ctx - reserve_for_reply - reserve_for_header - reserve_for_history
        )

        # Synthesize a memory section, preferring relevant first, then the legacy blob
        memory_section = ""
        if semantic_bits:
            memory_section += (
                "RELEVANT LONG-TERM CONTEXT:\n" + "\n".join(semantic_bits) + "\n\n"
            )
        memory_section += "CROSS‑SESSION MEMORY (condensed):\n" + pmm_context

        # Apply memory_chars limit before token trimming for deep mode expansion
        if semantic_bits:
            memory_section = (
                "RELEVANT LONG-TERM CONTEXT:\n" + "\n".join(semantic_bits) + "\n\n"
            )
        memory_section += (
            "CROSS‑SESSION MEMORY (condensed):\n" + pmm_context[:memory_chars]
        )

        memory_section = _trim_to_tokens(memory_section, max(256, available_for_memory))

        # Dynamic header trimming when token pressure is high
        trim_ratio = 1.0
        try:
            original_memory = (
                "RELEVANT LONG-TERM CONTEXT:\n"
                + "\n".join(semantic_bits)
                + "\n\nCROSS‑SESSION MEMORY (condensed):\n"
                + pmm_context
            )
            trim_ratio = _approx_tokens(memory_section) / max(
                1, _approx_tokens(original_memory)
            )
        except Exception:
            pass

        # If we're trimming >25% of memory, shorten the header to preserve content
        if trim_ratio < 0.75:
            # Collapse commitments to top 2 and drop Top Patterns for this turn
            open_commitments_str = (
                "\n".join([f"- {c['text']}" for c in (opens[:2] if opens else [])])
                or "none"
            )
            mind_policy_lines = [
                "MIND POLICY",
                f"Identity: {agent_name}",
                "Be direct and concise. If uncertain, state uncertainty briefly and answer your best.",
                "You may propose next steps and form commitments when helpful. If the user opts out, stop.",
                "Ask clarifying questions only when necessary to proceed.",
                "Open Commitments:",
                f"{open_commitments_str}",
            ]
            if identity_nudge:
                mind_policy_lines.append(
                    "Identity Nudge: Speak in first-person as this PMM agent. Prefer 'I' statements, reference my open commitments explicitly, avoid generic assistant voice."
                )
            # Carry forward identity-mode instruction in trimmed header too
            try:
                active_idents = get_identity_turn_commitments(pmm_memory.pmm)
                if any(
                    (it or {}).get("remaining_turns", 0) > 0
                    for it in (active_idents or [])
                ):
                    mind_policy_lines.append(
                        "Identity Mode: In the first sentence, include a brief explicit self-ascription (e.g., 'I am currently …' or 'My name is …'), then continue naturally. Only once."
                    )
            except Exception:
                pass
            mind_policy_lines.append("— end policy —")
            mind_policy = "\n".join(mind_policy_lines)

        return (
            f"You are {agent_name}.\n"
            "You have access to persistent memory (below). Use it as context; otherwise answer directly and plainly.\n"
            f"{mind_policy}{loop_hint}\n\n"
            f"{persona}\n\n"
            f"{memory_section}"
        )

    print(f"\n🤖 PMM is ready! Using {model_name} ({model_config.provider})")
    print(
        "💡 Hint: type --@help for commands. Try: --@identity list | --@probe list | --@track on"
    )
    print(
        "🧪 Auto deep mode: activates automatically for analysis, code, long inputs, or low emergence"
    )
    print("Start chatting...")

    # Initialize conversation history with PMM system prompt
    conversation_history = [
        {"role": "system", "content": get_pmm_system_prompt(identity_nudge_flag)}
    ]

    # Optional feedback collection (interactive only when enabled)
    FEEDBACK_ENABLE = os.getenv("PMM_FEEDBACK_ENABLE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    FEEDBACK_AUTOPROMPT = os.getenv("PMM_FEEDBACK_AUTOPROMPT", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def invoke_model(messages, use_deep_mode=False):
        """Invoke model with proper format based on provider type."""
        current_config = get_model_config(model_name)  # Get current model config
        llm_active = llm_deep if use_deep_mode else llm_normal

        if current_config.provider == "ollama":
            # Ollama expects a single string, so format the conversation
            formatted_prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted_prompt += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    formatted_prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"Assistant: {msg['content']}\n"
            formatted_prompt += "Assistant: "
            return llm_active.invoke(formatted_prompt)
        else:
            # OpenAI chat models expect message list
            return llm_active.invoke(messages)

    # Setup for potentially mixed input modes
    stdin_is_pipe = not sys.stdin.isatty()
    tty_file = None
    if stdin_is_pipe and not args.noninteractive:
        try:
            tty_file = open("/dev/tty", "r")
            print(
                "🎯 Piped input detected. After consuming piped messages, will switch to keyboard input."
            )
        except Exception:
            print("🎯 Piped input detected. Running in non-interactive mode.")
            tty_file = None

    def get_user_input():
        """Get user input from appropriate source."""
        if tty_file:
            print("\n👤 You: ", end="", flush=True)
            return tty_file.readline().strip()
        return input("\n👤 You: ").strip()

    # CLI-controlled logging toggles (off by default)
    debug_on = bool(getattr(args, "debug", False))
    telemetry_flag = bool(getattr(args, "telemetry", False))
    # If telemetry is enabled, print the actual SQLite store path for verification
    if telemetry_flag or (
        os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")
    ):
        try:
            conn = getattr(
                getattr(pmm_memory.pmm, "sqlite_store", object()), "conn", None
            )
            if conn:
                rows = conn.execute("PRAGMA database_list").fetchall()
                if rows:
                    store_path = rows[-1][2]
                    print(f"[STORE] {store_path}")
        except Exception as e:
            print(f"[STORE] error: {e}")
    # Unified runtime tracking toggle (seeded from CLI or env)
    track_enabled = telemetry_flag or (
        os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")
    )

    # Pretty printer for TRACK lines
    def _track_fmt(
        scores: dict, cd_status: dict, s0c: int, s0thr: int, stage: str
    ) -> list[str]:
        def fnum(x, d=3):
            try:
                if x is None:
                    return "-"
                return f"{float(x):.{d}f}"
            except Exception:
                return str(x)

        def hms(sec):
            try:
                if sec is None:
                    return "-"
                sec = int(float(sec))
                m, s = divmod(sec, 60)
                h, m = divmod(m, 60)
                if h:
                    return f"{h}h{m:02d}m"
                if m:
                    return f"{m}m{s:02d}s"
                return f"{s}s"
            except Exception:
                return "-"

        # Friendly labels
        stage_full = stage or "?"
        stage_short = (stage_full.split(":")[0] or stage_full).strip()

        def band(x):
            try:
                x = float(x)
                if x < 0.20:
                    return "low"
                if x < 0.50:
                    return "moderate"
                return "high"
            except Exception:
                return "-"

        ias = scores.get("IAS")
        gas = scores.get("GAS")
        # Use authoritative close rate derived from evidence-linked commitments
        try:
            analyzer = EmergenceAnalyzer(storage_manager=getattr(pmm_memory.pmm, "sqlite_store", None))
            close_rate = float(analyzer.commitment_close_rate(window=50))
        except Exception:
            close_rate = 0.0

        # Color helpers
        def c(text: str, lvl: str) -> str:
            col = {"low": "31", "moderate": "33", "high": "32"}.get(lvl)
            return f"\033[{col}m{text}\033[0m" if col else text

        ident_band = band(ias)
        growth_band = band(gas)
        ident_label = f"identity {c(ident_band.upper(), ident_band)} {fnum(ias)}"
        growth_label = f"growth {c(growth_band.upper(), growth_band)} {fnum(gas)}"
        close_label = f"close {fnum(close_rate,2)}"

        line1 = f"[TRACK] {stage_short} • {ident_label} • {growth_label} • {close_label} • s0={s0c}/{s0thr}"

        # Cooldown / readiness line
        cd = cd_status or {}
        tg = cd.get("time_gate_passed")
        ug = cd.get("turns_gate_passed")
        tg_mark = "✓" if tg else "✗"
        ug_mark = "✓" if ug else "✗"
        tsl = cd.get("turns_since_last", cd.get("turns_since_last:", None))
        tss = hms(cd.get("time_since_last_seconds"))

        # Simple guidance
        hint = None
        try:
            # Identity hint
            if float(ias or 0) < 0.20:
                hint = "Hint: --@identity open 3"
            # Reflection readiness hint
            if (tg and ug) and not hint:
                hint = "Ready to reflect"
            elif (not ug) and not hint:
                # If we know a min_turns, estimate remaining
                need = None
                mt = cd.get("min_turns_required") or cd.get(
                    "min_turns_required_seconds"
                )
                if isinstance(mt, int) and isinstance(tsl, int):
                    need = max(0, int(mt) - int(tsl))
                if need is not None and need > 0:
                    hint = f"Need {need} more turn(s)"
        except Exception:
            pass
        ready = "✓" if (tg and ug) else "✗"
        hint_str = f" • {hint}" if hint else ""
        turns_str = str(tsl) if tsl is not None else "-"
        line2 = f"[TRACK] Reflection: since {tss} • turns {turns_str} • time {tg_mark} • turns {ug_mark} • ready {ready}{hint_str}"
        return [line1, line2]

    # Simple tag colorizer for chat diagnostics
    def _tag(label: str, color: str | None = None) -> str:
        # Always colorize in terminal (simple ANSI)
        codes = {"green": "32", "cyan": "36", "yellow": "33", "magenta": "35"}
        code = codes.get(color)
        return f"\033[{code}m[{label}]\033[0m" if code else f"[{label}]"

    def _print_track_legend():
        print("\n🧭 TRACK quick guide")
        print("  • Stage S0–S4: S0 stuck → S4 generative")
        print(
            "  • identity: how much it speaks as ‘I’ / cites its commitments (LOW/MODERATE/HIGH)"
        )
        print("      Fix LOW: --@identity open 3")
        print("  • growth: novelty/experiments in replies (LOW/MODERATE/HIGH)")
        print("  • close: fraction of commitments closed with evidence (0–1)")
        print("  • Reflection line: since/turns/time/turns/ready + a hint when blocked")

    def _print_track_explain():
        print("\n🧒 TRACK explained (plain English)")
        print("  • After every answer, TRACK is a tiny dashboard about the AI’s mind.")
        print("  • Stage: where it is on a 0–4 scale. 0 = stuck, 4 = thriving.")
        print("  • Identity: are we speaking as ‘I’ and remembering our promises?")
        print("  • Growth: are we trying new things and not repeating ourselves?")
        print("  • Close: are we finishing the things we promised to do?")
        print("  • Reflection line: are we ready to do a quick self‑check now?")
        print(
            "      Two lights: time and turns. Both ✓ = ready. If not, it tells you what to do."
        )
        print(
            "  • If Identity is LOW: type --@identity open 3 to nudge speaking as ‘I’."
        )
        print(
            "  • If Growth is LOW: ask for a different style, new examples, or alternatives."
        )

    while True:
        try:
            # Get user input
            user_input = get_user_input()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("👋 Goodbye! Your conversation is saved with persistent memory.")
                break
            # Unified --@ router for quick, discoverable UI commands
            elif user_input.startswith("--@"):
                at_cmd = user_input[3:].strip().lower()
                # Registry of supported --@ things
                AT_REG = {
                    "identity": "Show identity. '--@identity list' for options",
                    "commitments": "Show open commitments. '--@commitments list' for options",
                    "traits": "Show Big Five trait snapshot",
                    "emergence": "Show emergence stage (IAS/GAS)",
                    "events": "Show recent events. '--@events list' for options",
                    "find": "Search events, commitments, reflections (e.g., --@find text)",
                    "tasks": "Dev tasks. '--@tasks list' for options",
                    "status": "Show system status (events, last kind, stage)",
                    "memory": "Show cross-session memory excerpt",
                    "track": "Real-time telemetry. '--@track list' for options",
                    "probe": "Probe API: --@probe list for more information",
                    "bandit": "Bandit utilities. '--@bandit debug' for quick test",
                }
                if at_cmd in ("help", "h", "?", "list"):
                    print("\n🧭 --@ commands")
                    for k, v in AT_REG.items():
                        print(f"  • --@{k:<12} {v}")
                    # Highlight useful subcommands in plain language
                    print("\n🔎 Tips (pasteable):")
                    print(
                        "  • --@identity open 3   Turn identity mode ON for 3 replies (speak as ‘I’)"
                    )
                    print("  • --@identity clear    Turn identity mode OFF")
                    print("  • --@probe list        Show available status queries")
                    print("  • --@probe start       Start the status server")
                    print("  • --@probe identity    Show current identity via server")
                    print("  • --@find something    Search everywhere for text")
                    print("  • --@track on|off      Toggle real-time telemetry")
                    print(
                        "  • --@bandit debug      Show bandit Qs/eps; simulate outcomes"
                    )
                    continue
                # Contextual help: --@help identity|probe
                if at_cmd.startswith("help "):
                    topic = at_cmd.split(None, 1)[1].strip()
                    if topic == "identity":
                        print("\n📘 --@identity help")
                        print(
                            "  • --@identity           Show your current identity and whether identity mode is ON"
                        )
                        print(
                            "  • --@identity open N    Turn identity mode ON for N replies (default 3)"
                        )
                        print("  • --@identity clear     Turn identity mode OFF")
                        continue
                    if topic == "probe":
                        print("\n📘 --@probe help")
                        print(
                            "  • --@probe list         Show available status queries with examples"
                        )
                        print(
                            "  • --@probe start        Start the status server in the background"
                        )
                        print(
                            "  • --@probe <path>       Run a query, e.g. --@probe commitments?limit=10"
                        )
                        continue
                # --@probe [start|<path>]
                if at_cmd.startswith("probe"):
                    parts = at_cmd.split(None, 1)

                    # Helper to parse base URL -> (host, port)
                    def _parse_host_port(default=("127.0.0.1", 8000)):
                        try:
                            from urllib.parse import urlparse

                            base = os.getenv("PMM_PROBE_URL", "http://127.0.0.1:8000")
                            u = urlparse(base)
                            host = u.hostname or default[0]
                            port = u.port or default[1]
                            return host, int(port)
                        except Exception:
                            return default

                    if len(parts) > 1 and parts[1].strip().lower() in {
                        "start",
                        "on",
                        "serve",
                        "run",
                    }:
                        # Start probe server in the background (best-effort)
                        host, port = _parse_host_port()
                        try:
                            import subprocess

                            cmd = [
                                sys.executable,
                                "-m",
                                "uvicorn",
                                "pmm.api.probe:app",
                                "--host",
                                host,
                                "--port",
                                str(port),
                            ]
                            subprocess.Popen(
                                cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            print(f"\n🚀 Probe started at http://{host}:{port}")
                            print("   Tip: set PMM_PROBE_URL to change host/port")
                        except Exception as e:
                            print(f"\n❌ Failed to start probe: {e}")
                        continue

                    # Special discovery helper: --@probe list|endpoints
                    if len(parts) > 1 and parts[1].strip().lower() in {
                        "list",
                        "endpoints",
                    }:
                        base = os.getenv("PMM_PROBE_URL", "http://127.0.0.1:8000")
                        url = f"{base}/endpoints"
                        try:
                            import urllib.request
                            import json as _json

                            with urllib.request.urlopen(url, timeout=4) as resp:
                                data = resp.read()
                                obj = _json.loads(data.decode("utf-8", "ignore"))
                            items = (
                                obj.get("items", []) if isinstance(obj, dict) else []
                            )
                            print("\n📚 Probe endpoints:")
                            for it in items:
                                print(
                                    f"  • {it.get('path'):<24} {it.get('desc')}\n    e.g. --@probe {it.get('example')}"
                                )
                        except Exception as e:
                            print(f"\n❌ Probe list error: {e} (url={url})")
                        continue

                    path = parts[1] if len(parts) > 1 else "identity"
                    # Sanitize path
                    path = path.lstrip("/")
                    base = os.getenv("PMM_PROBE_URL", "http://127.0.0.1:8000")
                    url = f"{base}/{path}"
                    try:
                        import urllib.request
                        import json as _json

                        with urllib.request.urlopen(url, timeout=4) as resp:
                            data = resp.read()
                            try:
                                obj = _json.loads(data.decode("utf-8", "ignore"))
                            except Exception:
                                obj = {"raw": data.decode("utf-8", "ignore")}
                        # Print compact summary for common endpoints
                        if path.startswith("identity") and isinstance(obj, dict):
                            print("\n👤 Probe /identity:")
                            print(f"  name: {obj.get('name')}")
                            items = obj.get("identity_commitments") or []
                            if items:
                                print("  identity commitments:")
                                for it in items:
                                    print(
                                        f"    • {it.get('policy')} {it.get('remaining_turns')}/{it.get('ttl_turns')} id={it.get('id')}"
                                    )
                            else:
                                print("  identity commitments: none")
                            # If user invoked bare '--@probe', offer a tip
                            if at_cmd.strip() == "probe":
                                print("\n  Tip: --@probe list  (see more endpoints)")
                        else:
                            print(f"\n🔎 Probe {path}:\n{obj}")
                    except Exception as e:
                        print(f"\n❌ Probe error: {e} (url={url})")
                    continue
                elif at_cmd.startswith("identity"):
                    # Subcommands: open [ttl], clear
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {"list", "help"}:
                        print("\n📘 --@identity options")
                        print(
                            "  • --@identity            Show your current identity and whether identity mode is ON"
                        )
                        print(
                            "  • --@identity open N     Turn identity mode ON for N replies (default 3)"
                        )
                        print("  • --@identity clear      Turn identity mode OFF")
                        continue
                    if sub == "open":
                        ttl = 3
                        # Allow raising the cap via env PMM_IDENTITY_MAX_TTL (default 10)
                        try:
                            max_ttl_cap = int(os.getenv("PMM_IDENTITY_MAX_TTL", "10"))
                        except Exception:
                            max_ttl_cap = 10
                        if len(parts) > 2 and parts[2].isdigit():
                            ttl = max(1, min(max_ttl_cap, int(parts[2])))
                        try:
                            h = open_identity_commitment(
                                pmm_memory.pmm,
                                policy="express_core_principles",
                                ttl_turns=ttl,
                                note="manual open via --@identity",
                            )
                            print(
                                f"\n✅ Opened identity commitment for {ttl} turns (id={h[:8]})"
                            )
                        except Exception as e:
                            print(f"\n❌ Failed to open identity commitment: {e}")
                        continue
                    if sub in {"clear", "close"}:
                        try:
                            n = close_identity_turn_commitments(pmm_memory.pmm)
                            print(f"\n✅ Closed {n} identity commitment(s)")
                        except Exception as e:
                            print(f"\n❌ Failed to close identity commitments: {e}")
                        continue
                    # Default: show identity + active commitments
                    try:
                        name = pmm_memory.pmm.model.core_identity.name
                    except Exception:
                        name = "(unknown)"
                    print(f"\n👤 Identity: {name}")
                    try:
                        items = get_identity_turn_commitments(pmm_memory.pmm)
                    except Exception:
                        items = []
                    if not items:
                        print("No active identity commitments.")
                        print(
                            "  Tip: --@identity open 3   Turn identity mode ON for 3 replies"
                        )
                        print("  Tip: --@identity list     Show available options")
                    else:
                        print("Active Identity Commitments:")
                        for it in items:
                            short = (it.get("event_hash", "") or "")[:8]
                            print(
                                f"  • policy={it.get('policy','')}, remaining={it.get('remaining_turns',0)}/{it.get('ttl_turns',0)}, id={short}"
                            )
                        print("  Tip: --@identity clear    Turn identity mode OFF")
                        print("  Tip: --@identity list     Show available options")
                    continue
                elif at_cmd.startswith("commitments"):
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {"list", "help"}:
                        print("\n📘 --@commitments options")
                        print(
                            "  • --@commitments            Show open commitments (top 5)"
                        )
                        print(
                            "  • --@commitments search X   Search recent commitment events for text X"
                        )
                        print(
                            "  • --@commitments close CID  Close a commitment by ID (e.g., c12)"
                        )
                        print(
                            "  • --@commitments clear      Close all open (legacy) commitments"
                        )
                        print(
                            "  • Note: identity turn-scoped commitments are managed by --@identity"
                        )
                        continue
                    if sub == "search" and len(parts) > 2:
                        query = at_cmd.split(" ", 2)[2].strip()
                        try:
                            store = pmm_memory.pmm.sqlite_store
                            rows = list(
                                store.conn.execute(
                                    "SELECT id,ts,content,hash FROM events WHERE kind='commitment' AND content LIKE ? ORDER BY id DESC LIMIT 10",
                                    (f"%{query}%",),
                                )
                            )
                            if not rows:
                                print(f"\n🔎 No commitment events matching: {query}")
                            else:
                                print(f"\n🔎 Commitments matching '{query}':")
                                for r in rows:
                                    preview = (r[2] or "")[:80].replace("\n", " ")
                                    print(
                                        f"  • {r[0]:>4} {r[1]} #{str(r[3])[:8]} :: {preview}"
                                    )
                        except Exception as e:
                            print(f"\n❌ Search error: {e}")
                        continue
                    if sub == "close" and len(parts) > 2:
                        user_token = parts[2].strip()
                        # Optional evidence text after ID/hash
                        evidence_text = (
                            at_cmd.split(" ", 3)[3].strip()
                            if len(at_cmd.split(" ")) >= 4
                            else ""
                        )

                        # Resolve to canonical commit hash when possible
                        def resolve_commit_ref(token: str) -> str:
                            """Map display id (e.g., c2) or unique hash prefix to full commit hash.

                            Falls back to the raw token if no mapping is found.
                            """
                            try:
                                open_list = pmm_memory.pmm.get_open_commitments() or []
                            except Exception:
                                open_list = []

                            # If looks like display id 'cN'
                            if _is_display_cid(token):
                                for c in open_list:
                                    if str(c.get("cid", "")) == token:
                                        h = str(c.get("hash", ""))
                                        if h:
                                            return h
                                return token

                            # If looks like hex prefix -> try to expand uniquely among open hashes
                            if _is_hex_prefix(token):
                                candidates = [str(c.get("hash", "")) for c in open_list]
                                matches = [h for h in candidates if h.startswith(token)]
                                if len(matches) == 1:
                                    return matches[0]
                                # If multiple or none, leave as provided (may be full hash already)
                                return token

                            return token

                        try:
                            commit_ref = resolve_commit_ref(user_token)

                            if evidence_text:
                                ok = False
                                try:
                                    # Use resolved commit_ref for evidence-based closure
                                    ok = pmm_memory.pmm.commitment_tracker.close_commitment_with_evidence(
                                        commit_ref,
                                        evidence_type="done",
                                        description=evidence_text,
                                        artifact="chat.py",
                                    )
                                except Exception:
                                    ok = False
                                if ok:
                                    # Mirror to SQLite so analyzer sees closure with evidence
                                    try:
                                        smm = pmm_memory.pmm
                                        # Append evidence event
                                        evidence_content = {
                                            "type": "done",
                                            "summary": evidence_text,
                                            "artifact": {"source": "chat.py"},
                                            "confidence": 0.85,
                                        }
                                        smm.sqlite_store.append_event(
                                            kind="evidence",
                                            content=json.dumps(
                                                evidence_content, ensure_ascii=False
                                            ),
                                            meta={
                                                "commit_ref": commit_ref,
                                                "subsystem": "chat",
                                            },
                                        )
                                        # Append commitment.close event
                                        smm.sqlite_store.append_event(
                                            kind="commitment.close",
                                            content=json.dumps(
                                                {"reason": "chat"}, ensure_ascii=False
                                            ),
                                            meta={
                                                "commit_ref": commit_ref,
                                                "subsystem": "chat",
                                            },
                                        )
                                    except Exception as ee:
                                        print(f"\n⚠️ SQLite mirror failed: {ee}")
                                    print(
                                        f"\n✅ Closed commitment {user_token} (ref {commit_ref[:8]}) with evidence"
                                    )
                                    continue
                                else:
                                    # Fall back to legacy close by CID (if user_token is a cid)
                                    pmm_memory.pmm.mark_commitment(
                                        user_token,
                                        "closed",
                                        "Closed via --@commitments",
                                    )
                                    print(f"\n✅ Closed commitment {user_token}")
                                    continue
                            else:
                                # No evidence provided: legacy close by cid
                                pmm_memory.pmm.mark_commitment(
                                    user_token, "closed", "Closed via --@commitments"
                                )
                                print(f"\n✅ Closed commitment {user_token}")
                                continue
                        except Exception as e:
                            print(f"\n❌ Failed to close {user_token}: {e}")
                        continue
                    if sub == "clear":
                        try:
                            open_list = pmm_memory.pmm.get_open_commitments() or []
                            count = 0
                            for c in open_list:
                                cid = c.get("cid")
                                if cid:
                                    pmm_memory.pmm.mark_commitment(
                                        cid,
                                        "closed",
                                        "Bulk close via --@commitments clear",
                                    )
                                    count += 1
                            print(f"\n✅ Closed {count} commitment(s)")
                        except Exception as e:
                            print(f"\n❌ Failed to clear commitments: {e}")
                        continue
                    # Default: show open commitments + identity scoped
                    try:
                        open_list = pmm_memory.pmm.get_open_commitments() or []
                    except Exception:
                        open_list = []
                    if not open_list:
                        print("No open commitments.")
                        print("  Tip: --@commitments list   Show available options")
                    else:
                        print("\n📌 Open Commitments:")
                        for c in open_list[:5]:
                            try:
                                cid = c.get("cid", "?")
                                status = c.get("status", "open")
                                tier = c.get("tier", "permanent")
                                h = (c.get("hash", "") or "")[:8]
                                txt = (c.get("text", "") or "").replace("\n", " ")
                                preview = txt[:160] + ("…" if len(txt) > 160 else "")
                                print(f"  • {cid} [{status}/{tier}] #{h} :: {preview}")
                            except Exception:
                                print(f"  • {c.get('text','')}")
                    try:
                        items = get_identity_turn_commitments(pmm_memory.pmm)
                        if items:
                            print("\n🎭 Identity (turn-scoped):")
                            for it in items:
                                short = (it.get("event_hash", "") or "")[:8]
                                print(
                                    f"  • {it.get('policy','')} {it.get('remaining_turns',0)}/{it.get('ttl_turns',0)} id={short}"
                                )
                    except Exception:
                        pass
                    continue
                elif at_cmd in ("traits",):
                    personality = pmm_memory.get_personality_summary()
                    print("\n🎭 Current Personality State:")
                    for trait, score in personality["personality_traits"].items():
                        print(f"   • {trait.title():<15} : {score:>6.2f}")
                    continue
                elif at_cmd in ("emergence",):
                    try:
                        scores = compute_emergence_scores(
                            window=15, storage_manager=pmm_memory.pmm.sqlite_store
                        )
                        stage = scores.get("stage", "Unknown")
                        print(
                            f"\n🌱 Emergence: stage={stage}, IAS={scores.get('IAS')}, GAS={scores.get('GAS')}"
                        )
                    except Exception as e:
                        print(f"\n🌱 Emergence: error: {e}")
                    continue
                elif at_cmd.startswith("events"):
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {"list", "help"}:
                        print("\n📘 --@events options")
                        print("  • --@events              Show last 5 events")
                        print("  • --@events recent N     Show last N events")
                        print(
                            "  • --@events kind K N     Show last N events of kind K (e.g., response, reflection)"
                        )
                        print(
                            "  • --@events search X     Search content for text X in recent events"
                        )
                        continue
                    try:
                        store = pmm_memory.pmm.sqlite_store
                        if sub == "recent" and len(parts) > 2 and parts[2].isdigit():
                            limit = max(1, min(50, int(parts[2])))
                            rows = store.recent_events(limit=limit)
                        elif sub == "kind" and len(parts) > 3 and parts[3].isdigit():
                            kind = parts[2]
                            limit = max(1, min(50, int(parts[3])))
                            # Select the same columns as recent_events so _row_to_dict works
                            rows = list(
                                store.conn.execute(
                                    "SELECT id,ts,kind,content,meta,prev_hash,hash,summary,keywords,embedding FROM events WHERE kind=? ORDER BY id DESC LIMIT ?",
                                    (kind, limit),
                                )
                            )
                            # Normalize to dicts like recent_events
                            rows = [store._row_to_dict(r) for r in rows]
                        elif sub == "search" and len(parts) > 2:
                            query = at_cmd.split(" ", 2)[2].strip()
                            rows = list(
                                store.conn.execute(
                                    "SELECT id,ts,kind,content FROM events WHERE content LIKE ? ORDER BY id DESC LIMIT 10",
                                    (f"%{query}%",),
                                )
                            )
                            print(f"\n🗂️  Events matching '{query}':")
                            if not rows:
                                print("  (none)")
                                continue
                            for r in rows:
                                preview = (r[3] or "")[:80].replace("\n", " ")
                                print(f"  • {r[0]:>4} {r[2]:<16} {r[1]} :: {preview}")
                            continue
                        else:
                            rows = store.recent_events(limit=5)
                        print("\n🗂️  Recent Events:")
                        for r in rows:
                            preview = (r.get("content") or "")[:80].replace("\n", " ")
                            print(
                                f"  • {r['id']:>4} {r['kind']:<16} {r['ts']} :: {preview}"
                            )
                    except Exception as e:
                        print(f"\n🗂️  Events error: {e}")
                    continue
                elif at_cmd.startswith("tasks"):
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {None, "list", "help"}:
                        print("\n📘 --@tasks options")
                        print(
                            "  • --@tasks list             Show open tasks (from task_* events)"
                        )
                        print("  • --@tasks open KIND TITLE  Open a dev task (ttl 8h)")
                        print(
                            "  • --@tasks close ID         Close a task by id (e.g., dt1)"
                        )
                        continue
                    if sub == "open" and len(parts) >= 4:
                        kind = parts[2]
                        title = at_cmd.split(" ", 3)[3]
                        try:
                            dtm = DevTaskManager(pmm_memory.pmm.sqlite_store)
                            tid = dtm.open_task(
                                kind=kind,
                                title=title,
                                ttl_hours=8,
                                policy={"source": "chat"},
                            )
                            print(f"\n✅ Opened task {tid}: {title}")
                        except Exception as e:
                            print(f"\n❌ Failed to open task: {e}")
                        continue
                    if sub == "close" and len(parts) >= 3:
                        tid = parts[2]
                        try:
                            DevTaskManager(pmm_memory.pmm.sqlite_store).close_task(
                                tid, reason="manual_close"
                            )
                            print(f"\n✅ Closed task {tid}")
                        except Exception as e:
                            print(f"\n❌ Failed to close task {tid}: {e}")
                        continue
                    # Default: list open tasks by folding events
                    try:
                        rows = pmm_memory.pmm.sqlite_store.conn.execute(
                            "SELECT id,ts,kind,content,meta FROM events WHERE kind IN ('task_created','task_progress','task_closed') ORDER BY id"
                        ).fetchall()
                        import json as _json

                        tasks = {}
                        for rid, ts, kind, content, meta in rows:
                            try:
                                m = (
                                    _json.loads(meta)
                                    if isinstance(meta, str)
                                    else (meta or {})
                                )
                            except Exception:
                                m = {}
                            tid = str(m.get("task_id", ""))
                            if not tid:
                                continue
                            rec = tasks.setdefault(
                                tid,
                                {
                                    "task_id": tid,
                                    "status": "open",
                                    "title": None,
                                    "kind": None,
                                    "progress": [],
                                },
                            )
                            if kind == "task_created":
                                try:
                                    c = (
                                        _json.loads(content)
                                        if isinstance(content, str)
                                        else (content or {})
                                    )
                                except Exception:
                                    c = {}
                                rec["title"] = c.get("title")
                                rec["kind"] = c.get("kind")
                            elif kind == "task_progress":
                                rec["progress"].append({"ts": ts, "content": content})
                            elif kind == "task_closed":
                                rec["status"] = "closed"
                        open_tasks = [
                            t for t in tasks.values() if t["status"] == "open"
                        ]
                        if not open_tasks:
                            print("\n🗂️  No open tasks.")
                        else:
                            print("\n🗂️  Open tasks:")
                            for t in open_tasks:
                                print(f"  • {t['task_id']}: [{t['kind']}] {t['title']}")
                    except Exception as e:
                        print(f"\n❌ Tasks error: {e}")
                    continue
                elif at_cmd.startswith("bandit"):
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {None, "list", "help"}:
                        print("\n📘 --@bandit options")
                        print(
                            "  • --@bandit debug  Print Qs/eps, simulate outcomes, show updated values"
                        )
                        print(
                            "  • --@bandit runbook  Run a 12-turn verification with hot segments"
                        )
                        continue
                    if sub == "debug":
                        try:
                            # Wire bandit to chat's SQLite store
                            pmm_bandit.set_store(pmm_memory.pmm.sqlite_store)
                            # Read current status
                            pol_before = pmm_bandit.load_policy()
                            stat_before = pmm_bandit.get_status(
                                pmm_memory.pmm.sqlite_store
                            )

                            def _row(name):
                                r = pol_before.get(name, {"value": 0.0, "pulls": 0})
                                return float(r.get("value", 0.0)), int(
                                    r.get("pulls", 0)
                                )

                            qr_b, nr_b = _row("reflect_now")
                            qc_b, nc_b = _row("continue")
                            print("\n[bandit] BEFORE:")
                            print(
                                f"  q_reflect={qr_b:.3f} pulls={nr_b} | q_continue={qc_b:.3f} pulls={nc_b} | eps={stat_before.get('eps'):.3f}"
                            )

                            # Simulate outcomes: one strong reflect win (+0.7/+0.3), one inert2 (-0.15)
                            import os as _os

                            def _f(name, default):
                                try:
                                    return float(_os.getenv(name, str(default)))
                                except Exception:
                                    return default

                            pos_acc = _f("PMM_BANDIT_POS_ACCEPTED", 0.7)
                            pos_close = _f("PMM_BANDIT_POS_CLOSE", 0.3)
                            neg_in2 = _f("PMM_BANDIT_NEG_INERT2", 0.15)
                            # Good reflect
                            pmm_bandit.record_outcome(
                                {"debug": True},
                                "reflect_now",
                                pos_acc + pos_close,
                                horizon=int(
                                    _os.getenv("PMM_BANDIT_HORIZON_TURNS", "5")
                                ),
                                notes="debug: accepted=True close=True",
                            )
                            # Inert2
                            pmm_bandit.record_outcome(
                                {"debug": True},
                                "reflect_now",
                                -neg_in2,
                                horizon=int(
                                    _os.getenv("PMM_BANDIT_HORIZON_TURNS", "5")
                                ),
                                notes="debug: inert2=True",
                            )

                            # Read updated status
                            pol_after = pmm_bandit.load_policy()
                            stat_after = pmm_bandit.get_status(
                                pmm_memory.pmm.sqlite_store
                            )

                            def _row2(name):
                                r = pol_after.get(name, {"value": 0.0, "pulls": 0})
                                return float(r.get("value", 0.0)), int(
                                    r.get("pulls", 0)
                                )

                            qr_a, nr_a = _row2("reflect_now")
                            qc_a, nc_a = _row2("continue")
                            print("[bandit] AFTER:")
                            print(
                                f"  q_reflect={qr_a:.3f} pulls={nr_a} | q_continue={qc_a:.3f} pulls={nc_a} | eps={stat_after.get('eps'):.3f}"
                            )
                            print("  (Inserted: +accepted+close, -inert2)")
                        except Exception as e:
                            print(f"\n❌ Bandit debug error: {e}")
                        continue
                    if sub == "runbook":
                        # Step 10 – verification loop (simulated)
                        try:
                            import os as _os
                            from pmm.logging_config import pmm_tlog as _tlog
                            from pmm.policy.bandit import build_context as _bctx

                            # 1) Set envs for the run
                            env_overrides = {
                                "PMM_BANDIT_ENABLED": "1",
                                "PMM_TELEMETRY": "1",
                                "PMM_IAS_IDENTITY_EXTENDED": "1",
                                "PMM_IAS_IDENTITY_EVIDENCE_MULT": "0.02",
                                "PMM_IAS_IDENTITY_MAX_BOOST": "0.12",
                                "PMM_IAS_ID_COMMIT_BONUS": "0.03",
                                "PMM_REFLECTION_COOLDOWN_SECONDS": "30",
                                "PMM_REFLECTION_HOT_FACTOR": "0.35",
                            }
                            for k, v in env_overrides.items():
                                _os.environ[k] = v

                            # Wire bandit to our store
                            pmm_bandit.set_store(pmm_memory.pmm.sqlite_store)

                            # Helper: add a commitment.close event
                            def _append_close(note: str = "runbook"):
                                try:
                                    pmm_memory.pmm.sqlite_store.append_event(
                                        kind="commitment.close",
                                        content=note,
                                        meta={"source": "runbook"},
                                    )
                                    # Let bandit index recent close
                                    pmm_memory._bandit_scan_commit_closes()
                                except Exception:
                                    pass

                            # 2) Run 12 simulated turns
                            turns = 12
                            # Craft three hot segments at turns 1, 5, 9
                            hot_turns = {1, 5, 9}
                            # Ensure horizon comes due by the end
                            try:
                                int(_os.getenv("PMM_BANDIT_HORIZON_TURNS", "5"))
                            except Exception:
                                pass

                            for t in range(1, turns + 1):
                                # Increment turn counter like save_context would
                                pmm_memory._bandit_turn_id = (
                                    int(getattr(pmm_memory, "_bandit_turn_id", 0) or 0)
                                    + 1
                                )

                                # Build a synthetic context snapshot
                                is_hot = t in hot_turns
                                ctx = _bctx(
                                    gas=0.92 if is_hot else 0.50,
                                    ias=0.40,
                                    close=0.80 if is_hot else 0.30,
                                    hot=is_hot,
                                    identity_signal_count=3,
                                    time_since_last_reflection_sec=25,
                                    dedup_threshold=0.94,
                                    inert_streak=0,
                                )

                                # Select and record action into ring buffer
                                try:
                                    action, eps, qrf, qct = (
                                        pmm_bandit.select_action_info(ctx)
                                    )
                                except Exception:
                                    action = pmm_bandit.select_action(ctx)
                                    eps = float(
                                        _os.getenv("PMM_BANDIT_EPSILON", "0.10")
                                    )
                                    qrf = qct = 0.0
                                rec = {
                                    "turn": int(pmm_memory._bandit_turn_id),
                                    "action": action,
                                    "ctx": dict(ctx),
                                    "finalized": False,
                                    "events": [],
                                }
                                pmm_memory._bandit_events.append(rec)
                                _tlog(
                                    f"[PMM_TELEMETRY] bandit_select: action={action}, eps={eps:.3f}, q_reflect={qrf:.3f}, q_continue={qct:.3f}, ctx={ctx}"
                                )

                                # Orchestrate outcomes:
                                # - Turn 1 hot: reflect_now accepted, and a commitment close
                                if t == 1:
                                    # Force accepted; if action was continue, still tag accepted to show positive update
                                    pmm_memory._bandit_note_event("reflection_accepted")
                                    _append_close("runbook: close after accepted")
                                # - Turn 2 and 3: two inert events to trigger inert2 penalty for a reflect_now action
                                if t in (2, 3):
                                    pmm_memory._bandit_note_event("reflection_inert")
                                # - Turn 5 hot: ensure a close without acceptance to reward 'continue' restraint if it happens
                                if t == 5:
                                    _append_close("runbook: close without acceptance")

                                # Periodically evaluate rewards so earlier actions cross horizon
                                pmm_memory._bandit_evaluate_rewards()

                            # Final evaluation pass
                            pmm_memory._bandit_evaluate_rewards()

                            # Print final status and recent rewards
                            stat = pmm_bandit.get_status(pmm_memory.pmm.sqlite_store)
                            print("\n[bandit] FINAL STATUS:")
                            print(
                                f"  q_reflect={stat.get('q_reflect'):.3f} | q_continue={stat.get('q_continue'):.3f} | eps={stat.get('eps'):.3f}"
                            )
                            # Show last few reward rows
                            try:
                                rows = list(
                                    pmm_memory.pmm.sqlite_store.conn.execute(
                                        "SELECT id, ts, action, reward, horizon, notes FROM bandit_rewards ORDER BY id DESC LIMIT 5"
                                    )
                                )
                                print("[bandit] recent outcomes:")
                                for r in rows[::-1]:
                                    print(
                                        f"  • #{r[0]} {r[1]} {r[2]} reward={float(r[3]):.3f} horizon={r[4]} notes={r[5]}"
                                    )
                            except Exception:
                                pass

                            print(
                                "\nRunbook complete. Check console for bandit_reward lines and /autonomy/status for bandit_q_*"
                            )
                        except Exception as e:
                            print(f"\n❌ Bandit runbook error: {e}")
                        continue
                elif at_cmd.startswith("find"):
                    # Unified search across events (content/summary), commitment events, and open commitments
                    if len(at_cmd.split(None, 1)) < 2:
                        print("\n📘 --@find usage: --@find <text>")
                        continue
                    query = at_cmd.split(" ", 2)[1].strip().strip("\"'")
                    try:
                        store = pmm_memory.pmm.sqlite_store
                        # Events search (content or summary)
                        ev_rows = list(
                            store.conn.execute(
                                "SELECT id,ts,kind,COALESCE(summary, content) AS c FROM events WHERE (content LIKE ? OR summary LIKE ?) ORDER BY id DESC LIMIT 15",
                                (f"%{query}%", f"%{query}%"),
                            )
                        )
                        # Open commitments (JSON model)
                        try:
                            open_list = pmm_memory.pmm.get_open_commitments() or []
                        except Exception:
                            open_list = []
                        oc_hits = [
                            c
                            for c in open_list
                            if query.lower() in (c.get("text", "").lower())
                        ]

                        print(f"\n🔎 Results for '{query}':")
                        # Print commitment hits first
                        if oc_hits:
                            print("  • Open Commitments:")
                            for c in oc_hits[:5]:
                                txt = (c.get("text", "") or "")[:100].replace("\n", " ")
                                print(f"    - {txt}")
                        # Then events
                        if ev_rows:
                            print("  • Events:")
                            for r in ev_rows:
                                preview = (r[3] or "")[:100].replace("\n", " ")
                                print(f"    - {r[0]:>4} {r[2]:<12} {r[1]} :: {preview}")
                        if not oc_hits and not ev_rows:
                            print("  (no matches)")
                    except Exception as e:
                        print(f"\n❌ Find error: {e}")
                    continue
                elif at_cmd.startswith("track"):
                    parts = at_cmd.split()
                    sub = parts[1] if len(parts) > 1 else None
                    if sub in {None, "list", "help"}:
                        print("\n📘 --@track options")
                        print(
                            "  • --@track on        Turn real-time telemetry ON (prints after each reply)"
                        )
                        print("  • --@track off       Turn real-time telemetry OFF")
                        print(
                            "  • --@track status    Show whether telemetry is ON or OFF"
                        )
                        print("  • --@track legend    Quick guide to the fields")
                        print(
                            "  • --@track explain   Plain-English explanation with what to do"
                        )
                        continue
                    if sub == "on":
                        track_enabled = True
                        print(
                            "\n✅ Tracking ON — you will see a [TRACK] line after each reply"
                        )
                        _print_track_legend()
                        continue
                    if sub == "off":
                        track_enabled = False
                        print("\n✅ Tracking OFF")
                        continue
                    if sub == "status":
                        print(f"\n🧭 Tracking: {'ON' if track_enabled else 'OFF'}")
                        continue
                    if sub == "legend":
                        _print_track_legend()
                        continue
                    if sub == "explain":
                        _print_track_explain()
                        continue
                elif at_cmd in ("status",):
                    # Minimal status snapshot
                    try:
                        rows = pmm_memory.pmm.sqlite_store.all_events()
                        total = len(rows)
                        last = rows[-1]["kind"] if rows else None
                    except Exception:
                        total, last = 0, None
                    try:
                        scores = compute_emergence_scores(
                            window=15, storage_manager=pmm_memory.pmm.sqlite_store
                        )
                        stage = scores.get("stage", "Unknown")
                    except Exception:
                        stage = "Unknown"
                    print(f"\n🧩 Status: events={total}, last={last}, stage={stage}")
                    continue
                elif at_cmd in ("memory",):
                    pmm_context = pmm_memory.load_memory_variables({}).get(
                        "history", ""
                    )
                    print("\n🧠 Cross-Session Memory Context:")
                    print(
                        pmm_context[:500]
                        if pmm_context
                        else "No cross-session memory yet"
                    )
                    continue
            elif user_input.lower() == "personality":
                personality = pmm_memory.get_personality_summary()
                print("\n🎭 Current Personality State:")
                for trait, score in personality["personality_traits"].items():
                    print(f"   • {trait.title():<15} : {score:>6.2f}")
                print(
                    f"\n📊 Stats: {personality['total_events']} events, {personality['open_commitments']} commitments"
                )
                continue
            elif user_input.lower() == "memory":
                pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
                print("\n🧠 Cross-Session Memory Context:")
                print(
                    pmm_context[:500] if pmm_context else "No cross-session memory yet"
                )
                continue
            elif user_input.lower() in ("identity", "commitments"):
                # Show current identity and active identity turn-scoped commitments
                try:
                    name = pmm_memory.pmm.model.core_identity.name
                except Exception:
                    name = "(unknown)"
                print(f"\n👤 Identity: {name}")
                try:
                    items = get_identity_turn_commitments(pmm_memory.pmm)
                except Exception:
                    items = []
                if not items:
                    print("No active identity commitments.")
                else:
                    print("Active Identity Commitments:")
                    for it in items:
                        short = (it.get("event_hash", "") or "")[:8]
                        print(
                            f"  • policy={it.get('policy','')}, remaining={it.get('remaining_turns',0)}/{it.get('ttl_turns',0)}, id={short}"
                        )
                continue
            elif user_input.lower() == "models":
                print("\n" + "=" * 50)
                # For piped sessions, allow inline model selection
                if stdin_is_pipe:
                    print("🎯 Select a model by typing the number:")
                    available_models = list_available_models()
                    for i, model in enumerate(available_models, 1):
                        marker = "⭐" if model == model_name else f"{i:2d}."
                        config = get_model_config(model)
                        cost_str = (
                            f"${config.cost_per_1k_tokens:.4f}/1K"
                            if config.cost_per_1k_tokens > 0
                            else "Free"
                        )
                        print(f"{marker} {model} ({config.provider}) - {cost_str}")
                    print(
                        f"\n💡 Type a number (1-{len(available_models)}) or press ENTER for current model"
                    )

                    # Get next input for model selection
                    try:
                        model_choice = get_user_input().strip()
                        if not model_choice:
                            new_model = model_name  # Keep current
                            print(f"✅ Keeping current model: {model_name}")
                        elif model_choice.isdigit():
                            idx = int(model_choice)
                            if 1 <= idx <= len(available_models):
                                new_model = available_models[idx - 1]
                                print(f"✅ Selected model {idx}: {new_model}")
                            else:
                                print(
                                    f"❌ Invalid number. Please choose 1-{len(available_models)}"
                                )
                                new_model = None
                        else:
                            print(f"❌ Please enter a number 1-{len(available_models)}")
                            new_model = None
                    except Exception as e:
                        print(f"❌ Error reading model choice: {e}")
                        new_model = None
                else:
                    new_model = show_model_selection(force_tty=not args.noninteractive)

                if new_model and new_model != model_name:
                    print(f"🔄 Switching to {new_model}... Please wait...")

                    # Update model configuration
                    model_name = new_model
                    model_config = get_model_config(model_name)

                    # Recreate dual LLMs with new model based on provider
                    if model_config.provider == "ollama":
                        llm_normal = OllamaLLM(model=model_config.name, temperature=0.7)
                        llm_deep = OllamaLLM(model=model_config.name, temperature=0.2)
                    else:  # openai
                        llm_normal = ChatOpenAI(
                            model=model_config.name, temperature=0.7
                        )
                        llm_deep = ChatOpenAI(model=model_config.name, temperature=0.2)

                    # Update active config in LLM factory for reflection system
                    from pmm.llm_factory import get_llm_factory
                    from pmm.embodiment import extract_model_family

                    llm_factory = get_llm_factory()

                    # Extract family from model name
                    family = extract_model_family(model_name)

                    # Get previous config for bridge handover
                    prev_config = llm_factory.get_active_config()

                    # Update active config with family info
                    enhanced_config = {
                        "name": model_name,
                        "provider": model_config.provider,
                        "family": family,
                        "version": "unknown",
                        "epoch": llm_factory.get_current_epoch(),
                    }
                    llm_factory.set_active_config(enhanced_config)

                    # Handle model switch through bridge manager
                    if prev_config:
                        prev_model_config = ModelConfig(
                            provider=prev_config.get("provider", "unknown"),
                            name=prev_config.get("name", "unknown"),
                            family=prev_config.get("family", "unknown"),
                            version=prev_config.get("version", "unknown"),
                            epoch=prev_config.get("epoch", 0),
                        )
                        curr_model_config = ModelConfig(
                            provider=enhanced_config["provider"],
                            name=enhanced_config["name"],
                            family=enhanced_config["family"],
                            version=enhanced_config["version"],
                            epoch=enhanced_config["epoch"],
                        )
                        bridge_manager.on_switch(prev_model_config, curr_model_config)

                    # Refresh conversation history with updated system prompt
                    conversation_history[0] = {
                        "role": "system",
                        "content": get_pmm_system_prompt(identity_nudge_flag),
                    }

                    print(
                        f"✅ Successfully switched to {model_name} ({model_config.provider})"
                    )
                    print(f"🔧 Max tokens: {model_config.max_tokens:,}")
                    if model_config.cost_per_1k_tokens > 0:
                        print(
                            f"💰 Cost: ${model_config.cost_per_1k_tokens:.4f}/1K tokens"
                        )
                    else:
                        print("💰 Cost: Free (local model)")
                    print("🧠 PMM context refreshed for new model")
                elif new_model == model_name:
                    print(f"✅ Already using {model_name}")
                else:
                    print("❌ Model selection cancelled")
                print("=" * 50 + "\n")
                continue
            elif user_input.lower() == "status":
                # Report feature toggles and DB stats
                try:
                    rows = pmm_memory.pmm.sqlite_store.all_events()
                    total_events = len(rows)
                    events_with_summaries = sum(1 for r in rows if len(r) >= 8 and r[7])
                except Exception:
                    total_events = len(
                        pmm_memory.pmm.model.self_knowledge.autobiographical_events
                    )
                    events_with_summaries = sum(
                        1
                        for e in pmm_memory.pmm.model.self_knowledge.autobiographical_events
                        if getattr(e, "summary", None)
                    )
                db_path = "pmm.db"
                try:
                    size_bytes = (
                        os.path.getsize(db_path) if os.path.exists(db_path) else 0
                    )
                except Exception:
                    size_bytes = 0
                size_kb = size_bytes / 1024.0
                print("\n📊 PMM Status:")
                print(
                    f"   • Thought Summarization: {'ON' if SUMMARY_ENABLED else 'OFF'}"
                )
                print(
                    f"   • Semantic Embeddings: {'ON' if EMBEDDINGS_ENABLED else 'OFF'}"
                )
                print(f"   • Database file: {db_path} ({size_kb:.1f} KB)")
                print(f"   • Total events: {total_events}")
                print(f"   • Events with summaries: {events_with_summaries}")
                continue

            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # CRITICAL FIX: Refresh system prompt with latest memory context before each response
            # --- One-turn identity anchor for S0/S1 with low momentum, gated on real context ---
            try:
                # Use the same storage and a wider window so we read the same snapshot as EMERGENCE logs
                storage = (
                    getattr(pmm_memory, "pmm", None).sqlite_store
                    if hasattr(pmm_memory.pmm, "sqlite_store")
                    else None
                )
                scores = compute_emergence_scores(window=15, storage_manager=storage)

                stage = scores.get("stage", "")
                ias = float(scores.get("ias", 0.0))
                gas = float(scores.get("gas", 0.0))
                ev_n = int(scores.get("events_analyzed", 0))
            except Exception:
                stage, ias, gas, ev_n = "", 0.0, 0.0, 0

            # Optional: 1-turn cooldown to avoid immediate re-fire on the very next turn
            # Track turns locally to avoid global state dependencies
            current_turn += 1
            cooldown_ok = (last_anchor_turn is None) or (
                current_turn - last_anchor_turn >= 1
            )

            add_identity_anchor = (
                ev_n > 0  # don’t fire on cold/empty context
                and cooldown_ok  # avoid back-to-back anchors
                and stage in ("S0: Substrate", "S1: Resistance")
                and (ias < 0.25 or gas < 0.50)
            )

            identity_anchor_text = (
                "\n\n[identity-anchor] In your next single reply, use first-person once "
                "and mention one PMM concept (commitments, memory, or drift) briefly. "
                "Do not change tone; keep it natural."
            )

            # Auto-detect deep mode and adjust memory allocation
            deep_now = should_deep_mode(user_input, scores)
            mem_chars = 2600 if deep_now else 1800

            # Pass the latest user text for semantic retrieval
            last_user_text = user_input
            system_prompt = get_pmm_system_prompt(
                identity_nudge_flag, last_user_text, mem_chars
            )
            if add_identity_anchor:
                system_prompt = (system_prompt or "") + identity_anchor_text
                if debug_on:
                    pmm_tlog(
                        f"[ANCHOR] stage={stage} IAS={ias:.3f} GAS={gas:.3f} events={ev_n} -> applied=1"
                    )
                # remember we fired this turn (in-memory only; no persistence)
                last_anchor_turn = current_turn

            conversation_history[0] = {
                "role": "system",
                "content": system_prompt,
            }

            # Show API call info
            current_config = get_model_config(model_name)
            provider_name = current_config.provider.upper()
            if debug_on:
                print(
                    f"🤖 PMM: {_tag('API','cyan')} Calling {provider_name} with prompt: {user_input[:50]}..."
                )

            # Show deep mode notification when active
            if deep_now:
                print("[PMM] deep mode: temp=0.2, memory+ ≈+40%")

            # Token telemetry: estimate prompt token usage
            try:
                sys_tokens = _approx_tokens(conversation_history[0]["content"])
                msg_count = len(conversation_history)
                token_meta = {
                    "system_tokens": sys_tokens,
                    "messages": msg_count,
                    "deep_mode": deep_now,
                    "stage": scores.get("stage"),
                    "ias": scores.get("IAS"),
                    "gas": scores.get("GAS"),
                }
                try:
                    pmm_memory.pmm.add_event(
                        summary=f"Token telemetry: sys={sys_tokens} msgs={msg_count}",
                        etype="telemetry",
                        evidence=token_meta,
                    )
                except Exception:
                    pass
            except Exception:
                pass

            response = invoke_model(conversation_history, use_deep_mode=deep_now)

            # Handle response format differences
            if current_config.provider == "ollama":
                response_text = response  # Ollama returns string directly
            else:
                response_text = response.content  # OpenAI returns message object

            if debug_on:
                print(
                    f"{_tag('API','cyan')} Response received: {len(response_text)} chars"
                )
            # Removed: keyword enforcement/rewrites for 'Next:' to allow autonomous, semantic commitments

            print(response_text)

            # Add AI response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})

            # Save to PMM memory system synchronously to ensure next turn sees LTM
            try:
                pmm_memory.save_context(
                    {"input": user_input}, {"response": response_text}
                )
            except Exception as _e:
                print(f"[warn] save_context failed: {_e}")

            # Auto-tick turn-scoped identity commitments and close when TTL hits 0
            try:
                tick_turn_scoped_identity_commitments(pmm_memory.pmm, response_text)
                # Show concise status for any active identity turn commitments
                try:
                    _items = get_identity_turn_commitments(pmm_memory.pmm)
                    if _items:
                        parts = []
                        for it in _items[:2]:
                            short = (it.get("event_hash", "") or "")[:8]
                            parts.append(
                                f"{it.get('policy','')} {it.get('remaining_turns',0)}/{it.get('ttl_turns',0)}#{short}"
                            )
                        more = "" if len(_items) <= 2 else f" (+{len(_items)-2} more)"
                        print("[PMM] identity commitments: " + ", ".join(parts) + more)
                except Exception:
                    pass
            except Exception:
                pass

            # Optional feedback collection and event logging
            try:
                if FEEDBACK_ENABLE:
                    # Attempt to capture feedback from user when interactive autoprompt is on
                    rating = None
                    note = None
                    if FEEDBACK_AUTOPROMPT and sys.stdin.isatty():
                        try:
                            raw = input(
                                "[PMM] Rate clarity 1-5 (ENTER to skip): "
                            ).strip()
                            if raw:
                                rating = int(raw)
                        except Exception:
                            rating = None
                        try:
                            if rating is not None:
                                note = input("[PMM] Optional note: ").strip() or None
                        except Exception:
                            note = None

                    # If we have feedback, or if autoprompt is off but environment provided defaults, log it
                    if rating is not None or note is not None:
                        # Look up the most recent response event id to reference
                        try:
                            store = pmm_memory.pmm.sqlite_store
                            row = store.conn.execute(
                                "SELECT id, ts, content FROM events WHERE kind='response' ORDER BY id DESC LIMIT 1"
                            ).fetchone()
                            resp_id = int(row[0]) if row else None
                        except Exception:
                            resp_id = None

                        meta = {"rating": rating, "response_ref": resp_id}
                        if note:
                            meta["note"] = note
                        try:
                            pmm_memory.pmm.sqlite_store.append_event(
                                kind="feedback", content=(note or ""), meta=meta
                            )
                        except Exception:
                            pass
            except Exception:
                pass

            # --- Autonomous S0 Recovery & Telemetry ---
            try:
                # Reuse the precomputed emergence snapshot from earlier this turn
                # to avoid divergence between analyzers/telemetry and prevent
                # spurious S0 streak escalations.
                # The snapshot was computed with window=15 and the active storage.
                stage = scores.get("stage", "Unknown")
                is_s0 = stage.startswith("S0")

                # Track consecutive S0
                if is_s0:
                    s0_consecutive += 1
                else:
                    s0_consecutive = 0

                # Real-time telemetry snapshot (pretty)
                telemetry_env = os.getenv("PMM_TELEMETRY", "").lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                telemetry_on = track_enabled or telemetry_env
                if telemetry_on:
                    try:
                        cd_status = pmm_memory.reflection_cooldown.get_status()
                    except Exception:
                        cd_status = {}
                    lines = _track_fmt(
                        scores, cd_status, s0_consecutive, s0_streak_threshold, stage
                    )
                    if track_enabled:
                        for ln in lines:
                            print(ln)
                    else:
                        # Legacy label
                        for ln in lines:
                            print(ln.replace("[TRACK]", "[PMM_TELEMETRY]", 1))

                    # Trigger reflection when ready
                    try:
                        if cd_status.get("time_gate_passed") and cd_status.get(
                            "turns_gate_passed"
                        ):
                            from pmm.atomic_reflection import AtomicReflectionManager

                            reflection_manager = AtomicReflectionManager(pmm_memory.pmm)
                            reflection_manager.run_once(user_input)
                    except Exception as e:
                        if track_enabled:
                            print(f"[TRACK] reflection_trigger_error: {e}")
                        pass

                # Decide on recovery actions when stuck in S0
                if s0_consecutive >= s0_streak_threshold:
                    identity_nudge_flag = True  # Nudge next turn

                    # Ensure an identity anchoring commitment exists
                    try:
                        existing = pmm_memory.pmm.get_open_commitments()
                        anchor_exists = any(
                            isinstance(c, dict)
                            and "identity" in c.get("text", "").lower()
                            and "anchor" in c.get("text", "").lower()
                            for c in (existing or [])
                        )
                        if not anchor_exists:
                            pmm_memory.pmm.add_commitment(
                                text="Short-term identity anchoring: I will speak in first-person as my PMM identity and reference my open commitments over the next 3 turns.",
                                source_insight_id="system:emergence_s0_recovery",
                                due=None,
                            )
                            # Also open a turn-scoped identity commitment (deduped by policy/scope/ttl)
                            try:
                                open_identity_commitment(
                                    pmm_memory.pmm,
                                    policy="express_core_principles",
                                    ttl_turns=3,
                                    note="S0 recovery identity anchoring",
                                )
                            except Exception:
                                pass
                            if telemetry_on:
                                msg = "commitment: opened identity anchoring commitment"
                                print(
                                    f"{'[TRACK]' if track_enabled else '[PMM_TELEMETRY]'} {msg}"
                                )
                    except Exception as _e:
                        if telemetry_on:
                            lbl = "[TRACK]" if track_enabled else "[PMM_TELEMETRY]"
                            print(f"{lbl} commitment_error: {str(_e)}")

                    # Force a reflection to break substrate inertia
                    try:
                        allowed, reason = pmm_memory.reflection_cooldown.should_reflect(
                            current_context=response_text,
                            force_reasons=["emergence_s0_stuck"],
                        )
                        if allowed:
                            # Use active model config for adapter selection
                            try:
                                active_cfg = llm_factory.get_active_config()
                            except Exception:
                                active_cfg = None
                            _ins = reflect_once(
                                pmm_memory.pmm, active_model_config=active_cfg
                            )
                            if telemetry_on:
                                lbl = "[TRACK]" if track_enabled else "[PMM_TELEMETRY]"
                                print(
                                    f"{lbl} reflection: forced reason=emergence_s0_stuck, accepted={getattr(_ins, 'meta', {}).get('accepted') if _ins else None}"
                                )
                    except Exception as _e:
                        if telemetry_on:
                            lbl = "[TRACK]" if track_enabled else "[PMM_TELEMETRY]"
                            print(f"{lbl} reflection_error: {str(_e)}")

                else:
                    # When not stuck, disable nudge to avoid oversteer
                    identity_nudge_flag = False

            except Exception as _e:
                # Never let recovery logic break the chat loop
                if telemetry_on:
                    lbl = "[TRACK]" if track_enabled else "[PMM_TELEMETRY]"
                    print(f"{lbl} s0_recovery_error: {type(_e).__name__}: {_e}")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Your conversation is saved!")
            break
        except EOFError:
            # Gracefully exit when running noninteractive or stdin is exhausted
            print("\n👋 End of input. Exiting chat.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing chat...")

    # Clean up
    if tty_file:
        tty_file.close()


if __name__ == "__main__":
    main()
