#!/usr/bin/env python3
"""
PMM Chat - Interactive interface for your Persistent Mind Model
Main entry point for chatting with your autonomous AI personality.
"""

import os
import sys
import argparse

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
from pmm.emergence import compute_emergence_scores
from pmm.reflection import reflect_once
from pmm.logging_config import pmm_tlog


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PMM Chat - Interactive AI personality interface"
    )
    parser.add_argument("--model", help="Model name or number from the menu")
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
    EMBEDDINGS_ENABLED = os.getenv(
        "PMM_ENABLE_EMBEDDINGS", "true"
    ).strip().lower() in ("1", "true", "yes", "on")

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

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print(f"🔄 {model_name} selected... Loading model... Please wait...")
    print()

    # Initialize PMM with selected model
    model_config = get_model_config(model_name)

    pmm_memory = PersistentMindMemory(
        agent_path="persistent_self_model.json",
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

    # Initialize LangChain components based on provider
    if model_config.provider == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.7)
    else:  # openai
        llm = ChatOpenAI(model=model_name, temperature=0.7)

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

    # Create enhanced system prompt with PMM context
    def get_pmm_system_prompt(identity_nudge: bool = False):
        # Always pull fresh memory right before each call
        raw_context = pmm_memory.load_memory_variables({}).get("history", "")

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
            f"Top Patterns (awareness only): {top_patterns_str}",
            "Open Commitments:",
            f"{open_commitments_str}",
        ]
        if identity_nudge:
            mind_policy_lines.append(
                "Identity Nudge: Speak in first-person as this PMM agent. Prefer 'I' statements, reference my open commitments explicitly, avoid generic assistant voice."
            )
        mind_policy_lines.append("— end policy —")
        mind_policy = "\n".join(mind_policy_lines)

        # Personality + cross‑session memory for rich context (truncated for prompt safety)
        persona = (
            f"PERSONALITY (Big Five): "
            f"O {traits['openness']:.2f} | C {traits['conscientiousness']:.2f} | "
            f"E {traits['extraversion']:.2f} | A {traits['agreeableness']:.2f} | "
            f"N {traits['neuroticism']:.2f}"
        )

        return (
            f"You are {agent_name}.\n"
            "You have access to persistent memory (below). Use it as context; otherwise answer directly and plainly.\n"
            f"{mind_policy}{loop_hint}\n\n"
            f"{persona}\n\n"
            "CROSS‑SESSION MEMORY (condensed):\n"
            f"{pmm_context[:1800]}"
        )

    print(f"\n🤖 PMM is ready! Using {model_name} ({model_config.provider})")
    print(
        "💡 Commands: 'quit' to exit, 'personality' for traits, 'memory' for context, 'models' to switch, 'status' for PMM status"
    )
    print("Start chatting...")

    # Initialize conversation history with PMM system prompt
    conversation_history = [
        {"role": "system", "content": get_pmm_system_prompt(identity_nudge_flag)}
    ]

    def invoke_model(messages):
        """Invoke model with proper format based on provider type."""
        current_config = get_model_config(model_name)  # Get current model config
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
            return llm.invoke(formatted_prompt)
        else:
            # OpenAI chat models expect message list
            return llm.invoke(messages)

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

                    # Recreate LLM with new model based on provider
                    if model_config.provider == "ollama":
                        llm = OllamaLLM(model=model_config.name, temperature=0.7)
                    else:  # openai
                        llm = ChatOpenAI(model=model_config.name, temperature=0.7)

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

            system_prompt = get_pmm_system_prompt(identity_nudge_flag)
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
                    f"🤖 PMM: [API] Calling {provider_name} with prompt: {user_input[:50]}..."
                )
            response = invoke_model(conversation_history)

            # Handle response format differences
            if current_config.provider == "ollama":
                response_text = response  # Ollama returns string directly
            else:
                response_text = response.content  # OpenAI returns message object

            if debug_on:
                print(f"[API] Response received: {len(response_text)} chars")
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

                # Telemetry snapshot (uses the same precomputed scores)
                telemetry_on = telemetry_flag or (
                    os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on")
                )
                if telemetry_on:
                    try:
                        cd_status = pmm_memory.reflection_cooldown.get_status()
                    except Exception:
                        cd_status = {}
                    print(
                        f"[PMM_TELEMETRY] emergence: stage={stage}, IAS={scores.get('IAS')}, GAS={scores.get('GAS')}, pmmspec={scores.get('pmmspec_avg')}, selfref={scores.get('selfref_avg')}, novelty={scores.get('novelty')}, commit_close_rate={scores.get('commit_close_rate')}, s0_streak={s0_consecutive}/{s0_streak_threshold}, cooldown={cd_status}"
                    )

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
                            if telemetry_on:
                                print(
                                    "[PMM_TELEMETRY] commitment: opened identity anchoring commitment"
                                )
                    except Exception as _e:
                        if telemetry_on:
                            print(f"[PMM_TELEMETRY] commitment_error: {str(_e)}")

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
                                print(
                                    f"[PMM_TELEMETRY] reflection: forced reason=emergence_s0_stuck, accepted={getattr(_ins, 'meta', {}).get('accepted') if _ins else None}"
                                )
                    except Exception as _e:
                        if telemetry_on:
                            print(f"[PMM_TELEMETRY] reflection_error: {str(_e)}")

                else:
                    # When not stuck, disable nudge to avoid oversteer
                    identity_nudge_flag = False

            except Exception as _e:
                # Never let recovery logic break the chat loop
                if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                    print(
                        f"[PMM_TELEMETRY] s0_recovery_error: {type(_e).__name__}: {_e}"
                    )

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Your conversation is saved!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing chat...")

    # Clean up
    if tty_file:
        tty_file.close()


if __name__ == "__main__":
    main()
