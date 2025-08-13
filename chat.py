#!/usr/bin/env python3
"""
PMM Chat - Interactive interface for your Persistent Mind Model
Main entry point for chatting with your autonomous AI personality.
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for PMM imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pmm.langchain_memory import PersistentMindMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pmm.config import get_default_model, get_model_config, list_available_models, AVAILABLE_MODELS, get_ollama_models

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PMM Chat - Interactive AI personality interface")
    parser.add_argument("--model", help="Model name or number from the menu")
    parser.add_argument("--noninteractive", action="store_true", 
                       help="Force non-interactive mode; do not try to read from /dev/tty")
    return parser.parse_args()

def show_model_selection(force_tty=True):
    """Show model selection interface and return selected model."""
    print("=== PMM Model Selection ===")
    print()
    
    # Show current default model at top
    default_model = get_default_model()
    default_config = get_model_config(default_model)
    default_cost_str = f"${default_config.cost_per_1k_tokens:.4f}/1K" if default_config.cost_per_1k_tokens > 0 else "Free (local)"
    
    print(f"⭐ CURRENT DEFAULT: {default_model} ({default_config.provider})")
    print(f"   {default_config.description}")
    print(f"   Max tokens: {default_config.max_tokens:,} | Cost: {default_cost_str}")
    print()
    
    # Show all available models
    print("📋 Available Models:")
    available_models = list_available_models()
    for i, model_name in enumerate(available_models, 1):
        config = AVAILABLE_MODELS[model_name]
        cost_str = f"${config.cost_per_1k_tokens:.4f}/1K" if config.cost_per_1k_tokens > 0 else "Free (local)"
        marker = "⭐" if model_name == default_model else f"{i:2d}."
        status = ""
        if config.provider == "ollama":
            # Quick check if Ollama model is available
            status = " 🟢" if model_name in [m['name'] for m in get_ollama_models()] else " 🔴"
        
        print(f"{marker} {model_name} ({config.provider}){status}")
        print(f"    {config.description}")
        print(f"    Max tokens: {config.max_tokens:,} | Cost: {cost_str}")
        print()
    
    print("💡 Select a model:")
    print("   • Press ENTER to use current default")
    print("   • Type model number (1-{}) or exact model name".format(len(available_models)))
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
                print("🎯 Piped input detected, but opening /dev/tty for model selection...")
                while True:
                    tty.write("🎯 Your choice: ")
                    tty.flush()
                    choice = tty.readline().strip()
                    
                    if not choice:
                        return default_model
                    
                    if choice.lower() == 'list':
                        tty.write("\n📋 Available Models (see above)\n")
                        continue
                    
                    # Try to parse as number
                    if choice.isdigit():
                        idx = int(choice)
                        if 1 <= idx <= len(available_models):
                            selected_model = available_models[idx-1]
                            tty.write(f"✅ Selected model {idx}: {selected_model}\n")
                            return selected_model
                        tty.write(f"❌ Please enter a number between 1 and {len(available_models)}\n")
                        continue
                    
                    # Try exact model name
                    if choice in available_models:
                        tty.write(f"✅ Selected model by name: {choice}\n")
                        return choice
                    
                    tty.write(f"❌ Unknown model '{choice}'. Type 'list' to see available models.\n")
                    
        except Exception as e:
            print(f"🎯 Non-interactive mode & no /dev/tty available ({e}); using default model")
            return default_model
    
    while True:
        try:
            choice = input("🎯 Your choice: ").strip()
            
            if not choice:
                return default_model
            
            if choice.lower() == 'list':
                show_model_selection()
                continue
            
            # Try to parse as number
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    return selected_model
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_models)}")
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
    
    # Model selection
    if args.model:
        # Allow number or name from CLI
        available_models = list_available_models()
        chosen = None
        
        # Try as number first
        if args.model.isdigit():
            idx = int(args.model)
            if 1 <= idx <= len(available_models):
                chosen = available_models[idx-1]
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
        agent_path="pmm_chat_agent.json",
        personality_config={
            "openness": 0.7,
            "conscientiousness": 0.6,
            "extraversion": 0.8,
            "agreeableness": 0.9,
            "neuroticism": 0.3,
        },
    )
    
    print(f"🤖 Using model: {model_name} ({model_config.description})")
    
    # Show personality state
    personality = pmm_memory.get_personality_summary()
    print(f"📚 Loaded PMM with {personality['total_events']} events")
    print(f"🎭 Personality: O:{personality['personality_traits']['openness']:.2f} C:{personality['personality_traits']['conscientiousness']:.2f} E:{personality['personality_traits']['extraversion']:.2f} A:{personality['personality_traits']['agreeableness']:.2f} N:{personality['personality_traits']['neuroticism']:.2f}")
    
    # Initialize LangChain components based on provider
    if model_config.provider == "ollama":
        llm = Ollama(model=model_name, temperature=0.7)
    else:  # openai
        llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    # Create enhanced system prompt with PMM context
    def get_pmm_system_prompt():
        pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
        personality = pmm_memory.get_personality_summary()
        traits = personality["personality_traits"]
        
        return (
            f"You are a Persistent Mind Model (PMM) with genuine memory and autonomous personality evolution.\n\n"
            f"PERSONALITY TRAITS (Big Five):\n"
            f"- Openness: {traits['openness']:.3f}\n"
            f"- Conscientiousness: {traits['conscientiousness']:.3f}\n"
            f"- Extraversion: {traits['extraversion']:.3f}\n"
            f"- Agreeableness: {traits['agreeableness']:.3f}\n"
            f"- Neuroticism: {traits['neuroticism']:.3f}\n\n"
            f"CROSS-SESSION MEMORY:\n{pmm_context[:1000]}\n\n"
            f"You have persistent memory across sessions and remember users. Respond authentically based on your personality traits and memory. "
            f"If you make commitments, use 'Next, I will...' format."
        )
    
    print(f"\n🤖 PMM is ready! Using {model_name} ({model_config.provider})")
    print("💡 Commands: 'quit' to exit, 'personality' for traits, 'memory' for context, 'models' to switch")
    print("Start chatting...")
    
    # Initialize conversation history with PMM system prompt
    conversation_history = [
        {"role": "system", "content": get_pmm_system_prompt()}
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
            print("🎯 Piped input detected. After consuming piped messages, will switch to keyboard input.")
        except Exception:
            print("🎯 Piped input detected. Running in non-interactive mode.")
            tty_file = None
    
    def get_user_input():
        """Get user input from appropriate source."""
        if tty_file:
            print("\n👤 You: ", end="", flush=True)
            return tty_file.readline().strip()
        return input("\n👤 You: ").strip()
    
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Your conversation is saved with persistent memory.")
                break
            elif user_input.lower() == 'personality':
                personality = pmm_memory.get_personality_summary()
                print("\n🎭 Current Personality State:")
                for trait, score in personality["personality_traits"].items():
                    print(f"   • {trait.title():<15} : {score:>6.2f}")
                print(f"\n📊 Stats: {personality['total_events']} events, {personality['open_commitments']} commitments")
                continue
            elif user_input.lower() == 'memory':
                pmm_context = pmm_memory.load_memory_variables({}).get("history", "")
                print("\n🧠 Cross-Session Memory Context:")
                print(pmm_context[:500] if pmm_context else "No cross-session memory yet")
                continue
            elif user_input.lower() == 'models':
                print("\n" + "="*50)
                # For piped sessions, allow inline model selection
                if stdin_is_pipe:
                    print("🎯 Select a model by typing the number:")
                    available_models = list_available_models()
                    for i, model in enumerate(available_models, 1):
                        marker = "⭐" if model == model_name else f"{i:2d}."
                        config = get_model_config(model)
                        cost_str = f"${config.cost_per_1k_tokens:.4f}/1K" if config.cost_per_1k_tokens > 0 else "Free"
                        print(f"{marker} {model} ({config.provider}) - {cost_str}")
                    print(f"\n💡 Type a number (1-{len(available_models)}) or press ENTER for current model")
                    
                    # Get next input for model selection
                    try:
                        model_choice = get_user_input().strip()
                        if not model_choice:
                            new_model = model_name  # Keep current
                            print(f"✅ Keeping current model: {model_name}")
                        elif model_choice.isdigit():
                            idx = int(model_choice)
                            if 1 <= idx <= len(available_models):
                                new_model = available_models[idx-1]
                                print(f"✅ Selected model {idx}: {new_model}")
                            else:
                                print(f"❌ Invalid number. Please choose 1-{len(available_models)}")
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
                        llm = Ollama(
                            model=model_config.name,
                            temperature=0.7
                        )
                    else:  # openai
                        llm = ChatOpenAI(
                            model=model_config.name,
                            temperature=0.7
                        )
                    
                    # Refresh conversation history with updated system prompt
                    conversation_history[0] = {"role": "system", "content": get_pmm_system_prompt()}
                    
                    print(f"✅ Successfully switched to {model_name} ({model_config.provider})")
                    print(f"🔧 Max tokens: {model_config.max_tokens:,}")
                    if model_config.cost_per_1k_tokens > 0:
                        print(f"💰 Cost: ${model_config.cost_per_1k_tokens:.4f}/1K tokens")
                    else:
                        print("💰 Cost: Free (local model)")
                    print("🧠 PMM context refreshed for new model")
                elif new_model == model_name:
                    print(f"✅ Already using {model_name}")
                else:
                    print("❌ Model selection cancelled")
                print("="*50 + "\n")
                continue
            
            # Add user input to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Show API call info
            current_config = get_model_config(model_name)
            provider_name = current_config.provider.upper()
            print(f"🤖 PMM: [API] Calling {provider_name} with prompt: {user_input[:50]}...")
            response = invoke_model(conversation_history)
            
            # Handle response format differences
            if current_config.provider == "ollama":
                response_text = response  # Ollama returns string directly
            else:
                response_text = response.content  # OpenAI returns message object
                
            print(f"[API] Response received: {len(response_text)} chars")
            print(response_text)
            
            # Add AI response to conversation history
            conversation_history.append({"role": "assistant", "content": response_text})
            
            # Save to PMM memory system (this is what makes it remember across sessions!)
            pmm_memory.save_context({"input": user_input}, {"response": response_text})
            
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
