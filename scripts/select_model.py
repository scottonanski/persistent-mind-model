#!/usr/bin/env python3
"""
Model selection utility for PMM.
Set your preferred model across all scripts and applications.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.config import print_model_info, list_available_models, get_default_model, get_model_config

def main():
    """Interactive model selection."""
    print("=== PMM Model Selection ===")
    
    current_model = get_default_model()
    current_config = get_model_config(current_model)
    
    print(f"Current model: {current_model}1")
    print(f"Description: {current_config.description}")
    print(f"Cost: ${current_config.cost_per_1k_tokens:.4f} per 1K tokens")
    print()
    
    print("Available models:")
    print_model_info()
    
    print("Options:")
    print("1. Keep current model")
    print("2. Change model")
    print("3. Set via environment variable")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        print(f"✅ Keeping {current_model}")
        return
    
    elif choice == "2":
        models = list_available_models()
        print("\nSelect new model:")
        for i, model in enumerate(models, 1):
            config = get_model_config(model)
            cost_str = f"${config.cost_per_1k_tokens:.4f}/1K" if config.cost_per_1k_tokens > 0 else "Free"
            print(f"{i}. {model} - {config.description} ({cost_str})")
        
        try:
            selection = int(input(f"Enter number (1-{len(models)}): ")) - 1
            if 0 <= selection < len(models):
                new_model = models[selection]
                
                # Update .env file
                env_file = Path(__file__).parent.parent / '.env'
                env_lines = []
                
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        env_lines = f.readlines()
                
                # Remove existing PMM_DEFAULT_MODEL line
                env_lines = [line for line in env_lines if not line.startswith('PMM_DEFAULT_MODEL=')]
                
                # Add new model
                env_lines.append(f'PMM_DEFAULT_MODEL={new_model}\n')
                
                with open(env_file, 'w') as f:
                    f.writelines(env_lines)
                
                print(f"✅ Model changed to {new_model}")
                print(f"✅ Updated {env_file}")
                print("Restart your PMM applications to use the new model.")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")
    
    elif choice == "3":
        print("\nTo set model via environment variable:")
        print("export PMM_DEFAULT_MODEL=gpt-4o-mini")
        print("# or add to your .env file:")
        print("echo 'PMM_DEFAULT_MODEL=gpt-4o-mini' >> .env")
        print()
        models = list_available_models()
        print(f"Available models: {', '.join(models)}")
    
    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main()
