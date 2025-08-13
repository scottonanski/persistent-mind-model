#!/usr/bin/env python3
"""
Centralized model configuration for PMM.
Single source of truth for all model selection across the entire codebase.
"""

import os
import subprocess
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration with provider and settings."""
    name: str
    provider: str
    max_tokens: int = 4096
    temperature: float = 0.7
    cost_per_1k_tokens: float = 0.0
    description: str = ""

def check_ollama_running() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_ollama_models() -> List[Dict]:
    """Get list of available Ollama models if running."""
    if not check_ollama_running():
        return []
    
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []
        
        models = []
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    size = parts[2] if len(parts) > 2 else "Unknown"
                    models.append({
                        'name': name,
                        'size': size,
                        'full_line': line.strip()
                    })
        return models
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return []

def build_dynamic_models() -> Dict[str, ModelConfig]:
    """Build models registry with dynamic Ollama detection."""
    models = {
        # OpenAI Models (always available if API key set)
        "gpt-4o-mini": ModelConfig(
            name="gpt-4o-mini",
            provider="openai",
            max_tokens=16384,
            temperature=0.7,
            cost_per_1k_tokens=0.00015,
            description="Fast, cost-effective model for most PMM tasks"
        ),
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            provider="openai", 
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.005,
            description="High-quality model for complex reasoning"
        ),
        "gpt-3.5-turbo": ModelConfig(
            name="gpt-3.5-turbo",
            provider="openai",
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.0015,
            description="Legacy model, still capable"
        ),
    }
    
    # Add Ollama models if available
    ollama_models = get_ollama_models()
    for model_info in ollama_models:
        model_name = model_info['name']
        models[model_name] = ModelConfig(
            name=model_name,
            provider="ollama",
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.0,
            description=f"Local Ollama model ({model_info['size']})"
        )
    
    return models

# Available models registry (built dynamically)
AVAILABLE_MODELS: Dict[str, ModelConfig] = build_dynamic_models()

def get_default_model() -> str:
    """Get default model from environment or fallback."""
    return os.getenv("PMM_DEFAULT_MODEL", "gpt-4o-mini")

def get_model_config(model_name: str = None) -> ModelConfig:
    """Get configuration for specified model or default."""
    model_name = model_name or get_default_model()
    
    if model_name not in AVAILABLE_MODELS:
        print(f"⚠️  Unknown model '{model_name}', falling back to gpt-4o-mini")
        model_name = "gpt-4o-mini"
    
    return AVAILABLE_MODELS[model_name]

def list_available_models() -> List[str]:
    """List all available model names."""
    return list(AVAILABLE_MODELS.keys())

def get_models_by_provider(provider: str) -> List[str]:
    """Get all models for a specific provider."""
    return [name for name, config in AVAILABLE_MODELS.items() 
            if config.provider == provider]

def estimate_cost(model_name: str, token_count: int) -> float:
    """Estimate cost for token usage with specified model."""
    config = get_model_config(model_name)
    return (token_count / 1000) * config.cost_per_1k_tokens

def print_model_info():
    """Print information about all available models."""
    print("=== PMM Available Models ===")
    
    # Show current default model at the top
    default_model = get_default_model()
    default_config = get_model_config(default_model)
    default_cost_str = f"${default_config.cost_per_1k_tokens:.4f}/1K" if default_config.cost_per_1k_tokens > 0 else "Free (local)"
    
    print(f"⭐ CURRENT DEFAULT: {default_model} ({default_config.provider})")
    print(f"   {default_config.description}")
    print(f"   Max tokens: {default_config.max_tokens:,} | Cost: {default_cost_str}")
    print()
    
    # Check Ollama status
    ollama_running = check_ollama_running()
    ollama_models = get_ollama_models()
    
    if ollama_running:
        print(f"🟢 Ollama: Running ({len(ollama_models)} models available)")
    else:
        print("🔴 Ollama: Not running (no local models available)")
    print()
    
    print("All Available Models:")
    for name, config in AVAILABLE_MODELS.items():
        cost_str = f"${config.cost_per_1k_tokens:.4f}/1K" if config.cost_per_1k_tokens > 0 else "Free (local)"
        status = ""
        if config.provider == "ollama":
            status = " 🟢" if ollama_running else " 🔴"
        
        # Mark current default with star
        marker = "⭐" if name == default_model else "•"
        print(f"{marker} {name} ({config.provider}){status}")
        print(f"  {config.description}")
        print(f"  Max tokens: {config.max_tokens:,} | Cost: {cost_str}")
        print()

if __name__ == "__main__":
    print_model_info()
