"""
Local Inference Engine - Layer 6 Implementation

Implements offline-first AI inference with support for Ollama, LM Studio, llama.cpp,
and hybrid API fallback for self-sovereign AI consciousness.
"""

from __future__ import annotations
import json
import requests
import subprocess
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .enhanced_model import ProviderConfig


@dataclass
class InferenceResult:
    """
    Result of an inference operation.
    """
    
    response: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    fallback_used: bool = False
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class BaseInferenceProvider(ABC):
    """
    Abstract base class for inference providers.
    """
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text response from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and ready."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        pass


class OllamaProvider(BaseInferenceProvider):
    """
    Provider for Ollama local LLM inference.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.ollama_base_url
        self.model = config.ollama_model
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using Ollama API."""
        start_time = time.time()
        
        try:
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                }
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                result_data = response.json()
                latency = (time.time() - start_time) * 1000
                
                return InferenceResult(
                    response=result_data.get("response", ""),
                    provider="ollama",
                    model=self.model,
                    tokens_used=result_data.get("eval_count", 0),
                    latency_ms=latency,
                    success=True
                )
            else:
                return InferenceResult(
                    response="",
                    provider="ollama",
                    model=self.model,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return InferenceResult(
                response="",
                provider="ollama",
                model=self.model,
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model.get("name", "").startswith(self.model) for model in models)
            return False
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return {"error": "Model info not available"}
        except Exception as e:
            return {"error": str(e)}


class LMStudioProvider(BaseInferenceProvider):
    """
    Provider for LM Studio local inference.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.base_url = config.lm_studio_base_url
        self.model = config.lm_studio_model
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using LM Studio OpenAI-compatible API."""
        start_time = time.time()
        
        try:
            # LM Studio uses OpenAI-compatible API
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.config.timeout_seconds
            )
            
            if response.status_code == 200:
                result_data = response.json()
                latency = (time.time() - start_time) * 1000
                
                choice = result_data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                usage = result_data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                
                return InferenceResult(
                    response=content,
                    provider="lm_studio",
                    model=self.model,
                    tokens_used=tokens_used,
                    latency_ms=latency,
                    success=True
                )
            else:
                return InferenceResult(
                    response="",
                    provider="lm_studio",
                    model=self.model,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            return InferenceResult(
                response="",
                provider="lm_studio",
                model=self.model,
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if LM Studio is running."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get LM Studio model information."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {"error": "Model info not available"}
        except Exception as e:
            return {"error": str(e)}


class LlamaCppProvider(BaseInferenceProvider):
    """
    Provider for llama.cpp direct inference.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model_path = config.llamacpp_model_path
        self.n_ctx = config.llamacpp_n_ctx
        self.n_threads = config.llamacpp_n_threads
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using llama.cpp command line."""
        start_time = time.time()
        
        if not self.model_path or not os.path.exists(self.model_path):
            return InferenceResult(
                response="",
                provider="llamacpp",
                model=self.model_path or "unknown",
                success=False,
                error_message="Model path not found"
            )
        
        try:
            # Prepare llama.cpp command
            cmd = [
                "llama.cpp",  # Assumes llama.cpp is in PATH
                "-m", self.model_path,
                "-p", prompt,
                "-n", str(kwargs.get("max_tokens", self.config.max_tokens)),
                "-c", str(self.n_ctx),
                "-t", str(self.n_threads),
                "--temp", str(kwargs.get("temperature", self.config.temperature)),
                "--no-display-prompt"
            ]
            
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            latency = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                return InferenceResult(
                    response=result.stdout.strip(),
                    provider="llamacpp",
                    model=os.path.basename(self.model_path),
                    latency_ms=latency,
                    success=True
                )
            else:
                return InferenceResult(
                    response="",
                    provider="llamacpp",
                    model=os.path.basename(self.model_path),
                    success=False,
                    error_message=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return InferenceResult(
                response="",
                provider="llamacpp",
                model=os.path.basename(self.model_path) if self.model_path else "unknown",
                success=False,
                error_message="Inference timeout"
            )
        except Exception as e:
            return InferenceResult(
                response="",
                provider="llamacpp",
                model=os.path.basename(self.model_path) if self.model_path else "unknown",
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if llama.cpp and model are available."""
        try:
            # Check if llama.cpp is in PATH
            result = subprocess.run(["which", "llama.cpp"], capture_output=True)
            if result.returncode != 0:
                return False
            
            # Check if model file exists
            return self.model_path and os.path.exists(self.model_path)
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get llama.cpp model information."""
        if not self.model_path or not os.path.exists(self.model_path):
            return {"error": "Model not found"}
        
        return {
            "model_path": self.model_path,
            "model_size": os.path.getsize(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads
        }


class HuggingFaceProvider(BaseInferenceProvider):
    """
    Provider for HuggingFace Transformers local inference.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.model_name = config.hf_model_name
        self.device = config.hf_device
        self.max_length = config.hf_max_length
        self.pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model and pipeline."""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"Failed to initialize HuggingFace model: {e}")
            self.pipeline = None
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using HuggingFace pipeline."""
        start_time = time.time()
        
        if not TRANSFORMERS_AVAILABLE:
            return InferenceResult(
                response="",
                provider="huggingface",
                model=self.model_name,
                success=False,
                error_message="Transformers library not available"
            )
        
        if not self.pipeline:
            return InferenceResult(
                response="",
                provider="huggingface",
                model=self.model_name,
                success=False,
                error_message="Model not initialized"
            )
        
        try:
            # Generate text
            result = self.pipeline(
                prompt,
                max_length=kwargs.get("max_tokens", self.max_length),
                temperature=kwargs.get("temperature", self.config.temperature),
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            latency = (time.time() - start_time) * 1000
            
            # Extract generated text (remove prompt)
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return InferenceResult(
                response=generated_text,
                provider="huggingface",
                model=self.model_name,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            return InferenceResult(
                response="",
                provider="huggingface",
                model=self.model_name,
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if HuggingFace model is available."""
        return TRANSFORMERS_AVAILABLE and self.pipeline is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        if not self.pipeline:
            return {"error": "Model not available"}
        
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "tokenizer_vocab_size": len(self.pipeline.tokenizer) if self.pipeline.tokenizer else 0
        }


class OpenAIProvider(BaseInferenceProvider):
    """
    Provider for OpenAI API inference (fallback).
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = config.openai_model
        self.base_url = config.openai_base_url
        
        if OPENAI_AVAILABLE and self.api_key:
            if self.base_url:
                openai.api_base = self.base_url
            openai.api_key = self.api_key
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """Generate text using OpenAI API."""
        start_time = time.time()
        
        if not OPENAI_AVAILABLE:
            return InferenceResult(
                response="",
                provider="openai",
                model=self.model,
                success=False,
                error_message="OpenAI library not available"
            )
        
        if not self.api_key:
            return InferenceResult(
                response="",
                provider="openai",
                model=self.model,
                success=False,
                error_message="OpenAI API key not provided"
            )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=self.config.timeout_seconds
            )
            
            latency = (time.time() - start_time) * 1000
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return InferenceResult(
                response=content,
                provider="openai",
                model=self.model,
                tokens_used=tokens_used,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            return InferenceResult(
                response="",
                provider="openai",
                model=self.model,
                success=False,
                error_message=str(e)
            )
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "model": self.model,
            "api_key_configured": bool(self.api_key),
            "base_url": self.base_url
        }


class LocalInferenceEngine:
    """
    Main engine for local and hybrid AI inference.
    
    Manages multiple providers with automatic fallback and provider selection
    based on availability and performance.
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.providers = {}
        self.provider_stats = {}
        
        # Initialize providers
        self._initialize_providers()
        
        # Determine provider order
        self.provider_order = self._determine_provider_order()
    
    def _initialize_providers(self):
        """Initialize all available providers."""
        # Local providers
        self.providers["ollama"] = OllamaProvider(self.config)
        self.providers["lm_studio"] = LMStudioProvider(self.config)
        self.providers["llamacpp"] = LlamaCppProvider(self.config)
        self.providers["huggingface"] = HuggingFaceProvider(self.config)
        
        # API provider (fallback)
        self.providers["openai"] = OpenAIProvider(self.config)
        
        # Initialize stats
        for provider_name in self.providers:
            self.provider_stats[provider_name] = {
                "success_count": 0,
                "failure_count": 0,
                "total_latency": 0.0,
                "avg_latency": 0.0,
                "last_used": None
            }
    
    def _determine_provider_order(self) -> List[str]:
        """Determine provider priority order based on config and availability."""
        order = []
        
        # Primary provider first (if available)
        if self.providers[self.config.primary_provider].is_available():
            order.append(self.config.primary_provider)
        
        # Local providers (if use_local_first is enabled)
        if self.config.use_local_first:
            local_providers = ["ollama", "lm_studio", "llamacpp", "huggingface"]
            for provider in local_providers:
                if provider not in order and self.providers[provider].is_available():
                    order.append(provider)
        
        # Fallback provider
        if (self.config.api_fallback_enabled and 
            self.config.fallback_provider not in order and
            self.providers[self.config.fallback_provider].is_available()):
            order.append(self.config.fallback_provider)
        
        return order
    
    def generate_text(self, prompt: str, **kwargs) -> InferenceResult:
        """
        Generate text using the best available provider.
        
        Tries providers in order until one succeeds or all fail.
        """
        if not self.provider_order:
            return InferenceResult(
                response="",
                provider="none",
                model="none",
                success=False,
                error_message="No providers available"
            )
        
        last_error = None
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            
            # Skip if provider is not available
            if not provider.is_available():
                continue
            
            # Try inference
            result = provider.generate_text(prompt, **kwargs)
            
            # Update stats
            self._update_provider_stats(provider_name, result)
            
            if result.success:
                # Mark if fallback was used
                if provider_name != self.config.primary_provider:
                    result.fallback_used = True
                
                return result
            else:
                last_error = result.error_message
                continue
        
        # All providers failed
        return InferenceResult(
            response="",
            provider="failed",
            model="none",
            success=False,
            error_message=f"All providers failed. Last error: {last_error}"
        )
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        
        for provider_name, provider in self.providers.items():
            status[provider_name] = {
                "available": provider.is_available(),
                "model_info": provider.get_model_info(),
                "stats": self.provider_stats[provider_name].copy()
            }
        
        return status
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get comprehensive inference statistics."""
        total_requests = sum(
            stats["success_count"] + stats["failure_count"] 
            for stats in self.provider_stats.values()
        )
        
        successful_requests = sum(
            stats["success_count"] for stats in self.provider_stats.values()
        )
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
            "provider_order": self.provider_order,
            "provider_stats": self.provider_stats.copy(),
            "config": {
                "primary_provider": self.config.primary_provider,
                "fallback_provider": self.config.fallback_provider,
                "use_local_first": self.config.use_local_first,
                "api_fallback_enabled": self.config.api_fallback_enabled
            }
        }
    
    def _update_provider_stats(self, provider_name: str, result: InferenceResult):
        """Update statistics for a provider."""
        stats = self.provider_stats[provider_name]
        
        if result.success:
            stats["success_count"] += 1
            stats["total_latency"] += result.latency_ms
            stats["avg_latency"] = stats["total_latency"] / stats["success_count"]
        else:
            stats["failure_count"] += 1
        
        stats["last_used"] = result.timestamp
    
    def refresh_provider_availability(self):
        """Refresh provider availability and update order."""
        self.provider_order = self._determine_provider_order()
    
    def set_primary_provider(self, provider_name: str):
        """Change the primary provider."""
        if provider_name in self.providers:
            self.config.primary_provider = provider_name
            self.refresh_provider_availability()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
