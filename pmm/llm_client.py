#!/usr/bin/env python3
"""
Production-ready LLM client with retry logic, rate limiting, and proper error handling.
"""

import time
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import get_config
from .logging_config import get_logger

logger = get_logger("llm")


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    response_time_ms: int


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Raised when rate limit is exceeded."""
    pass


class TokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass


class LLMClient:
    """Production-ready LLM client with comprehensive error handling."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """Initialize LLM client with configuration."""
        self.config = get_config().llm
        
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        self.total_tokens_used = 0
        self.request_count = 0
        
        logger.info(f"Initialized LLM client with model: {self.config.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    def chat(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat completion request with retry logic."""
        start_time = time.time()
        
        try:
            response = self._make_request(
                system=system,
                user=user,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            
            # Track usage
            if response.usage:
                tokens_used = response.usage.total_tokens
                self.total_tokens_used += tokens_used
                logger.debug(f"Tokens used: {tokens_used}, Total: {self.total_tokens_used}")
            
            self.request_count += 1
            
            logger.info(
                f"LLM request completed in {response_time_ms}ms, "
                f"finish_reason: {choice.finish_reason}"
            )
            
            return content
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise RateLimitError(f"Rate limit exceeded: {e}")
        
        except openai.APITimeoutError as e:
            logger.warning(f"Request timeout: {e}")
            raise LLMError(f"Request timeout: {e}")
        
        except openai.BadRequestError as e:
            if "maximum context length" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise TokenLimitError(f"Token limit exceeded: {e}")
            else:
                logger.error(f"Bad request: {e}")
                raise LLMError(f"Bad request: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def _make_request(
        self,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int
    ) -> Any:
        """Make the actual API request."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        logger.debug(f"Making LLM request: model={self.config.model}, temp={temperature}")
        
        return self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.config.timeout
        )
    
    async def chat_async(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Async version of chat method."""
        # Run sync method in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            system,
            user,
            temperature,
            max_tokens
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "request_count": self.request_count,
            "model": self.config.model,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count
                if self.request_count > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.total_tokens_used = 0
        self.request_count = 0
        logger.info("Usage statistics reset")


class BatchLLMClient:
    """Client for batch processing multiple LLM requests."""
    
    def __init__(self, client: LLMClient, batch_size: int = 5, delay_ms: int = 1000):
        self.client = client
        self.batch_size = batch_size
        self.delay_ms = delay_ms
        
    async def process_batch(
        self,
        requests: List[Dict[str, str]]
    ) -> List[str]:
        """Process multiple requests with rate limiting."""
        results = []
        
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            
            # Process batch concurrently
            tasks = [
                self.client.chat_async(
                    system=req["system"],
                    user=req["user"]
                )
                for req in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch request failed: {result}")
                    results.append("")
                else:
                    results.append(result)
            
            # Rate limiting delay between batches
            if i + self.batch_size < len(requests):
                await asyncio.sleep(self.delay_ms / 1000)
        
        return results
