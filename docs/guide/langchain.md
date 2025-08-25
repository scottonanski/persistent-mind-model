---
title: LangChain Integration
---

# LangChain Integration

This guide shows how to use PersistentMindMemory to add persistent personality and directive tracking to any LangChain app.

Refer to LANGCHAIN_INTEGRATION.md at the repository root for the full, canonical version. A condensed version is provided here for convenience.

## Quick Start

1. Install dependencies (repo root):
   - pip install -r requirements.txt
   - pip install langchain openai

2. Basic usage:

```python
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from pmm.langchain_memory import PersistentMindMemory

memory = PersistentMindMemory(
    agent_path="my_agent.json",
    personality_config={
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.9,
        "neuroticism": 0.2,
    },
)

chain = ConversationChain(llm=OpenAI(), memory=memory)
print(chain.predict(input="Hello! What are you interested in?"))
```

## Features

- Big Five personality that evolves over time
- Automatic commitment extraction and tracking
- Model-agnostic: OpenAI, Ollama, HF, etc.
- Rich behavioral context for prompts

## Advanced

- Custom prompts with {history}
- Monitor evolution via memory.get_personality_summary()
- Trigger manual reflection with memory.trigger_reflection()

For extended examples and troubleshooting, see the repository’s LANGCHAIN_INTEGRATION.md and examples/.