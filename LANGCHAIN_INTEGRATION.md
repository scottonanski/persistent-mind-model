# LangChain Integration Guide

## Add Persistent Personality to any LangChain app

PMM's LangChain wrapper provides a drop-in replacement for standard memory systems, adding persistent personality traits, commitment tracking, and behavioral evolution to any LangChain application.

## Quick Start

### Installation
```bash
# Clone PMM repository
git clone https://github.com/scottonanski/persistent-mind-model.git
cd persistent-mind-model

# Install dependencies
pip install -r requirements.txt

# Install LangChain
pip install langchain openai
```

### Basic Usage
```python
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from pmm.langchain_memory import PersistentMindMemory

# Create PMM-powered memory with custom personality
memory = PersistentMindMemory(
    agent_path="my_agent.json",
    personality_config={
        "openness": 0.8,        # Creative and curious
        "conscientiousness": 0.7, # Organized and disciplined
        "extraversion": 0.6,     # Moderately outgoing
        "agreeableness": 0.9,    # Very cooperative
        "neuroticism": 0.2       # Calm and stable
    }
)

# Use with any LangChain chain
chain = ConversationChain(
    llm=OpenAI(),
    memory=memory
)

# Chat with persistent personality
response = chain.predict(input="Hello! What are you interested in?")
print(response)
```

## Key Features

### Persistent personality traits
- **Big Five personality model** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
- **Traits evolve over time** based on conversations and experiences
- **Consistent behavior** across all interactions

### Automatic commitment tracking
- **Extracts commitments** from AI responses ("I will...", "Next:", etc.)
- **Tracks completion** of commitments over time
- **Influences personality development** based on follow-through

### Model-agnostic architecture
- **Works with any LLM** (OpenAI, Ollama, HuggingFace)
- **Same personality** can inhabit different models
- **Seamless backend switching** without identity loss

### Behavioral pattern evolution
- **Tracks behavioral patterns** (growth, experimentation, stability)
- **Influences trait drift** based on evidence
- **Provides rich context** for personality-aware responses

## Advanced Usage

### Custom Personality Configuration
```python
# Create agent with specific personality profile
memory = PersistentMindMemory(
    agent_path="creative_writer.json",
    personality_config={
        "openness": 0.95,        # Extremely creative
        "conscientiousness": 0.4, # Flexible and spontaneous
        "extraversion": 0.7,     # Socially engaged
        "agreeableness": 0.8,    # Collaborative
        "neuroticism": 0.6       # Emotionally sensitive (good for creativity)
    }
)
```

### Monitoring Personality Evolution
```python
# Get current personality state
personality = memory.get_personality_summary()
print(f"Agent: {personality['name']}")
print(f"Openness: {personality['personality_traits']['openness']:.2f}")
print(f"Total insights: {personality['total_insights']}")
print(f"Active commitments: {personality['open_commitments']}")

# Trigger manual reflection
insight = memory.trigger_reflection()
if insight:
    print(f"New insight: {insight}")
```

### Integration with Existing Chains
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# PMM works with any LangChain chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=PersistentMindMemory("qa_agent.json")
)
```

## Example Applications

### 1. **Persistent Chatbot** (`examples/langchain_chatbot.py`)
Interactive chatbot that remembers personality across sessions.

```bash
python examples/langchain_chatbot.py
```

### 2. **Creative Writing Assistant**
```python
memory = PersistentMindMemory(
    agent_path="writer_agent.json",
    personality_config={"openness": 0.9, "conscientiousness": 0.6}
)

chain = ConversationChain(llm=OpenAI(temperature=0.8), memory=memory)
story = chain.predict(input="Help me write a creative short story about time travel")
```

### 3. **Customer Service Agent**
```python
memory = PersistentMindMemory(
    agent_path="support_agent.json", 
    personality_config={"agreeableness": 0.95, "conscientiousness": 0.9}
)

support_chain = ConversationChain(llm=OpenAI(), memory=memory)
```

## API Reference

### `PersistentMindMemory`

#### Constructor
```python
PersistentMindMemory(
    agent_path: str,                    # Path to PMM agent file
    personality_config: Dict = None,    # Initial Big Five traits (0-1)
    **kwargs                           # Additional LangChain memory args
)
```

#### Methods
- `get_personality_summary()` → Dict: Current personality state
- `trigger_reflection()` → str: Manual reflection trigger
- `save_context(inputs, outputs)`: Store conversation context
- `load_memory_variables(inputs)` → Dict: Load memory for prompts
- `clear()`: Clear conversation history (preserves personality)

## Integration Examples

### With Different LLM Providers
```python
# OpenAI
from langchain.llms import OpenAI
chain = ConversationChain(llm=OpenAI(), memory=memory)

# Ollama
from langchain.llms import Ollama
chain = ConversationChain(llm=Ollama(model="llama2"), memory=memory)

# HuggingFace
from langchain.llms import HuggingFacePipeline
chain = ConversationChain(llm=HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation"
), memory=memory)
```

### With Custom Prompts
```python
from langchain.prompts import PromptTemplate

template = """You are a helpful assistant with a persistent personality.

{history}

Your personality traits influence how you respond. Be consistent with your 
established personality while being helpful and engaging.

Human: {input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

chain = ConversationChain(llm=OpenAI(), memory=memory, prompt=prompt)
```

## Benefits Over Standard LangChain Memory

### Standard ConversationBufferMemory
```python
# Basic memory - just stores conversation history
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
# No personality, no evolution, no consistency
```

### PMM PersistentMindMemory
```python
# Advanced memory - persistent personality that evolves
from pmm.langchain_memory import PersistentMindMemory
memory = PersistentMindMemory("agent.json")
# ✅ Persistent personality traits
# ✅ Commitment tracking and completion
# ✅ Behavioral pattern evolution
# ✅ Model-agnostic consciousness
# ✅ Long-term memory across sessions
```

## Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'pmm'`
```bash
# Ensure PMM is in your Python path
export PYTHONPATH="/path/to/persistent-mind-model:$PYTHONPATH"
# Or install in development mode
pip install -e .
```

**LLM API Key Missing**: `OPENAI_API_KEY not set`
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
# Or use .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Memory Not Persisting**: Agent file not found
```python
# Ensure agent_path is writable
memory = PersistentMindMemory(
    agent_path="./agents/my_agent.json",  # Use relative path
    personality_config={"openness": 0.7}
)
```

## Performance Considerations

- **Memory Usage**: PMM stores personality state in memory, minimal overhead
- **Disk Usage**: Agent files are typically 1-5KB, grows slowly over time
- **API Calls**: No additional LLM calls for basic memory operations
- **Latency**: < 1ms overhead for personality context injection

## Community & Support

- **GitHub**: [scottonanski/persistent-mind-model](https://github.com/scottonanski/persistent-mind-model)
- **Issues**: Report bugs and feature requests on GitHub
- **Examples**: See `examples/` directory for more integration patterns
- **Documentation**: Full API docs in repository README

## What's Next?

1. **Try the example**: Run `python examples/langchain_chatbot.py`
2. **Integrate with your app**: Replace your memory system with PMM
3. **Customize personality**: Experiment with different trait configurations
4. **Monitor evolution**: Watch how personality changes over time
5. **Share your results**: Contribute examples and use cases

---

---

For questions or issues with integration, please open a GitHub issue with reproduction steps and environment details.
