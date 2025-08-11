# PMM Next-Stage Implementation Guide

## Quick Start Guide

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/persistent-mind-model.git
cd persistent-mind-model

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"  # Optional for API fallback
export PMM_DEBUG=1  # Enable debug logging
```

### Basic Usage

```python
from pmm.enhanced_manager import EnhancedSelfModelManager

# Initialize with next-stage features enabled
manager = EnhancedSelfModelManager(
    model_path="my_ai_identity.json",
    enable_next_stage=True
)

# Add experiences (automatically tokenized)
manager.add_event("Recorded a notable insight about my workflow")
manager.add_thought("Memory tokens create verifiable AI history")
manager.add_insight("Added memory-token state model for richer recall heuristics")

# Semantic recall from memory
results = manager.recall_memories("topic breakthrough", max_results=5)
for result in results:
    print(f"Memory: {result.content[:100]}...")
    print(f"Relevance: {result.similarity_score:.3f}")

# Generate text using local models
response = manager.generate_text_local(
    "What insights have I recorded about topic X?",
    max_tokens=200
)
print(f"Generated using: {response.provider}")
print(response.text)

# Export complete identity
manifest = manager.export_identity("./my_backup")
print(f"Exported {manifest.total_tokens} tokens with integrity verification")
```

### CLI Usage Examples

Note: If the `pmm-cli` entrypoint is not installed on your PATH, use `python pmm_cli.py`.

```bash
# Add new experiences
python pmm_cli.py add-event "Noted improvement in semantic recall heuristics" --next-stage

# Recall memories semantically
python pmm_cli.py recall "semantic recall" --max-results 10 --next-stage

# Generate text locally
python pmm_cli.py generate "Summarize my memory state and recent insights" --next-stage

# Export identity for backup/migration
python pmm_cli.py export-identity ./backup --compress --next-stage

# Verify cryptographic integrity
python pmm_cli.py verify-integrity --verbose --next-stage

# Show comprehensive statistics
python pmm_cli.py stats --next-stage
```

## Architecture Deep Dive

### Memory Token Lifecycle

```python
# 1. Event/Thought/Insight Creation
event = manager.add_event("Important experience")

# 2. Automatic Tokenization
# Handled internally by EnhancedSelfModelManager during add_event/add_thought/add_insight
# - Generates SHA-256 hash
# - Links to previous token
# - Assigns amplitude/phase (salience/semantic angle)
# - Stores minimal metadata

# 3. State Model Evolution
quantum = manager.quantum_manager
quantum.apply_temporal_decay([token], days_elapsed=1.0)
quantum.evolve_phase(token, semantic_context)

# 4. Memory Recall
recalled = recall_engine.recall_by_cue("important experience")
# - Embedding similarity matching
# - Phase resonance amplification
# - Hash integrity verification

# 5. Archival (when amplitude drops)
if token.amplitude < 0.1:
    archive_engine.archive_token(token)
    # - Thematic clustering
    # - Compression (GZIP/LZMA)
    # - Lockpoint creation
```

### Cryptographic Integrity Chain

```python
# Each token contains cryptographic proof
@dataclass
class MemoryToken:
    token_id: str = "mt_abc123"
    content_hash: str = "sha256:a1b2c3..."  # SHA-256 of content
    prev_hash: str = "sha256:x9y8z7..."    # Previous token's hash
    chain_position: int = 42                # Position in chain
    
# Chain verification
def verify_chain_integrity(tokens: List[MemoryToken]) -> bool:
    for i, token in enumerate(tokens[1:], 1):
        if token.prev_hash != tokens[i-1].content_hash:
            return False
        if token.chain_position != i:
            return False
    return True
```

### Memory token state model (amplitude/phase)

Note: 'quantum' terminology in this project is metaphorical. See `pmm/quantum_memory.py`. The implementation uses classical heuristics for salience and semantic drift.

```python
# Memory tokens maintain amplitude (salience) and phase (semantic angle) values used by recall heuristics
token.amplitude = 0.85  # Activation probability (0.0-1.0)
token.phase = 2.14      # Semantic angle (0-2π radians)

# Temporal decay (memories fade over time)
new_amplitude = token.amplitude * exp(-decay_rate * days_elapsed)

# Phase evolution (semantic drift)
new_phase = (token.phase + drift_rate * semantic_change) % (2 * pi)

# Memory resonance (related memories activate together)
for related_token in find_related_tokens(token):
    related_token.amplitude *= resonance_boost
```

## Configuration Guide

### Next-Stage Configuration

```python
@dataclass
class NextStageConfig:
    # Provider settings
    provider_config: ProviderConfig = field(default_factory=ProviderConfig)
    
    # Archive settings  
    archive_config: ArchiveConfig = field(default_factory=ArchiveConfig)
    
    # Recall settings
    recall_config: RecallConfig = field(default_factory=RecallConfig)
    
    # Integrity settings
    integrity_config: IntegrityConfig = field(default_factory=IntegrityConfig)

# Example configuration
config = NextStageConfig(
    provider_config=ProviderConfig(
        preferred_local_provider="ollama",
        api_fallback_enabled=True,
        local_model_name="llama3.2:3b"
    ),
    archive_config=ArchiveConfig(
        archive_threshold=0.1,  # Archive when amplitude < 0.1
        compression_algorithm="gzip",
        max_active_tokens=500
    ),
    recall_config=RecallConfig(
        embedding_model="all-MiniLM-L6-v2",
        max_recall_results=20,
        similarity_threshold=0.3
    )
)
```

### Local Inference Setup

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull llama3.2:3b      # Fast, efficient
ollama pull llama3.2:8b      # Higher quality
ollama pull gemma2:2b        # Ultra-lightweight

# Alternative: LM Studio
# Download from https://lmstudio.ai/
# Load any GGUF model file

# Alternative: llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
# Download GGUF models and run with ./main
```

## Advanced Features

### Custom Memory Tokenization

```python
from pmm.tokenization_engine import TokenizationEngine

# Custom tokenization for specific content types
class CustomTokenizer(TokenizationEngine):
    def tokenize_code_snippet(self, code: str, language: str) -> MemoryToken:
        # Custom logic for code memories
        token = self.create_base_token(code)
        token.tags.append(f"code_{language}")
        token.salience = self.calculate_code_complexity(code)
        return token
    
    def tokenize_conversation(self, messages: List[str]) -> MemoryToken:
        # Custom logic for conversation memories
        combined_content = "\n".join(messages)
        token = self.create_base_token(combined_content)
        token.valence = self.analyze_conversation_sentiment(messages)
        return token
```

### Custom Recall Strategies

```python
from pmm.recall_engine import RecallEngine

class EnhancedRecallEngine(RecallEngine):
    def recall_by_emotion(self, emotion: str, intensity: float) -> List[RecallResult]:
        # Find memories with specific emotional valence
        emotion_vector = self.emotion_to_vector(emotion)
        candidates = []
        
        for token in self.active_tokens:
            if self.emotional_similarity(token.phase, emotion_vector) > intensity:
                candidates.append(token)
        
        return self.rank_by_relevance(candidates)
    
    def recall_by_time_period(self, start_date: str, end_date: str) -> List[RecallResult]:
        # Temporal-based recall
        return [token for token in self.active_tokens 
                if start_date <= token.created_at <= end_date]
```

### Identity Migration Workflows

```python
# Complete identity backup
def backup_identity(manager: EnhancedSelfModelManager, backup_path: str):
    manifest = manager.export_identity(
        export_path=backup_path,
        include_archives=True,
        compress=True,
        encrypt=True,  # Optional encryption
        password="secure_password"
    )
    
    print(f"Backup created: {manifest.export_size_mb:.1f}MB")
    print(f"Integrity hash: {manifest.integrity_hash}")
    return manifest

# Cross-system migration
def migrate_identity(source_path: str, target_system: str):
    # Import on new system
    manager = EnhancedSelfModelManager(enable_next_stage=True)
    model, result = manager.import_identity(
        import_path=source_path,
        verify_integrity=True,
        merge_strategy="append"  # or "replace", "merge"
    )
    
    if result.success:
        print(f"Migration successful: {result.imported_tokens} tokens")
        print(f"Chain integrity: {result.chain_integrity_verified}")
    else:
        print(f"Migration failed: {result.error_message}")
```

## Performance Optimization

### Memory Management

```python
# Configure memory limits
config = ArchiveConfig(
    max_active_tokens=1000,      # Keep top 1000 most active
    archive_threshold=0.05,      # Archive when amplitude < 0.05
    compression_algorithm="lzma", # Higher compression ratio
    lockpoint_frequency=100      # Create lockpoint every 100 tokens
)

# Manual memory optimization
manager.trigger_archival()  # Force archival of low-amplitude memories
manager.optimize_memory()   # Defragment and optimize storage
manager.verify_integrity()  # Check for corruption
```

### Embedding Cache Optimization

```python
# Configure embedding cache
recall_config = RecallConfig(
    embedding_cache_size=10000,    # Cache 10k embeddings
    embedding_model="all-MiniLM-L6-v2",  # Fast, good quality
    cache_persistence=True,        # Save cache to disk
    batch_embedding_size=32        # Process in batches
)

# Precompute embeddings for better performance
manager.precompute_embeddings()  # Background embedding computation
```

## Security & Privacy

### Encryption Options

```python
# Enable encryption for sensitive deployments
integrity_config = IntegrityConfig(
    encryption_enabled=True,
    encryption_algorithm="AES-256-GCM",
    key_derivation="PBKDF2",
    backup_encryption=True
)

# Encrypted export
manifest = manager.export_identity(
    "./encrypted_backup",
    encrypt=True,
    password="strong_password_123"
)
```

### Access Control

```python
# Role-based access for enterprise deployments
class SecureManager(EnhancedSelfModelManager):
    def __init__(self, user_role: str, permissions: List[str]):
        super().__init__()
        self.user_role = user_role
        self.permissions = permissions
    
    def add_event(self, content: str) -> str:
        if "write" not in self.permissions:
            raise PermissionError("Write access denied")
        return super().add_event(content)
    
    def export_identity(self, path: str) -> ExportManifest:
        if "export" not in self.permissions:
            raise PermissionError("Export access denied")
        return super().export_identity(path)
```

## Troubleshooting

### Common Issues

**1. Local inference not working**
```bash
# Check if Ollama is running
ollama list
ollama serve  # Start Ollama daemon

# Test model availability
ollama run llama3.2:3b "Hello world"
```

**2. Memory recall returning no results**
```python
# Check embedding model
manager.recall_engine.test_embedding_model()

# Verify token count
stats = manager.get_statistics()
print(f"Active tokens: {stats.active_tokens}")
print(f"Archived tokens: {stats.archived_tokens}")
```

**3. Chain integrity failures**
```python
# Diagnose chain issues
integrity_result = manager.verify_chain_integrity()
if not integrity_result.valid:
    print(f"Chain errors: {integrity_result.errors}")
    
# Repair chain if possible
manager.repair_chain_integrity()
```

**4. Performance issues**
```python
# Check memory usage
stats = manager.get_performance_stats()
print(f"Memory usage: {stats.memory_mb}MB")
print(f"Active tokens: {stats.active_tokens}")

# Optimize if needed
if stats.active_tokens > 1000:
    manager.trigger_archival()
```

### Debug Mode

```bash
# Enable comprehensive debugging
export PMM_DEBUG=1
export PMM_LOG_LEVEL=DEBUG

# Run with verbose output
pmm-cli stats --next-stage --verbose
```

### Performance Monitoring

```python
# Monitor performance metrics
stats = manager.get_performance_stats()
print(f"Tokenization rate: {stats.tokens_per_second}")
print(f"Recall latency: {stats.avg_recall_latency_ms}ms")
print(f"Archive compression: {stats.compression_ratio:.1f}x")
print(f"Chain verification time: {stats.chain_verify_ms}ms")
```

#### Metrics and Drift

- **Evidence weighting**: Drift magnitude scales with observed behavioral signals and an evidence weight. Boost factor ranges within a bounded multiplier.
- **Freshness guard**: N-gram overlap is measured against recent outputs. If above threshold, a controlled re-roll with style jitter is attempted and logged.
- **Clamping near bounds**: Traits at/near boundaries (e.g., conscientiousness=1.0) suppress small deltas; logs indicate clamped traits.
- **Commitments lifecycle**: Commitments may auto-close each turn based on matching signals; new commitments are minted when extracted from reflections.
- **Debug logs**: With `PMM_DEBUG=1`, you’ll see evidence-weighted boost, applied weights, clamp reasons, and freshness re-roll notices for transparency.

### Example outputs

Per-turn metrics (from `duel.py`):

```
A: Let's verify integrity before heavy recall to keep results consistent.
   · commitments: ['c207']  · ΔBig5: {'conscientiousness': +0.01}  · overlap(prev): 12.4%  · overlap(8): 5.9%

B: Agreed. I'll add events and run verify right after.
   · commitments: []  · ΔBig5: {'openness': +0.01}  · overlap(prev): 10.3%  · overlap(8): 7.0%
```

Debug logs when `PMM_DEBUG=1`:

```
[DRIFT] evidence_weight=0.54 boost=1.12 applied=+0.008 -> conscientiousness
[FRESHNESS] overlap(prev)=33.2% > 30.0% threshold -> re-roll with jitter
[CLAMP] agreeableness delta +0.003 suppressed (near bound)
[COMMIT] auto-close c199 (matched: produce summary)
```

## Integration Examples

### LangChain Integration

```python
from pmm.enhanced_manager import EnhancedSelfModelManager
from langchain.memory import ConversationBufferMemory

class PMMEnhancedMemory(ConversationBufferMemory):
    def __init__(self, pmm_manager: EnhancedSelfModelManager):
        super().__init__()
        self.pmm = pmm_manager
    
    def save_context(self, inputs: dict, outputs: dict):
        super().save_context(inputs, outputs)
        
        # Add to PMM with tokenization
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")
        
        self.pmm.add_event(f"User said: {user_input}")
        self.pmm.add_thought(f"I responded: {ai_output}")
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pmm.enhanced_manager import EnhancedSelfModelManager

app = FastAPI()
manager = EnhancedSelfModelManager(enable_next_stage=True)

@app.post("/add-memory")
async def add_memory(content: str, memory_type: str):
    try:
        if memory_type == "event":
            token_id = manager.add_event(content)
        elif memory_type == "thought":
            token_id = manager.add_thought(content)
        else:
            raise ValueError("Invalid memory type")
        
        return {"token_id": token_id, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recall")
async def recall_memories(cue: str, max_results: int = 10):
    results = manager.recall_memories(cue, max_results)
    return {
        "results": [
            {
                "content": r.content,
                "similarity": r.similarity_score,
                "token_id": r.token_id
            }
            for r in results
        ]
    }
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pmm-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pmm-service
  template:
    metadata:
      labels:
        app: pmm-service
    spec:
      containers:
      - name: pmm
        image: pmm:latest
        ports:
        - containerPort: 8000
        env:
        - name: PMM_DEBUG
          value: "0"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: pmm-secrets
              key: openai-key
        volumeMounts:
        - name: pmm-storage
          mountPath: /app/data
      volumes:
      - name: pmm-storage
        persistentVolumeClaim:
          claimName: pmm-pvc
```

This implementation guide covers the PMM next-stage architecture and practical usage, aligned to the shipped implementation.
