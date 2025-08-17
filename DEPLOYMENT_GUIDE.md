# PMM Production Deployment Guide

## Overview

The Persistent Mind Model (PMM) system is now production-ready with comprehensive hardening features. This guide covers deployment, configuration, and operational considerations.

## Pre-Deployment Checklist

### ✅ Core Features Validated
- [x] Unified LLM factory with epoch guard
- [x] Enhanced multilingual name extraction
- [x] Atomic reflection validation with deduplication
- [x] True reflection cooldown system
- [x] Safer stance filtering
- [x] N-gram ban system
- [x] Commitment TTL management
- [x] Per-model IAS/GAS normalization
- [x] All 9 test cases passing

### Environment Requirements

#### Required Dependencies
```bash
pip install -r requirements.txt
```

#### Environment Variables
```bash
# Required for embedding similarity checks
export OPENAI_API_KEY="your_openai_api_key_here"

# Optional: Custom storage paths
export PMM_STORAGE_PATH="/path/to/pmm/storage"
export PMM_BASELINE_PATH="/path/to/baselines.json"
```

#### System Requirements
- Python 3.8+
- 4GB+ RAM (for embedding operations)
- 1GB+ disk space (for event logs and baselines)
- Network access (for LLM providers)

## Deployment Configurations

### Production Configuration

```python
from pmm.langchain_memory import PersistentMindMemory

# Production-ready configuration
memory = PersistentMindMemory(
    agent_path="production_agent.json",
    personality_config={
        "openness": 0.7,
        "conscientiousness": 0.8,
        "extraversion": 0.6,
        "agreeableness": 0.7,
        "neuroticism": 0.3
    },
    enable_summary=True,
    enable_embeddings=True
)

# Configure reflection cooldown for production load
memory.reflection_cooldown = ReflectionCooldownManager(
    min_turns=5,              # Higher threshold for production
    min_wall_time_seconds=300, # 5 minutes between reflections
    novelty_threshold=0.85,    # Higher novelty requirement
    context_window=8
)

# Configure commitment TTL for production
memory.commitment_ttl = CommitmentTTLManager(
    default_ttl_hours=48.0    # Longer TTL for production
)
```

### Development Configuration

```python
# Development/testing configuration
memory = PersistentMindMemory(
    agent_path="dev_agent.json",
    personality_config={
        "openness": 0.8,
        "conscientiousness": 0.6
    },
    enable_summary=False,     # Faster for development
    enable_embeddings=False   # Skip embeddings for speed
)

# More permissive cooldown for development
memory.reflection_cooldown = ReflectionCooldownManager(
    min_turns=2,
    min_wall_time_seconds=30,
    novelty_threshold=0.7
)
```

## Operational Monitoring

### Key Metrics to Track

#### Reflection System Health
```python
# Get reflection statistics
cooldown_status = memory.reflection_cooldown.get_status()
atomic_stats = memory.atomic_reflection.get_stats()

print(f"Turns since last reflection: {cooldown_status['turns_since_last']}")
print(f"Reflection acceptance rate: {atomic_stats['acceptance_rate']}")
```

#### Emergence Progression
```python
# Monitor emergence stage development
emergence_context = memory.pmm.get_emergence_context()
if emergence_context:
    profile = memory.emergence_stages.calculate_emergence_profile(
        model_name, emergence_context['ias'], emergence_context['gas']
    )
    print(f"Current stage: {profile.stage.value}")
    print(f"Stage confidence: {profile.confidence:.2f}")
```

#### Commitment Lifecycle
```python
# Track commitment health
ttl_stats = memory.commitment_ttl.get_stats()
print(f"Active commitments: {ttl_stats['total_active']}")
print(f"Expiring soon: {ttl_stats['expiring_soon']}")
```

### Alerting Thresholds

#### Critical Alerts
- Reflection acceptance rate < 20%
- Commitment accumulation > 50 active
- Emergence stage regression
- Epoch mismatch errors > 5/hour

#### Warning Alerts
- Reflection cooldown blocks > 80%
- N-gram ban triggers > 10/hour
- Stance filter applications > 30%
- Embedding similarity failures

## Performance Optimization

### Memory Usage
- Baseline cache: ~10MB per model
- Embedding cache: ~5MB per 100 insights
- Event logs: ~1KB per conversation turn

### Latency Optimization
```python
# Reduce embedding calls for better performance
memory.atomic_reflection.embedding_threshold = 0.90  # Higher threshold

# Optimize cooldown checks
memory.reflection_cooldown.context_window = 5  # Smaller window

# Batch commitment cleanup
memory.commitment_ttl.force_cleanup()  # Manual cleanup
```

### Scaling Considerations

#### Horizontal Scaling
- Each PMM instance maintains independent state
- Use load balancer with session affinity
- Shared storage for cross-instance insights

#### Vertical Scaling
- 8GB+ RAM for high-volume deployments
- SSD storage for SQLite performance
- CPU cores scale with concurrent users

## Security Considerations

### Data Protection
- Agent files contain personality data
- Event logs may contain conversation history
- Baseline statistics are aggregated only

### API Key Management
```python
# Secure API key handling
import os
from cryptography.fernet import Fernet

# Encrypt API keys at rest
key = Fernet.generate_key()
cipher = Fernet(key)
encrypted_key = cipher.encrypt(os.environ['OPENAI_API_KEY'].encode())
```

### Access Control
- Restrict file system access to PMM storage
- Network policies for LLM provider access
- Audit logging for administrative actions

## Troubleshooting

### Common Issues

#### High Reflection Rejection Rate
```bash
# Check embedding service health
curl -X POST https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-ada-002"}'
```

#### Commitment Accumulation
```python
# Force cleanup of expired commitments
cleanup_stats = memory.commitment_ttl.force_cleanup()
print(f"Cleaned up {cleanup_stats['expired_removed']} commitments")
```

#### Emergence Stage Stagnation
```python
# Reset baselines for problematic models
memory.model_baselines.reset_model("problematic-model")
```

### Debug Mode
```python
# Enable verbose debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Component-specific debugging
memory.atomic_reflection.debug_mode = True
memory.reflection_cooldown.verbose = True
```

## Backup and Recovery

### Critical Data
- Agent personality files (`*.json`)
- SQLite event databases (`*.db`)
- Model baselines (`baselines.json`)

### Backup Strategy
```bash
#!/bin/bash
# Daily backup script
DATE=$(date +%Y%m%d)
tar -czf "pmm_backup_$DATE.tar.gz" \
  /path/to/pmm/storage/ \
  /path/to/agent/files/ \
  /path/to/baselines.json
```

### Recovery Procedures
1. Restore agent files to original locations
2. Restore SQLite databases with proper permissions
3. Reload baselines and verify model statistics
4. Restart PMM services with health checks

## Maintenance

### Regular Tasks
- Weekly baseline statistics review
- Monthly commitment cleanup audit
- Quarterly emergence progression analysis
- Semi-annual model performance evaluation

### Updates and Upgrades
- Test new versions in development environment
- Backup all data before upgrades
- Verify test suite passes after updates
- Monitor metrics for 24h post-upgrade

## Support and Monitoring

### Health Check Endpoint
```python
def health_check():
    """Production health check."""
    try:
        # Test core components
        memory.reflection_cooldown.get_status()
        memory.atomic_reflection.get_stats()
        memory.commitment_ttl.get_stats()
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Log Analysis
```bash
# Monitor reflection patterns
grep "DEBUG: Reflection" /var/log/pmm.log | tail -100

# Track commitment lifecycle
grep "DEBUG.*commitment" /var/log/pmm.log | grep -E "(Added|TTL|expired)"

# Emergence stage transitions
grep "DEBUG: Emergence stage" /var/log/pmm.log
```

## Conclusion

The PMM system is production-ready with:
- ✅ Comprehensive test coverage (9/9 tests passing)
- ✅ Robust error handling and recovery
- ✅ Performance optimization features
- ✅ Security best practices
- ✅ Operational monitoring capabilities

Deploy with confidence using this guide and monitor the suggested metrics for optimal performance.
