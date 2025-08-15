# Phase 3C Testing Guide

This guide provides comprehensive testing strategies for Phase 3C features, from unit tests to production validation.

## Testing Levels

### 1. Foundation Tests ✅ (Already Complete)
**Status**: All passing (4/4 tests)
**Command**: `python3 tests/test_phase3c_foundation.py`

Tests core component functionality:
- EmergenceAnalyzer.commitment_close_rate with real SQLite integration
- AdaptiveTrigger decision logic (time, events, emergence factors)
- SemanticAnalyzer novelty detection and similarity scoring
- MetaReflectionAnalyzer pattern analysis

### 2. Integration Tests (Recommended Next Step)

#### Quick Integration Test
```bash
cd /home/scott/Documents/Projects/Business-Development/persistent-mind-model
source venv/bin/activate
python3 tests/test_phase3c_integration.py
```

Tests full Phase 3C pipeline with mocked dependencies.

#### Live Integration Test (Requires OpenAI API Key)
```bash
export OPENAI_API_KEY="your-api-key-here"
python3 tests/test_phase3c_integration.py
```

### 3. Manual Conversation Testing

#### A. Simple Chat Test
```bash
# Start PMM chat with Phase 3C features
export OPENAI_API_KEY="your-api-key"
export PMM_REFLECTION_CADENCE="0.1"  # Reflect every 2.4 hours for testing
python3 -m pmm.chat
```

**Test Scenarios:**
1. **Adaptive Triggers**: Have a conversation with 4+ exchanges, watch for reflection triggers
2. **Commitment Tracking**: Say "I commit to testing Phase 3C thoroughly"
3. **Evidence Events**: Later say "Done: I've completed thorough Phase 3C testing"
4. **Semantic Novelty**: Make similar statements and see if reflections avoid duplication

#### B. Probe API Testing
```bash
# Start probe server
python3 -m pmm.api.probe

# In another terminal, test endpoints:
curl http://localhost:8000/reflection/quality
curl http://localhost:8000/emergence/trends
curl http://localhost:8000/personality/adaptation
curl http://localhost:8000/meta-cognition
```

### 4. Production Validation Testing

#### A. Long-Running Conversation Test
**Goal**: Validate adaptive triggers over extended periods

```bash
# Set realistic reflection cadence
export PMM_REFLECTION_CADENCE="7"  # 7 days like production
export OPENAI_API_KEY="your-api-key"

# Have conversations over several days
python3 -m pmm.chat
```

**What to Monitor:**
- Reflections should trigger based on time + event accumulation + emergence signals
- High emergence (IAS/GAS > 0.7) should delay reflections
- Low emergence (IAS/GAS < 0.3) should accelerate reflections

#### B. Semantic Deduplication Test
**Goal**: Verify reflection hygiene prevents repetitive insights

**Test Process:**
1. Have similar conversations multiple times
2. Check that reflections don't repeat identical insights
3. Validate novelty scoring in `/reflection/quality` endpoint

#### C. Meta-Reflection Validation
**Goal**: Confirm AI self-awareness improves over time

**Test Process:**
1. Generate 10+ reflections over time
2. Check `/meta-cognition` endpoint for pattern recognition
3. Verify recommendations improve reflection quality

## Testing Checklist

### ✅ Foundation Components
- [x] EmergenceAnalyzer real SQLite integration
- [x] AdaptiveTrigger decision logic
- [x] SemanticAnalyzer novelty detection
- [x] MetaReflectionAnalyzer pattern analysis

### 🔄 Integration Testing
- [ ] Adaptive triggers in live PMM conversations
- [ ] Semantic novelty in reflection hygiene
- [ ] Enhanced probe endpoints with real data
- [ ] Multi-LLM embedding provider switching

### 🎯 Production Validation
- [ ] Extended conversation testing (multi-day)
- [ ] Commitment-evidence loop validation
- [ ] Reflection quality improvement over time
- [ ] Probe API performance under load

## Common Issues & Solutions

### Issue: "OpenAI API key not set"
**Solution**: Export your API key: `export OPENAI_API_KEY="sk-..."`

### Issue: "Module not found" errors
**Solution**: 
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Reflections not triggering
**Solutions**:
1. Check reflection cadence: `export PMM_REFLECTION_CADENCE="0.1"` (for testing)
2. Verify adaptive trigger logic in debug output
3. Check emergence scores aren't too high (delaying reflections)

### Issue: Semantic analysis not working
**Solutions**:
1. Verify OpenAI API key for embeddings
2. Check embedding provider configuration
3. Test with mock provider first

## Performance Benchmarks

### Expected Performance
- **Reflection Trigger Decision**: < 10ms
- **Semantic Novelty Scoring**: < 500ms (with OpenAI API)
- **Emergence Score Calculation**: < 50ms
- **Probe Endpoint Response**: < 200ms

### Memory Usage
- **SQLite Database Growth**: ~1KB per event
- **Embedding Cache**: ~4KB per cached embedding
- **In-Memory Structures**: < 10MB for typical usage

## Debugging Tools

### 1. Database Inspection
```bash
sqlite3 pmm.db
.tables
SELECT kind, content, meta FROM events ORDER BY id DESC LIMIT 10;
```

### 2. Reflection Trigger Debug
Add debug prints in `pmm/langchain_memory.py` around line 570:
```python
print(f"DEBUG: Adaptive trigger decision: {should_reflect} ({reason})")
print(f"DEBUG: IAS={ias}, GAS={gas}, Events since reflection={events_since_reflection}")
```

### 3. Probe API Debug
```bash
# Check probe server logs
python3 -m pmm.api.probe --debug
```

### 4. Semantic Analysis Debug
```python
from pmm.semantic_analysis import get_semantic_analyzer
analyzer = get_semantic_analyzer()
novelty = analyzer.semantic_novelty_score("test text", ["reference text"])
print(f"Novelty score: {novelty}")
```

## Success Criteria

### Phase 3C is working correctly if:

1. **Adaptive Triggers**: Reflections occur based on time + events + emergence (not just every 4 events)
2. **Semantic Novelty**: Similar reflections get low novelty scores (< 0.5), novel ones get high scores (> 0.7)
3. **Meta-Reflection**: `/meta-cognition` endpoint shows pattern recognition and improvement recommendations
4. **Emergence Integration**: IAS/GAS scores influence reflection timing appropriately
5. **Probe Transparency**: All enhanced endpoints return meaningful data

### Red Flags (Indicates Issues):
- Reflections still triggering every 4 events exactly
- Identical reflections appearing repeatedly
- Probe endpoints returning empty or error responses
- Semantic novelty scores always the same regardless of content
- No meta-cognitive insights after 10+ reflections

## Next Steps After Testing

1. **If tests pass**: Deploy to production with monitoring
2. **If issues found**: Use debugging tools above to isolate problems
3. **Performance optimization**: Add caching, async processing for embeddings
4. **Extended validation**: Multi-week testing with real usage patterns

---

**Quick Start Testing Command:**
```bash
cd /home/scott/Documents/Projects/Business-Development/persistent-mind-model
source venv/bin/activate
python3 tests/test_phase3c_foundation.py && echo "✅ Foundation tests passed - ready for integration testing!"
```
