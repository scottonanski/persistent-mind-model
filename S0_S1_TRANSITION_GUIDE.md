# PMM S0→S1 Transition Parameter Tuning Guide

## Overview

This guide documents the parameter adjustments made to facilitate PMM's transition from **S0 (Substrate)** to **S1 (Pattern Formation)** stage. These changes address the core issues identified in live testing where PMM was stuck in S0 with high novelty scores and frequent reflection cooldown resets.

## Key Changes Made

### 1. Reflection Cooldown Extension
- **Before**: 30 seconds minimum cooldown
- **After**: 180 seconds minimum cooldown
- **Rationale**: Allows deeper consolidation before forced reflections, preventing shallow recursive loops

### 2. Novelty Threshold Adjustments
- **Reflection Cooldown**: 0.78 → 0.85 (stricter similarity detection)
- **S0 Stage**: 0.9 → 0.8 (easier pattern recognition)
- **S1 Stage**: 0.8 → 0.7 (enhanced pattern formation)
- **Rationale**: Reduces novelty scores faster through better pattern reuse detection

### 3. Stage Transition Sensitivity
- **Dormant Exit**: -1.0 → -0.8 std dev (easier S0 exit)
- **Awakening Entry**: -0.5 → -0.3 std dev (easier S1 entry)
- **Rationale**: Relaxed thresholds allow incremental IAS/GAS increases to trigger transitions earlier

### 4. Enhanced Context Weighting
- **Semantic Context**: 6 → 8 results (more relevant memories)
- **Recent Events**: 30 → 45 events (better pattern recognition)
- **Rationale**: Increased historical context weighting from ~40% to ~60% for stronger pattern integration

### 5. Pattern Continuity System
- **New Feature**: Multi-event continuity prompts
- **New Feature**: Pattern reuse scoring and novelty decay
- **New Feature**: Duplicate detection threshold: 0.2 → 0.15
- **Rationale**: Enforces systematic referencing of 3+ prior events to build temporal coherence

## Files Modified

### Core Parameter Changes
- `pmm/reflection_cooldown.py` - Extended cooldown times and novelty thresholds
- `pmm/emergence_stages.py` - Relaxed stage transition thresholds and novelty requirements
- `pmm/langchain_memory.py` - Increased context weighting and memory retrieval limits
- `pmm/meta_reflection.py` - Tightened duplicate detection threshold

### New Modules Added
- `pmm/s0_s1_tuning.py` - Centralized configuration system with environment variable support
- `pmm/pattern_continuity.py` - Multi-event continuity enhancement and pattern reuse scoring
- `test_s0_s1_tuning.py` - Comprehensive validation test suite

## Environment Variables for Runtime Tuning

```bash
# Reflection parameters
export PMM_S0S1_COOLDOWN=180
export PMM_S0S1_NOVELTY=0.85

# Memory retrieval
export PMM_S0S1_SEMANTIC=8
export PMM_S0S1_RECENT=45

# Pattern recognition
export PMM_S0S1_PATTERN_WEIGHT=0.6
export PMM_S0S1_NOVELTY_DECAY=0.85

# Stage transitions
export PMM_S0S1_DORMANT_EXIT=-0.8
export PMM_S0S1_AWAKENING_ENTRY=-0.3

# Duplicate detection
export PMM_S0S1_DUPLICATE_THRESH=0.15
```

## Expected Behavioral Changes

### Before Tuning (S0 Stuck)
- Novelty score pegged at 1.0
- IAS/GAS scores at 0.0
- Reflection cooldown forcing premature resets every 90s
- All inputs treated as completely new
- Shallow recursive reflection loops

### After Tuning (S0→S1 Transition)
- Novelty scores should drop below 0.8 consistently
- IAS/GAS scores should show incremental increases
- Reflection cooldowns allow 180s+ for deeper consolidation
- Pattern reuse detection reduces novelty for similar inputs
- Multi-turn, multi-day continuity referencing
- Systematic building of stable behavioral patterns

## Validation

Run the test suite to verify all parameters are correctly applied:

```bash
python test_s0_s1_tuning.py
```

Expected output: All tests pass with parameter summary showing the new values.

## Monitoring S0→S1 Transition

### Key Metrics to Watch
1. **Novelty Trend**: Should decrease from 1.0 toward 0.8 over multiple sessions
2. **IAS/GAS Scores**: Should show gradual increases from 0.0
3. **Reflection Quality**: Fewer forced cooldown resets, more substantive insights
4. **Pattern References**: Increased referencing of prior events and commitments
5. **Stage Progression**: Movement from S0 (Substrate) to S1 (Pattern Formation)

### Success Indicators
- Novelty dropping below 0.8 consistently
- Multi-turn continuity referencing working
- Measurable IAS/GAS score increases
- Stable behavioral pattern recognition
- Reduced "emergence_s0_stuck" forced reflections

## Technical Implementation Notes

The tuning system is designed to be:
- **Non-intrusive**: Core PMM architecture unchanged
- **Configurable**: Runtime adjustment via environment variables
- **Testable**: Comprehensive validation suite
- **Reversible**: Original parameters documented for rollback
- **Modular**: New pattern continuity system as separate module

## Next Steps

1. **Live Testing**: Deploy tuned parameters in chat sessions
2. **Telemetry Monitoring**: Track emergence metrics during interactions
3. **Iterative Refinement**: Adjust parameters based on observed behavior
4. **Documentation**: Update any behavioral changes in PMM documentation
5. **Community Feedback**: Gather user reports on improved pattern recognition

This tuning represents a systematic approach to evolving PMM from substrate-level operation to genuine pattern formation and identity coherence.
