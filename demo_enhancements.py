#!/usr/bin/env python3
"""
Demonstration of GPT-5 Enhanced Persistent Mind Model Features

This script showcases the major improvements implemented based on GPT-5's recommendations:
1. Commitment lifecycle management
2. Evidence-weighted drift stabilization  
3. Enhanced n-gram cache for language freshness
4. Improved provenance tracking
5. Pattern-driven behavioral steering
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from pmm.commitments import CommitmentTracker
from pmm.self_model_manager import SelfModelManager
from pmm.model import PersistentMindModel


def demo_commitment_lifecycle():
    """Demonstrate commitment extraction, tracking, and auto-closing."""
    print("🎯 === COMMITMENT LIFECYCLE DEMO ===")
    
    tracker = CommitmentTracker()
    
    # Simulate a series of reflections with commitments
    reflections = [
        "I notice my responses lack specificity. Next: I will include concrete examples in explanations.",
        "My error handling is inconsistent. I will implement systematic validation checks.",
        "I successfully added concrete examples to my last 3 responses. The validation system is working well.",
        "I completed the systematic validation implementation. Done with example improvements."
    ]
    
    for i, reflection in enumerate(reflections, 1):
        print(f"\n📝 Reflection {i}: {reflection}")
        
        # Extract and add commitments
        commitment, ngrams = tracker.extract_commitment(reflection)
        if commitment:
            cid = tracker.add_commitment(reflection, f"in{i}")
            print(f"   ✅ Extracted commitment {cid}: {commitment}")
            print(f"   🔍 Key n-grams: {ngrams[:2]}")
        
        # Auto-close based on completion signals
        closed = tracker.auto_close_from_reflection(reflection)
        if closed:
            print(f"   🔒 Auto-closed commitments: {closed}")
    
    # Show final metrics
    metrics = tracker.get_commitment_metrics()
    print(f"\n📊 Final Commitment Metrics:")
    print(f"   Total: {metrics['commitments_total']}")
    print(f"   Open: {metrics['commitments_open']}")
    print(f"   Closed: {metrics['commitments_closed']}")
    print(f"   Close Rate: {metrics['close_rate']:.1%}")
    
    return tracker


def demo_evidence_weighted_drift():
    """Demonstrate evidence-weighted drift calculations."""
    print("\n⚖️ === EVIDENCE-WEIGHTED DRIFT DEMO ===")
    
    # Create test scenarios
    scenarios = [
        {
            "name": "Low Evidence",
            "experimentation": 2,
            "user_goal_alignment": 1,
            "close_rate": 0.2,
            "expected": "Minimal drift boost"
        },
        {
            "name": "Medium Evidence", 
            "experimentation": 5,
            "user_goal_alignment": 3,
            "close_rate": 0.5,
            "expected": "Moderate drift boost"
        },
        {
            "name": "High Evidence",
            "experimentation": 8,
            "user_goal_alignment": 6,
            "close_rate": 0.8,
            "expected": "Strong drift boost"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🧪 Scenario: {scenario['name']}")
        
        # Calculate evidence-weighted signals (GPT-5's formula)
        exp_delta = max(0, scenario['experimentation'] - 3)
        align_delta = max(0, scenario['user_goal_alignment'] - 2)
        close_rate_delta = max(0, scenario['close_rate'] - 0.3)
        
        signals = exp_delta + align_delta + close_rate_delta
        evidence_weight = min(1, signals / 3)
        boost_factor = 1 + (0.5 * evidence_weight) if evidence_weight > 0.3 else 1.0
        
        print(f"   📈 Experimentation delta: {exp_delta}")
        print(f"   🎯 Goal alignment delta: {align_delta}")
        print(f"   ✅ Close rate delta: {close_rate_delta:.1f}")
        print(f"   🔢 Combined signals: {signals:.1f}")
        print(f"   ⚖️ Evidence weight: {evidence_weight:.2f}")
        print(f"   🚀 Drift boost factor: {boost_factor:.2f}x")
        print(f"   💭 Expected: {scenario['expected']}")


def demo_ngram_freshness():
    """Demonstrate n-gram cache for language freshness."""
    print("\n🔄 === N-GRAM FRESHNESS DEMO ===")
    
    # Simulate recent insights
    recent_insights = [
        "I need to improve my response quality and focus on user needs.",
        "My goal is to enhance user experience through better responses.",
        "I should focus on providing more detailed and helpful information."
    ]
    
    # Build n-gram cache
    ngram_cache = set()
    for insight in recent_insights:
        words = insight.lower().split()
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                gram = ' '.join(words[i:i+n])
                if all(len(w) > 2 for w in words[i:i+n]):
                    ngram_cache.add(gram)
    
    print(f"📚 Built n-gram cache from {len(recent_insights)} recent insights")
    print(f"🔍 Cache contains {len(ngram_cache)} unique n-grams")
    print(f"📝 Sample n-grams: {list(ngram_cache)[:5]}")
    
    # Test new reflection for overlap
    test_reflection = "I need to focus on better user experience and response quality."
    test_words = test_reflection.lower().split()
    test_ngrams = set()
    
    for n in range(2, 5):
        for i in range(len(test_words) - n + 1):
            gram = ' '.join(test_words[i:i+n])
            if all(len(w) > 2 for w in test_words[i:i+n]):
                test_ngrams.add(gram)
    
    overlap = test_ngrams & ngram_cache
    overlap_ratio = len(overlap) / len(test_ngrams) if test_ngrams else 0
    
    print(f"\n🧪 Testing reflection: '{test_reflection}'")
    print(f"📊 Overlap ratio: {overlap_ratio:.2%}")
    print(f"🔄 Overlapping n-grams: {list(overlap)[:3]}")
    
    if overlap_ratio > 0.35:
        print("⚠️ HIGH OVERLAP - Would trigger re-roll with style constraint")
    else:
        print("✅ ACCEPTABLE OVERLAP - Reflection would be accepted")


def demo_pattern_driven_steering():
    """Demonstrate behavioral pattern-driven trait steering."""
    print("\n🎛️ === PATTERN-DRIVEN STEERING DEMO ===")
    
    # Create model with test patterns (use existing model to avoid file creation)
    try:
        mgr = SelfModelManager("persistent_self_model.json")
    except:
        # Fallback: just demonstrate the logic without file I/O
        print("📊 Simulating pattern-driven steering logic:")
        demo_pattern_logic()
        return
    
    # Set up test behavioral patterns
    patterns = {
        "experimentation": 7,
        "user_goal_alignment": 5,
        "calibration": 6,
        "error_correction": 4,
        "source_citation": 3
    }
    
    mgr.model.self_knowledge.behavioral_patterns = patterns
    
    print("🧠 Current behavioral patterns:")
    for pattern, count in patterns.items():
        print(f"   {pattern}: {count}")
    
    # Show how patterns would influence drift
    print("\n🎯 Pattern-driven drift influences:")
    
    if patterns["experimentation"] > 5:
        print("   📈 High experimentation → Boost openness drift")
    
    if patterns["user_goal_alignment"] > 3:
        print("   🎯 Strong goal alignment → Boost conscientiousness drift")
    
    if patterns["calibration"] > 3 and patterns["error_correction"] > 2:
        print("   📉 Good calibration + error correction → Reduce neuroticism")
    
    if patterns["source_citation"] < 4:
        print("   📚 Low source citation → Trigger citation stimulus")
    
    # Show current Big Five scores
    big5 = mgr.get_big5()
    print(f"\n🧬 Current Big Five traits:")
    for trait, score in big5.items():
        print(f"   {trait}: {score:.3f}")


def main():
    """Run all enhancement demonstrations."""
    print("🚀 GPT-5 Enhanced Persistent Mind Model Demo")
    print("=" * 50)
    
    # Run demonstrations
    commitment_tracker = demo_commitment_lifecycle()
    demo_evidence_weighted_drift()
    demo_ngram_freshness()
    demo_pattern_driven_steering()
    
    print("\n🎉 === DEMO COMPLETE ===")
    print("\nKey enhancements implemented:")
    print("✅ Commitment lifecycle with auto-closing")
    print("✅ Evidence-weighted drift stabilization")
    print("✅ N-gram cache for language freshness")
    print("✅ Pattern-driven behavioral steering")
    print("✅ Enhanced provenance tracking")
    
    print(f"\n📊 Final commitment metrics: {commitment_tracker.get_commitment_metrics()}")


def demo_pattern_logic():
    """Fallback demo for pattern-driven steering without file I/O."""
    patterns = {
        "experimentation": 7,
        "user_goal_alignment": 5,
        "calibration": 6,
        "error_correction": 4,
        "source_citation": 3
    }
    
    print("🧠 Test behavioral patterns:")
    for pattern, count in patterns.items():
        print(f"   {pattern}: {count}")
    
    print("\n🎯 Pattern-driven influences:")
    if patterns["experimentation"] > 5:
        print("   📈 High experimentation → Boost openness drift")
    if patterns["user_goal_alignment"] > 3:
        print("   🎯 Strong goal alignment → Boost conscientiousness drift")
    if patterns["calibration"] > 3 and patterns["error_correction"] > 2:
        print("   📉 Good calibration + error correction → Reduce neuroticism")
    if patterns["source_citation"] < 4:
        print("   📚 Low source citation → Trigger citation stimulus")
    
    print("\n🧬 Simulated Big Five trait effects:")
    print("   openness: 0.542 → 0.567 (+0.025 from experimentation)")
    print("   conscientiousness: 0.615 → 0.635 (+0.020 from goal alignment)")
    print("   neuroticism: 0.423 → 0.413 (-0.010 from calibration)")


if __name__ == "__main__":
    main()
