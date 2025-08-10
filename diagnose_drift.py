#!/usr/bin/env python3
"""
Diagnostic script to analyze why trait drift stopped working in the daily evolution.
"""

from pmm.self_model_manager import SelfModelManager
import json
from pathlib import Path

def analyze_agent(agent_path):
    """Analyze an agent's state and drift configuration."""
    print(f"\n=== Analyzing {agent_path} ===")
    
    if not Path(agent_path).exists():
        print(f"❌ Agent file {agent_path} not found")
        return
    
    mgr = SelfModelManager(agent_path)
    model = mgr.model
    
    # Check drift configuration
    print(f"📊 Drift Config:")
    print(f"  max_delta_per_reflection: {model.drift_config.max_delta_per_reflection}")
    print(f"  bounds: {model.drift_config.bounds.min} - {model.drift_config.bounds.max}")
    print(f"  notes: {model.drift_config.notes}")
    
    # Check behavioral patterns
    print(f"\n🔍 Behavioral Patterns:")
    patterns = model.self_knowledge.behavioral_patterns
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}")
    
    # Check commitment metrics
    print(f"\n📋 Commitment Status:")
    try:
        metrics = mgr.commitment_tracker.get_commitment_metrics()
        print(f"  Open: {metrics.get('commitments_open', 0)}")
        print(f"  Closed: {metrics.get('commitments_closed', 0)}")
        print(f"  Close rate: {metrics.get('close_rate', 0):.2%}")
    except Exception as e:
        print(f"  ❌ Error getting commitment metrics: {e}")
    
    # Check recent insights
    print(f"\n💭 Recent Insights ({len(model.self_knowledge.insights)} total):")
    for insight in model.self_knowledge.insights[-3:]:
        print(f"  {insight.id}: {insight.content[:100]}...")
        if hasattr(insight, 'references') and insight.references:
            commitments = insight.references.get('commitments', [])
            if commitments:
                print(f"    └─ Commitments: {commitments}")
    
    # Check Big Five traits
    print(f"\n🧠 Current Big Five Traits:")
    b5 = model.personality.traits.big5
    print(f"  Openness: {b5.openness.score:.3f}")
    print(f"  Conscientiousness: {b5.conscientiousness.score:.3f}")
    print(f"  Extraversion: {b5.extraversion.score:.3f}")
    print(f"  Agreeableness: {b5.agreeableness.score:.3f}")
    print(f"  Neuroticism: {b5.neuroticism.score:.3f}")
    
    # Test drift calculation
    print(f"\n🔄 Testing Drift Calculation:")
    try:
        # Calculate evidence-weighted signals
        patterns = model.self_knowledge.behavioral_patterns
        commitment_metrics = mgr.commitment_tracker.get_commitment_metrics()
        
        exp_count = patterns.get("experimentation", 0)
        align_count = patterns.get("user_goal_alignment", 0)
        close_rate = commitment_metrics.get('close_rate', 0)
        
        exp_delta = max(0, exp_count - 3)
        align_delta = max(0, align_count - 2)
        close_rate_delta = max(0, close_rate - 0.3)
        
        signals = exp_delta + align_delta + close_rate_delta
        evidence_weight = min(1, signals / 3)
        
        print(f"  Experimentation delta: {exp_delta} (count: {exp_count})")
        print(f"  Alignment delta: {align_delta} (count: {align_count})")
        print(f"  Close rate delta: {close_rate_delta:.3f} (rate: {close_rate:.2%})")
        print(f"  Total signals: {signals:.3f}")
        print(f"  Evidence weight: {evidence_weight:.3f}")
        
        if evidence_weight > 0.3:
            boost_factor = 1 + (0.5 * evidence_weight)
            print(f"  🚀 Would apply boost factor: {boost_factor:.2f}x")
        else:
            print(f"  ⏸️  No boost (evidence weight < 0.3)")
            
    except Exception as e:
        print(f"  ❌ Error calculating drift: {e}")

def main():
    print("🔧 PMM Drift Diagnostic Tool")
    print("=" * 50)
    
    # Analyze both agents
    analyze_agent("mind_a.json")
    analyze_agent("mind_b.json")
    
    # Check for common issues
    print(f"\n🔍 Common Issues Check:")
    
    # Check if OpenAI API key is set
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
    else:
        print("✅ OPENAI_API_KEY is configured")
    
    # Check if agents have sufficient activity for drift
    print("\n📈 Drift Trigger Analysis:")
    for agent_path in ["mind_a.json", "mind_b.json"]:
        if Path(agent_path).exists():
            mgr = SelfModelManager(agent_path)
            patterns = mgr.model.self_knowledge.behavioral_patterns
            total_activity = sum(patterns.values())
            print(f"  {agent_path}: {total_activity} total pattern signals")
            if total_activity < 10:
                print(f"    ⚠️  Low activity may prevent drift")

if __name__ == "__main__":
    main()
