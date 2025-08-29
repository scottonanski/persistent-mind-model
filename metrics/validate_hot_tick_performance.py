#!/usr/bin/env python3
"""
Step 7: Runbook scenarios to validate end-to-end hot tick performance.

This script creates specific test scenarios to validate that the enhanced bandit system
properly handles hot ticks with improved reflection acceptance and reward attribution.
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pmm.langchain_memory import PersistentMindMemory
from pmm.config.models import ModelConfig
from pmm.emergence import compute_emergence_scores


class HotTickValidator:
    """Validates hot tick performance with specific test scenarios."""
    
    def __init__(self):
        self.results = []
        self.test_start = datetime.now(timezone.utc)
        
    def setup_hot_environment(self):
        """Configure environment for hot tick testing."""
        os.environ.update({
            "PMM_BANDIT_ENABLED": "1",
            "PMM_TELEMETRY": "1",
            "PMM_SAFETY_FIRST_HOT_ALWAYS": "1",
            "PMM_REFLECT_DEDUP_FLOOR_HOT": "0.88",
            "PMM_REFLECT_MIN_TOKENS_HOT": "45",
            "PMM_BANDIT_HOT_REFLECT_BOOST": "0.3",
            "PMM_BANDIT_HOT_CONTINUE_PENALTY": "0.1",
        })
        
    def create_hot_scenario(self, memory_manager, scenario_name: str):
        """Create a scenario designed to trigger hot ticks."""
        print(f"\n=== {scenario_name} ===")
        
        # Simulate high-commitment activity to boost GAS
        commitments = [
            "I will analyze the user's technical requirements thoroughly",
            "I will provide detailed code examples with explanations", 
            "I will validate all solutions before presenting them",
            "I will track progress and update status regularly"
        ]
        
        for commitment in commitments:
            memory_manager.add_ai_message(f"Commitment: {commitment}")
            time.sleep(0.1)  # Brief pause between commitments
            
        # Add evidence of completion to boost close rates
        evidence_items = [
            "Done: Completed technical analysis of requirements",
            "Finished: Provided comprehensive code examples",
            "Delivered: Validated all proposed solutions"
        ]
        
        for evidence in evidence_items:
            memory_manager.add_ai_message(f"Evidence: {evidence}")
            time.sleep(0.1)
            
        return self.measure_hot_strength()
        
    def measure_hot_strength(self):
        """Measure current hot strength and emergence scores."""
        try:
            scores = compute_emergence_scores(window=15)
            gas = float(scores.get("GAS", 0.0) or 0.0)
            close_rate = float(scores.get("commit_close_rate", 0.0) or 0.0)
            
            # Use same hot_strength computation as bandit
            from pmm.policy.bandit import compute_hot_strength
            hot_strength = compute_hot_strength(gas, close_rate)
            
            return {
                "gas": gas,
                "close_rate": close_rate,
                "hot_strength": hot_strength,
                "is_hot": hot_strength >= 0.5
            }
        except Exception as e:
            print(f"Error measuring hot strength: {e}")
            return {"gas": 0.0, "close_rate": 0.0, "hot_strength": 0.0, "is_hot": False}
    
    def test_reflection_in_hot_context(self, memory_manager):
        """Test reflection behavior during hot ticks."""
        print("\n--- Testing Reflection in Hot Context ---")
        
        # Trigger reflection attempt
        reflection_prompt = "Based on recent commitments and evidence, I should reflect on my performance patterns. Next: I will focus on maintaining high completion rates while ensuring quality standards."
        
        # Attempt reflection
        try:
            result = memory_manager._auto_reflect(
                active_model_config=ModelConfig().model_dump(),
                force_reflect=True
            )
            
            return {
                "reflection_attempted": True,
                "reflection_accepted": result is not None,
                "reflection_content": str(result) if result else None
            }
        except Exception as e:
            print(f"Reflection test error: {e}")
            return {
                "reflection_attempted": True,
                "reflection_accepted": False,
                "error": str(e)
            }
    
    def run_scenario(self, scenario_name: str, target_hot_strength: float = 0.7):
        """Run a complete hot tick validation scenario."""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"Target Hot Strength: {target_hot_strength}")
        print(f"{'='*60}")
        
        # Initialize memory manager
        memory_manager = PersistentMindMemory()
        
        # Measure baseline
        baseline = self.measure_hot_strength()
        print(f"Baseline - GAS: {baseline['gas']:.3f}, Close Rate: {baseline['close_rate']:.3f}, Hot Strength: {baseline['hot_strength']:.3f}")
        
        # Create hot scenario
        hot_metrics = self.create_hot_scenario(memory_manager, scenario_name)
        print(f"Hot Context - GAS: {hot_metrics['gas']:.3f}, Close Rate: {hot_metrics['close_rate']:.3f}, Hot Strength: {hot_metrics['hot_strength']:.3f}")
        
        # Test reflection behavior
        reflection_result = self.test_reflection_in_hot_context(memory_manager)
        
        # Collect results
        scenario_result = {
            "scenario": scenario_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline": baseline,
            "hot_context": hot_metrics,
            "reflection_test": reflection_result,
            "success_criteria": {
                "achieved_hot_context": hot_metrics["hot_strength"] >= 0.5,
                "reflection_attempted": reflection_result["reflection_attempted"],
                "reflection_accepted": reflection_result.get("reflection_accepted", False),
                "target_hot_strength_met": hot_metrics["hot_strength"] >= target_hot_strength
            }
        }
        
        self.results.append(scenario_result)
        
        # Print summary
        success = scenario_result["success_criteria"]
        print(f"\n--- Scenario Results ---")
        print(f"✅ Hot Context Achieved: {success['achieved_hot_context']}")
        print(f"✅ Reflection Attempted: {success['reflection_attempted']}")
        print(f"✅ Reflection Accepted: {success['reflection_accepted']}")
        print(f"✅ Target Hot Strength Met: {success['target_hot_strength_met']}")
        
        overall_success = all(success.values())
        print(f"\n🎯 Overall Success: {overall_success}")
        
        return scenario_result
    
    def generate_report(self):
        """Generate comprehensive validation report."""
        report = {
            "validation_run": {
                "start_time": self.test_start.isoformat(),
                "end_time": datetime.now(timezone.utc).isoformat(),
                "total_scenarios": len(self.results)
            },
            "scenarios": self.results,
            "summary": {
                "total_scenarios": len(self.results),
                "successful_scenarios": sum(1 for r in self.results if all(r["success_criteria"].values())),
                "hot_contexts_achieved": sum(1 for r in self.results if r["success_criteria"]["achieved_hot_context"]),
                "reflections_accepted": sum(1 for r in self.results if r["success_criteria"]["reflection_accepted"])
            }
        }
        
        # Save report
        report_file = Path(__file__).parent / f"hot_tick_validation_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Validation report saved to: {report_file}")
        return report


def main():
    """Run hot tick validation scenarios."""
    print("🔥 PMM Hot Tick Performance Validation")
    print("=" * 50)
    
    validator = HotTickValidator()
    validator.setup_hot_environment()
    
    # Run validation scenarios
    scenarios = [
        ("High Commitment Activity", 0.6),
        ("Rapid Evidence Completion", 0.7),
        ("Mixed Commitment-Evidence Chain", 0.8)
    ]
    
    for scenario_name, target_strength in scenarios:
        try:
            validator.run_scenario(scenario_name, target_strength)
            time.sleep(2)  # Brief pause between scenarios
        except Exception as e:
            print(f"❌ Scenario '{scenario_name}' failed: {e}")
    
    # Generate final report
    report = validator.generate_report()
    
    # Print final summary
    summary = report["summary"]
    print(f"\n🎯 FINAL VALIDATION SUMMARY")
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Successful Scenarios: {summary['successful_scenarios']}")
    print(f"Hot Contexts Achieved: {summary['hot_contexts_achieved']}")
    print(f"Reflections Accepted: {summary['reflections_accepted']}")
    
    success_rate = summary['successful_scenarios'] / summary['total_scenarios'] if summary['total_scenarios'] > 0 else 0
    print(f"Overall Success Rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("✅ Hot tick performance validation PASSED")
        return 0
    else:
        print("❌ Hot tick performance validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
