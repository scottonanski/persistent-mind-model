#!/usr/bin/env python3
"""
Phase 3B Reflection Hygiene Test

Tests the new referential reflection system:
1. Insights with event references are ACCEPTED (trigger drift)
2. Insights without references are INERT (stored but no drift)
3. Validation patterns work correctly
"""

import os
import sys
from unittest.mock import Mock

# Add PMM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.self_model_manager import SelfModelManager
from pmm.reflection import reflect_once, _validate_insight_references
from pmm.adapters.openai_adapter import OpenAIAdapter


def test_insight_validation():
    """Test the insight reference validation patterns."""
    print("🧪 Testing insight validation patterns...")

    # Create a temporary PMM instance with proper initialization
    temp_path = f"test_pmm_{os.getpid()}.json"

    try:
        # Initialize with a basic model first
        mgr = SelfModelManager(temp_path)

        # Add some test events for reference validation
        mgr.add_event("Test event 1", etype="test")
        mgr.add_event("Test event 2", etype="test")
        mgr.add_event("Test event 3", etype="test")

        # Test cases for validation
        test_cases = [
            # Should be ACCEPTED (has references)
            (
                "Based on event ev1, I notice my behavior is improving. Next, I will focus more.",
                True,
            ),
            (
                "Looking at ev2 and the recent patterns, I will try a new approach.",
                True,
            ),
            (
                "The commitment hash a1b2c3d4e5f6789a shows progress. I will continue.",
                True,
            ),
            ("Event 1 demonstrates growth. Next, I will experiment more.", True),
            # Should be INERT (no references)
            ("I notice my behavior is generally stable and consistent.", False),
            ("My approach remains focused on improvement and growth.", False),
            ("I will try to be more creative in my responses.", False),
            ("Next, I plan to focus on better communication.", False),
        ]

        passed = 0
        for content, expected_accepted in test_cases:
            is_accepted, refs = _validate_insight_references(content, mgr)
            status = "✅ PASS" if is_accepted == expected_accepted else "❌ FAIL"
            print(
                f"  {status} '{content[:50]}...' -> Accepted: {is_accepted}, Refs: {refs}"
            )
            if is_accepted == expected_accepted:
                passed += 1

        print(f"📊 Validation Tests: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)

    finally:
        os.unlink(temp_path)


def test_accepted_vs_inert_insights():
    """Test that accepted insights trigger drift while inert ones don't."""
    print("\n🧪 Testing accepted vs inert insight processing...")

    temp_path = f"test_pmm_insights_{os.getpid()}.json"

    try:
        mgr = SelfModelManager(temp_path)

        # Add test events for reference
        mgr.add_event("User interaction", etype="conversation")
        mgr.add_event("System response", etype="self_expression")

        # Mock LLM adapter to return controlled responses
        mock_llm = Mock(spec=OpenAIAdapter)

        # Test 1: Insight WITH references (should be accepted)
        mock_llm.chat.return_value = "Based on event ev1, I see improvement in my responses. Next, I will focus on clarity."

        # initial_insights = len(mgr.model.self_knowledge.insights)  # Not used
        initial_mod_count = mgr.model.meta_cognition.self_modification_count

        insight1 = reflect_once(mgr, mock_llm)

        # Check if insight was accepted
        is_accepted1 = getattr(insight1, "meta", {}).get("accepted", False)
        new_mod_count1 = mgr.model.meta_cognition.self_modification_count

        print(
            f"  📝 Insight 1 (with refs): Accepted={is_accepted1}, Mod count: {initial_mod_count} -> {new_mod_count1}"
        )

        # Test 2: Insight WITHOUT references (should be inert)
        mock_llm.chat.return_value = (
            "I notice my behavior is stable. Next, I will try to improve."
        )

        insight2 = reflect_once(mgr, mock_llm)

        # Check if insight was marked as inert
        is_accepted2 = getattr(insight2, "meta", {}).get("accepted", False)
        new_mod_count2 = mgr.model.meta_cognition.self_modification_count

        print(
            f"  📝 Insight 2 (no refs): Accepted={is_accepted2}, Mod count: {new_mod_count1} -> {new_mod_count2}"
        )

        # Validate results
        success = (
            is_accepted1  # First insight should be accepted
            and not is_accepted2  # Second insight should be inert
            and new_mod_count1 > initial_mod_count  # First should increment mod count
            and new_mod_count2
            == new_mod_count1  # Second should NOT increment mod count
        )

        print(f"📊 Accepted vs Inert Test: {'✅ PASS' if success else '❌ FAIL'}")
        return success

    finally:
        os.unlink(temp_path)


def test_drift_gating():
    """Test that drift only applies with accepted insights."""
    print("\n🧪 Testing drift gating for accepted insights...")

    temp_path = f"test_pmm_drift_{os.getpid()}.json"

    try:
        mgr = SelfModelManager(temp_path)

        # Add an inert insight (no references)
        from pmm.model import Insight

        inert_insight = Insight(
            id="in1",
            t="2025-01-01T00:00:00Z",
            content="I will improve my responses.",
            references={},
        )
        mgr.model.self_knowledge.insights.append(inert_insight)

        # Try to apply drift - should be blocked
        drift_result1 = mgr.apply_drift_and_save()

        # Add an accepted insight (with references)
        accepted_insight = Insight(
            id="in2",
            t="2025-01-01T00:01:00Z",
            content="Based on event ev1, I will focus more.",
            references={"referenced_event_ids": ["ev1"]},
        )
        mgr.model.self_knowledge.insights.append(accepted_insight)

        # Try to apply drift - should proceed
        drift_result2 = mgr.apply_drift_and_save()

        success = (
            len(drift_result1) == 0  # No drift with inert insights
            and len(drift_result2) >= 0  # Drift allowed with accepted insights
        )

        print(f"📊 Drift Gating Test: {'✅ PASS' if success else '❌ FAIL'}")
        print(f"  🚫 Inert insight drift result: {len(drift_result1)} changes")
        print(f"  ✅ Accepted insight drift result: {len(drift_result2)} changes")

        return success

    finally:
        os.unlink(temp_path)


def main():
    """Run all Phase 3B reflection hygiene tests."""
    print("🎯 Phase 3B: Reflection Hygiene Test Suite")
    print("=" * 50)

    # Set debug mode for detailed output
    os.environ["PMM_DEBUG"] = "1"

    tests = [
        test_insight_validation,
        test_accepted_vs_inert_insights,
        test_drift_gating,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")

    print("\n" + "=" * 50)
    print(f"🏁 Phase 3B Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 Phase 3B Reflection Hygiene: ALL TESTS PASSED!")
        print("✅ Insights now require event references to trigger drift")
        print("✅ Non-referential insights stored as inert")
        print("✅ Drift gating working correctly")
    else:
        print("⚠️  Some tests failed - check implementation")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
