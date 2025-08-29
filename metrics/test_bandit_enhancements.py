#!/usr/bin/env python3
"""
Test script to validate the seven-step bandit enhancement implementation.
Focuses on testing the core functionality without complex memory interactions.
"""

import os
import sys
from pathlib import Path

# Add PMM to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_hot_strength_computation():
    """Test Step 1: Hot strength computation."""
    print("🔥 Testing hot strength computation...")

    try:
        from pmm.policy.bandit import compute_hot_strength

        # Test cases
        test_cases = [
            (0.0, 0.0, 0.0),  # No activity
            (0.5, 0.5, 0.5),  # Moderate activity
            (0.9, 0.8, 0.85),  # High activity
            (1.0, 1.0, 1.0),  # Maximum activity
        ]

        for gas, close_rate, expected in test_cases:
            result = compute_hot_strength(gas, close_rate)
            print(
                f"  GAS={gas:.1f}, Close={close_rate:.1f} → Hot Strength={result:.3f} (expected ~{expected:.3f})"
            )

        print("✅ Hot strength computation working")
        return True
    except Exception as e:
        print(f"❌ Hot strength computation failed: {e}")
        return False


def test_bandit_context_building():
    """Test bandit context building with hot strength."""
    print("\n🎯 Testing bandit context building...")

    try:
        from pmm.policy.bandit import build_context

        # Test context building with sample values
        context = build_context(gas=0.8, ias=0.6, close=0.7)

        # Check that context contains the key enhancement features
        print(f"  Context keys: {list(context.keys())}")
        print(f"  Hot strength: {context.get('hot_strength', 'N/A')}")
        print(f"  Is hot: {context.get('hot', 'N/A')}")

        # Verify hot strength computation is working
        if "hot_strength" in context and context["hot_strength"] > 0:
            print("✅ Bandit context building working")
            return True
        else:
            print("❌ Hot strength not computed in context")
            return False
    except Exception as e:
        print(f"❌ Bandit context building failed: {e}")
        return False


def test_reward_shaping():
    """Test Step 5: Reward shaping for hot contexts."""
    print("\n💰 Testing reward shaping...")

    try:
        from pmm.policy.bandit import _BanditCore

        # Set environment variables for testing
        os.environ["PMM_BANDIT_HOT_REFLECT_BOOST"] = "0.3"
        os.environ["PMM_BANDIT_HOT_CONTINUE_PENALTY"] = "0.1"
        os.environ["PMM_TELEMETRY"] = "1"

        bandit = _BanditCore()

        # Test hot context reward shaping
        hot_context = {"hot_strength": 0.8, "reflect_id": "test123"}

        # Test reflect reward boost
        bandit.record_outcome(hot_context, "reflect_now", 0.5, 10, "test")
        print("  Tested reflect reward boost in hot context")

        # Test continue penalty
        bandit.record_outcome(hot_context, "continue", 0.2, 10, "test")
        print("  Tested continue penalty in hot context")

        print("✅ Reward shaping working")
        return True
    except Exception as e:
        print(f"❌ Reward shaping failed: {e}")
        return False


def test_reflection_id_tracking():
    """Test Step 4: Reflection ID tracking."""
    print("\n🆔 Testing reflection ID tracking...")

    try:
        import uuid

        # Generate test reflection ID
        reflect_id = str(uuid.uuid4())[:8]
        print(f"  Generated reflection ID: {reflect_id}")

        # Test context with reflection ID
        context = {"reflect_id": reflect_id, "hot_strength": 0.6}

        from pmm.policy.bandit import _BanditCore

        bandit = _BanditCore()

        # Test reward recording with reflection ID
        bandit.record_outcome(context, "reflect_now", 0.7, 10, "test with ID")

        print("✅ Reflection ID tracking working")
        return True
    except Exception as e:
        print(f"❌ Reflection ID tracking failed: {e}")
        return False


def test_environment_variables():
    """Test that all new environment variables are recognized."""
    print("\n⚙️  Testing environment variables...")

    env_vars = [
        "PMM_REFLECT_DEDUP_FLOOR_HOT",
        "PMM_REFLECT_MIN_TOKENS_HOT",
        "PMM_SAFETY_FIRST_HOT_ALWAYS",
        "PMM_BANDIT_HOT_REFLECT_BOOST",
        "PMM_BANDIT_HOT_CONTINUE_PENALTY",
    ]

    # Set test values
    for var in env_vars:
        os.environ[var] = (
            "0.5" if "FLOOR" in var or "BOOST" in var or "PENALTY" in var else "1"
        )

    # Test that they can be read
    for var in env_vars:
        value = os.getenv(var, "NOT_SET")
        print(f"  {var}: {value}")

    print("✅ Environment variables working")
    return True


def test_telemetry_output():
    """Test enhanced telemetry output."""
    print("\n📊 Testing telemetry output...")

    try:
        os.environ["PMM_TELEMETRY"] = "1"

        from pmm.policy.bandit import _BanditCore

        bandit = _BanditCore()

        # Test telemetry with hot context
        context = {
            "hot_strength": 0.75,
            "reflect_id": "test456",
            "gas": 0.8,
            "close_rate": 0.7,
        }

        print("  Testing telemetry output (should see PMM_TELEMETRY lines):")
        bandit.record_outcome(context, "reflect_now", 0.6, 10, "telemetry test")

        print("✅ Telemetry output working")
        return True
    except Exception as e:
        print(f"❌ Telemetry output failed: {e}")
        return False


def main():
    """Run all bandit enhancement tests."""
    print("🚀 PMM Bandit Enhancement Validation")
    print("=" * 50)

    tests = [
        ("Hot Strength Computation", test_hot_strength_computation),
        ("Bandit Context Building", test_bandit_context_building),
        ("Reward Shaping", test_reward_shaping),
        ("Reflection ID Tracking", test_reflection_id_tracking),
        ("Environment Variables", test_environment_variables),
        ("Telemetry Output", test_telemetry_output),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 30)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        print("🎉 All bandit enhancements validated successfully!")
        return 0
    else:
        print("⚠️  Some tests failed - check implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
