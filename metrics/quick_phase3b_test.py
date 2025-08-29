#!/usr/bin/env python3
"""
Quick Phase 3B Validation Test
"""

import os
import sys

# Add PMM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmm.reflection import _validate_insight_references
from pmm.self_model_manager import SelfModelManager


def main():
    print("🎯 Quick Phase 3B Validation")
    print("=" * 40)

    # Test 1: Direct validation function test
    print("🧪 Testing insight validation patterns...")

    # Create a minimal manager for testing
    try:
        mgr = SelfModelManager("test_quick.json")

        # Test cases
        test_cases = [
            ("Based on event ev1, I will improve.", True),
            ("Looking at ev2, I notice patterns.", True),
            ("I will try to be better.", False),
            ("My approach is stable.", False),
        ]

        passed = 0
        for content, expected in test_cases:
            is_accepted, refs = _validate_insight_references(content, mgr)
            status = "✅" if is_accepted == expected else "❌"
            print(f"  {status} '{content}' -> {is_accepted} (refs: {refs})")
            if is_accepted == expected:
                passed += 1

        print(f"\n📊 Validation: {passed}/{len(test_cases)} passed")

        # Cleanup
        if os.path.exists("test_quick.json"):
            os.unlink("test_quick.json")

        if passed == len(test_cases):
            print("🎉 Phase 3B reflection hygiene is working!")
            print("✅ Insights with event references are accepted")
            print("✅ Insights without references are marked inert")
            return True
        else:
            print("⚠️ Some validation tests failed")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
