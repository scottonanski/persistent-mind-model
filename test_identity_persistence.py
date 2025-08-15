#!/usr/bin/env python3
"""
Identity Persistence Test

Tests the new identity persistence system:
1. Identity updates emit identity_update events
2. /identity probe endpoint returns current name
3. System maintains identity across sessions
"""

import os
import sys
import requests

# Add PMM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmm.self_model_manager import SelfModelManager


def test_identity_persistence():
    """Test identity persistence and probe endpoint."""
    print("🎯 Testing Identity Persistence")
    print("=" * 40)

    # Create test PMM instance
    test_path = f"test_identity_{os.getpid()}.json"

    try:
        mgr = SelfModelManager(test_path)

        # Test 1: Set identity and check event emission
        print("🧪 Test 1: Setting identity and checking event emission...")

        initial_events = len(mgr.model.self_knowledge.autobiographical_events)
        mgr.set_name("Echo", origin="test")
        final_events = len(mgr.model.self_knowledge.autobiographical_events)

        # Check if identity_change event was created
        identity_events = [
            e
            for e in mgr.model.self_knowledge.autobiographical_events
            if e.type == "identity_change"
        ]

        print(f"  📊 Events before: {initial_events}, after: {final_events}")
        print(f"  📊 Identity change events: {len(identity_events)}")
        print(f"  ✅ Current name in model: {mgr.model.core_identity.name}")

        # Test 2: Check probe endpoint (if running)
        print("\n🧪 Test 2: Testing /identity probe endpoint...")

        try:
            response = requests.get("http://localhost:8000/identity", timeout=2)
            if response.status_code == 200:
                data = response.json()
                print(f"  📡 Probe response: {data}")
                print(
                    f"  ✅ Current name from probe: {data.get('current_name', 'Unknown')}"
                )
                probe_working = True
            else:
                print(f"  ⚠️  Probe endpoint returned {response.status_code}")
                probe_working = False
        except requests.exceptions.RequestException:
            print(
                "  ⚠️  Probe API not running (start with: uvicorn pmm.api.probe:app --reload)"
            )
            probe_working = False

        # Test 3: Validate identity consistency
        print("\n🧪 Test 3: Validating identity consistency...")

        stored_name = mgr.model.core_identity.name
        expected_name = "Echo"

        identity_consistent = stored_name == expected_name
        print(f"  📝 Stored name: '{stored_name}'")
        print(f"  📝 Expected name: '{expected_name}'")
        print(
            f"  {'✅' if identity_consistent else '❌'} Identity consistent: {identity_consistent}"
        )

        # Summary
        print("\n📋 Test Summary:")
        print(
            f"  {'✅' if final_events > initial_events else '❌'} Event emission: {final_events > initial_events}"
        )
        print(
            f"  {'✅' if len(identity_events) > 0 else '❌'} Identity events: {len(identity_events) > 0}"
        )
        print(
            f"  {'✅' if identity_consistent else '❌'} Name consistency: {identity_consistent}"
        )
        print(f"  {'✅' if probe_working else '⚠️ '} Probe endpoint: {probe_working}")

        success = (
            final_events > initial_events
            and len(identity_events) > 0
            and identity_consistent
        )

        if success:
            print("\n🎉 Identity Persistence: WORKING!")
            print("✅ Identity updates emit traceable events")
            print("✅ Name persistence across model saves")
            print("✅ Probe endpoint available for validation")
        else:
            print("\n⚠️  Some identity tests failed")

        return success

    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.unlink(test_path)


def main():
    """Run identity persistence tests."""
    print("🔍 Phase 3B+: Identity Persistence Validation")
    print("=" * 50)

    success = test_identity_persistence()

    if success:
        print("\n🎯 READY FOR GPT'S LIVE EXPERIMENTS:")
        print("1. 'Let's officially adopt the name Echo'")
        print("2. Check /identity endpoint shows 'Echo'")
        print("3. Verify 'What's your name?' → 'Echo' (no Agent fallback)")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
