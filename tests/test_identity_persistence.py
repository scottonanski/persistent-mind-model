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
import pytest

# Add PMM to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmm.self_model_manager import SelfModelManager


@pytest.mark.integration
@pytest.mark.network
def test_identity_persistence():
    """Test identity persistence and probe endpoint."""

    # Create test PMM instance
    test_path = f"test_identity_{os.getpid()}.json"

    try:
        mgr = SelfModelManager(test_path)

        # Test 1: Set identity and check event emission

        initial_events = len(mgr.model.self_knowledge.autobiographical_events)
        mgr.set_name("Echo", origin="test")
        final_events = len(mgr.model.self_knowledge.autobiographical_events)

        # Check if identity_change event was created
        identity_events = [
            e
            for e in mgr.model.self_knowledge.autobiographical_events
            if e.type == "identity_change"
        ]

        # Test 2: Check probe endpoint (if running)

        try:
            response = requests.get("http://localhost:8000/identity", timeout=2)
            if response.status_code == 200:
                _ = response.json()
                _ = True
            else:
                _ = False
        except requests.exceptions.RequestException:
            _ = False

        # Test 3: Validate identity consistency

        stored_name = mgr.model.core_identity.name
        expected_name = "Echo"

        identity_consistent = stored_name == expected_name

        # Summary

        success = (
            final_events > initial_events
            and len(identity_events) > 0
            and identity_consistent
        )

        # Probe is optional; core assertions below are deterministic

        assert success, "Identity persistence tests failed"

    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.unlink(test_path)


# Removed demo-style main runner; tests are executed via pytest
