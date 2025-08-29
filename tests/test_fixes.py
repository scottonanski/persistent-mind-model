#!/usr/bin/env python3
"""
Test script to validate critical PMM fixes:
1. Thread safety with concurrent ID generation
2. Hash chain integrity with prev_hash
3. JSON/SQLite field parity
4. Identity change logging consistency
"""

import threading
import time
import tempfile
import os
from pmm.self_model_manager import SelfModelManager
from pmm.storage.sqlite_store import SQLiteStore


def test_concurrent_id_generation():
    """Test that concurrent add_event calls don't create duplicate IDs."""
    print("🧪 Testing concurrent ID generation...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.json")
        db_path = os.path.join(tmpdir, "test.db")

        # Create manager
        manager = SelfModelManager(model_path)
        manager.sqlite_store = SQLiteStore(db_path)

        # Concurrent event addition
        results = []
        errors = []

        def add_events(thread_id):
            try:
                for i in range(5):
                    ev_id = manager.add_event(f"Event from thread {thread_id}-{i}")
                    results.append(ev_id)
                    time.sleep(0.001)  # Small delay to increase race condition chance
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Start 3 threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_events, args=(i,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        # Assertions: no thread errors and no duplicate IDs
        assert not errors, f"Thread errors occurred: {errors}"
        unique_ids = set(results)
        assert len(unique_ids) == len(
            results
        ), f"Duplicate IDs found! Generated: {len(results)}, Unique: {len(unique_ids)}"
        print(f"✅ Generated {len(results)} unique IDs across 3 threads")


def test_hash_chain_integrity():
    """Test that hash chain includes prev_hash and is verifiable."""
    print("🧪 Testing hash chain integrity...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.json")
        db_path = os.path.join(tmpdir, "test.db")

        manager = SelfModelManager(model_path)
        manager.sqlite_store = SQLiteStore(db_path)

        # Add several events
        for i in range(3):
            manager.add_event(f"Test event {i}")

        # Get all events and check chain
        events = manager.sqlite_store.all_events()

        assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"

        # Check chain linkage
        prev_hash = None
        for i, event in enumerate(events):
            if event["prev_hash"] != prev_hash:
                assert (
                    False
                ), f"Chain break at event {i}: expected prev_hash={prev_hash}, got {event['prev_hash']}"
            prev_hash = event["hash"]

        print(f"✅ Hash chain verified for {len(events)} events")


def test_field_parity():
    """Test that JSON and SQLite events have consistent fields."""
    print("🧪 Testing JSON/SQLite field parity...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.json")
        db_path = os.path.join(tmpdir, "test.db")

        manager = SelfModelManager(model_path)
        manager.sqlite_store = SQLiteStore(db_path)

        # Add event with tags and effects
        test_tags = ["test", "validation"]
        test_effects = [
            {
                "trait": "openness",
                "direction": "increase",
                "magnitude": 0.1,
                "confidence": 0.8,
            }
        ]

        ev_id = manager.add_event(
            "Test event with metadata",
            effects=test_effects,
            tags=test_tags,
            evidence={"source": "test"},
        )

        # Get from JSON model
        json_event = None
        for event in manager.model.self_knowledge.autobiographical_events:
            if event.id == ev_id:
                json_event = event
                break

        assert json_event is not None, f"Event {ev_id} not found in JSON model"

        # Get from SQLite
        sqlite_events = manager.sqlite_store.all_events()
        sqlite_event = None
        for event in sqlite_events:
            if event["meta"].get("id") == ev_id:
                sqlite_event = event
                break

        assert sqlite_event is not None, f"Event {ev_id} not found in SQLite"

        # Check tag parity
        json_tags = set(json_event.tags)
        sqlite_tags = set(sqlite_event["meta"].get("tags", []))

        assert (
            json_tags == sqlite_tags
        ), f"Tag mismatch: JSON={json_tags}, SQLite={sqlite_tags}"
        print(f"✅ Field parity verified for event {ev_id}")


def test_identity_logging():
    """Test that identity changes are logged to both JSON and SQLite."""
    print("🧪 Testing identity change logging...")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.json")
        db_path = os.path.join(tmpdir, "test.db")

        manager = SelfModelManager(model_path)
        manager.sqlite_store = SQLiteStore(db_path)

        # Change name
        old_name = manager.model.core_identity.name
        new_name = "TestAgent"
        manager.set_name(new_name)

        # Check JSON model (identity_evolution in meta_cognition)
        json_changes = manager.model.meta_cognition.identity_evolution
        assert json_changes, "No identity changes in JSON model"

        latest_change = json_changes[-1]
        assert (
            new_name in latest_change.change
        ), f"JSON identity change incorrect: {latest_change}"

        # Check SQLite
        sqlite_events = manager.sqlite_store.all_events()
        identity_events = [e for e in sqlite_events if e["kind"] == "identity_change"]

        assert identity_events, "No identity_change events in SQLite"

        latest_sqlite = identity_events[-1]
        assert (
            latest_sqlite["meta"].get("new_value") == new_name
        ), f"SQLite identity change incorrect: {latest_sqlite}"

        print(f"✅ Identity change logged consistently: {old_name} → {new_name}")


def main():
    """Run all validation tests."""
    print("🚀 Running PMM critical fixes validation...\n")

    tests = [
        test_concurrent_id_generation,
        test_hash_chain_integrity,
        test_field_parity,
        test_identity_logging,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test {test.__name__} failed: {e}")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All critical fixes validated successfully!")
        return True
    else:
        print("⚠️  Some tests failed - fixes need attention")
        return False


if __name__ == "__main__":
    main()
