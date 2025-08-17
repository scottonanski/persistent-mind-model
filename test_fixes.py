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
        if errors:
            print(f"❌ Errors occurred: {errors}")
            return False

        # Check for duplicate IDs
        unique_ids = set(results)
        if len(unique_ids) != len(results):
            print(
                f"❌ Duplicate IDs found! Generated: {len(results)}, Unique: {len(unique_ids)}"
            )
            print(f"IDs: {sorted(results)}")
            return False

        print(f"✅ Generated {len(results)} unique IDs across 3 threads")
        return True


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

        if len(events) < 3:
            print(f"❌ Expected at least 3 events, got {len(events)}")
            return False

        # Check chain linkage
        prev_hash = None
        for i, event in enumerate(events):
            if event["prev_hash"] != prev_hash:
                print(
                    f"❌ Chain break at event {i}: expected prev_hash={prev_hash}, got {event['prev_hash']}"
                )
                return False
            prev_hash = event["hash"]

        print(f"✅ Hash chain verified for {len(events)} events")
        return True


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

        if not json_event:
            print(f"❌ Event {ev_id} not found in JSON model")
            return False

        # Get from SQLite
        sqlite_events = manager.sqlite_store.all_events()
        sqlite_event = None
        for event in sqlite_events:
            if event["meta"].get("id") == ev_id:
                sqlite_event = event
                break

        if not sqlite_event:
            print(f"❌ Event {ev_id} not found in SQLite")
            return False

        # Check tag parity
        json_tags = set(json_event.tags)
        sqlite_tags = set(sqlite_event["meta"].get("tags", []))

        if json_tags != sqlite_tags:
            print(f"❌ Tag mismatch: JSON={json_tags}, SQLite={sqlite_tags}")
            return False

        print(f"✅ Field parity verified for event {ev_id}")
        return True


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
        if not json_changes:
            print("❌ No identity changes in JSON model")
            return False

        latest_change = json_changes[-1]
        if new_name not in latest_change.change:
            print(f"❌ JSON identity change incorrect: {latest_change}")
            return False

        # Check SQLite
        sqlite_events = manager.sqlite_store.all_events()
        identity_events = [e for e in sqlite_events if e["kind"] == "identity_change"]

        if not identity_events:
            print("❌ No identity_change events in SQLite")
            return False

        latest_sqlite = identity_events[-1]
        if latest_sqlite["meta"].get("new_value") != new_name:
            print(f"❌ SQLite identity change incorrect: {latest_sqlite}")
            return False

        print(f"✅ Identity change logged consistently: {old_name} → {new_name}")
        return True


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
            if test():
                passed += 1
            else:
                print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            print()

    print(f"📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All critical fixes validated successfully!")
        return True
    else:
        print("⚠️  Some tests failed - fixes need attention")
        return False


if __name__ == "__main__":
    main()
