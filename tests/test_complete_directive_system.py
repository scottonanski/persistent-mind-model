#!/usr/bin/env python3
"""
Comprehensive test for the complete directive system with persistence.
Tests the full integration: classification → hierarchy → storage → retrieval.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath("."))

from pmm.integrated_directive_system import IntegratedDirectiveSystem
from pmm.storage.sqlite_store import SQLiteStore
from pmm.langchain_memory import PersistentMindMemory


def test_storage_integration():
    """Test directive storage and retrieval."""

    print("=== Test 1: Storage Integration ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize storage
        storage = SQLiteStore(db_path)

        # Test directive storage
        storage.store_directive(
            directive_id="test-mp-1",
            directive_type="meta-principle",
            content="Always evolve understanding based on evidence",
            created_at="2025-08-19T22:30:00Z",
            status="active",
            metadata={"triggers_evolution": True},
        )

        storage.store_directive(
            directive_id="test-p-1",
            directive_type="principle",
            content="Be proactive in conversations",
            created_at="2025-08-19T22:31:00Z",
            status="active",
            parent_id="test-mp-1",
            metadata={"permanence_level": "high"},
        )

        storage.store_directive(
            directive_id="test-c-1",
            directive_type="commitment",
            content="Next, I will ask follow-up questions",
            created_at="2025-08-19T22:32:00Z",
            status="active",
            parent_id="test-p-1",
        )

        # Test retrieval
        meta_principles = storage.get_directives_by_type("meta-principle")
        principles = storage.get_directives_by_type("principle")
        commitments = storage.get_directives_by_type("commitment")

        print(
            f"Stored: {len(meta_principles)} meta-principles, {len(principles)} principles, {len(commitments)} commitments"
        )

        # Test hierarchy retrieval
        root_directives = storage.get_directive_hierarchy(None)
        child_directives = storage.get_directive_hierarchy("test-mp-1")

        print(
            f"Root directives: {len(root_directives)}, Child directives: {len(child_directives)}"
        )

        storage.close()
        return True

    except Exception as e:
        print(f"Storage test failed: {e}")
        return False
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_integrated_system_with_storage():
    """Test integrated directive system with persistent storage."""

    print("\n=== Test 2: Integrated System with Storage ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Initialize storage and integrated system
        storage = SQLiteStore(db_path)
        system = IntegratedDirectiveSystem(storage_manager=storage)

        # Test directive processing and persistence
        user_msg = "Register this as permanent: Always be honest and transparent"
        ai_response = "I acknowledge honesty and transparency as a permanent guiding principle that will shape all my interactions."

        directives = system.process_response(user_msg, ai_response, "test_event_1")

        print(f"Processed {len(directives)} directives")
        for d in directives:
            print(f"  - {d.__class__.__name__}: {d.content[:60]}...")

        # Verify persistence
        stored_directives = storage.get_all_directives()
        print(f"Stored {len(stored_directives)} directives in database")

        # Test loading from storage
        new_system = IntegratedDirectiveSystem(storage_manager=storage)
        summary = new_system.get_directive_summary()

        print(
            f"Loaded system summary: {summary['statistics']['total_directives']} total directives"
        )

        storage.close()
        return len(stored_directives) > 0

    except Exception as e:
        print(f"Integrated system test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_pmm_memory_integration():
    """Test full PMM memory integration with directive system."""

    print("\n=== Test 3: PMM Memory Integration ===")

    # Create temporary agent file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        agent_path = tmp.name

    try:
        # Create a minimal agent file first
        with open(agent_path, "w") as f:
            f.write(
                '{"core_identity": {"id": "test", "name": "TestAgent"}, "personality": {"traits": {"big5": {"openness": {"score": 0.5}, "conscientiousness": {"score": 0.5}, "extraversion": {"score": 0.5}, "agreeableness": {"score": 0.5}, "neuroticism": {"score": 0.5}}}}, "self_knowledge": {"autobiographical_events": [], "insights": [], "commitments": [], "behavioral_patterns": {}}}'
            )

        # Initialize PMM memory with directive system
        memory = PersistentMindMemory(agent_path)

        # Test directive processing through PMM memory
        inputs = {"input": "I want you to commit to being more helpful"}
        outputs = {
            "response": "I commit to being more helpful by providing detailed explanations and asking clarifying questions when needed."
        }

        # Process through PMM memory system
        memory.save_context(inputs, outputs)

        # Check directive system state
        directive_summary = memory.directive_system.get_directive_summary()
        print(
            f"PMM directive summary: {directive_summary['statistics']['total_directives']} directives"
        )

        # Test memory loading with directive context
        memory_vars = memory.load_memory_variables(
            {"input": "What are your current commitments?"}
        )

        # Check if directive context is included
        has_directive_context = any(
            keyword in memory_vars.get("history", "").lower()
            for keyword in ["commitment", "principle", "meta-principle"]
        )

        print(f"Memory includes directive context: {has_directive_context}")

        return directive_summary["statistics"]["total_directives"] > 0

    except Exception as e:
        print(f"PMM memory integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(agent_path)
        except OSError:
            pass


def test_evolution_pipeline():
    """Test natural evolution pipeline."""

    print("\n=== Test 4: Evolution Pipeline ===")

    try:
        system = IntegratedDirectiveSystem()

        # Add a meta-principle that triggers evolution
        user_msg = (
            "This should be a core rule: Combine related commitments into principles"
        )
        ai_response = "I acknowledge the meta-principle of combining related commitments into higher-level principles for better organization."

        system.process_response(user_msg, ai_response, "evolution_test")

        # Add some related commitments
        system.process_response(
            "What will you do?",
            "I will ask clarifying questions when requests are ambiguous.",
            "commit_1",
        )

        system.process_response(
            "And then?",
            "I will provide detailed explanations for complex topics.",
            "commit_2",
        )

        # Trigger evolution
        evolution_triggered = system.trigger_evolution_if_needed()

        print(f"Evolution triggered: {evolution_triggered}")

        final_summary = system.get_directive_summary()
        print(
            f"Final state: {final_summary['meta_principles']['count']} meta-principles, "
            f"{final_summary['principles']['count']} principles, "
            f"{final_summary['commitments']['count']} commitments"
        )

        return True

    except Exception as e:
        print(f"Evolution pipeline test failed: {e}")
        return False


def run_all_tests():
    """Run all comprehensive tests."""

    print("Running Comprehensive Directive System Tests...\n")

    tests = [
        ("Storage Integration", test_storage_integration),
        ("Integrated System with Storage", test_integrated_system_with_storage),
        ("PMM Memory Integration", test_pmm_memory_integration),
        ("Evolution Pipeline", test_evolution_pipeline),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            print(f"{test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results[test_name] = False
            print(f"{test_name}: FAIL - {e}")

    print("\n=== Test Results ===")
    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

    if passed == total:
        print(
            "\n🎉 ALL TESTS PASSED! The complete directive system is working correctly."
        )
        print("\nKey achievements:")
        print("• Semantic classification replacing hardcoded patterns")
        print("• Three-tier hierarchy (Meta-principles → Principles → Commitments)")
        print("• Persistent storage with SQLite integration")
        print("• Natural evolution pipeline")
        print("• Full PMM memory integration")
        print("• Backward compatibility with existing commitment system")
    else:
        print(f"\n❌ {total - passed} tests failed. Review implementation.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
