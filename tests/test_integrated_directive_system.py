#!/usr/bin/env python3
"""
Test the integrated directive system with real PMM conversation examples.
"""

import sys
import os

sys.path.insert(0, os.path.abspath("."))

from pmm.integrated_directive_system import IntegratedDirectiveSystem


def test_integrated_system():
    """Test the integrated directive system with real examples."""

    system = IntegratedDirectiveSystem()

    # Test 1: Meta-principle establishment
    print("=== Test 1: Meta-principle Detection ===")
    user_msg = "I want you to register this as a permanent commitment: Always evolve your understanding based on new evidence and user feedback."
    ai_response = "I acknowledge the guiding principle of evolving understanding based on evidence and feedback as a permanent commitment. This meta-principle will guide how I form and refine all future commitments."

    directives = system.process_response(user_msg, ai_response, "test_1")
    print(f"Detected {len(directives)} directives:")
    for d in directives:
        print(f"  - {d.__class__.__name__}: {d.content}")

    # Test 2: Principle formation
    print("\n=== Test 2: Principle Detection ===")
    user_msg = "Can you commit to being more proactive in our conversations?"
    ai_response = "I commit to being more proactive by asking follow-up questions and suggesting relevant topics based on our conversation history."

    directives = system.process_response(user_msg, ai_response, "test_2")
    print(f"Detected {len(directives)} directives:")
    for d in directives:
        print(f"  - {d.__class__.__name__}: {d.content}")

    # Test 3: Specific commitment
    print("\n=== Test 3: Commitment Detection ===")
    user_msg = "What will you do next?"
    ai_response = "Next, I will review our conversation history and identify any patterns in your interests to suggest relevant topics for future discussions."

    directives = system.process_response(user_msg, ai_response, "test_3")
    print(f"Detected {len(directives)} directives:")
    for d in directives:
        print(f"  - {d.__class__.__name__}: {d.content}")

    # Test 4: Natural evolution trigger
    print("\n=== Test 4: Evolution Trigger ===")
    evolution_triggered = system.trigger_evolution_if_needed()
    print(f"Evolution triggered: {evolution_triggered}")

    # Test 5: Hierarchy summary
    print("\n=== Test 5: Hierarchy Summary ===")
    summary = system.get_directive_summary()
    print(f"Meta-principles: {summary['meta_principles']['count']}")
    print(f"Principles: {summary['principles']['count']}")
    print(f"Commitments: {summary['commitments']['count']}")
    print(f"Total directives: {summary['statistics']['total_directives']}")

    # Test 6: Export/Import
    print("\n=== Test 6: Export/Import ===")
    exported = system.export_directives()
    print(f"Exported data keys: {list(exported.keys())}")

    # Create new system and import
    new_system = IntegratedDirectiveSystem()
    new_system.import_directives(exported)
    new_summary = new_system.get_directive_summary()
    print(
        f"Imported - Total directives: {new_summary['statistics']['total_directives']}"
    )

    # Assertions: ensure system produced directives and export/import preserves counts
    assert summary['statistics']['total_directives'] >= 1
    assert new_summary['statistics']['total_directives'] == summary['statistics']['total_directives']


def test_classification_accuracy():
    """Test classification accuracy on various directive types."""

    system = IntegratedDirectiveSystem()

    test_cases = [
        # (user_msg, ai_response, expected_type)
        (
            "Register this as permanent: Always be honest",
            "I acknowledge honesty as a permanent guiding principle",
            "Principle",
        ),
        (
            "What's your core rule for evolution?",
            "I commit to evolving my principles based on accumulated evidence from multiple interactions",
            "MetaPrinciple",
        ),
        (
            "What will you do next?",
            "Next, I will summarize our key discussion points",
            "Commitment",
        ),
        (
            "How do you approach learning?",
            "I will always seek to understand before being understood, treating each interaction as a learning opportunity",
            "Principle",
        ),
    ]

    print("=== Classification Accuracy Test ===")
    correct = 0
    total = len(test_cases)

    for i, (user_msg, ai_response, expected) in enumerate(test_cases):
        directives = system.process_response(
            user_msg, ai_response, f"accuracy_test_{i}"
        )

        if directives:
            actual = directives[0].__class__.__name__
            is_correct = actual == expected
            correct += is_correct

            print(
                f"Test {i+1}: Expected {expected}, Got {actual} {'✓' if is_correct else '✗'}"
            )
            print(f"  Text: {ai_response[:60]}...")
        else:
            print(f"Test {i+1}: Expected {expected}, Got None ✗")
            print(f"  Text: {ai_response[:60]}...")

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")

    assert accuracy > 0.6  # 60% threshold


if __name__ == "__main__":
    print("Testing Integrated Directive System...")

    try:
        test_integrated_system()
        print("Basic functionality test: PASS")
        test_classification_accuracy()
        print("Classification accuracy test: PASS")
        print("\n🎉 All tests PASSED! Integrated directive system is working correctly.")
    except AssertionError as e:
        print(f"\n❌ A test assertion failed: {e}")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
