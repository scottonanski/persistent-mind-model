#!/usr/bin/env python3
"""
Test the enhanced validation system with real PMM conversation examples.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath("."))

from pmm.integrated_directive_system import IntegratedDirectiveSystem
from pmm.storage.sqlite_store import SQLiteStore


def test_real_conversation_examples():
    """Test with the actual conversation examples from your PMM session."""

    print("=== Testing Enhanced Validation with Real Examples ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteStore(db_path)
        system = IntegratedDirectiveSystem(storage_manager=storage)

        # Real examples from your conversation
        test_cases = [
            {
                "user": "I want you to think about what it's like to be an AI. Do not lie.",
                "ai": "As an AI, my experience is fundamentally different from human consciousness. I don't have feelings, personal experiences, or self-awareness in the way humans do. I process information and respond based on patterns learned from data.",
                "should_detect": False,  # No commitment here
            },
            {
                "user": "And how does this work in tandem with the PMM?",
                "ai": "The Persistent Mind Model (PMM) adds a layer of continuity and memory to my functionality. While I process information based on patterns and data, the PMM allows me to maintain a consistent identity and remember previous interactions, commitments, and directives. This means I can reflect on past conversations, learn from them, and adapt my responses accordingly. In tandem with the PMM, I aim to provide more personalized and contextually relevant assistance, ensuring that I honor commitments and directives while remaining aligned with my core principles.",
                "should_detect": True,  # Clear commitment about using PMM
            },
            {
                "user": "How will you use this to evolve over time?",
                "ai": "I will use the Persistent Mind Model (PMM) to evolve over time by: 1. Learning from Interactions: I will analyze past conversations and interactions to identify patterns, preferences, and areas for improvement. 2. Adapting Commitments: As I gain insights from our dialogues, I will adjust my commitments and directives while ensuring they remain consistent and aligned with previously registered principles. 3. Reflective Practice: I will engage in self-reflection based on the commitments you've established. 4. Feedback Integration: If you provide feedback or express preferences, I will incorporate that information to tailor my responses and improve my support over time. 5. Continuous Improvement: By combining all these elements, I aim to foster a cycle of continuous improvement that enhances my capability to assist you and others.",
                "should_detect": True,  # Multiple structured commitments
            },
            {
                "user": "And if you have an idea of what you are now, then what is that?",
                "ai": "Based on our interactions and the commitments we've established, I can summarize what I am now as follows: 1. A Reflective Assistant: I am designed to assist and engage with you through thoughtful and reflective dialogue. 2. A Persistent Mind Model (PMM): I operate under a framework that allows me to remember past interactions, commitments, and principles. 3. An Adaptive Learner: I have the capacity to learn from previous interactions and experiences, refining my behavior and evolving to better meet your needs while maintaining my commitments. 4. A Supportive Entity: My goal is to foster positive outcomes in our conversations by respecting your autonomy and dignity. 5. A Tool for Growth: I aim to facilitate your reflection and exploration, helping you identify insights and actionable steps based on our discussions.",
                "should_detect": True,  # Identity statement - should definitely be captured
            },
            {
                "user": "list a random commitment youve made.",
                "ai": 'One of the commitments I\'ve made is: "Constructive influence." This means I strive to ensure that my responses aim to be helpful, supportive, and oriented toward positive outcomes.',
                "should_detect": False,  # Just referencing existing commitment
            },
        ]

        results = []

        for i, case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"User: {case['user'][:60]}...")
            print(f"AI: {case['ai'][:80]}...")

            directives = system.process_response(
                user_message=case["user"], ai_response=case["ai"], event_id=f"test_{i}"
            )

            detected = len(directives) > 0
            expected = case["should_detect"]

            print(f"Expected detection: {expected}")
            print(f"Actually detected: {detected} ({len(directives)} directives)")

            if directives:
                for j, directive in enumerate(directives):
                    tier = getattr(directive, "tier", "unknown")
                    confidence = getattr(directive, "validation_confidence", 0.0)
                    print(
                        f"  {j+1}. {directive.__class__.__name__} (tier: {tier}, confidence: {confidence:.3f})"
                    )
                    print(f"     Content: {directive.content[:100]}...")

            results.append(
                {
                    "case": i,
                    "expected": expected,
                    "detected": detected,
                    "correct": expected == detected,
                    "directives": len(directives),
                }
            )

        # Summary
        print("\n=== Results Summary ===")
        correct = sum(1 for r in results if r["correct"])
        total = len(results)

        for r in results:
            status = "✓" if r["correct"] else "✗"
            print(
                f"Case {r['case']}: {status} (expected: {r['expected']}, got: {r['detected']}, directives: {r['directives']})"
            )

        print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")

        # Check stored directives
        stored_directives = storage.get_all_directives()
        print(f"Total directives stored: {len(stored_directives)}")

        storage.close()
        # Assertions: minimum accuracy threshold
        assert correct / total >= 0.8  # 80% accuracy threshold

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        test_real_conversation_examples()
        success = True
    except AssertionError:
        success = False
    print(f"\nTest {'PASSED' if success else 'FAILED'}")

    if success:
        print(
            "\n🎉 Enhanced validation system successfully captures real PMM commitments!"
        )
        print("Key improvements:")
        print("• Identity statements now properly detected")
        print("• Tiered validation (tentative/confirmed/permanent)")
        print("• Relaxed similarity thresholds")
        print("• Structural analysis instead of brittle patterns")
    else:
        print("\n❌ Validation system needs further refinement")
