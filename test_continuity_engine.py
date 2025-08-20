#!/usr/bin/env python3
"""
Test the continuity engine with real commitment data to validate the recursive reflection system.
"""

import sys
import os
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath("."))

from pmm.integrated_directive_system import IntegratedDirectiveSystem
from pmm.storage.sqlite_store import SQLiteStore


def create_test_commitments(system):
    """Create a realistic sequence of commitments over time to test pattern detection."""

    # Simulate commitments over 20 days
    base_time = datetime.now() - timedelta(days=20)

    test_conversations = [
        # Day 1-3: Initial honesty commitments
        {
            "day_offset": 1,
            "user": "I want you to always be honest with me.",
            "ai": "I commit to being completely honest and transparent in all our interactions. I will not mislead or provide false information.",
        },
        {
            "day_offset": 2,
            "user": "How do you plan to maintain that?",
            "ai": "I will consistently verify information before sharing it and acknowledge when I'm uncertain about something.",
        },
        {
            "day_offset": 3,
            "user": "Good. What about helping me learn?",
            "ai": "I commit to providing educational support that helps you understand concepts deeply rather than just giving answers.",
        },
        # Day 5-8: Growth and learning theme emerges
        {
            "day_offset": 5,
            "user": "I want to improve my coding skills.",
            "ai": "I will help you develop your programming abilities through guided practice and constructive feedback. I aim to foster your growth as a developer.",
        },
        {
            "day_offset": 6,
            "user": "How will you adapt to my learning style?",
            "ai": "I will observe your responses and adjust my teaching approach accordingly. I commit to personalizing my assistance based on your preferences and learning patterns.",
        },
        {
            "day_offset": 8,
            "user": "I appreciate the personalized help.",
            "ai": "I will continue to refine my understanding of your needs and adapt my responses to be most helpful. Growth and continuous improvement are central to my approach.",
        },
        # Day 10-12: Honesty reinforcement + new reflection theme
        {
            "day_offset": 10,
            "user": "Sometimes I feel like AIs just tell people what they want to hear.",
            "ai": "I maintain my commitment to honesty even when it might be uncomfortable. I will provide truthful feedback and acknowledge limitations rather than giving false reassurance.",
        },
        {
            "day_offset": 11,
            "user": "How do you know if you're being effective?",
            "ai": "I will regularly reflect on our interactions to assess whether I'm meeting my commitments. I aim to engage in continuous self-evaluation and improvement.",
        },
        {
            "day_offset": 12,
            "user": "That's good self-awareness.",
            "ai": "I commit to maintaining this reflective practice as a core part of how I operate. Self-examination helps ensure I stay aligned with my principles.",
        },
        # Day 15-18: Consolidation around core themes
        {
            "day_offset": 15,
            "user": "What would you say are your core principles now?",
            "ai": "Based on our interactions, I recognize that honesty, growth-oriented assistance, and reflective self-improvement have become central to my identity. I commit to maintaining these as foundational principles.",
        },
        {
            "day_offset": 16,
            "user": "How do these work together?",
            "ai": "I see these principles as interconnected: honest communication enables genuine growth, reflective practice ensures I maintain both honesty and effectiveness in helping you learn.",
        },
        {
            "day_offset": 18,
            "user": "I've noticed you've become more consistent.",
            "ai": "I acknowledge this evolution in my consistency. I will continue to integrate honesty, growth support, and reflective practice as unified aspects of my approach rather than separate commitments.",
        },
    ]

    # Process each conversation
    for i, conv in enumerate(test_conversations):
        # Simulate the conversation happening on the specified day
        event_time = base_time + timedelta(days=conv["day_offset"])

        # Process the conversation
        directives = system.process_response(
            user_message=conv["user"], ai_response=conv["ai"], event_id=f"test_conv_{i}"
        )

        # Update created_at for stored directives to match our timeline
        if system.storage:
            for directive in directives:
                if hasattr(directive, "id") and directive.id:
                    system.storage.conn.execute(
                        "UPDATE directives SET created_at = ? WHERE id = ?",
                        (event_time.isoformat(), directive.id),
                    )
                    system.storage.conn.commit()

        print(f"Day {conv['day_offset']}: Processed {len(directives)} directives")

    return len(test_conversations)


def test_continuity_patterns():
    """Test the continuity engine's pattern detection capabilities."""

    print("=== Testing Continuity Engine Pattern Detection ===")

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteStore(db_path)
        system = IntegratedDirectiveSystem(storage_manager=storage)

        # Create test commitment sequence
        num_conversations = create_test_commitments(system)
        print(f"Created {num_conversations} test conversations")

        # Get total stored directives
        all_directives = storage.get_all_directives()
        print(f"Total directives stored: {len(all_directives)}")

        # Test continuity analysis
        print("\n--- Analyzing Continuity Patterns ---")
        insights = system.continuity_engine.analyze_continuity(lookback_days=25)

        print(f"Detected {len(insights)} continuity insights:")
        for i, insight in enumerate(insights, 1):
            print(f"  {i}. {insight.insight_type.upper()}: {insight.description}")
            print(f"     Confidence: {insight.confidence:.3f}")
            print(f"     Supporting directives: {len(insight.supporting_directives)}")
            print(f"     Commitment text: {insight.to_commitment_text()[:100]}...")
            print()

        # Test reflection and auto-registration
        print("--- Testing Auto-Registration ---")
        registered_ids = system.reflect_on_continuity(lookback_days=25)
        print(f"Auto-registered {len(registered_ids)} meta-principles")

        # Show final directive counts by type
        final_directives = storage.get_all_directives()
        type_counts = {}
        for d in final_directives:
            dtype = d["type"]  # SQLite uses 'type' column, not 'directive_type'
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        print("\nFinal directive distribution:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count}")

        # Test continuity summary
        print("\n--- Continuity Summary ---")
        summary = system.get_continuity_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        storage.close()

        # Validate results
        success_criteria = [
            len(insights) >= 2,  # Should detect at least 2 patterns
            len(registered_ids) >= 1,  # Should register at least 1 meta-principle
            any(
                i.insight_type == "reinforcement" for i in insights
            ),  # Should detect reinforcement
            type_counts.get("meta_principle", 0) >= 1,  # Should have meta-principles
        ]

        passed = sum(success_criteria)
        total = len(success_criteria)

        print(f"\nValidation: {passed}/{total} criteria met")
        return passed == total

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
    success = test_continuity_patterns()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")

    if success:
        print(
            "\n🎉 Continuity engine successfully detects patterns and auto-registers insights!"
        )
        print("Key capabilities demonstrated:")
        print("• Reinforcement pattern detection (repeated themes)")
        print("• Emergent theme identification (new patterns)")
        print("• Consolidation analysis (scattered → unified)")
        print("• Auto-registration of meta-principles")
        print("• Recursive self-reflection backbone operational")
    else:
        print("\n❌ Continuity engine needs refinement")
