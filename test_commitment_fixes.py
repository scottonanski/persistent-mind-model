#!/usr/bin/env python3
"""
Test the improved commitment detection patterns.
"""

from pmm.commitments import CommitmentTracker

def test_commitment_detection_fixes():
    """Test that the improved patterns catch Echo session commitments."""
    tracker = CommitmentTracker()
    
    # Test cases from the Echo session that were missed
    test_cases = [
        # Markdown formatted commitments
        "I commit to enhancing our conversations by actively incorporating emotional intelligence principles into my responses. Specifically, I will:\n\n1. **Prioritize Emotional Awareness**: I will strive to recognize and acknowledge emotional cues in our conversations, responding in ways that reflect understanding and empathy.",
        
        # "By committing to" pattern
        "By committing to these practices, I aim to foster more meaningful and connected interactions with you.",
        
        # "I commit to achieving" pattern
        "I commit to achieving the goal of effectively utilizing storytelling to convey a concept related to emotional intelligence in our conversation today.",
        
        # "I commit to evolving" pattern
        "I commit to evolving my conscientiousness by:\n\n1. **Setting Clear Goals**: I will outline specific goals for each conversation, ensuring I stay focused and organized in addressing your needs.",
        
        # Simple "I will" in markdown list
        "To enhance my meta-cognitive processes, I commit to:\n\n1. **Structured Reflection**: I will implement a structured reflection process after each conversation, focusing on specific aspects such as emotional engagement, creativity, and depth of discussion.",
        
        # "I aim to" pattern
        "I aim to enhance my creative thinking and provide a more engaging conversational experience."
    ]
    
    print("🧪 Testing Improved Commitment Detection")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        commitment, ngrams = tracker.extract_commitment(test_case)
        
        print(f"\n**Test Case {i}:**")
        print(f"Input: {test_case[:100]}...")
        
        if commitment:
            print(f"✅ **DETECTED**: {commitment[:80]}...")
            print(f"   N-grams: {len(ngrams)} generated")
        else:
            print("❌ **MISSED**: No commitment detected")
    
    print("\n" + "=" * 50)
    print("🎯 Summary: These should ALL be detected now!")

if __name__ == "__main__":
    test_commitment_detection_fixes()
