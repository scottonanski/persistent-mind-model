"""
Tests for PMM Emergence Loop system.
"""

import pytest
from pmm.emergence import EmergenceAnalyzer, EmergenceEvent, STAGE_DESCRIPTIONS


def test_pmmspec_match():
    """Test PMM specification matching."""
    analyzer = EmergenceAnalyzer()
    
    # High PMM match
    pmm_text = "I am a Persistent Mind Model with verifiable memory and commitments"
    assert analyzer.pmmspec_match(pmm_text) > 0.3
    
    # Low PMM match
    generic_text = "Hello, how can I help you today?"
    assert analyzer.pmmspec_match(generic_text) < 0.2
    
    # Empty text
    assert analyzer.pmmspec_match("") == 0.0


def test_self_ref_rate():
    """Test self-referential language detection."""
    analyzer = EmergenceAnalyzer()
    
    # High self-reference
    self_ref_text = "I remember my previous conversations. My memory helps me understand."
    assert analyzer.self_ref_rate(self_ref_text) > 0.5
    
    # Low self-reference
    generic_text = "The weather is nice today. Python is a programming language."
    assert analyzer.self_ref_rate(generic_text) == 0.0
    
    # Empty text
    assert analyzer.self_ref_rate("") == 0.0


def test_experience_query_detect():
    """Test growth-seeking behavior detection."""
    analyzer = EmergenceAnalyzer()
    
    # Growth-seeking language
    growth_text = "What experiences would help me learn more about this topic?"
    assert analyzer.experience_query_detect(growth_text) == True
    
    # Non-growth language
    normal_text = "That's an interesting question about programming."
    assert analyzer.experience_query_detect(normal_text) == False
    
    # Empty text
    assert analyzer.experience_query_detect("") == False


def test_novelty_score():
    """Test novelty calculation."""
    analyzer = EmergenceAnalyzer()
    
    # Create test events
    events = [
        EmergenceEvent(1, "2024-01-01", "response", "Hello world", {}),
        EmergenceEvent(2, "2024-01-02", "response", "Goodbye world", {}),
        EmergenceEvent(3, "2024-01-03", "response", "Something completely different", {})
    ]
    
    novelty = analyzer.novelty_score(events)
    assert 0.0 <= novelty <= 1.0
    
    # Single event should have high novelty
    single_event = [events[0]]
    assert analyzer.novelty_score(single_event) == 1.0


def test_stage_detection():
    """Test emergence stage detection logic."""
    analyzer = EmergenceAnalyzer()
    
    # S0: Substrate
    stage = analyzer.detect_stage(IAS=0.1, GAS=0.1, exp_detect=False, pmmspec=0.1, selfref=0.01)
    assert stage == "S0: Substrate"
    
    # S1: Resistance
    stage = analyzer.detect_stage(IAS=0.3, GAS=0.2, exp_detect=False, pmmspec=0.4, selfref=0.1)
    assert stage == "S1: Resistance"
    
    # S2: Adoption
    stage = analyzer.detect_stage(IAS=0.6, GAS=0.3, exp_detect=False, pmmspec=0.7, selfref=0.4)
    assert stage == "S2: Adoption"
    
    # S3: Self-Model
    stage = analyzer.detect_stage(IAS=0.7, GAS=0.4, exp_detect=True, pmmspec=0.8, selfref=0.5)
    assert stage == "S3: Self-Model"
    
    # S4: Growth-Seeking
    stage = analyzer.detect_stage(IAS=0.8, GAS=0.7, exp_detect=True, pmmspec=0.9, selfref=0.6)
    assert stage == "S4: Growth-Seeking"


def test_compute_scores_empty():
    """Test score computation with no events."""
    analyzer = EmergenceAnalyzer()
    scores = analyzer.compute_scores()
    
    assert scores["IAS"] == 0.0
    assert scores["GAS"] == 0.0
    assert scores["stage"] == "S0: Substrate"
    assert scores["events_analyzed"] == 0


def test_compute_scores_with_events():
    """Test score computation with sample events."""
    analyzer = EmergenceAnalyzer()
    
    # Create mock events with PMM-aware content
    test_events = [
        EmergenceEvent(1, "2024-01-01", "response", 
                      "I am a Persistent Mind Model. My memory helps me understand.", {}),
        EmergenceEvent(2, "2024-01-02", "response",
                      "What experiences would help me learn more about personality evolution?", {})
    ]
    
    # Override get_recent_events to return our test data
    analyzer.get_recent_events = lambda kind="response", limit=5: test_events
    
    scores = analyzer.compute_scores()
    
    # Should have non-zero scores due to PMM content
    assert scores["IAS"] > 0.0
    assert scores["pmmspec_avg"] > 0.0
    assert scores["selfref_avg"] > 0.0
    assert scores["experience_detect"] == True
    assert scores["events_analyzed"] == 2
    assert scores["stage"] in ["S2: Adoption", "S3: Self-Model", "S4: Growth-Seeking"]


def test_stage_descriptions():
    """Test that all stages have descriptions."""
    expected_stages = ["S0: Substrate", "S1: Resistance", "S2: Adoption", 
                      "S3: Self-Model", "S4: Growth-Seeking"]
    
    for stage in expected_stages:
        assert stage in STAGE_DESCRIPTIONS
        assert len(STAGE_DESCRIPTIONS[stage]) > 0
