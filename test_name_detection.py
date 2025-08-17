#!/usr/bin/env python3
"""
Unit tests for name detection to prevent false positive regressions.
"""

import pytest
from pmm.name_detect import (
    extract_agent_name_command,
    _too_soon_since_last_name_change,
    _utcnow_str,
)
from datetime import datetime, timezone, timedelta


class TestNameDetection:
    """Test suite for strict name detection"""

    def test_does_not_rename_on_im_just(self):
        """Should not detect name from 'I'm just...' patterns"""
        assert extract_agent_name_command("I'm just trying to help", "user") is None
        assert extract_agent_name_command("I'm just continuing to work", "user") is None
        assert extract_agent_name_command("I'm just testing this", "user") is None

    def test_does_not_rename_on_im_continuing(self):
        """Should not detect name from 'I'm continuing...' patterns"""
        assert (
            extract_agent_name_command("I'm continuing to develop the PMM.", "user")
            is None
        )
        assert (
            extract_agent_name_command("I'm continuing with the project", "user")
            is None
        )

    def test_does_not_rename_on_common_stopwords(self):
        """Should not detect names from common stopwords"""
        stopword_phrases = [
            "I'm well, thank you",
            "I'm okay with that",
            "I'm sure about this",
            "I'm trying to understand",
            "I'm working on it",
            "I'm hello there",  # weird but possible
        ]
        for phrase in stopword_phrases:
            assert extract_agent_name_command(phrase, "user") is None

    def test_user_explicit_names_agent(self):
        """Should detect explicit user commands to name the agent"""
        assert extract_agent_name_command("Your name is Echo", "user") == "Echo"
        assert extract_agent_name_command("Let's call you Echo", "user") == "Echo"
        assert extract_agent_name_command("I'll call you Echo", "user") == "Echo"
        assert extract_agent_name_command("From now on, you are Echo", "user") == "Echo"
        assert extract_agent_name_command("Your name shall be Echo", "user") == "Echo"

    def test_assistant_self_declaration_ok(self):
        """Should detect valid assistant self-declarations"""
        assert extract_agent_name_command("I'm Echo", "assistant") == "Echo"
        assert extract_agent_name_command("I am Echo", "assistant") == "Echo"
        assert extract_agent_name_command("  I'm Echo  ", "assistant") == "Echo"

    def test_assistant_self_declaration_rejects_stopwords(self):
        """Should reject assistant self-declarations with stopwords"""
        assert extract_agent_name_command("I'm just", "assistant") is None
        assert extract_agent_name_command("I'm continuing", "assistant") is None
        assert extract_agent_name_command("I'm trying", "assistant") is None

    def test_user_commands_reject_stopwords(self):
        """Should reject user commands with stopwords"""
        assert extract_agent_name_command("Your name is just", "user") is None
        assert extract_agent_name_command("Let's call you continuing", "user") is None

    def test_case_insensitive_patterns(self):
        """Should work case-insensitively"""
        assert extract_agent_name_command("your name is Echo", "user") == "Echo"
        assert extract_agent_name_command("YOUR NAME IS Echo", "user") == "Echo"
        assert extract_agent_name_command("i'm Echo", "assistant") == "Echo"

    def test_name_validation_length(self):
        """Should validate name length constraints"""
        # Too short (single char)
        assert extract_agent_name_command("Your name is A", "user") is None
        # Valid length
        assert extract_agent_name_command("Your name is Echo", "user") == "Echo"
        # Very long name (should still work within 63 char limit)
        long_name = "A" * 63
        assert (
            extract_agent_name_command(f"Your name is {long_name}", "user") == long_name
        )

    def test_speaker_role_matters(self):
        """Should only detect self-declarations from assistant, commands from user"""
        # User saying "I'm Echo" should not be detected as agent naming
        assert extract_agent_name_command("I'm Echo", "user") is None
        # Assistant saying "Your name is Echo" should not be detected
        assert extract_agent_name_command("Your name is Echo", "assistant") is None

    def test_cooldown_functionality(self):
        """Should properly implement cooldown logic"""
        # No previous change - should allow
        assert not _too_soon_since_last_name_change(None, days=1)

        # Recent change - should block
        recent_time = _utcnow_str()
        assert _too_soon_since_last_name_change(recent_time, days=1)

        # Old change - should allow
        old_time = (datetime.now(timezone.utc) - timedelta(days=2)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        assert not _too_soon_since_last_name_change(old_time, days=1)

        # Invalid timestamp - should allow (fail safe)
        assert not _too_soon_since_last_name_change("invalid-timestamp", days=1)

    def test_empty_or_none_input(self):
        """Should handle empty or None inputs gracefully"""
        assert extract_agent_name_command("", "user") is None
        assert extract_agent_name_command(None, "user") is None
        assert extract_agent_name_command("", "assistant") is None
        assert extract_agent_name_command(None, "assistant") is None

    def test_complex_sentences(self):
        """Should work in complex sentences"""
        assert (
            extract_agent_name_command(
                "Well, I think your name is Echo and that's final.", "user"
            )
            == "Echo"
        )
        assert (
            extract_agent_name_command(
                "Hello everyone, I'm Echo and I'm here to help.", "assistant"
            )
            == "Echo"
        )

    def test_punctuation_handling(self):
        """Should handle punctuation correctly"""
        assert extract_agent_name_command("Your name is Echo.", "user") == "Echo"
        assert extract_agent_name_command("I'm Echo!", "assistant") == "Echo"
        assert extract_agent_name_command("Let's call you Echo?", "user") == "Echo"


if __name__ == "__main__":
    pytest.main([__file__])
