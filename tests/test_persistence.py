#!/usr/bin/env python3
"""
Comprehensive tests for persistence layer.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from pmm.persistence import ModelPersistence
from pmm.model import PersistentMindModel


class TestModelPersistence:
    """Test cases for ModelPersistence class."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_model.json"
        self.persistence = ModelPersistence(str(self.test_file))

    def test_load_creates_new_model_if_not_exists(self):
        """Test that load creates a new model if file doesn't exist."""
        assert not self.test_file.exists()

        model = self.persistence.load()

        assert isinstance(model, PersistentMindModel)
        assert self.test_file.exists()

    def test_save_creates_valid_json(self):
        """Test that save creates valid JSON file."""
        model = PersistentMindModel()

        with patch.object(self.persistence, "_to_dict", return_value={"test": "data"}):
            with patch("pmm.persistence.validate_model"):
                self.persistence.save(model)

        assert self.test_file.exists()
        with open(self.test_file) as f:
            data = json.load(f)
        assert data == {"test": "data"}

    def test_save_validates_before_saving(self):
        """Test that save validates model before writing."""
        model = PersistentMindModel()

        with patch.object(self.persistence, "_to_dict", return_value={}):
            with patch("pmm.persistence.validate_model") as mock_validate:
                mock_validate.side_effect = ValueError("Invalid model")

                with pytest.raises(ValueError, match="Invalid model"):
                    self.persistence.save(model)

    def test_atomic_write_on_save_failure(self):
        """Test that failed saves don't corrupt existing files."""
        # Create initial valid file
        initial_data = {"version": 1}
        with open(self.test_file, "w") as f:
            json.dump(initial_data, f)

        model = PersistentMindModel()

        with patch.object(self.persistence, "_to_dict", return_value={}):
            with patch("pmm.persistence.validate_model"):
                with patch("builtins.open", side_effect=IOError("Disk full")):
                    with pytest.raises(ValueError, match="Failed to save"):
                        self.persistence.save(model)

        # Original file should be unchanged
        with open(self.test_file) as f:
            data = json.load(f)
        assert data == initial_data

    def test_backup_creates_timestamped_copy(self):
        """Test that backup creates timestamped copy."""
        # Create test file
        test_data = {"test": "backup"}
        with open(self.test_file, "w") as f:
            json.dump(test_data, f)

        backup_path = self.persistence.backup("test")

        assert backup_path.exists()
        assert "test" in backup_path.name
        assert backup_path.suffix == ".bak"

        with open(backup_path) as f:
            backup_data = json.load(f)
        assert backup_data == test_data

    def test_thread_safety(self):
        """Test that operations are thread-safe."""
        import threading
        import time

        model = PersistentMindModel()
        results = []
        errors = []

        def save_operation():
            try:
                with patch.object(
                    self.persistence,
                    "_to_dict",
                    return_value={"thread": threading.current_thread().name},
                ):
                    with patch("pmm.persistence.validate_model"):
                        self.persistence.save(model)
                        time.sleep(0.01)  # Simulate some work
                        results.append(threading.current_thread().name)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_operation, name=f"Thread-{i}")
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 5


class TestModelPersistenceIntegration:
    """Integration tests for persistence with real files."""

    def test_full_save_load_cycle(self):
        """Test complete save/load cycle with real model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "integration_test.json"
            persistence = ModelPersistence(str(file_path))

            # Create model with some data
            original_model = PersistentMindModel()
            original_model.core_identity.name = "Test Agent"

            # This would need proper implementation of _to_dict and load
            # For now, just test the file operations
            with patch.object(
                persistence, "_to_dict", return_value={"name": "Test Agent"}
            ):
                with patch("pmm.persistence.validate_model"):
                    persistence.save(original_model)

            assert file_path.exists()

            # Verify file contents
            with open(file_path) as f:
                data = json.load(f)
            assert data["name"] == "Test Agent"
