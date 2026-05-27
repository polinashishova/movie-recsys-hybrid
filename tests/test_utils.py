"""Tests for utility functions."""

import pytest
import json
from pathlib import Path
import tempfile
from mrh.utils import load_json, save_json, inverse_dict, setup_logging


class TestInverseDict:
    """Tests for inverse_dict function."""
    
    def test_normal_dict(self):
        """Test: normal dictionary."""
        d = {1: 'a', 2: 'b', 3: 'c'}
        expected = {'a': 1, 'b': 2, 'c': 3}
        assert inverse_dict(d) == expected
    
    def test_empty_dict(self):
        """Test: empty dictionary."""
        assert inverse_dict({}) == {}
    
    def test_dict_with_duplicate_values(self):
        """Test: dictionary with duplicate values."""
        d = {1: 'a', 2: 'a', 3: 'b'}
        result = inverse_dict(d)
        assert len(result) == 2
        assert result['a'] == 2  # Last value
        assert result['b'] == 3
    
    def test_invalid_input(self):
        """Test: invalid input (not a dictionary)."""
        with pytest.raises(TypeError, match="Expected dictionary"):
            inverse_dict([1, 2, 3])


class TestJsonOperations:
    """Tests for JSON operations."""
    
    def test_save_and_load_json(self):
        """Test: save and load JSON."""
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            save_json(data, tmp_path)
            loaded_data = load_json(tmp_path)
            assert loaded_data == data
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test: load non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_json(Path("/nonexistent/path/file.json"))
    
    def test_load_invalid_json(self):
        """Test: load invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write("{invalid json}")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_json(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


class TestSetupLogging:
    """Tests for logging setup."""
    
    def test_setup_logging_creates_log_dir(self, tmp_path):
        """Test: log directory is created."""
        log_dir = tmp_path / "logs"
        assert not log_dir.exists()
        
        setup_logging(level="INFO", log_dir=log_dir, log_filename="test.log")
        
        assert log_dir.exists()
    
    def test_setup_logging_returns_logger(self, tmp_path):
        """Test: function returns a logger."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(level="INFO", log_dir=log_dir, log_filename="test.log")
        
        assert logger is not None
        assert logger.level == 20  # logging.INFO = 20