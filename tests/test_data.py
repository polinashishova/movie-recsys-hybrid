"""Tests for data processing functions."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import tempfile
from mrh.data import (
    preprocess_tags, 
    get_ratings_by_threshold,
    save_data,
    load_data
)


class TestPreprocessTags:
    """Tests for tag preprocessing."""
    
    def test_normal_string(self):
        """Test: normal string."""
        assert preprocess_tags("Hello, World!") == "hello world"
    
    def test_string_with_punctuation(self):
        """Test: string with punctuation."""
        assert preprocess_tags("Fantasy! Action? Drama...") == "fantasy action drama"
    
    def test_uppercase_string(self):
        """Test: uppercase string."""
        assert preprocess_tags("COMEDY ROMANCE") == "comedy romance"
    
    def test_empty_string(self):
        """Test: empty string."""
        assert preprocess_tags("") == ""
    
    def test_nan_value(self):
        """Test: NaN value."""
        assert preprocess_tags(float('nan')) == ""
    
    def test_none_value(self):
        """Test: None value."""
        assert preprocess_tags(None) == ""
    
    def test_russian_letters(self):
        """Test: Russian letters."""
        assert preprocess_tags("Привет, Мир! 123") == "привет мир 123"
    
    def test_numbers_only(self):
        """Test: numbers only."""
        assert preprocess_tags("123 456") == "123 456"
    
    def test_invalid_type(self):
        """Test: invalid type."""
        with pytest.raises(TypeError, match="must be str, float or None"):
            preprocess_tags(["list", "is", "invalid"])


class TestGetRatingsByThreshold:
    """Tests for rating filtering by threshold."""
    
    @pytest.fixture
    def sample_ratings(self):
        """Fixture: sample rating data."""
        return pd.DataFrame({
            'userId': [1, 1, 2, 2, 3, 3],
            'movieId': [101, 102, 201, 202, 301, 302],
            'rating': [2.5, 4.0, 3.5, 5.0, 1.5, 4.5],
            'timestamp': [1000, 1001, 2000, 2001, 3000, 3001]
        })
    
    def test_split_at_threshold(self, sample_ratings):
        """Test: split by threshold."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=4.0)
        
        assert len(neg) == 3
        assert all(rating < 4.0 for rating in neg['rating'])
        
        assert len(pos) == 3
        assert all(rating >= 4.0 for rating in pos['rating'])
    
    def test_threshold_3_5(self, sample_ratings):
        """Test: threshold 3.5."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=3.5)
        
        assert len(neg) == 2
        assert len(pos) == 4
    
    def test_threshold_at_min(self, sample_ratings):
        """Test: threshold at minimum value."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=1.5)
        
        assert len(neg) == 0  
        assert len(pos) == 6   
    
    def test_threshold_at_max(self, sample_ratings):
        """Test: threshold at maximum value."""
        neg, pos = get_ratings_by_threshold(sample_ratings, threshold=5.0)
        
        assert len(neg) == 5 
        assert len(pos) == 1
    
    def test_invalid_ratings_type(self):
        """Test: ratings is not a DataFrame."""
        with pytest.raises(TypeError):
            get_ratings_by_threshold([1, 2, 3], 4.0)
    
    def test_missing_columns(self):
        """Test: missing required columns."""
        df = pd.DataFrame({'wrong_col': [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            get_ratings_by_threshold(df, 4.0)
    
    def test_empty_dataframe(self):
        """Test: empty DataFrame."""
        df = pd.DataFrame(columns=['movieId', 'rating'])
        with pytest.raises(ValueError, match="empty"):
            get_ratings_by_threshold(df, 4.0)


class TestSaveLoadData:
    """Tests for saving and loading data."""
    
    def test_save_and_load_dataframe(self, tmp_path):
        """Test: save and load DataFrame."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        file_path = tmp_path / "test_df.csv"
        
        save_data(file_path, df)
        assert file_path.exists()
        
        loaded = load_data(file_path)
        pd.testing.assert_frame_equal(df, loaded)
    
    def test_save_and_load_numpy_array(self, tmp_path):
        """Test: save and load numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        file_path = tmp_path / "test_array.csv"
        
        save_data(file_path, arr)
        assert file_path.exists()
        
        loaded = load_data(file_path)
        assert isinstance(loaded, pd.DataFrame)
        np.testing.assert_array_equal(arr, loaded.values)
    
    def test_save_existing_file_skip(self, tmp_path, caplog):
        """Test: existing file is skipped."""
        caplog.set_level(logging.INFO)
        df = pd.DataFrame({'col': [1, 2]})
        file_path = tmp_path / "existing.csv"

        save_data(file_path, df)
        assert file_path.exists()
        
        save_data(file_path, df)
        
        assert "already exists" in caplog.text
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test: load non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.npz")
    
    def test_load_unsupported_format(self, tmp_path):
        """Test: load unsupported format."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("some text")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(file_path)