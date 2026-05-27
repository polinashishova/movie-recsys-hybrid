"""Tests for models and recommendations."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from implicit.cpu.als import AlternatingLeastSquares
import scipy.sparse as sparse

from mrh.models import (
    recommend_by_movieId,
    recommend_by_userId,
    save_model,
    load_model
)


class TestRecommendByMovieId:
    """Tests for recommendations by movie ID."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture: test data for recommendations."""
        # 5 movies with 3 features
        movie_features = np.array([
            [1.0, 0.0, 0.0],  # Movie 1
            [0.0, 1.0, 0.0],  # Movie 2
            [0.0, 0.0, 1.0],  # Movie 3
            [0.8, 0.2, 0.0],  # Movie 4 (similar to movie 1)
            [0.0, 0.7, 0.3],  # Movie 5 (similar to movie 2)
        ])
        
        movieId_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        idx_to_movieId = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
        
        return movie_features, movieId_to_idx, idx_to_movieId
    
    def test_recommend_single_movie(self, sample_data):
        """Test: recommendations for a single movie."""
        features, mid_to_idx, idx_to_mid = sample_data
        
        recs = recommend_by_movieId(1, features, mid_to_idx, idx_to_mid, k=2)
        
        assert len(recs) == 1
        movie_ids, scores = recs[0]
        assert len(movie_ids) == 2
        assert len(scores) == 2
        # Movie 4 should be the most similar to movie 1
        assert movie_ids[0] == 4
    
    def test_recommend_multiple_movies(self, sample_data):
        """Test: recommendations for multiple movies."""
        features, mid_to_idx, idx_to_mid = sample_data
        
        recs = recommend_by_movieId([1, 2], features, mid_to_idx, idx_to_mid, k=2)
        
        assert len(recs) == 2
        # First movie
        movie_ids_1, scores_1 = recs[0]
        assert len(movie_ids_1) == 2
        # Second movie
        movie_ids_2, scores_2 = recs[1]
        assert len(movie_ids_2) == 2
    
    def test_recommend_k_bigger_than_available(self, sample_data):
        """Test: k larger than available number of movies."""
        features, mid_to_idx, idx_to_mid = sample_data
        k = 10
        
        recs = recommend_by_movieId(1, features, mid_to_idx, idx_to_mid, k=k)
        
        movie_ids, scores = recs[0]
        # Should return all available movies (except itself)
        assert len(movie_ids) == 4  # 5 movies - 1 requested
        assert len(scores) == 4
    
    def test_invalid_movie_id(self, sample_data):
        """Test: non-existent movie ID."""
        features, mid_to_idx, idx_to_mid = sample_data
        
        with pytest.raises(KeyError, match="Movie 999 not found"):
            recommend_by_movieId(999, features, mid_to_idx, idx_to_mid, k=2)
    
    def test_invalid_k(self, sample_data):
        """Test: invalid k value."""
        features, mid_to_idx, idx_to_mid = sample_data
        
        with pytest.raises(ValueError, match="k must be greater than 0"):
            recommend_by_movieId(1, features, mid_to_idx, idx_to_mid, k=0)
        
        with pytest.raises(ValueError, match="k must be greater than 0"):
            recommend_by_movieId(1, features, mid_to_idx, idx_to_mid, k=-5)
    
    def test_empty_features(self, sample_data):
        """Test: empty feature matrix."""
        features, mid_to_idx, idx_to_mid = sample_data
        empty_features = np.array([])
        
        with pytest.raises(ValueError, match="movie_features matrix is empty"):
            recommend_by_movieId(1, empty_features, mid_to_idx, idx_to_mid, k=2)


class TestRecommendByUserId:
    """Tests for recommendations by user ID."""
    
    @pytest.fixture
    def sample_cf_data(self):
        """Fixture: test data for CF recommendations."""
        # Simple ALS model
        model = AlternatingLeastSquares(factors=2, iterations=5, random_state=42)
        
        # 3 users x 4 movies matrix
        user_item_matrix = sparse.csr_matrix([
            [1, 0, 1, 0],  # User 1: liked movies 1 and 3
            [0, 1, 0, 1],  # User 2: liked movies 2 and 4
            [1, 1, 0, 0],  # User 3: liked movies 1 and 2
        ])
        
        model.fit(user_item_matrix)
        
        movieId_to_idx = {101: 0, 102: 1, 103: 2, 104: 3}
        userId_to_idx = {1001: 0, 1002: 1, 1003: 2}
        idx_to_movieId = {0: 101, 1: 102, 2: 103, 3: 104}
        idx_to_userId = {0: 1001, 1: 1002, 2: 1003}
        
        return model, user_item_matrix, movieId_to_idx, userId_to_idx, idx_to_movieId, idx_to_userId
    
    def test_recommend_single_user(self, sample_cf_data):
        """Test: recommendations for a single user."""
        (model, matrix, mid_to_idx, uid_to_idx, idx_to_mid, idx_to_uid) = sample_cf_data
        
        recs = recommend_by_userId(
            userIds=1001,
            als_model=model,
            user_item_matrix=matrix,
            movieId_to_idx=mid_to_idx,
            userId_to_idx=uid_to_idx,
            idx_to_movieId=idx_to_mid,
            idx_to_userId=idx_to_uid,
            k=2
        )
        
        assert len(recs) == 1
        movie_ids, scores = recs[0]
        assert len(movie_ids) == 2
        assert len(scores) == 2
    
    def test_recommend_multiple_users(self, sample_cf_data):
        """Test: recommendations for multiple users."""
        (model, matrix, mid_to_idx, uid_to_idx, idx_to_mid, idx_to_uid) = sample_cf_data
        
        recs = recommend_by_userId(
            userIds=[1001, 1002],
            als_model=model,
            user_item_matrix=matrix,
            movieId_to_idx=mid_to_idx,
            userId_to_idx=uid_to_idx,
            idx_to_movieId=idx_to_mid,
            idx_to_userId=idx_to_uid,
            k=2
        )
        
        assert len(recs) == 2
    
    def test_recommend_with_neg_movies(self, sample_cf_data):
        """Test: recommendations with movie exclusion."""
        (model, matrix, mid_to_idx, uid_to_idx, idx_to_mid, idx_to_uid) = sample_cf_data
        
        # Exclude movie 101 for user 1001
        recs = recommend_by_userId(
            userIds=1001,
            als_model=model,
            user_item_matrix=matrix,
            movieId_to_idx=mid_to_idx,
            userId_to_idx=uid_to_idx,
            idx_to_movieId=idx_to_mid,
            idx_to_userId=idx_to_uid,
            k=3,
            neg_movieIds=101
        )
        
        movie_ids, scores = recs[0]
        # Movie 101 should not be in recommendations
        assert 101 not in movie_ids
    
    def test_invalid_user_id(self, sample_cf_data):
        """Test: non-existent user ID."""
        (model, matrix, mid_to_idx, uid_to_idx, idx_to_mid, idx_to_uid) = sample_cf_data
        
        with pytest.raises(KeyError, match="User 9999 not found"):
            recommend_by_userId(
                userIds=9999,
                als_model=model,
                user_item_matrix=matrix,
                movieId_to_idx=mid_to_idx,
                userId_to_idx=uid_to_idx,
                idx_to_movieId=idx_to_mid,
                idx_to_userId=idx_to_uid,
                k=2
            )


class TestSaveLoadModel:
    """Tests for saving and loading models."""
    
    @pytest.fixture
    def sample_pipeline(self):
        """Fixture: simple Pipeline for testing."""
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD(n_components=2, random_state=42))
        ])
        
        texts = ["movie action thriller", "comedy romance", "drama biography"]
        pipeline.fit(texts)
        
        return pipeline
    
    def test_save_and_load_pipeline(self, tmp_path, sample_pipeline):
        """Test: save and load Pipeline."""
        file_path = tmp_path / "test_pipeline.joblib"
        
        save_model(sample_pipeline, file_path)
        assert file_path.exists()
        
        loaded = load_model(file_path)
        assert isinstance(loaded, Pipeline)
        
        texts = ["new movie"]
        original_pred = sample_pipeline.transform(texts)
        loaded_pred = loaded.transform(texts)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_save_and_load_als_model(self, tmp_path):
        """Test: save and load ALS model."""
        model = AlternatingLeastSquares(factors=2, iterations=5, random_state=42)

        matrix = sparse.csr_matrix([[1, 0], [0, 1], [1, 1]])
        model.fit(matrix)
        
        file_path = tmp_path / "test_als.joblib"
        
        save_model(model, file_path)
        assert file_path.exists()
        
        loaded = load_model(file_path)
        assert isinstance(loaded, AlternatingLeastSquares)
        assert loaded.user_factors is not None
        assert loaded.item_factors is not None
    
    def test_save_existing_file_skip(self, tmp_path, sample_pipeline, caplog):
        """Test: existing file is skipped."""
        caplog.set_level(logging.INFO)
        file_path = tmp_path / "existing.joblib"

        save_model(sample_pipeline, file_path)
        assert file_path.exists()

        save_model(sample_pipeline, file_path)
        
        assert "already exists" in caplog.text
    
    def test_load_nonexistent_model(self, tmp_path):
        """Test: load non-existent model."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")
    
    def test_save_invalid_model(self, tmp_path):
        """Test: save invalid model."""
        with pytest.raises(TypeError):
            save_model("not a model", tmp_path / "invalid.joblib")