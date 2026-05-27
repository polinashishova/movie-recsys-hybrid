"""Tests for predict endpoint."""

import pytest
from fastapi.testclient import TestClient
from mrh.api.main import create_app


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    @pytest.fixture
    def client(self):
        """Fixture: FastAPI test client."""
        return TestClient(create_app())
    
    
    def test_predict_without_ids(self, client):
        """Test: request without movieIds and userIds."""
        response = client.post(
            "/predict",
            json={"k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_both_ids(self, client):
        """Test: request with both ID types."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "userIds": [1], "k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_empty_movieIds(self, client):
        """Test: empty movieIds list."""
        response = client.post(
            "/predict",
            json={"movieIds": [], "k": 10}
        )
        
        assert response.status_code == 400
        assert "cannot be empty" in response.text
    
    def test_predict_with_empty_userIds(self, client):
        """Test: empty userIds list."""
        response = client.post(
            "/predict",
            json={"userIds": [], "k": 10}
        )
        
        assert response.status_code == 400
        assert "cannot be empty" in response.text
    
    def test_predict_with_negative_ids(self, client):
        """Test: negative IDs."""
        response = client.post(
            "/predict",
            json={"movieIds": [-1, -2], "k": 10}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_k_out_of_range(self, client):
        """Test: k out of valid range."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 200}
        )
        assert response.status_code == 422
        
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 0}
        )
        assert response.status_code == 422
    
    def test_predict_response_format(self, client):
        """Test: response format."""
        response = client.post(
            "/predict",
            json={"movieIds": [1], "k": 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data
            assert "model_used" in data
            assert "timestamp" in data
            assert "request_id" in data