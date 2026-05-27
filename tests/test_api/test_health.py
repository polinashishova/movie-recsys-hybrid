"""Tests for health endpoint."""

import pytest
from fastapi.testclient import TestClient
from mrh.api.main import create_app


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    @pytest.fixture
    def client(self):
        """Fixture: FastAPI test client."""
        return TestClient(create_app())
    
    def test_health_endpoint_returns_503(self, client):
        """Test: health endpoint returns 503 because models are not ready."""
        response = client.get("/health")
        assert response.status_code == 503
    
    def test_health_response_format(self, client):
        """Test: health endpoint response format."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "models_loaded" in data
        assert "timestamp" in data
        assert data["service"] == "movie-recsys-hybrid"
    
    def test_root_endpoint(self, client):
        """Test: root endpoint."""
        response = client.get("/")
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert "predict" in data