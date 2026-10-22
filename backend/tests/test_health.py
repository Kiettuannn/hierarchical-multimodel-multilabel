from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_healthy():
    """Verify health endpoint returns HTTP 200 with expected schema."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "inference_mode" in data
    assert "app_name" in data
