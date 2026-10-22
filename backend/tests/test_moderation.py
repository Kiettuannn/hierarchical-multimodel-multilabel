import io
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_list_categories():
    """Verify categories endpoint returns all seven predefined Harm Categories."""
    response = client.get("/api/v1/moderation/categories")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 7
    category_names = [item["category"] for item in data]
    assert "information_harm" in category_names
    assert "sexual_harm" in category_names
    assert "psychological_harm" in category_names


def test_submit_job_and_get_status():
    """Verify video submission creates a job and returns accepted status."""
    dummy_video = io.BytesIO(b"fake video data")
    response = client.post(
        "/api/v1/moderation/jobs",
        files={"file": ("sample_test.mp4", dummy_video, "video/mp4")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["video_filename"] == "sample_test.mp4"
    assert data["status"] in ["PENDING", "PROCESSING", "COMPLETED"]

    # Verify retrieval
    job_id = data["job_id"]
    get_res = client.get(f"/api/v1/moderation/jobs/{job_id}")
    assert get_res.status_code == 200
    assert get_res.json()["job_id"] == job_id
