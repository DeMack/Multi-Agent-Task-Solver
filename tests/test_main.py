from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_create_task_accepts_valid_request():
    response = client.post("/task", json={"request": "Summarise Q3 financials"})
    assert response.status_code == 200
    body = response.json()
    assert "task_id" in body
    assert body["status"] == "pending"


def test_create_task_rejects_missing_request_field():
    response = client.post("/task", json={})
    assert response.status_code == 422


def test_clarify_route_exists():
    response = client.post("/task/some-id/clarify", json={"answers": ["answer 1"]})
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "some-id"
    assert body["status"] == "resumed"


def test_stream_route_exists():
    # SSE endpoints return 200 with text/event-stream content type
    with client.stream("GET", "/task/some-id/stream") as response:
        assert response.status_code == 200


def test_outputs_static_mount():
    # The /outputs path should be mounted (404 on a missing file, not a routing error)
    response = client.get("/outputs/nonexistent.png")
    assert response.status_code == 404
