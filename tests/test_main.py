import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import src.events as events_mod
from src.events import create_queue, get_queue, publish
from src.main import app
from src.models import SSEEvent

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    events_mod._reset()
    yield
    events_mod._reset()


def _event(task_id: str, event: str = "test") -> SSEEvent:
    return SSEEvent(
        event=event,
        task_id=task_id,
        timestamp=datetime.now(UTC).isoformat(),
        data={"msg": "hello"},
    )


# --- existing smoke tests (kept as-is) ---


def test_create_task_accepts_valid_request():
    with patch("src.main.Orchestrator") as mock_orch:
        mock_orch.return_value.run = AsyncMock()
        response = client.post("/task", json={"request": "Summarize Q3 financials"})
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


def test_outputs_static_mount():
    response = client.get("/outputs/nonexistent.png")
    assert response.status_code == 404


# --- Phase 4: orchestrator wiring ---


def test_create_task_creates_event_queue():
    with patch("src.main.Orchestrator") as mock_orch:
        mock_orch.return_value.run = AsyncMock()
        response = client.post("/task", json={"request": "What is AI?"})
    task_id = response.json()["task_id"]
    assert get_queue(task_id) is not None


def test_create_task_launches_orchestrator():
    with patch("src.main.Orchestrator") as mock_orch:
        mock_orch.return_value.run = AsyncMock()
        client.post("/task", json={"request": "What is AI?"})

    assert mock_orch.return_value.run.called


def test_create_task_passes_clarifications_to_context():
    captured_contexts = []

    async def capture_run(ctx):
        captured_contexts.append(ctx)

    with patch("src.main.Orchestrator") as mock_orch:
        mock_orch.return_value.run = capture_run
        client.post(
            "/task",
            json={"request": "Analyze sales", "clarifications": ["Q3 only", "US market"]},
        )

    assert len(captured_contexts) == 1
    assert captured_contexts[0].clarifications == ["Q3 only", "US market"]
    assert captured_contexts[0].original_request == "Analyze sales"


def test_clarify_calls_submit_clarification():
    with patch("src.main.submit_clarification") as mock_submit:
        client.post("/task/abc-123/clarify", json={"answers": ["answer 1", "answer 2"]})
    mock_submit.assert_called_once_with("abc-123", ["answer 1", "answer 2"])


def test_stream_returns_404_for_unknown_task():
    with client.stream("GET", "/task/nonexistent-id/stream") as response:
        assert response.status_code == 404


def test_stream_sends_events_from_queue():
    task_id = "stream-test-task"
    create_queue(task_id)

    async def seed_queue():
        await publish(task_id, _event(task_id, "plan_ready"))
        q = get_queue(task_id)
        if q:
            await q.put(None)  # sentinel

    asyncio.run(seed_queue())

    with client.stream("GET", f"/task/{task_id}/stream") as response:
        assert response.status_code == 200
        lines = list(response.iter_lines())

    data_lines = [ln for ln in lines if ln.startswith("data:")]
    assert len(data_lines) >= 1
    payload = json.loads(data_lines[0].removeprefix("data:").strip())
    assert payload["event"] == "plan_ready"
    assert payload["task_id"] == task_id


def test_stream_content_type_is_sse():
    task_id = "ct-test"
    create_queue(task_id)

    async def close_queue():
        q = get_queue(task_id)
        if q:
            await q.put(None)

    asyncio.run(close_queue())

    with client.stream("GET", f"/task/{task_id}/stream") as response:
        assert "text/event-stream" in response.headers["content-type"]
