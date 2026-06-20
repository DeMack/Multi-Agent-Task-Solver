"""
End-to-end integration test — exercises the full pipeline against the real API.

Run with: pytest --integration tests/integration/test_e2e.py -v -s

This test calls the Anthropic API and may take several minutes to complete.
"""

import os
import uuid

import pytest
from anthropic import Anthropic

import src.events as events_mod
from src.config import config
from src.events import arm_user_messages, create_queue, get_queue
from src.models import SubTask, TaskContext, TaskGraph
from src.orchestrator import Orchestrator

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(autouse=True)
def reset_events():
    events_mod._reset()
    yield
    events_mod._reset()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_summary_with_chart():
    """
    Submits a request that requires research (data gathering) and code (chart creation).
    Verifies the complete event sequence and final result shape.
    """
    client = Anthropic()
    task_id = "e2e-" + uuid.uuid4().hex[:8]

    # Pre-seed clarifications so the orchestrator skips the clarification phase
    # and proceeds straight to planning — avoids hanging on human input.
    context = TaskContext(
        task_id=task_id,
        original_request=(
            "Research the approximate annual returns of the S&P 500, NASDAQ, and Dow Jones "
            "for 2023 and 2024, then write Python code to create a grouped bar chart "
            "comparing their returns and save it as a PNG file."
        ),
        clarifications=["Use approximate, publicly known annual return figures."],
    )
    create_queue(task_id)

    await Orchestrator(client, config).run(context)

    # Drain the queue. run() always closes it (finally block), so all events
    # plus the None sentinel are present before we start reading.
    queue = get_queue(task_id)
    assert queue is not None, "Queue was garbage-collected before drain"

    events = []
    while True:
        item = await queue.get()
        if item is None:
            break
        events.append(item)

    event_types = [e.event for e in events]

    # ── Required events ──────────────────────────────────────────────
    assert "plan_ready" in event_types, f"No plan_ready. Got: {event_types}"
    assert "result_ready" in event_types, f"No result_ready. Got: {event_types}"

    # ── At least one agent ran ───────────────────────────────────────
    started = [e for e in events if e.event == "agent_started"]
    assert len(started) >= 1, "No agent_started events found"

    # ── Event ordering: plan before agents before result ─────────────
    plan_idx = event_types.index("plan_ready")
    result_idx = event_types.index("result_ready")
    assert plan_idx < result_idx, "plan_ready must precede result_ready"
    for e in started:
        assert event_types.index(e.event) > plan_idx, "agent_started before plan_ready"

    # ── Final result shape ───────────────────────────────────────────
    result_event = next(e for e in events if e.event == "result_ready")
    result = result_event.data["result"]

    assert isinstance(result.get("answer"), str), "result.answer must be a string"
    assert len(result["answer"]) > 20, "result.answer is suspiciously short"
    assert isinstance(result.get("artifacts"), list), "result.artifacts must be a list"
    assert isinstance(result.get("warnings"), list), "result.warnings must be a list"


@pytest.mark.integration
async def test_handle_user_message_calls_claude_and_returns_ack():
    """
    Verifies the mid-run message path makes a real Claude API call and returns
    a valid acknowledgment. Calls _handle_user_messages directly to avoid
    timing-dependent injection during a live execution.
    """
    client = Anthropic()
    task_id = "msg-" + uuid.uuid4().hex[:8]
    create_queue(task_id)
    arm_user_messages(task_id)

    context = TaskContext(
        task_id=task_id,
        original_request="Research AI trends and create a bar chart",
        clarifications=[],
        plan=TaskGraph(
            subtasks=[
                SubTask(id="t1", description="research AI trends", agent="research", depends_on=[]),
                SubTask(id="t2", description="create bar chart", agent="code", depends_on=["t1"]),
                SubTask(id="t3", description="aggregate", agent="aggregator", depends_on=["t2"]),
            ]
        ),
    )

    orch = Orchestrator(client, config)
    directive = await orch._handle_user_messages(
        messages=["skip the chart, I only need the research summary"],
        context=context,
        pending={"t2", "t3"},
        completed={"t1"},
    )

    # Drain SSE queue to find the ack event
    q = get_queue(task_id)
    assert q is not None
    events = []
    while not q.empty():
        item = await q.get()
        if item is not None:
            events.append(item)

    assert isinstance(directive.acknowledgment, str)
    assert len(directive.acknowledgment) > 0

    # Drain SSE queue to find the ack event
    q = get_queue(task_id)
    assert q is not None
    events = []
    while not q.empty():
        item = await q.get()
        if item is not None:
            events.append(item)

    ack_events = [e for e in events if e.event == "user_message_ack"]
    assert len(ack_events) == 1, f"Expected 1 user_message_ack, got: {[e.event for e in events]}"
    ack = ack_events[0].data
    assert isinstance(ack["acknowledgment"], str)
    assert len(ack["acknowledgment"]) > 0
    assert isinstance(ack["reuse_plan"], bool)
    assert isinstance(ack["restarted_subtask_ids"], list)
    assert isinstance(ack["skipped_subtask_ids"], list)
