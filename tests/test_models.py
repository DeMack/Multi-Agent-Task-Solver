import pytest
from pydantic import ValidationError

from src.models import (
    ClarifyRequest,
    ClarifyResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    SubTask,
    TaskContext,
    TaskGraph,
    TaskStatus,
)


def test_task_status_values():
    assert TaskStatus.pending == "pending"
    assert TaskStatus.running == "running"
    assert TaskStatus.completed == "completed"
    assert TaskStatus.failed == "failed"


def test_subtask_valid():
    task = SubTask(
        id="t1", description="Research quarterly financials", agent="research", depends_on=[]
    )
    assert task.id == "t1"
    assert task.agent == "research"
    assert task.depends_on == []


def test_subtask_all_valid_agent_types():
    for agent in ("research", "code", "summary", "aggregator"):
        task = SubTask(id="t", description="x", agent=agent, depends_on=[])
        assert task.agent == agent


def test_subtask_rejects_invalid_agent():
    with pytest.raises(ValidationError):
        SubTask(id="t1", description="...", agent="unknown_agent", depends_on=[])  # type: ignore[arg-type]


def test_task_graph_holds_subtasks():
    graph = TaskGraph(
        subtasks=[
            SubTask(id="t1", description="Research", agent="research", depends_on=[]),
            SubTask(id="t2", description="Summarise", agent="summary", depends_on=["t1"]),
        ]
    )
    assert len(graph.subtasks) == 2
    assert graph.subtasks[1].depends_on == ["t1"]


def test_task_graph_empty_is_valid():
    graph = TaskGraph(subtasks=[])
    assert graph.subtasks == []


def test_task_context_required_fields():
    ctx = TaskContext(task_id="abc-123", original_request="Do something", clarifications=[])
    assert ctx.task_id == "abc-123"
    assert ctx.original_request == "Do something"
    assert ctx.clarifications == []


def test_task_context_defaults():
    ctx = TaskContext(task_id="abc", original_request="x", clarifications=[])
    assert ctx.plan is None
    assert ctx.agent_outputs == {}
    assert ctx.status == {}


def test_task_context_accepts_plan():
    graph = TaskGraph(
        subtasks=[
            SubTask(id="t1", description="Research", agent="research", depends_on=[]),
        ]
    )
    ctx = TaskContext(
        task_id="abc",
        original_request="Analyse something",
        clarifications=["Q1 answer"],
        plan=graph,
    )
    assert ctx.plan is not None
    assert len(ctx.plan.subtasks) == 1


def test_task_context_status_uses_enum():
    ctx = TaskContext(
        task_id="abc",
        original_request="x",
        clarifications=[],
        status={"t1": TaskStatus.running},
    )
    assert ctx.status["t1"] == TaskStatus.running


def test_create_task_request_requires_request_field():
    req = CreateTaskRequest(request="Summarise Q3 financials")
    assert req.request == "Summarise Q3 financials"
    assert req.clarifications == []


def test_create_task_request_missing_request_raises():
    with pytest.raises(ValidationError):
        CreateTaskRequest()  # type: ignore[call-arg]


def test_create_task_response_shape():
    resp = CreateTaskResponse(task_id="xyz", status="pending")
    assert resp.task_id == "xyz"
    assert resp.status == "pending"


def test_clarify_request_shape():
    req = ClarifyRequest(answers=["Q1", "Q2"])
    assert req.answers == ["Q1", "Q2"]


def test_clarify_response_shape():
    resp = ClarifyResponse(task_id="xyz", status="resumed")
    assert resp.task_id == "xyz"
    assert resp.status == "resumed"
