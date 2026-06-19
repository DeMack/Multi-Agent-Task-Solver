import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.events as events_mod
from src.config import Config
from src.events import create_queue, get_queue, submit_clarification
from src.models import SSEEvent, SubTask, TaskContext, TaskGraph
from src.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def reset_state():
    events_mod._reset()
    yield
    events_mod._reset()


def _config(max_retries: int = 0) -> Config:
    return Config(
        anthropic_api_key="test-key",
        agent_timeout_seconds=5,
        code_execution_timeout_seconds=5,
        max_agent_retries=max_retries,
        outputs_dir=Path("outputs"),
    )


def _context(request: str = "What is AI?") -> TaskContext:
    return TaskContext(task_id="test-task", original_request=request, clarifications=[])


def _clear_client() -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = "CLEAR"
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _questions_client(questions: list[str]) -> MagicMock:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    block = MagicMock()
    block.type = "text"
    block.text = numbered
    resp = MagicMock()
    resp.content = [block]
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


def _agg_result() -> dict:
    return {"answer": "42", "artifacts": [], "warnings": []}


def _simple_graph() -> TaskGraph:
    return TaskGraph(
        subtasks=[
            SubTask(id="t1", description="aggregate results", agent="aggregator", depends_on=[]),
        ]
    )


def _research_graph() -> TaskGraph:
    return TaskGraph(
        subtasks=[
            SubTask(id="t1", description="research AI", agent="research", depends_on=[]),
            SubTask(id="t2", description="aggregate", agent="aggregator", depends_on=["t1"]),
        ]
    )


async def _drain(task_id: str) -> list[SSEEvent]:
    q = get_queue(task_id)
    results: list[SSEEvent] = []
    if q is None:
        return results
    while not q.empty():
        item = await q.get()
        if item is None:
            break
        results.append(item)
    return results


# --- _parse_questions (pure unit tests) ---


def test_parse_questions_returns_empty_for_clear():
    orch = Orchestrator(_clear_client(), _config())
    assert orch._parse_questions("CLEAR") == []


def test_parse_questions_case_insensitive():
    orch = Orchestrator(_clear_client(), _config())
    assert orch._parse_questions("clear") == []


def test_parse_questions_extracts_numbered_questions():
    orch = Orchestrator(_clear_client(), _config())
    result = orch._parse_questions("1. What time period?\n2. Which region?")
    assert result == ["What time period?", "Which region?"]


def test_parse_questions_strips_leading_number_and_dot():
    orch = Orchestrator(_clear_client(), _config())
    result = orch._parse_questions("1. First?\n2. Second?")
    assert result[0] == "First?"
    assert result[1] == "Second?"


# --- run() integration-style unit tests ---


@patch("src.orchestrator.Planner")
async def test_run_invokes_planner(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    mock_planner_class.return_value.plan.assert_called_once_with(ctx)


@patch("src.orchestrator.Planner")
async def test_run_emits_plan_ready_event(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    assert any(e.event == "plan_ready" for e in events)


@patch("src.orchestrator.Planner")
async def test_run_emits_result_ready_with_aggregator_output(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = {"answer": "42", "artifacts": [], "warnings": []}
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    result_evts = [e for e in events if e.event == "result_ready"]
    assert len(result_evts) == 1
    assert result_evts[0].data["result"]["answer"] == "42"


@patch("src.orchestrator.Planner")
async def test_run_skips_clarification_when_request_is_clear(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    assert not any(e.event == "clarification_needed" for e in events)


@patch("src.orchestrator.Planner")
async def test_run_emits_clarification_needed_for_ambiguous_request(mock_planner_class: MagicMock):
    questions = ["What time period?", "Which market?"]
    ctx = _context()
    create_queue(ctx.task_id)

    async def respond() -> None:
        await asyncio.sleep(0.1)
        submit_clarification(ctx.task_id, ["Last 3 months", "US market"])

    asyncio.create_task(respond())
    mock_planner_class.return_value.plan.return_value = _simple_graph()

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_questions_client(questions), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    clarify_evts = [e for e in events if e.event == "clarification_needed"]
    assert len(clarify_evts) == 1
    assert clarify_evts[0].data["questions"] == questions


@patch("src.orchestrator.Planner")
async def test_run_skips_clarification_when_preanswered(mock_planner_class: MagicMock):
    """If clarifications were submitted upfront, skip the detection LLM call."""
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = TaskContext(
        task_id="test-task",
        original_request="What is AI?",
        clarifications=["already answered"],
    )
    create_queue(ctx.task_id)
    client = _questions_client(["Should not be asked"])  # would ask if detection ran

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(client, _config()).run(ctx)

    # client.messages.create should NOT have been called (clarification skipped)
    client.messages.create.assert_not_called()


@patch("src.orchestrator.Planner")
async def test_run_dispatches_research_agent(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.return_value = "research findings"
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    mock_research.return_value.run.assert_called_once()


@patch("src.orchestrator.Planner")
async def test_run_emits_agent_started_and_completed(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    event_types = [e.event for e in events]
    assert "agent_started" in event_types
    assert "agent_completed" in event_types


@patch("src.orchestrator.Planner")
async def test_run_handles_agent_failure(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = RuntimeError("search failed")
        mock_agg.return_value.run.return_value = {
            "answer": "partial",
            "artifacts": [],
            "warnings": ["t1 failed"],
        }
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    failed_evts = [e for e in events if e.event == "agent_failed"]
    assert len(failed_evts) >= 1
    assert failed_evts[0].data["subtask_id"] == "t1"


@patch("src.orchestrator.Planner")
async def test_run_retries_failed_subtask(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    call_count = 0

    def flaky(*_args):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("temporary failure")
        return "success"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = flaky
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config(max_retries=2)).run(ctx)

    assert call_count == 3  # 1 initial + 2 retries


@patch("src.orchestrator.Planner")
async def test_run_respects_subtask_dependencies(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = TaskGraph(
        subtasks=[
            SubTask(id="t1", description="research", agent="research", depends_on=[]),
            SubTask(id="t2", description="summary", agent="summary", depends_on=["t1"]),
            SubTask(id="t3", description="aggregate", agent="aggregator", depends_on=["t2"]),
        ]
    )
    ctx = _context()
    create_queue(ctx.task_id)
    execution_order: list[str] = []

    def make_runner(sid: str, return_value: object):
        def run(*_args):
            execution_order.append(sid)
            return return_value

        return run

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.SummaryAgent") as mock_summary,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = make_runner("t1", "research")
        mock_summary.return_value.run.side_effect = make_runner("t2", "summary")
        mock_agg.return_value.run.side_effect = make_runner("t3", _agg_result())
        await Orchestrator(_clear_client(), _config()).run(ctx)

    assert execution_order.index("t1") < execution_order.index("t2")
    assert execution_order.index("t2") < execution_order.index("t3")
