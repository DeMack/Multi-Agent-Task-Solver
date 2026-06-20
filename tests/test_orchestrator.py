import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.events as events_mod
from src.config import Config
from src.events import (
    create_queue,
    get_context,
    get_queue,
    submit_clarification,
    submit_user_message,
)
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


# --- S1: mid-run user messages ---


def _steer_payload(
    acknowledgment: str,
    reuse_plan: bool = True,
    reuse_ids: list[str] | None = None,
    skip_ids: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "acknowledgment": acknowledgment,
            "reuse_plan": reuse_plan,
            "reuse_subtask_ids": reuse_ids or [],
            "skip_subtask_ids": skip_ids or [],
        }
    )


def _clear_then_message_client(
    acknowledgment: str,
    skip_ids: list[str],
    *,
    reuse_plan: bool = True,
    reuse_ids: list[str] | None = None,
    fenced: bool = False,
) -> MagicMock:
    """Client mock: first call returns CLEAR (clarification), second returns steer directive."""
    clear_block = MagicMock()
    clear_block.type = "text"
    clear_block.text = "CLEAR"
    clear_resp = MagicMock()
    clear_resp.content = [clear_block]

    raw = _steer_payload(acknowledgment, reuse_plan, reuse_ids, skip_ids)
    msg_block = MagicMock()
    msg_block.type = "text"
    msg_block.text = f"```json\n{raw}\n```" if fenced else raw
    msg_resp = MagicMock()
    msg_resp.content = [msg_block]

    mock = MagicMock()
    mock.messages.create.side_effect = [clear_resp, msg_resp]
    return mock


def _steer_client(
    acknowledgment: str,
    *,
    reuse_plan: bool = True,
    reuse_ids: list[str] | None = None,
    skip_ids: list[str] | None = None,
    fenced: bool = False,
) -> MagicMock:
    """Single-call steer client — no CLEAR step. Use with pre-clarified context."""
    raw = _steer_payload(acknowledgment, reuse_plan, reuse_ids, skip_ids)
    block = MagicMock()
    block.type = "text"
    block.text = f"```json\n{raw}\n```" if fenced else raw
    resp = MagicMock()
    resp.content = [block]
    mock = MagicMock()
    mock.messages.create.return_value = resp
    return mock


def _pre_clarified_context(request: str = "What is AI?") -> TaskContext:
    """Context with clarifications pre-set so the CLEAR API call is skipped."""
    return TaskContext(task_id="test-task", original_request=request, clarifications=["N/A"])


def _linear_graph() -> TaskGraph:
    return TaskGraph(
        subtasks=[
            SubTask(id="t1", description="research", agent="research", depends_on=[]),
            SubTask(id="t2", description="summary", agent="summary", depends_on=["t1"]),
            SubTask(id="t3", description="aggregate", agent="aggregator", depends_on=["t2"]),
        ]
    )


@patch("src.orchestrator.Planner")
async def test_run_emits_user_message_ack_when_message_received(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    def research_and_message(*_args):
        submit_user_message(ctx.task_id, "just checking in")
        return "findings"

    mock_client = _clear_then_message_client("Got it, continuing.", [])

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_message
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(mock_client, _config()).run(ctx)

    events = await _drain(ctx.task_id)
    ack_evts = [e for e in events if e.event == "user_message_ack"]
    assert len(ack_evts) == 1
    assert ack_evts[0].data["acknowledgment"] == "Got it, continuing."


@patch("src.orchestrator.Planner")
async def test_run_skips_subtask_and_emits_agent_skipped(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _linear_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    def research_and_request_skip(*_args):
        submit_user_message(ctx.task_id, "skip the summary")
        return "findings"

    mock_client = _clear_then_message_client("Skipping the summary.", ["t2"])

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.SummaryAgent") as mock_summary,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_request_skip
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(mock_client, _config()).run(ctx)

    mock_summary.return_value.run.assert_not_called()

    events = await _drain(ctx.task_id)
    skipped_evts = [e for e in events if e.event == "agent_skipped"]
    assert len(skipped_evts) == 1
    assert skipped_evts[0].data["subtask_id"] == "t2"


@patch("src.orchestrator.Planner")
async def test_run_handles_fenced_json_in_message_response(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    def research_and_message(*_args):
        submit_user_message(ctx.task_id, "just checking in")
        return "findings"

    mock_client = _clear_then_message_client("Got it, continuing.", [], fenced=True)

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_message
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(mock_client, _config()).run(ctx)

    events = await _drain(ctx.task_id)
    ack_evts = [e for e in events if e.event == "user_message_ack"]
    assert len(ack_evts) == 1
    assert ack_evts[0].data["acknowledgment"] == "Got it, continuing."


@patch("src.orchestrator.Planner")
async def test_run_does_not_skip_aggregator_even_if_requested(mock_planner_class: MagicMock):
    """Aggregator must always run; skip requests for it are silently ignored."""
    mock_planner_class.return_value.plan.return_value = _linear_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    def research_and_request_skip_all(*_args):
        submit_user_message(ctx.task_id, "skip everything")
        return "findings"

    mock_client = _clear_then_message_client("Skipping remaining tasks.", ["t2", "t3"])

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.SummaryAgent"),
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_request_skip_all
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(mock_client, _config()).run(ctx)

    mock_agg.return_value.run.assert_called_once()
    events = await _drain(ctx.task_id)
    assert any(e.event == "result_ready" for e in events)


@patch("src.orchestrator.Planner")
async def test_run_skipped_subtask_does_not_block_downstream(mock_planner_class: MagicMock):  # noqa: E501
    """A skipped subtask is treated as completed so its dependents can still run."""
    mock_planner_class.return_value.plan.return_value = _linear_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    def research_and_request_skip(*_args):
        submit_user_message(ctx.task_id, "skip the summary")
        return "findings"

    mock_client = _clear_then_message_client("Skipping the summary.", ["t2"])

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.SummaryAgent"),
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_request_skip
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(mock_client, _config()).run(ctx)

    mock_agg.return_value.run.assert_called_once()


# --- S1: restart-with-reuse tests ---


@patch("src.orchestrator.Planner")
async def test_run_accumulates_user_messages_in_context(mock_planner_class: MagicMock):
    """Messages sent mid-run are stored on context.user_messages for downstream use."""
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    def research_and_message(*_args):
        submit_user_message(ctx.task_id, "pivot to WW1")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_message
        mock_agg.return_value.run.return_value = _agg_result()
        # reuse_ids=["t1"] keeps research output so the loop doesn't restart indefinitely
        await Orchestrator(_steer_client("Got it.", reuse_ids=["t1"]), _config()).run(ctx)

    assert ctx.user_messages == ["pivot to WW1"]


@patch("src.orchestrator.Planner")
async def test_run_replans_when_directive_says_reuse_plan_false(mock_planner_class: MagicMock):
    """When reuse_plan=false, the pipeline re-plans before re-executing."""
    mock_planner_class.return_value.plan.side_effect = [_research_graph(), _simple_graph()]
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    def research_and_trigger(*_args):
        submit_user_message(ctx.task_id, "pivot to WW1")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_trigger
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_steer_client("Restarting for WW1.", reuse_plan=False), _config()).run(
            ctx
        )

    assert mock_planner_class.return_value.plan.call_count == 2


@patch("src.orchestrator.Planner")
async def test_run_emits_plan_reset_before_second_plan_ready(mock_planner_class: MagicMock):
    """plan_reset is emitted before the second plan_ready so the UI can clear."""
    mock_planner_class.return_value.plan.side_effect = [_research_graph(), _simple_graph()]
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    def research_and_trigger(*_args):
        submit_user_message(ctx.task_id, "pivot to WW1")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_trigger
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_steer_client("Restarting.", reuse_plan=False), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    types = [e.event for e in events]
    first_plan_idx = types.index("plan_ready")
    reset_idx = types.index("plan_reset")
    second_plan_idx = types.index("plan_ready", first_plan_idx + 1)
    assert reset_idx < second_plan_idx


@patch("src.orchestrator.Planner")
async def test_run_restarts_completed_subtask_not_in_reuse_ids(mock_planner_class: MagicMock):
    """A completed subtask absent from reuse_subtask_ids is moved back to pending and re-run."""
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    call_count = 0

    def research_trigger_then_run(*_args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            submit_user_message(ctx.task_id, "actually research WW1 instead")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_trigger_then_run
        mock_agg.return_value.run.return_value = _agg_result()
        # reuse_plan=True but reuse_ids=[] → t1 completed but not reused → re-run
        await Orchestrator(
            _steer_client("Re-running research for WW1.", reuse_ids=[]), _config()
        ).run(ctx)

    assert mock_research.return_value.run.call_count == 2


@patch("src.orchestrator.Planner")
async def test_run_restarted_task_emits_agent_restarted_event(mock_planner_class: MagicMock):
    mock_planner_class.return_value.plan.return_value = _research_graph()
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    call_count = 0

    def research_trigger_then_run(*_args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            submit_user_message(ctx.task_id, "restart research")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_trigger_then_run
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_steer_client("Re-running.", reuse_ids=[]), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    restarted = [e for e in events if e.event == "agent_restarted"]
    assert len(restarted) == 1
    assert restarted[0].data["subtask_id"] == "t1"


@patch("src.orchestrator.Planner")
async def test_run_keeps_reused_subtask_and_skips_pending(mock_planner_class: MagicMock):
    """Reused subtask is not re-run; skipped pending subtask is omitted."""
    mock_planner_class.return_value.plan.return_value = _linear_graph()
    ctx = _pre_clarified_context()
    create_queue(ctx.task_id)

    def research_and_trigger(*_args):
        submit_user_message(ctx.task_id, "keep research, skip summary")
        return "findings"

    with (
        patch("src.orchestrator.ResearchAgent") as mock_research,
        patch("src.orchestrator.SummaryAgent") as mock_summary,
        patch("src.orchestrator.AggregatorAgent") as mock_agg,
    ):
        mock_research.return_value.run.side_effect = research_and_trigger
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(
            _steer_client("Keeping research, skipping summary.", reuse_ids=["t1"], skip_ids=["t2"]),
            _config(),
        ).run(ctx)

    mock_research.return_value.run.assert_called_once()
    mock_summary.return_value.run.assert_not_called()


# --- S2: multi-turn refinement ---


@patch("src.orchestrator.Planner")
async def test_run_saves_context_after_completion(mock_planner_class: MagicMock):
    """After run() completes, get_context returns the TaskContext."""
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    saved = get_context(ctx.task_id)
    assert saved is ctx


@patch("src.orchestrator.Planner")
async def test_run_appends_aggregator_output_to_prior_results(mock_planner_class: MagicMock):
    """A successful run appends the aggregator's dict output to context.prior_results."""
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    create_queue(ctx.task_id)

    agg_output = {"answer": "42", "artifacts": [], "warnings": []}
    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = agg_output
        await Orchestrator(_clear_client(), _config()).run(ctx)

    assert ctx.prior_results == [agg_output]


@patch("src.orchestrator.Planner")
async def test_run_prior_results_accumulates_across_refinement_runs(
    mock_planner_class: MagicMock,
):
    """Each run appends its result; prior runs' results are preserved."""
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    ctx.prior_results = [{"answer": "first run", "artifacts": [], "warnings": []}]
    ctx.clarifications = ["N/A"]  # skip clarification phase
    create_queue(ctx.task_id)

    agg_output = {"answer": "second run", "artifacts": [], "warnings": []}
    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = agg_output
        await Orchestrator(_clear_client(), _config()).run(ctx)

    assert len(ctx.prior_results) == 2
    assert ctx.prior_results[1] == agg_output


@patch("src.orchestrator.Planner")
async def test_run_emits_plan_reset_before_plan_ready_on_refinement(
    mock_planner_class: MagicMock,
):
    """When prior_results is non-empty (refinement run), plan_reset precedes plan_ready."""
    mock_planner_class.return_value.plan.return_value = _simple_graph()
    ctx = _context()
    ctx.prior_results = [{"answer": "first run", "artifacts": [], "warnings": []}]
    ctx.clarifications = ["N/A"]
    create_queue(ctx.task_id)

    with patch("src.orchestrator.AggregatorAgent") as mock_agg:
        mock_agg.return_value.run.return_value = _agg_result()
        await Orchestrator(_clear_client(), _config()).run(ctx)

    events = await _drain(ctx.task_id)
    types = [e.event for e in events]
    assert "plan_reset" in types
    assert types.index("plan_reset") < types.index("plan_ready")
