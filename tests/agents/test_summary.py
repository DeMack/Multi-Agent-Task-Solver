from unittest.mock import MagicMock

from src.agents.summary import SummaryAgent
from src.models import SubTask, TaskContext

# --- helpers ---


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _end_turn(text: str) -> MagicMock:
    r = MagicMock()
    r.content = [_text_block(text)]
    r.stop_reason = "end_turn"
    return r


def _client(text: str = "A summary.") -> MagicMock:
    c = MagicMock()
    c.messages.create.return_value = _end_turn(text)
    return c


def _subtask(description: str = "Summarise the findings") -> SubTask:
    return SubTask(id="t1", description=description, agent="summary", depends_on=[])


def _ctx(**kwargs) -> TaskContext:
    return TaskContext(task_id="x", original_request="test", clarifications=[], **kwargs)


# --- model constant ---


def test_summary_model_is_sonnet():
    assert SummaryAgent.MODEL == "claude-sonnet-4-5"


# --- run() ---


def test_summary_calls_client_with_sonnet():
    client = _client()
    SummaryAgent(client).run(_subtask(), _ctx())
    assert client.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-5"


def test_summary_returns_text():
    client = _client("The quarterly results were strong.")
    result = SummaryAgent(client).run(_subtask(), _ctx())
    assert result == "The quarterly results were strong."


def test_summary_includes_subtask_description_in_prompt():
    client = _client()
    SummaryAgent(client).run(_subtask("Summarise EU market trends"), _ctx())
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Summarise EU market trends" in content


def test_summary_includes_prior_outputs_in_prompt():
    client = _client()
    ctx = _ctx(agent_outputs={"t0": "Revenue grew 15% YoY"})
    subtask = SubTask(id="t1", description="Summarise", agent="summary", depends_on=["t0"])
    SummaryAgent(client).run(subtask, ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Revenue grew 15% YoY" in content


def test_summary_omits_irrelevant_outputs_from_prompt():
    client = _client()
    ctx = _ctx(agent_outputs={"t0": "relevant data", "t99": "unrelated data"})
    subtask = SubTask(id="t1", description="Summarise", agent="summary", depends_on=["t0"])
    SummaryAgent(client).run(subtask, ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "relevant data" in content
    assert "unrelated data" not in content


def test_summary_makes_single_api_call():
    client = _client()
    SummaryAgent(client).run(_subtask(), _ctx())
    assert client.messages.create.call_count == 1
