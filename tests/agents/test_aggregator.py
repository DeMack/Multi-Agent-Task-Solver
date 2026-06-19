import json
from unittest.mock import MagicMock

from src.agents.aggregator import AggregatorAgent
from src.models import TaskContext

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


def _client(text: str) -> MagicMock:
    c = MagicMock()
    c.messages.create.return_value = _end_turn(text)
    return c


def _ctx(**kwargs) -> TaskContext:
    return TaskContext(
        task_id="x", original_request="What are AI trends?", clarifications=[], **kwargs
    )


VALID_RESULT = json.dumps(
    {
        "answer": "AI is advancing rapidly.",
        "artifacts": [],
        "warnings": [],
    }
)

RESULT_WITH_ARTIFACT = json.dumps(
    {
        "answer": "Here is your chart.",
        "artifacts": [{"type": "chart", "url": "/outputs/x/chart.png", "caption": "Revenue"}],
        "warnings": [],
    }
)

RESULT_WITH_WARNING = json.dumps(
    {
        "answer": "Partial results only.",
        "artifacts": [],
        "warnings": ["research subtask failed: timeout"],
    }
)


# --- model constant ---


def test_aggregator_model_is_sonnet():
    assert AggregatorAgent.MODEL == "claude-sonnet-4-5"


# --- run() ---


def test_aggregator_calls_client_with_sonnet():
    client = _client(VALID_RESULT)
    AggregatorAgent(client).run(_ctx())
    assert client.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-5"


def test_aggregator_returns_answer_field():
    client = _client(VALID_RESULT)
    result = AggregatorAgent(client).run(_ctx())
    assert result["answer"] == "AI is advancing rapidly."


def test_aggregator_returns_artifacts_field():
    client = _client(RESULT_WITH_ARTIFACT)
    result = AggregatorAgent(client).run(_ctx())
    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["url"] == "/outputs/x/chart.png"


def test_aggregator_returns_warnings_field():
    client = _client(RESULT_WITH_WARNING)
    result = AggregatorAgent(client).run(_ctx())
    assert "research subtask failed: timeout" in result["warnings"]


def test_aggregator_handles_fenced_json():
    fenced = f"```json\n{VALID_RESULT}\n```"
    client = _client(fenced)
    result = AggregatorAgent(client).run(_ctx())
    assert result["answer"] == "AI is advancing rapidly."


def test_aggregator_falls_back_on_invalid_json():
    client = _client("not valid json")
    result = AggregatorAgent(client).run(_ctx())
    assert "answer" in result
    assert result["artifacts"] == []
    assert result["warnings"] == []


def test_aggregator_includes_original_request_in_prompt():
    client = _client(VALID_RESULT)
    ctx = _ctx()
    AggregatorAgent(client).run(ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "What are AI trends?" in content


def test_aggregator_includes_agent_outputs_in_prompt():
    client = _client(VALID_RESULT)
    ctx = _ctx(
        agent_outputs={"t1": "Research found X", "t2": "Chart saved to /outputs/x/chart.png"}
    )
    AggregatorAgent(client).run(ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Research found X" in content
    assert "Chart saved to /outputs/x/chart.png" in content


def test_aggregator_makes_single_api_call():
    client = _client(VALID_RESULT)
    AggregatorAgent(client).run(_ctx())
    assert client.messages.create.call_count == 1
