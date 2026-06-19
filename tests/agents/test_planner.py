import json
from unittest.mock import MagicMock

import pytest

from src.agents.planner import Planner, PlannerError
from src.models import TaskContext

# --- helpers ---


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _response(text: str) -> MagicMock:
    r = MagicMock()
    r.content = [_text_block(text)]
    r.stop_reason = "end_turn"
    return r


def _client(*responses: MagicMock) -> MagicMock:
    c = MagicMock()
    c.messages.create.side_effect = list(responses)
    return c


VALID_GRAPH = {
    "subtasks": [
        {"id": "t1", "description": "Research X", "agent": "research", "depends_on": []},
        {"id": "t2", "description": "Aggregate", "agent": "aggregator", "depends_on": ["t1"]},
    ]
}
VALID_JSON = json.dumps(VALID_GRAPH)


def _ctx() -> TaskContext:
    return TaskContext(task_id="x", original_request="Tell me about AI", clarifications=[])


# --- model constant ---


def test_planner_model_is_opus():
    assert Planner.MODEL == "claude-opus-4-8"


# --- plan() ---


def test_plan_calls_client_with_opus():
    client = _client(_response(VALID_JSON))
    Planner(client).plan(_ctx())
    assert client.messages.create.call_args.kwargs["model"] == "claude-opus-4-8"


def test_plan_returns_task_graph_on_valid_json():
    client = _client(_response(VALID_JSON))
    graph = Planner(client).plan(_ctx())
    assert len(graph.subtasks) == 2
    assert graph.subtasks[0].agent == "research"
    assert graph.subtasks[1].agent == "aggregator"


def test_plan_includes_request_in_prompt():
    client = _client(_response(VALID_JSON))
    ctx = TaskContext(
        task_id="x", original_request="Summarise quarterly results", clarifications=[]
    )
    Planner(client).plan(ctx)
    messages = client.messages.create.call_args.kwargs["messages"]
    assert "Summarise quarterly results" in messages[0]["content"]


def test_plan_includes_clarifications_in_prompt():
    client = _client(_response(VALID_JSON))
    ctx = TaskContext(
        task_id="x",
        original_request="Do a thing",
        clarifications=["Focus on Europe", "Use Q3 data"],
    )
    Planner(client).plan(ctx)
    messages = client.messages.create.call_args.kwargs["messages"]
    content = messages[0]["content"]
    assert "Focus on Europe" in content
    assert "Use Q3 data" in content


def test_plan_strips_markdown_fences():
    fenced = f"```json\n{VALID_JSON}\n```"
    client = _client(_response(fenced))
    graph = Planner(client).plan(_ctx())
    assert len(graph.subtasks) == 2


def test_plan_retries_on_invalid_json():
    client = _client(_response("not valid json"), _response(VALID_JSON))
    graph = Planner(client).plan(_ctx())
    assert client.messages.create.call_count == 2
    assert len(graph.subtasks) == 2


def test_plan_retries_on_schema_error():
    bad_schema = json.dumps({"subtasks": [{"id": "t1", "agent": "unknown"}]})
    client = _client(_response(bad_schema), _response(VALID_JSON))
    graph = Planner(client).plan(_ctx())
    assert client.messages.create.call_count == 2
    assert len(graph.subtasks) == 2


def test_plan_raises_planner_error_after_two_failures():
    client = _client(_response("bad"), _response("also bad"))
    with pytest.raises(PlannerError):
        Planner(client).plan(_ctx())
    assert client.messages.create.call_count == 2


def test_plan_retry_includes_error_in_message():
    client = _client(_response("not json"), _response(VALID_JSON))
    Planner(client).plan(_ctx())
    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    # retry message should follow the failed assistant turn
    user_messages = [m for m in second_call_messages if m["role"] == "user"]
    assert len(user_messages) == 2
