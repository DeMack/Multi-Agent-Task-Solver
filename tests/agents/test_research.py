from unittest.mock import MagicMock, patch

from src.agents.research import ResearchAgent
from src.models import SubTask, TaskContext
from src.tools.search import SEARCH_TOOL_DEFINITION

# --- helpers ---


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name: str, tool_id: str, input_: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.id = tool_id
    b.input = input_
    return b


def _end_turn(text: str) -> MagicMock:
    r = MagicMock()
    r.content = [_text_block(text)]
    r.stop_reason = "end_turn"
    return r


def _tool_response(name: str, tool_id: str, input_: dict) -> MagicMock:
    r = MagicMock()
    r.content = [_tool_use_block(name, tool_id, input_)]
    r.stop_reason = "tool_use"
    return r


def _client(*responses: MagicMock) -> MagicMock:
    c = MagicMock()
    c.messages.create.side_effect = list(responses)
    return c


def _subtask(description: str = "Research AI") -> SubTask:
    return SubTask(id="t1", description=description, agent="research", depends_on=[])


def _ctx(request: str = "Tell me about AI") -> TaskContext:
    return TaskContext(task_id="x", original_request=request, clarifications=[])


# --- model constant ---


def test_research_model_is_sonnet():
    assert ResearchAgent.MODEL == "claude-sonnet-4-5"


# --- run() ---


def test_research_calls_client_with_sonnet():
    client = _client(_end_turn("result"))
    ResearchAgent(client).run(_subtask(), _ctx())
    assert client.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-5"


def test_research_provides_search_tool():
    client = _client(_end_turn("result"))
    ResearchAgent(client).run(_subtask(), _ctx())
    assert client.messages.create.call_args.kwargs["tools"] == [SEARCH_TOOL_DEFINITION]


def test_research_returns_text_on_end_turn():
    client = _client(_end_turn("Here are the findings."))
    result = ResearchAgent(client).run(_subtask(), _ctx())
    assert result == "Here are the findings."


def test_research_includes_request_in_prompt():
    client = _client(_end_turn("result"))
    ctx = TaskContext(task_id="x", original_request="AI trends", clarifications=[])
    ResearchAgent(client).run(_subtask(), ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "AI trends" in content


def test_research_includes_subtask_description_in_prompt():
    client = _client(_end_turn("result"))
    ResearchAgent(client).run(_subtask("Research DuckDuckGo usage"), _ctx())
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Research DuckDuckGo usage" in content


def test_research_handles_tool_call_and_returns_final_text():
    client = _client(
        _tool_response("search", "tu_1", {"query": "AI trends 2025"}),
        _end_turn("Found: AI is growing fast."),
    )
    with patch("src.agents.research.search") as mock_search:
        mock_search.return_value = [{"title": "T", "url": "u", "snippet": "s"}]
        result = ResearchAgent(client).run(_subtask(), _ctx())

    assert result == "Found: AI is growing fast."
    assert client.messages.create.call_count == 2


def test_research_calls_search_with_query():
    client = _client(
        _tool_response("search", "tu_1", {"query": "AI trends 2025"}),
        _end_turn("result"),
    )
    with patch("src.agents.research.search") as mock_search:
        mock_search.return_value = []
        ResearchAgent(client).run(_subtask(), _ctx())

    mock_search.assert_called_once_with(query="AI trends 2025", max_results=5)


def test_research_respects_max_results_in_tool_input():
    client = _client(
        _tool_response("search", "tu_1", {"query": "news", "max_results": 3}),
        _end_turn("result"),
    )
    with patch("src.agents.research.search") as mock_search:
        mock_search.return_value = []
        ResearchAgent(client).run(_subtask(), _ctx())

    mock_search.assert_called_once_with(query="news", max_results=3)


def test_research_includes_prior_outputs_in_prompt():
    client = _client(_end_turn("result"))
    ctx = TaskContext(
        task_id="x",
        original_request="test",
        clarifications=[],
        agent_outputs={"t0": "prior research data"},
    )
    subtask = SubTask(id="t1", description="analyse", agent="research", depends_on=["t0"])
    ResearchAgent(client).run(subtask, ctx)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "prior research data" in content
