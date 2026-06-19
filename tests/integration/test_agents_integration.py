import os
from pathlib import Path

import pytest
from anthropic import Anthropic

from src.agents.aggregator import AggregatorAgent
from src.agents.code import CodeAgent
from src.agents.planner import Planner
from src.agents.research import ResearchAgent
from src.agents.summary import SummaryAgent
from src.models import SubTask, TaskContext

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _client() -> Anthropic:
    return Anthropic()


def _ctx(request: str = "What is the capital of France?") -> TaskContext:
    return TaskContext(task_id="integration-test", original_request=request, clarifications=[])


@pytest.mark.integration
def test_planner_returns_valid_task_graph():
    graph = Planner(_client()).plan(_ctx())
    assert len(graph.subtasks) >= 1
    assert graph.subtasks[-1].agent == "aggregator"
    for subtask in graph.subtasks:
        assert subtask.id
        assert subtask.description
        assert subtask.agent in ("research", "code", "summary", "aggregator")


@pytest.mark.integration
def test_research_agent_returns_text():
    subtask = SubTask(
        id="t1", description="Find the capital of France", agent="research", depends_on=[]
    )
    result = ResearchAgent(_client()).run(subtask, _ctx())
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Paris" in result or "france" in result.lower()


@pytest.mark.integration
def test_code_agent_executes_code(tmp_path: Path):
    subtask = SubTask(
        id="t1",
        description="Write Python code that computes 2 + 2 and prints the result",
        agent="code",
        depends_on=[],
    )
    ctx = TaskContext(task_id="int-test", original_request="compute", clarifications=[])
    result = CodeAgent(_client()).run(subtask, ctx, tmp_path)
    assert "result" in result


@pytest.mark.integration
def test_summary_agent_returns_text():
    subtask = SubTask(
        id="t1",
        description="Summarize: Paris is the capital of France, known for the Eiffel Tower.",
        agent="summary",
        depends_on=[],
    )
    result = SummaryAgent(_client()).run(subtask, _ctx())
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_aggregator_returns_structured_result():
    ctx = TaskContext(
        task_id="int-test",
        original_request="What is the capital of France?",
        clarifications=[],
        agent_outputs={"t1": "Paris is the capital of France."},
    )
    result = AggregatorAgent(_client()).run(ctx)
    assert "answer" in result
    assert "artifacts" in result
    assert "warnings" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
