import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.code import CodeAgent
from src.models import SubTask, TaskContext
from src.tools.executor import EXECUTE_PYTHON_TOOL_DEFINITION, ExecutionResult

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


def _subtask(description: str = "Plot a bar chart") -> SubTask:
    return SubTask(id="t1", description=description, agent="code", depends_on=[])


def _ctx() -> TaskContext:
    return TaskContext(task_id="x", original_request="make a chart", clarifications=[])


def _exec_result(
    stdout: str = "", stderr: str = "", exit_code: int = 0, artifacts: list[str] | None = None
) -> ExecutionResult:
    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        artifact_paths=artifacts or [],
        timed_out=False,
    )


RESULT_JSON = json.dumps({"result": "chart created", "artifact_path": "/outputs/x/chart.png"})
NO_ARTIFACT_JSON = json.dumps({"result": "sum is 42", "artifact_path": None})


# --- model constant ---


def test_code_model_is_sonnet():
    assert CodeAgent.MODEL == "claude-sonnet-4-5"


# --- run() ---


def test_code_calls_client_with_sonnet(tmp_path: Path):
    client = _client(_end_turn(NO_ARTIFACT_JSON))
    CodeAgent(client).run(_subtask(), _ctx(), tmp_path)
    assert client.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-5"


def test_code_provides_execute_python_tool(tmp_path: Path):
    client = _client(_end_turn(NO_ARTIFACT_JSON))
    CodeAgent(client).run(_subtask(), _ctx(), tmp_path)
    assert client.messages.create.call_args.kwargs["tools"] == [EXECUTE_PYTHON_TOOL_DEFINITION]


def test_code_returns_parsed_json_on_end_turn(tmp_path: Path):
    client = _client(_end_turn(NO_ARTIFACT_JSON))
    result = CodeAgent(client).run(_subtask(), _ctx(), tmp_path)
    assert result == {"result": "sum is 42", "artifact_path": None}


def test_code_handles_fenced_json_response(tmp_path: Path):
    fenced = f"```json\n{NO_ARTIFACT_JSON}\n```"
    client = _client(_end_turn(fenced))
    result = CodeAgent(client).run(_subtask(), _ctx(), tmp_path)
    assert result["result"] == "sum is 42"


def test_code_handles_tool_call_and_returns_result(tmp_path: Path):
    code = "print(42)"
    client = _client(
        _tool_response("execute_python", "tu_1", {"code": code}),
        _end_turn(RESULT_JSON),
    )
    with patch("src.agents.code.execute_python") as mock_exec:
        mock_exec.return_value = _exec_result(stdout="42\n")
        result = CodeAgent(client).run(_subtask(), _ctx(), tmp_path)

    assert result["result"] == "chart created"
    assert client.messages.create.call_count == 2


def test_code_calls_execute_python_with_code(tmp_path: Path):
    code = "x = 1 + 1\nprint(x)"
    client = _client(
        _tool_response("execute_python", "tu_1", {"code": code}),
        _end_turn(NO_ARTIFACT_JSON),
    )
    with patch("src.agents.code.execute_python") as mock_exec:
        mock_exec.return_value = _exec_result()
        CodeAgent(client).run(_subtask(), _ctx(), tmp_path)

    call_args = mock_exec.call_args
    assert call_args.args[0] == code


def test_code_passes_output_dir_to_execute_python(tmp_path: Path):
    code = "pass"
    client = _client(
        _tool_response("execute_python", "tu_1", {"code": code}),
        _end_turn(NO_ARTIFACT_JSON),
    )
    with patch("src.agents.code.execute_python") as mock_exec:
        mock_exec.return_value = _exec_result()
        CodeAgent(client).run(_subtask(), _ctx(), tmp_path)

    call_args = mock_exec.call_args
    assert call_args.args[1] == tmp_path


def test_code_includes_subtask_description_in_prompt(tmp_path: Path):
    client = _client(_end_turn(NO_ARTIFACT_JSON))
    CodeAgent(client).run(_subtask("Compute average salary"), _ctx(), tmp_path)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Compute average salary" in content


def test_code_includes_prior_outputs_in_prompt(tmp_path: Path):
    client = _client(_end_turn(NO_ARTIFACT_JSON))
    ctx = TaskContext(
        task_id="x",
        original_request="analyse data",
        clarifications=[],
        agent_outputs={"t0": "sales: [100, 200, 300]"},
    )
    subtask = SubTask(id="t1", description="plot sales", agent="code", depends_on=["t0"])
    CodeAgent(client).run(subtask, ctx, tmp_path)
    content = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "sales: [100, 200, 300]" in content
