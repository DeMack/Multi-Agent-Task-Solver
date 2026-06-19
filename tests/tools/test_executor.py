from typing import Any

from src.tools.executor import EXECUTE_PYTHON_TOOL_DEFINITION, ExecutionResult, execute_python

# --- tool definition ---


def test_tool_definition_has_name():
    assert EXECUTE_PYTHON_TOOL_DEFINITION["name"] == "execute_python"


def test_tool_definition_has_description():
    assert "description" in EXECUTE_PYTHON_TOOL_DEFINITION
    assert len(EXECUTE_PYTHON_TOOL_DEFINITION["description"]) > 0


def test_tool_definition_has_input_schema():
    schema: dict[str, Any] = EXECUTE_PYTHON_TOOL_DEFINITION["input_schema"]  # type: ignore[assignment]
    assert schema["type"] == "object"
    assert "code" in schema["properties"]
    assert "code" in schema["required"]


# --- execution result shape ---


def test_execution_result_is_dataclass(tmp_path):
    result = execute_python("print('hello')", tmp_path)
    assert isinstance(result, ExecutionResult)
    assert hasattr(result, "stdout")
    assert hasattr(result, "stderr")
    assert hasattr(result, "exit_code")
    assert hasattr(result, "artifact_paths")
    assert hasattr(result, "timed_out")


# --- stdout / stderr capture ---


def test_stdout_is_captured(tmp_path):
    result = execute_python("print('hello world')", tmp_path)
    assert "hello world" in result.stdout


def test_stderr_is_captured(tmp_path):
    result = execute_python("import sys; sys.stderr.write('oops\\n')", tmp_path)
    assert "oops" in result.stderr


def test_successful_exit_code(tmp_path):
    result = execute_python("x = 1 + 1", tmp_path)
    assert result.exit_code == 0


def test_failed_exit_code_on_exception(tmp_path):
    result = execute_python("raise ValueError('bad')", tmp_path)
    assert result.exit_code != 0


def test_syntax_error_captured_in_stderr(tmp_path):
    result = execute_python("def broken(:", tmp_path)
    assert result.exit_code != 0
    assert result.stderr != ""


# --- artifacts ---


def test_artifact_written_to_output_dir_is_detected(tmp_path):
    code = """
import os
path = os.path.join(os.environ['OUTPUT_DIR'], 'chart.png')
open(path, 'w').close()
"""
    result = execute_python(code, tmp_path)
    assert result.exit_code == 0
    assert any("chart.png" in p for p in result.artifact_paths)


def test_no_artifacts_returns_empty_list(tmp_path):
    result = execute_python("x = 1", tmp_path)
    assert result.artifact_paths == []


# --- timeout ---


def test_timeout_sets_timed_out_flag(tmp_path):
    result = execute_python("import time; time.sleep(60)", tmp_path, timeout_seconds=1)
    assert result.timed_out is True


def test_timeout_returns_nonzero_exit_code(tmp_path):
    result = execute_python("import time; time.sleep(60)", tmp_path, timeout_seconds=1)
    assert result.exit_code != 0


def test_no_timeout_flag_on_fast_code(tmp_path):
    result = execute_python("print('fast')", tmp_path)
    assert result.timed_out is False


# --- isolation ---


def test_output_dir_is_available_via_env(tmp_path):
    code = """
import os
print(os.environ.get('OUTPUT_DIR', 'MISSING'))
"""
    result = execute_python(code, tmp_path)
    assert str(tmp_path) in result.stdout
