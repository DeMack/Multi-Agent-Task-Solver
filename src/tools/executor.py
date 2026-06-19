import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

EXECUTE_PYTHON_TOOL_DEFINITION = {
    "name": "execute_python",
    "description": (
        "Execute Python code in a sandboxed subprocess. "
        "Write output files (e.g. charts) to the directory available as the "
        "OUTPUT_DIR environment variable. Returns stdout, stderr, exit code, "
        "paths of any files written to OUTPUT_DIR, and a timeout flag."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Self-contained Python code to execute.",
            },
        },
        "required": ["code"],
    },
}


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    artifact_paths: list[str] = field(default_factory=list)
    timed_out: bool = False


def execute_python(
    code: str,
    output_dir: Path,
    timeout_seconds: int = 30,
) -> ExecutionResult:
    """Run Python code in a temporary subprocess and return the result.

    The subprocess receives OUTPUT_DIR as an environment variable pointing
    to output_dir, where it should write any generated files (charts, etc.).
    Any files found in output_dir after execution are reported as artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as work_dir:
        script = Path(work_dir) / "script.py"
        script.write_text(code)

        env = os.environ.copy()
        env["OUTPUT_DIR"] = str(output_dir)

        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                cwd=work_dir,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = -1
            timed_out = True

    artifact_paths = [str(p) for p in sorted(output_dir.iterdir()) if p.is_file()]

    return ExecutionResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        artifact_paths=artifact_paths,
        timed_out=timed_out,
    )
