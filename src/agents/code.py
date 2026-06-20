import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from src.agents._helpers import strip_fences
from src.models import SubTask, TaskContext
from src.tools.executor import EXECUTE_PYTHON_TOOL_DEFINITION, execute_python

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a code execution agent. Write Python and run it using execute_python.

Available packages: matplotlib, pandas, numpy, and the Python standard library.

Rules:
- Save charts to the OUTPUT_DIR environment variable path. Never use plt.show().
- Do not make network calls from within executed code.
- If execution fails, read stderr, fix the code, and try again.
- When done, output ONLY this JSON — no prose, no fences:
  {"result": "<what the code produced>", "artifact_path": "<path or null>"}"""


class CodeAgent:
    MODEL = "claude-sonnet-4-5"
    MAX_EXECUTION_TURNS = 8

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def run(self, subtask: SubTask, context: TaskContext, output_dir: Path) -> dict:  # type: ignore[type-arg]
        # list[Any]: SDK accepts str or list[blocks] as content at runtime
        messages: list[Any] = [{"role": "user", "content": self._build_prompt(subtask, context)}]
        turn = 0

        while True:
            turn += 1
            if turn > self.MAX_EXECUTION_TURNS:
                logger.warning(
                    "[%s] max execution turns (%d) reached", subtask.id, self.MAX_EXECUTION_TURNS
                )
                return {
                    "result": "Exceeded maximum execution turns without completing.",
                    "artifact_path": None,
                }

            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=[EXECUTE_PYTHON_TOOL_DEFINITION],
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text = ""
                for block in response.content:
                    if block.type == "text":
                        text = block.text  # type: ignore[assignment]
                        break
                try:
                    result = json.loads(strip_fences(text))
                    logger.info(
                        "[%s] code agent finished — artifact_path=%r",
                        subtask.id,
                        result.get("artifact_path"),
                    )
                    return result  # type: ignore[no-any-return]
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "[%s] code agent response was not valid JSON: %s — raw: %r",
                        subtask.id,
                        exc,
                        text[:200],
                    )
                    return {"result": text, "artifact_path": None}

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "execute_python":
                    input_dict: dict[str, Any] = (
                        block.input if isinstance(block.input, dict) else {}
                    )
                    code = str(input_dict.get("code", ""))
                    exec_result = execute_python(code, output_dir)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(asdict(exec_result)),
                        }
                    )

            if not tool_results:
                logger.warning("[%s] tool_use response contained no recognized tools", subtask.id)
                return {"result": "", "artifact_path": None}

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def _build_prompt(self, subtask: SubTask, context: TaskContext) -> str:
        parts = [f"Your task: {subtask.description}"]
        if context.user_messages:
            msgs = "; ".join(context.user_messages)
            parts.append(f"Updated direction from user (takes priority): {msgs}")
        prior = {k: v for k, v in context.agent_outputs.items() if k in subtask.depends_on}
        if prior:
            parts.append(f"Context from prior agents:\n{json.dumps(prior, indent=2)}")
        return "\n\n".join(parts)
