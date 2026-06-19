import json
from typing import Any

from anthropic import Anthropic
from pydantic import ValidationError

from src.agents._helpers import strip_fences
from src.models import TaskContext, TaskGraph

SYSTEM_PROMPT = """You are a task planner for a multi-agent pipeline.

Available agent types:
- research: Searches the web for current factual information.
- code: Writes and executes Python code for computation, data analysis, or chart generation.
- summary: Synthesises prose from multiple sources into a human-readable narrative.
- aggregator: Merges all outputs into the final result delivered to the user.

Output format:
Return ONLY a JSON object — no prose, no markdown fences:
{
  "subtasks": [
    {"id": "t1", "description": "...", "agent": "research", "depends_on": []},
    {"id": "t2", "description": "...", "agent": "aggregator", "depends_on": ["t1"]}
  ]
}

Planning rules:
- Every plan must end with exactly one aggregator subtask that depends on all others.
- Prefer fewer, well-scoped subtasks over many fine-grained ones.
- No circular dependencies.
- If the request is simple, a single summary subtask followed by an aggregator is acceptable."""

RETRY_PROMPT = (
    "Your previous response could not be parsed as a valid task plan.\n"
    "Error: {error}\n\n"
    "Return ONLY the JSON object — no prose, no code fences."
)


class PlannerError(Exception):
    pass


class Planner:
    MODEL = "claude-opus-4"

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def plan(self, context: TaskContext) -> TaskGraph:
        # list[Any]: the SDK accepts varying content types (str, list[blocks]) at runtime
        messages: list[Any] = [{"role": "user", "content": self._build_prompt(context)}]
        last_exc: Exception | None = None

        for attempt in range(2):
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            text = self._extract_text(response)
            try:
                return TaskGraph.model_validate(json.loads(strip_fences(text)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_exc = exc
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": RETRY_PROMPT.format(error=exc)},
                ]

        raise PlannerError(f"Planner failed after 2 attempts: {last_exc}") from last_exc

    def _build_prompt(self, context: TaskContext) -> str:
        parts = [f"Request: {context.original_request}"]
        if context.clarifications:
            clarifs = "\n".join(f"- {c}" for c in context.clarifications)
            parts.append(f"Clarifications:\n{clarifs}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_text(response) -> str:  # type: ignore[no-untyped-def]
        for block in response.content:
            if block.type == "text":
                return block.text  # type: ignore[no-any-return]
        return ""
