import json

from anthropic import Anthropic

from src.agents._helpers import strip_fences
from src.models import TaskContext

SYSTEM_PROMPT = """You are the final aggregator in a multi-agent pipeline.

Return ONLY this JSON — no prose, no code fences:
{
  "answer": "<direct response to the user's original request>",
  "artifacts": [{"type": "chart", "url": "/outputs/<filename>", "caption": "..."}],
  "warnings": ["<failed or insufficient subtask descriptions, if any>"]
}

Rules:
- answer must directly address the original request, not just summarize what the agents did.
- artifacts must only reference files that actually exist (paths provided in agent outputs).
- warnings must name any subtask that failed or produced insufficient data.
- Use empty arrays for artifacts and warnings when none apply."""


class AggregatorAgent:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def run(self, context: TaskContext) -> dict:  # type: ignore[type-arg]
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_prompt(context)}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text  # type: ignore[assignment]
                break
        try:
            return json.loads(strip_fences(text))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {"answer": text, "artifacts": [], "warnings": []}

    def _build_prompt(self, context: TaskContext) -> str:
        parts = [f"Original request: {context.original_request}"]
        if context.clarifications:
            parts.append(f"Clarifications: {'; '.join(context.clarifications)}")
        parts.append(f"Agent outputs:\n{json.dumps(context.agent_outputs, indent=2)}")
        return "\n\n".join(parts)
