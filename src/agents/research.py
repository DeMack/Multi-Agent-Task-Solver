import json
from typing import Any

from anthropic import Anthropic

from src.models import SubTask, TaskContext
from src.tools.search import SEARCH_TOOL_DEFINITION, search

SYSTEM_PROMPT = """You are a research agent. Use the search tool to find factual information.

Rules:
- Always use the search tool at least once. Never answer from training knowledge alone.
- Cite sources by including URLs inline with the claims they support.
- If results are irrelevant or insufficient, say so explicitly — do not fabricate information.
- Be factual and concise. Your output feeds downstream agents."""


class ResearchAgent:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def run(self, subtask: SubTask, context: TaskContext) -> str:
        # list[Any]: SDK accepts str or list[blocks] as content at runtime
        messages: list[Any] = [{"role": "user", "content": self._build_prompt(subtask, context)}]

        while True:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=[SEARCH_TOOL_DEFINITION],
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return block.text  # type: ignore[no-any-return]
                return ""

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "search":
                    input_dict: dict[str, Any] = (
                        block.input if isinstance(block.input, dict) else {}
                    )
                    results = search(
                        query=str(input_dict.get("query", "")),
                        max_results=int(input_dict.get("max_results", 5)),
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(results),
                        }
                    )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def _build_prompt(self, subtask: SubTask, context: TaskContext) -> str:
        parts = [
            f"Original request: {context.original_request}",
            f"Your research task: {subtask.description}",
        ]
        if context.clarifications:
            parts.append(f"Clarifications: {'; '.join(context.clarifications)}")
        prior = {k: v for k, v in context.agent_outputs.items() if k in subtask.depends_on}
        if prior:
            parts.append(f"Prior agent outputs:\n{json.dumps(prior, indent=2)}")
        return "\n\n".join(parts)
