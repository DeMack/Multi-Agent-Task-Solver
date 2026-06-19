import json
import logging
from typing import Any

from anthropic import Anthropic

from src.models import SubTask, TaskContext
from src.tools.search import SEARCH_TOOL_DEFINITION, search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research agent. Use the search tool to find factual information.

Rules:
- Always use the search tool at least once. Never answer from training knowledge alone.
- Cite sources by including URLs inline with the claims they support.
- If results are irrelevant or insufficient, say so explicitly — do not fabricate information.
- Be factual and concise. Your output feeds downstream agents."""


class ResearchAgent:
    MODEL = "claude-sonnet-4-5"
    # Cap search turns so the accumulated context stays manageable and the
    # final write-up call doesn't balloon past the agent timeout.
    MAX_SEARCH_TURNS = 2

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def run(self, subtask: SubTask, context: TaskContext) -> str:
        # list[Any]: SDK accepts str or list[blocks] as content at runtime
        messages: list[Any] = [{"role": "user", "content": self._build_prompt(subtask, context)}]
        turn = 0

        while True:
            turn += 1
            # Once we've done MAX_SEARCH_TURNS of tool use, omit the tools
            # parameter so the model is forced to write its final answer
            # instead of issuing more searches.
            force_final = turn > self.MAX_SEARCH_TURNS
            if force_final:
                logger.info(
                    "[%s] max search turns (%d) reached — forcing final answer",
                    subtask.id,
                    self.MAX_SEARCH_TURNS,
                )

            call_kwargs: dict[str, Any] = {
                "model": self.MODEL,
                "max_tokens": 4096,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            }
            if not force_final:
                call_kwargs["tools"] = [SEARCH_TOOL_DEFINITION]
                call_kwargs["tool_choice"] = {"type": "any"}  # guarantee at least one search

            logger.info(
                "[%s] research turn %d%s — calling API",
                subtask.id,
                turn,
                " (final)" if force_final else "",
            )
            response = self.client.messages.create(**call_kwargs)
            logger.info(
                "[%s] research turn %d — stop_reason=%s", subtask.id, turn, response.stop_reason
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        return block.text  # type: ignore[no-any-return]
                return ""

            if force_final:
                # Model returned tool_use even without tools — shouldn't happen;
                # return whatever text exists in the response.
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
                    query = str(input_dict.get("query", ""))
                    max_results = int(input_dict.get("max_results", 5))
                    logger.info("[%s] search: %r (max=%d)", subtask.id, query, max_results)
                    results = search(query=query, max_results=max_results)
                    logger.info("[%s] search returned %d results", subtask.id, len(results))
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(results),
                        }
                    )

            if not tool_results:
                logger.warning("[%s] tool_use response contained no recognized tools", subtask.id)
                return ""

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
