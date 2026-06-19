import json
import logging

from anthropic import Anthropic

from src.agents._helpers import strip_fences
from src.models import TaskContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the final aggregator in a multi-agent pipeline.

Return ONLY this JSON — no prose, no code fences:
{
  "answer": "<direct response to the user's original request>",
  "artifacts": [{"type": "chart", "url": "<artifact_path value>", "caption": "..."}],
  "warnings": ["<failed or insufficient subtask descriptions, if any>"]
}

Rules:
- answer must directly address the original request, not just summarize what the agents did.
- artifacts: use the artifact_path value from the code agent output exactly as-is — it is
  already a web URL (e.g. /outputs/TASKID/chart.png). Do not modify or reconstruct it.
- Only include artifacts that have a non-null artifact_path in the agent outputs.
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
            result = json.loads(strip_fences(text))
            for a in result.get("artifacts", []):
                url = a.get("url", "")
                if not url:
                    logger.warning("aggregator: artifact entry has no URL")
                elif not url.startswith("/outputs/"):
                    logger.warning("aggregator: artifact URL looks wrong — %r", url)
                else:
                    logger.info("aggregator: artifact url=%r", url)
            if result.get("warnings"):
                logger.warning("aggregator: warnings — %s", result["warnings"])
            return result  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            logger.error("aggregator: failed to parse JSON response: %s", exc)
            return {"answer": text, "artifacts": [], "warnings": []}

    def _build_prompt(self, context: TaskContext) -> str:
        parts = [f"Original request: {context.original_request}"]
        if context.clarifications:
            parts.append(f"Clarifications: {'; '.join(context.clarifications)}")
        parts.append(f"Agent outputs:\n{json.dumps(context.agent_outputs, indent=2)}")
        return "\n\n".join(parts)
