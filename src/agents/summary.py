import json

from anthropic import Anthropic

from src.models import SubTask, TaskContext

SYSTEM_PROMPT = """You are a synthesis agent. Produce a clear prose summary from provided context.

Rules:
- Only use information from the provided inputs. Do not introduce external facts.
- Synthesize — do not transcribe large chunks verbatim.
- If inputs are contradictory or insufficient, say so explicitly."""


class SummaryAgent:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, client: Anthropic) -> None:
        self.client = client

    def run(self, subtask: SubTask, context: TaskContext) -> str:
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._build_prompt(subtask, context)}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text  # type: ignore[no-any-return]
        return ""

    def _build_prompt(self, subtask: SubTask, context: TaskContext) -> str:
        parts = [f"Summary task: {subtask.description}"]
        if context.user_messages:
            msgs = "; ".join(context.user_messages)
            parts.append(f"Updated direction from user (takes priority): {msgs}")
        if context.prior_results:
            last = context.prior_results[-1]
            parts.append(f"Previous run result: {str(last)[:300]}")
        prior = {k: v for k, v in context.agent_outputs.items() if k in subtask.depends_on}
        if prior:
            parts.append(f"Inputs:\n{json.dumps(prior, indent=2)}")
        return "\n\n".join(parts)
