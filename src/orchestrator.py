import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from anthropic import Anthropic

from src.agents.aggregator import AggregatorAgent
from src.agents.code import CodeAgent
from src.agents.planner import Planner, PlannerError
from src.agents.research import ResearchAgent
from src.agents.summary import SummaryAgent
from src.config import Config
from src.events import arm_clarification, close, publish, wait_for_clarification
from src.models import SSEEvent, SubTask, TaskContext

_CLARIFICATION_SYSTEM = """You analyze user requests for a multi-agent AI task solver.

If the request has critical missing information that would prevent producing a useful result,
respond with numbered questions only:
1. First question?
2. Second question?

If the request is clear enough to proceed, respond with exactly: CLEAR

Be conservative — only ask when the ambiguity would fundamentally change the output.
Most requests should be answered with CLEAR."""


class Orchestrator:
    MODEL = "claude-sonnet-4-5"

    def __init__(self, client: Anthropic, config: Config) -> None:
        self.client = client
        self.config = config

    async def run(self, context: TaskContext) -> None:
        try:
            await self._clarification_phase(context)
            await self._planning_phase(context)
            await self._execution_phase(context)
        except Exception as exc:
            await self._emit(context.task_id, "error", {"message": str(exc)})
        finally:
            await close(context.task_id)

    # --- clarification ---

    async def _clarification_phase(self, context: TaskContext) -> None:
        if context.clarifications:
            return
        questions = await asyncio.to_thread(self._detect_ambiguities, context.original_request)
        if not questions:
            return
        await self._emit(context.task_id, "clarification_needed", {"questions": questions})
        arm_clarification(context.task_id)
        answers = await wait_for_clarification(context.task_id, timeout=300.0)
        context.clarifications = answers

    def _detect_ambiguities(self, request: str) -> list[str]:
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=512,
            system=_CLARIFICATION_SYSTEM,
            messages=[{"role": "user", "content": request}],
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text  # type: ignore[assignment]
                break
        return self._parse_questions(text)

    def _parse_questions(self, text: str) -> list[str]:
        text = text.strip()
        if text.upper() == "CLEAR":
            return []
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        return [re.sub(r"^\d+\.\s*", "", ln) for ln in lines if ln]

    # --- planning ---

    async def _planning_phase(self, context: TaskContext) -> None:
        planner = Planner(self.client)
        try:
            graph = await asyncio.to_thread(planner.plan, context)
        except PlannerError as exc:
            raise RuntimeError(f"Planning failed: {exc}") from exc
        context.plan = graph
        await self._emit(
            context.task_id,
            "plan_ready",
            {
                "subtasks": [
                    {
                        "id": s.id,
                        "description": s.description,
                        "agent": s.agent,
                        "depends_on": s.depends_on,
                    }
                    for s in graph.subtasks
                ]
            },
        )

    # --- execution ---

    async def _execution_phase(self, context: TaskContext) -> None:
        assert context.plan is not None
        subtasks = context.plan.subtasks
        pending = {s.id for s in subtasks}
        completed: set[str] = set()
        failed: set[str] = set()

        while pending:
            ready = [
                s
                for s in subtasks
                if s.id in pending
                and all(d in completed for d in s.depends_on)
                and not any(d in failed for d in s.depends_on)
            ]
            if not ready:
                for sid in list(pending):
                    context.agent_outputs[sid] = "SKIPPED: upstream failure"
                    failed.add(sid)
                pending.clear()
                break

            for s in ready:
                pending.discard(s.id)

            results: list[Any] = await asyncio.gather(
                *[self._run_subtask(s, context) for s in ready],
                return_exceptions=True,
            )

            for subtask, result in zip(ready, results):
                if isinstance(result, BaseException):
                    failed.add(subtask.id)
                    context.agent_outputs[subtask.id] = f"FAILED: {result}"
                else:
                    completed.add(subtask.id)
                    context.agent_outputs[subtask.id] = result

        aggregator_subtask = next((s for s in subtasks if s.agent == "aggregator"), None)
        if aggregator_subtask and aggregator_subtask.id in context.agent_outputs:
            output = context.agent_outputs[aggregator_subtask.id]
            if isinstance(output, dict):
                await self._emit(context.task_id, "result_ready", {"result": output})

    async def _run_subtask(self, subtask: SubTask, context: TaskContext) -> Any:
        await self._emit(
            context.task_id,
            "agent_started",
            {
                "subtask_id": subtask.id,
                "agent": subtask.agent,
                "description": subtask.description,
            },
        )
        last_exc: BaseException = RuntimeError("no attempts made")
        for attempt in range(self.config.max_agent_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._dispatch(subtask, context),
                    timeout=float(self.config.agent_timeout_seconds),
                )
                has_artifact = isinstance(result, dict) and result.get("artifact_path") is not None
                await self._emit(
                    context.task_id,
                    "agent_completed",
                    {
                        "subtask_id": subtask.id,
                        "agent": subtask.agent,
                        "summary": str(result)[:200],
                        "has_artifact": has_artifact,
                    },
                )
                return result
            except BaseException as exc:
                last_exc = exc
                if attempt < self.config.max_agent_retries:
                    continue

        await self._emit(
            context.task_id,
            "agent_failed",
            {
                "subtask_id": subtask.id,
                "agent": subtask.agent,
                "error": str(last_exc),
            },
        )
        raise last_exc

    async def _dispatch(self, subtask: SubTask, context: TaskContext) -> Any:
        match subtask.agent:
            case "research":
                return await asyncio.to_thread(ResearchAgent(self.client).run, subtask, context)
            case "code":
                output_dir = self.config.outputs_dir / context.task_id
                return await asyncio.to_thread(
                    CodeAgent(self.client).run, subtask, context, output_dir
                )
            case "summary":
                return await asyncio.to_thread(SummaryAgent(self.client).run, subtask, context)
            case "aggregator":
                return await asyncio.to_thread(AggregatorAgent(self.client).run, context)
            case _:
                raise ValueError(f"Unknown agent type: {subtask.agent!r}")

    async def _emit(self, task_id: str, event: str, data: dict[str, Any]) -> None:
        await publish(
            task_id,
            SSEEvent(
                event=event,
                task_id=task_id,
                timestamp=datetime.now(UTC).isoformat(),
                data=data,
            ),
        )
