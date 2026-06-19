import asyncio
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
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

logger = logging.getLogger(__name__)


class Orchestrator:
    MODEL = "claude-sonnet-4-5"
    _CLARIFICATION_SYSTEM = """You analyze user requests for a multi-agent AI task solver.

If the request has critical missing information that would prevent producing a useful result,
respond with numbered questions only:
1. First question?
2. Second question?

If the request is clear enough to proceed, respond with exactly: CLEAR

Be conservative — only ask when the ambiguity would fundamentally change the output.
Most requests should be answered with CLEAR."""

    def __init__(self, client: Anthropic, config: Config) -> None:
        self.client = client
        self.config = config

    async def run(self, context: TaskContext) -> None:
        logger.info("task %s: starting — %r", context.task_id, context.original_request[:80])
        try:
            await self._clarification_phase(context)
            await self._planning_phase(context)
            await self._execution_phase(context)
        except Exception as exc:
            logger.error("task %s: fatal error — %s", context.task_id, exc)
            await self._emit(context.task_id, "error", {"message": str(exc)})
        finally:
            await close(context.task_id)

    # --- clarification ---

    async def _clarification_phase(self, context: TaskContext) -> None:
        if context.clarifications:
            return
        logger.info("task %s: checking for ambiguities", context.task_id)
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
            system=self._CLARIFICATION_SYSTEM,
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
        logger.info("task %s: planning", context.task_id)
        planner = Planner(self.client)
        try:
            graph = await asyncio.to_thread(planner.plan, context)
        except PlannerError as exc:
            raise RuntimeError(f"Planning failed: {exc}") from exc
        logger.info(
            "task %s: plan ready — %d subtasks: %s",
            context.task_id,
            len(graph.subtasks),
            ", ".join(f"{s.id}({s.agent})" for s in graph.subtasks),
        )
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
        max_attempts = self.config.max_agent_retries + 1
        last_exc: BaseException = RuntimeError("no attempts made")
        for attempt in range(max_attempts):
            logger.info(
                "task %s: %s/%s attempt %d/%d",
                context.task_id,
                subtask.agent,
                subtask.id,
                attempt + 1,
                max_attempts,
            )
            try:
                result = await asyncio.wait_for(
                    self._dispatch(subtask, context),
                    timeout=float(self.config.agent_timeout_seconds),
                )
                has_artifact = isinstance(result, dict) and result.get("artifact_path") is not None
                logger.info(
                    "task %s: %s/%s completed (has_artifact=%s)",
                    context.task_id,
                    subtask.agent,
                    subtask.id,
                    has_artifact,
                )
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
            except TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "task %s: %s/%s timed out after %ds (attempt %d/%d)",
                    context.task_id,
                    subtask.agent,
                    subtask.id,
                    self.config.agent_timeout_seconds,
                    attempt + 1,
                    max_attempts,
                )
            except BaseException as exc:
                last_exc = exc
                logger.warning(
                    "task %s: %s/%s error (attempt %d/%d): %s",
                    context.task_id,
                    subtask.agent,
                    subtask.id,
                    attempt + 1,
                    max_attempts,
                    exc,
                )

        logger.error(
            "task %s: %s/%s permanently failed: %s",
            context.task_id,
            subtask.agent,
            subtask.id,
            last_exc,
        )
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
                result = await asyncio.to_thread(
                    CodeAgent(self.client).run, subtask, context, output_dir
                )
                # Build the web URL from the filename alone. Using relative_to()
                # is fragile — the LLM may report a path that differs from our
                # resolved output_dir (symlinks, CWD drift, stdout-constructed
                # paths). We own the directory structure, so just take the name.
                if isinstance(result, dict):
                    raw_path = result.get("artifact_path")
                    if raw_path:
                        filename = Path(raw_path).name
                        url = f"/outputs/{context.task_id}/{filename}"
                        logger.info("task %s: artifact %r → %s", context.task_id, raw_path, url)
                        result["artifact_path"] = url
                    else:
                        logger.info("task %s: code agent returned no artifact", context.task_id)
                return result
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
