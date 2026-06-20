import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
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
from src.events import (
    arm_clarification,
    arm_user_messages,
    close,
    drain_user_messages,
    publish,
    wait_for_clarification,
)
from src.models import SSEEvent, SubTask, TaskContext

logger = logging.getLogger(__name__)


@dataclass
class _SteerDirective:
    acknowledgment: str
    reuse_plan: bool
    reuse_subtask_ids: set[str] = field(default_factory=set)
    skip_subtask_ids: set[str] = field(default_factory=set)


class Orchestrator:
    MODEL = "claude-sonnet-4-5"
    _MESSAGE_SYSTEM = """You are coordinating a multi-agent task pipeline that is currently running.
A user has sent a mid-run message. You will see the current plan, which subtasks have completed
(with their outputs), and which are still pending.

Your job is to decide what to reuse vs. redo given the user's new intent.

RESPONSE FIELDS (JSON only — no prose, no markdown fences):
{
  "acknowledgment": "brief honest reply (1-2 sentences)",
  "reuse_plan": true,
  "reuse_subtask_ids": [],
  "skip_subtask_ids": []
}

RULES:
- reuse_plan: set false if the current plan no longer fits the new intent and a fresh plan
  is needed. Set true to keep the existing plan and selectively restart/skip subtasks.
- reuse_subtask_ids: IDs of COMPLETED subtasks whose outputs are still useful.
  Any completed subtask NOT listed here will be discarded and re-run.
  Never include the aggregator — it always re-runs when anything changes.
- skip_subtask_ids: IDs of PENDING subtasks the user explicitly does not want.
  Only use when the user clearly wants to omit specific work (e.g. "skip the chart").
  Never include the aggregator.
- If the user's message doesn't require any change, set reuse_plan=true, list all completed
  subtask IDs in reuse_subtask_ids, and leave skip_subtask_ids empty."""

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
            while True:
                await self._planning_phase(context)
                needs_replan = await self._execution_phase(context)
                if not needs_replan:
                    break
                logger.info("task %s: replanning with updated user intent", context.task_id)
                await self._emit(context.task_id, "plan_reset", {})
                context.plan = None
                context.agent_outputs = {}
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

    async def _execution_phase(self, context: TaskContext) -> bool:
        """Run subtasks. Returns True if the caller should re-plan and re-execute."""
        assert context.plan is not None
        arm_user_messages(context.task_id)
        subtasks = context.plan.subtasks
        aggregator_ids = {s.id for s in subtasks if s.agent == "aggregator"}
        pending = {s.id for s in subtasks}
        completed: set[str] = set()
        failed: set[str] = set()

        while pending:
            messages = drain_user_messages(context.task_id)
            if messages:
                context.user_messages.extend(messages)
                directive = await self._handle_user_messages(messages, context, pending, completed)
                if not directive.reuse_plan:
                    return True

                # Move completed tasks not in reuse_subtask_ids back to pending
                to_restart = completed - directive.reuse_subtask_ids - aggregator_ids
                for sid in to_restart:
                    completed.discard(sid)
                    pending.add(sid)
                    del context.agent_outputs[sid]
                    await self._emit(context.task_id, "agent_restarted", {"subtask_id": sid})

                # Skip explicitly unwanted pending tasks
                for sid in directive.skip_subtask_ids:
                    if sid in pending and sid not in aggregator_ids:
                        pending.discard(sid)
                        context.agent_outputs[sid] = "SKIPPED: user request"
                        completed.add(sid)
                        await self._emit(context.task_id, "agent_skipped", {"subtask_id": sid})

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

        return False

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

    async def _handle_user_messages(
        self,
        messages: list[str],
        context: TaskContext,
        pending: set[str],
        completed: set[str],
    ) -> _SteerDirective:
        assert context.plan is not None
        lines = [f"Original request: {context.original_request}", ""]

        done = [s for s in context.plan.subtasks if s.id in completed]
        lines.append("Completed subtasks (with outputs):")
        if done:
            for s in done:
                output = context.agent_outputs.get(s.id, "(no output)")
                preview = str(output)[:200]
                lines.append(f"  {s.id} ({s.agent}): {s.description}")
                lines.append(f"    output: {preview}")
        else:
            lines.append("  (none yet)")

        lines.append("")
        remaining = [s for s in context.plan.subtasks if s.id in pending]
        lines.append("Pending subtasks (not yet run):")
        if remaining:
            lines.extend(f"  {s.id} ({s.agent}): {s.description}" for s in remaining)
        else:
            lines.append("  (none)")

        lines += ["", "User message(s):"]
        lines.extend(f"  - {m}" for m in messages)

        response = await asyncio.to_thread(
            self.client.messages.create,
            model=self.MODEL,
            max_tokens=512,
            system=self._MESSAGE_SYSTEM,
            messages=[{"role": "user", "content": "\n".join(lines)}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "{}")

        acknowledgment = "Got it — continuing as planned."
        reuse_plan = True
        reuse_ids: set[str] = set(completed)
        skip_ids: set[str] = set()
        try:
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
            parsed = json.loads(clean)
            acknowledgment = parsed.get("acknowledgment", acknowledgment)
            reuse_plan = bool(parsed.get("reuse_plan", True))
            reuse_ids = {sid for sid in parsed.get("reuse_subtask_ids", []) if sid in completed}
            skip_ids = {sid for sid in parsed.get("skip_subtask_ids", []) if sid in pending}
        except (json.JSONDecodeError, AttributeError):
            logger.warning("task %s: failed to parse message response: %r", context.task_id, text)

        directive = _SteerDirective(
            acknowledgment=acknowledgment,
            reuse_plan=reuse_plan,
            reuse_subtask_ids=reuse_ids,
            skip_subtask_ids=skip_ids,
        )

        restarted = list(completed - reuse_ids) if reuse_plan else []
        await self._emit(
            context.task_id,
            "user_message_ack",
            {
                "acknowledgment": acknowledgment,
                "reuse_plan": reuse_plan,
                "restarted_subtask_ids": restarted,
                "skipped_subtask_ids": list(skip_ids),
            },
        )
        return directive

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
