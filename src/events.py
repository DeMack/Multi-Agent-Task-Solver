import asyncio

from src.models import SSEEvent

_queues: dict[str, asyncio.Queue[SSEEvent | None]] = {}
_clarify_events: dict[str, asyncio.Event] = {}
_clarify_answers: dict[str, list[str]] = {}


def create_queue(task_id: str) -> None:
    _queues[task_id] = asyncio.Queue()


def get_queue(task_id: str) -> asyncio.Queue[SSEEvent | None] | None:
    return _queues.get(task_id)


async def publish(task_id: str, event: SSEEvent) -> None:
    q = _queues.get(task_id)
    if q is not None:
        await q.put(event)


async def close(task_id: str) -> None:
    q = _queues.get(task_id)
    if q is not None:
        await q.put(None)


def arm_clarification(task_id: str) -> None:
    _clarify_events[task_id] = asyncio.Event()


def submit_clarification(task_id: str, answers: list[str]) -> None:
    _clarify_answers[task_id] = answers
    ev = _clarify_events.get(task_id)
    if ev is not None:
        ev.set()


async def wait_for_clarification(task_id: str, timeout: float) -> list[str]:
    ev = _clarify_events.get(task_id)
    if ev is None:
        raise RuntimeError(f"No clarification event armed for task {task_id!r}")
    await asyncio.wait_for(ev.wait(), timeout=timeout)
    answers = _clarify_answers.pop(task_id, [])
    _clarify_events.pop(task_id, None)
    return answers


def _reset() -> None:
    """Clear all state. For testing only."""
    _queues.clear()
    _clarify_events.clear()
    _clarify_answers.clear()
