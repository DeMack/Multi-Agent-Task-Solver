import asyncio
from datetime import UTC, datetime

import pytest

import src.events as events_mod
from src.events import (
    arm_clarification,
    arm_user_messages,
    cleanup,
    close,
    create_queue,
    drain_user_messages,
    get_queue,
    publish,
    submit_clarification,
    submit_user_message,
    wait_for_clarification,
)
from src.models import SSEEvent


@pytest.fixture(autouse=True)
def reset_state():
    events_mod._reset()
    yield
    events_mod._reset()


def _event(task_id: str = "t1") -> SSEEvent:
    return SSEEvent(
        event="test",
        task_id=task_id,
        timestamp=datetime.now(UTC).isoformat(),
        data={},
    )


# --- queue management ---


def test_get_queue_returns_none_before_create():
    assert get_queue("nonexistent") is None


def test_create_queue_registers_queue():
    create_queue("t1")
    assert get_queue("t1") is not None


def test_create_queue_creates_empty_queue():
    create_queue("t1")
    q = get_queue("t1")
    assert q is not None
    assert q.empty()


async def test_publish_puts_event_in_queue():
    create_queue("t1")
    evt = _event("t1")
    await publish("t1", evt)
    q = get_queue("t1")
    assert q is not None
    assert not q.empty()
    assert await q.get() == evt


async def test_publish_ignores_unknown_task():
    await publish("nonexistent", _event())  # must not raise


async def test_close_puts_sentinel_in_queue():
    create_queue("t1")
    await close("t1")
    q = get_queue("t1")
    assert q is not None
    assert await q.get() is None


async def test_close_sentinel_comes_after_events():
    create_queue("t1")
    evt = _event("t1")
    await publish("t1", evt)
    await close("t1")
    q = get_queue("t1")
    assert q is not None
    assert await q.get() == evt
    assert await q.get() is None


# --- clarification ---


def test_arm_clarification_does_not_raise():
    arm_clarification("t1")


async def test_wait_for_clarification_returns_answers():
    arm_clarification("t1")
    answers = ["Answer A", "Answer B"]

    async def submit_later() -> None:
        await asyncio.sleep(0.01)
        submit_clarification("t1", answers)

    asyncio.create_task(submit_later())
    result = await wait_for_clarification("t1", timeout=1.0)
    assert result == answers


async def test_wait_for_clarification_times_out():
    arm_clarification("t1")
    with pytest.raises(asyncio.TimeoutError):
        await wait_for_clarification("t1", timeout=0.05)


def test_submit_clarification_without_arm_does_not_raise():
    submit_clarification("t1", ["answer"])


# --- user messages ---


def test_submit_user_message_returns_false_when_not_armed():
    assert submit_user_message("t1", "hello") is False


def test_arm_user_messages_enables_submit():
    arm_user_messages("t1")
    assert submit_user_message("t1", "hello") is True


def test_drain_user_messages_returns_all_queued_messages():
    arm_user_messages("t1")
    submit_user_message("t1", "first")
    submit_user_message("t1", "second")
    assert drain_user_messages("t1") == ["first", "second"]


def test_drain_user_messages_clears_after_drain():
    arm_user_messages("t1")
    submit_user_message("t1", "hello")
    drain_user_messages("t1")
    assert drain_user_messages("t1") == []


def test_drain_user_messages_returns_empty_when_not_armed():
    assert drain_user_messages("t1") == []


def test_cleanup_removes_user_message_state():
    arm_user_messages("t1")
    submit_user_message("t1", "hello")
    cleanup("t1")
    assert submit_user_message("t1", "new") is False
