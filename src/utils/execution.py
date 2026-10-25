"""One monotonic deadline for nested stages, model calls and tools (Python 3.10+)."""
from __future__ import annotations

import asyncio
import inspect
import time
import threading
from contextlib import contextmanager
from contextvars import ContextVar

_deadline: ContextVar[float | None] = ContextVar("research_deadline", default=None)
_cancel_events: ContextVar[tuple] = ContextVar("research_cancel_events", default=())
_timer_deadline: ContextVar[float | None] = ContextVar("research_active_timer", default=None)


def remaining() -> float | None:
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def check_deadline() -> None:
    if any(event.is_set() for event in _cancel_events.get()):
        raise asyncio.CancelledError("Research operation cancelled")
    if remaining() == 0:
        raise asyncio.TimeoutError("Research deadline exceeded")


@contextmanager
def deadline_scope(*, seconds: float | None = None, deadline: float | None = None):
    parent = _deadline.get()
    if seconds is not None:
        local = time.monotonic() + max(0.0, seconds)
        deadline = local if deadline is None else min(deadline, local)
    if parent is not None:
        deadline = parent if deadline is None else min(parent, deadline)
    token = _deadline.set(deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


async def bounded(awaitable):
    """wait_for cancels and drains its child; context propagates to child tasks."""
    budget = remaining()
    if budget == 0:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError("Research deadline exceeded")
    active = _timer_deadline.get()
    deadline = _deadline.get()
    if budget is None or (active is not None and deadline >= active):
        # The enclosing wait_for already owns this deadline. Multiple timers for
        # the same instant could cancel a child's cleanup a second time.
        return await awaitable
    token = _timer_deadline.set(deadline)
    try:
        return await asyncio.wait_for(awaitable, budget)
    finally:
        _timer_deadline.reset(token)


async def run_sync(fn, *args, **kwargs):
    # Cancellation cannot kill arbitrary Python threads. Only use this for CPU work,
    # read-only snapshots, or isolated objects; never for orchestrator state changes.
    check_deadline()
    event = threading.Event()
    token = _cancel_events.set(_cancel_events.get() + (event,))
    try:
        return await bounded(asyncio.to_thread(fn, *args, **kwargs))
    except (asyncio.CancelledError, asyncio.TimeoutError):
        event.set()
        raise
    finally:
        _cancel_events.reset(token)


async def call_policy(policy, messages):
    from .recovery import (RecoveryState, current_recovery, classify_error,
                           ResearchAbort, RecoveryError, Failure, wait_retry)
    from .tracing import record_event
    state = current_recovery() or RecoveryState()
    for attempt in range(state.config.max_attempts):
        check_deadline()
        # This is a request-size estimate/reservation, NOT provider billed usage.
        estimate = len(str(messages)) // 3 + max(0, int(getattr(policy, "max_tokens", 0) or 0))
        state.model(estimate)
        try:
            if callable(getattr(policy, "acall", None)):
                response = await bounded(policy.acall(messages))
            else:
                response = await run_sync(policy, messages)
            calls = response.get("tool_calls") if isinstance(response, dict) else None
            if (not isinstance(response, dict)
                    or (response.get("content") is not None and not isinstance(response.get("content"), str))
                    or (calls is not None and (not isinstance(calls, list) or any(
                        not isinstance(tc, dict) or not isinstance(tc.get("function"), dict)
                        or not isinstance(tc["function"].get("name"), str)
                        or (tc.get("id") is not None and not isinstance(tc["id"], str))
                        for tc in calls)))):
                raise RecoveryError(Failure(2, "INVALID_MODEL_OUTPUT", "Invalid model message/tool_calls format", retryable=True))
            if calls:
                ids = [tc["id"] for tc in calls if tc.get("id")]
                if len(ids) != len(set(ids)):
                    raise RecoveryError(Failure(2, "INVALID_MODEL_OUTPUT", "Duplicate tool call IDs in one batch", retryable=True))
            return response
        except asyncio.CancelledError:
            # Do not retry an expired deadline or a still-running synchronous extension.
            policy.was_truncated = True
            raise
        except Exception as exc:
            if isinstance(exc, asyncio.TimeoutError):
                try:
                    check_deadline()
                except asyncio.TimeoutError:
                    policy.was_truncated = True
                    raise
                if not callable(getattr(policy, "acall", None)):
                    policy.was_truncated = True
                    raise
            failure = classify_error(exc, model=True)
            record_event("model_recovery", {"attempt": attempt + 1, "level": failure.level,
                                            "code": failure.code, "message": failure.message})
            if failure.level == 5:
                raise ResearchAbort(failure) from exc
            if not failure.retryable:
                raise
            if (attempt + 1 >= state.config.max_attempts or not state.action()
                    or not await wait_retry(failure, attempt, state.config)):
                raise RecoveryError(Failure(4, "MODEL_RETRIES_EXHAUSTED", failure.message, "task")) from exc
