"""Run-scoped local telemetry plus optional LangSmith; exporter failures never retry business work."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from enum import Enum
import functools
import hashlib
import inspect
import json
import logging
import re
import sys
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)
_recorder: ContextVar[Any] = ContextVar("research_telemetry", default=None)
_parent: ContextVar[str | None] = ContextVar("research_span", default=None)
_SECRET = re.compile(r"api.?key|authorization|password|secret|access_token|connection", re.I)


def safe_data(value: Any, *, limit: int = 2000, depth: int = 0) -> Any:
    """Bounded previews; never serialize arbitrary client/policy reprs or reasoning."""
    if depth > 6:
        return "<depth limit>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…<preview>"
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): "<redacted>" if _SECRET.search(str(k)) or k == "reasoning_content"
                else safe_data(v, limit=limit, depth=depth + 1)
                for k, v in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [safe_data(v, limit=limit, depth=depth + 1) for v in value[:100]]
    return f"<{type(value).__name__}>"


def config_fingerprint(config: dict) -> str:
    # Fingerprints must cover every item, unlike bounded span previews.
    def canonical(value):
        if isinstance(value, dict):
            return {str(k): "<redacted>" if _SECRET.search(str(k)) else canonical(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return f"<{type(value).__name__}>"
    return hashlib.sha256(json.dumps(canonical(config),
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def is_tracing_enabled() -> bool:
    if "langsmith.run_helpers" in sys.modules:
        from langsmith.run_helpers import get_tracing_context
        enabled = get_tracing_context().get("enabled")
        if enabled is not None:
            return bool(enabled)
    from .env_config import get_env
    return (get_env("LANGSMITH_TRACING") or "").lower() in ("true", "1", "yes")


def maybe_wrap_openai_client(client: Any) -> Any:
    if is_tracing_enabled():
        try:
            from langsmith.wrappers import wrap_openai
            return wrap_openai(client, chat_name="ChatOpenAI")
        except Exception as exc:
            logger.warning("LangSmith wrapping unavailable: %s", type(exc).__name__)
    return client


class RunTelemetry:
    def __init__(self, metadata: dict | None = None, run_id: str | None = None):
        self.run_id = run_id or str(uuid.uuid4())
        self.metadata = safe_data(metadata or {})
        self.spans: list[dict] = []
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.status = "running"
        self.trace_id: str | None = None
        self.trace_url: str | None = None
        self.export_errors: list[str] = []
        self.dropped = 0

    def append(self, collection: str, item: dict) -> None:
        with self.lock:
            target = getattr(self, collection)
            if len(target) < 10000:
                target.append(item)
            else:
                self.dropped += 1

    def snapshot(self) -> dict:
        with self.lock:
            spans, events = list(self.spans), list(self.events)
        usages = [e["data"] for e in events if e["name"] == "llm_usage"]
        return {
            "research_run_id": self.run_id, "trace_id": self.trace_id,
            "trace_url": self.trace_url, "status": self.status, "metadata": self.metadata,
            "elapsed_seconds": round(time.monotonic() - self.started, 3),
            "usage": {"llm_calls": len(usages),
                      "llm_requests": sum(e["name"] == "llm_request" for e in events),
                      "llm_errors": sum(e["name"] == "llm_error" for e in events),
                      "provider_usage_calls": sum(u["usage_source"] == "provider" for u in usages),
                      "unavailable_usage_calls": sum(u["usage_source"] != "provider" for u in usages),
                      **{k: sum(u.get(k, 0) or 0 for u in usages)
                         for k in ("prompt_tokens", "completion_tokens", "total_tokens")}},
            "spans": spans, "events": events, "dropped_records": self.dropped,
            "export_errors": list(self.export_errors),
        }


def current_run_id() -> str | None:
    recorder = _recorder.get()
    return recorder.run_id if recorder else None


def record_event(name: str, data: dict) -> None:
    recorder = _recorder.get()
    if recorder:
        recorder.append("events", {"name": name, "span_id": _parent.get(),
                                   "offset_seconds": round(time.monotonic() - recorder.started, 6),
                                   "data": safe_data(data)})


class _Span:
    def __init__(self, native=None):
        self.native = native
        self.outputs = {}

    def add_output(self, outputs: dict) -> None:
        self.outputs.update(safe_data(outputs))
        if self.native:
            try:
                self.native.add_outputs(self.outputs)
            except Exception:
                pass


@contextmanager
def trace_block(name: str, run_type: str = "chain", inputs: dict | None = None,
                tags: list[str] | None = None, metadata: dict | None = None,
                *, run_id: str | None = None):
    recorder = _recorder.get()
    span_id, parent_id, started = run_id or str(uuid.uuid4()), _parent.get(), time.monotonic()
    attrs = {**(recorder.metadata if recorder else {}), **(metadata or {})}
    if recorder:
        attrs["research_run_id"] = recorder.run_id
    native_cm = native = None
    if is_tracing_enabled():
        try:
            from langsmith import trace
            native_cm = trace(name, run_type=run_type, inputs=safe_data(inputs or {}),
                              tags=tags or [], metadata=safe_data(attrs), run_id=span_id)
            native = native_cm.__enter__()
            if recorder and run_id == recorder.run_id and native is not None:
                recorder.trace_id = str(native.trace_id)
                # SDK URL resolution may do sync HTTP; keep it off the hot path.
                import os
                project_url = os.getenv("LANGSMITH_PROJECT_URL", "").rstrip("/")
                if project_url:
                    recorder.trace_url = f"{project_url}/r/{native.id}?trace_id={native.trace_id}"
        except Exception as exc:
            if recorder:
                recorder.export_errors.append(type(exc).__name__)
            logger.warning("Trace setup failed: %s", type(exc).__name__)
    token = _parent.set(span_id)
    span, status, error, exc_info = _Span(native), "success", None, (None, None, None)
    try:
        yield span
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else (
            "timeout" if isinstance(exc, TimeoutError) else "failed")
        error, exc_info = type(exc).__name__, (type(exc), exc, exc.__traceback__)
        raise
    finally:
        _parent.reset(token)
        if native_cm is not None and native is not None:
            try:
                native_cm.__exit__(*exc_info)
            except Exception as exc:
                if recorder:
                    recorder.export_errors.append(type(exc).__name__)
                logger.warning("Trace export failed: %s", type(exc).__name__)
        if recorder:
            recorder.append("spans", {
                "span_id": span_id, "parent_span_id": parent_id, "name": name,
                "run_type": run_type, "status": status, "error_type": error,
                "offset_seconds": round(started - recorder.started, 6),
                "duration_seconds": round(time.monotonic() - started, 6),
                "metadata": safe_data(attrs), "inputs": safe_data(inputs or {}),
                "outputs": span.outputs,
            })


@contextmanager
def research_trace(metadata: dict | None = None):
    recorder = RunTelemetry(metadata)
    token, parent_token = _recorder.set(recorder), _parent.set(None)
    try:
        with trace_block("research.run", metadata=metadata, run_id=recorder.run_id):
            yield recorder
        if recorder.status == "running":
            recorder.status = "success"
    except BaseException as exc:
        recorder.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else (
            "timeout" if isinstance(exc, TimeoutError) else "failed")
        raise
    finally:
        _parent.reset(parent_token)
        _recorder.reset(token)


def traceable(run_type="chain", name=None, tags=None, metadata=None):
    def decorate(func):
        signature = inspect.signature(func)
        def inputs(args, kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            return {k: safe_data(v) for k, v in bound.arguments.items() if k not in ("self", "cls")}
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def awrapped(*args, **kwargs):
                if not _recorder.get() and not is_tracing_enabled():
                    return await func(*args, **kwargs)
                with trace_block(name or func.__name__, run_type, inputs(args, kwargs), tags, metadata) as span:
                    result = await func(*args, **kwargs)
                    span.add_output({"result": result})
                    return result
            return awrapped
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if not _recorder.get() and not is_tracing_enabled():
                return func(*args, **kwargs)
            with trace_block(name or func.__name__, run_type, inputs(args, kwargs), tags, metadata) as span:
                result = func(*args, **kwargs)
                span.add_output({"result": result})
                return result
        return wrapped
    return decorate


def trace_agent(name=None, tags=None):
    return traceable("chain", name, tags)


def trace_tool(name=None, tags=None):
    return traceable("tool", name, tags)


def trace_chain(name=None, tags=None):
    return traceable("chain", name, tags)


def trace_retriever(name=None, tags=None):
    return traceable("retriever", name, tags)


def flush_traces(timeout: float = 2.0) -> None:
    """Bounded shutdown outside the research deadline."""
    if is_tracing_enabled():
        try:
            from langsmith.run_trees import get_cached_client
            get_cached_client().flush(timeout=timeout)
        except Exception as exc:
            logger.warning("Trace flush incomplete: %s", type(exc).__name__)
