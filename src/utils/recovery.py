"""Five-level recovery: call repair, retry, fallback, replan, request termination.

No provider requests happen here without the caller's configured tools/policy.
RecoveryState is request-scoped, shared by asyncio children, never process-global.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import inspect
import json
import math
import random
import re
from typing import Any


@dataclass(frozen=True)
class RecoveryConfig:
    max_attempts: int = 3  # Includes the first execution.
    max_repairs: int = 1
    max_fallbacks: int = 2
    max_task_tool_calls: int = 12
    max_run_tool_calls: int = 96
    max_run_model_calls: int = 64
    max_recovery_actions: int = 32
    max_estimated_tokens: int = 1_000_000
    base_delay: float = 1.0
    max_delay: float = 8.0
    jitter: float = 0.2

    def __post_init__(self):
        for key in ("max_attempts", "max_task_tool_calls", "max_run_tool_calls",
                    "max_run_model_calls", "max_estimated_tokens"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"recovery.{key} must be a positive integer")
        for key in ("max_repairs", "max_fallbacks", "max_recovery_actions"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"recovery.{key} must be a nonnegative integer")
        for key in ("base_delay", "max_delay", "jitter"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"recovery.{key} must be finite and nonnegative")
        if self.jitter > 1 or self.max_delay < self.base_delay:
            raise ValueError("recovery requires jitter <= 1 and max_delay >= base_delay")

    @classmethod
    def from_dict(cls, values=None):
        if isinstance(values, cls):
            return values
        return cls(**(values or {}))


@dataclass
class Failure:
    level: int
    code: str
    message: str
    scope: str = "call"
    retryable: bool = False
    retry_after: float | None = None


class RecoveryError(RuntimeError):
    def __init__(self, failure: Failure):
        self.failure = failure
        super().__init__(f"[{failure.code}] {failure.message}")


class ResearchAbort(BaseException):
    """L5 control signal, like cancellation: ordinary fallback handlers must not swallow it.

    Caught at Orchestrator.run; standalone callers may explicitly catch it.
    """
    def __init__(self, failure: Failure):
        self.failure = failure
        super().__init__(f"[{failure.code}] {failure.message}")


def public_message(value):
    """Remove credentials and query-string secrets before model/trace feedback."""
    text = str(value)
    text = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|authorization|secret)[\"']?\s*[=:]\s*[\"']?)[^\s&\",;}']+",
                  r"\1[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:1500]


def retry_after_seconds(value):
    if value is None:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = (parsedate_to_datetime(str(value)) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0., seconds) if math.isfinite(seconds) else None


async def http_error_payload(response, query="", source=""):
    """Preserve HTTP status and Retry-After even for non-JSON error bodies."""
    return {"error": public_message(f"HTTP {response.status}: {(await response.text())[:1000]}"),
            "status_code": response.status, "retry_after": response.headers.get("Retry-After"),
            "query": query, "source": source, "results": [], "papers": []}


async def close_tools(tools):
    """Try every task-owned transport; cleanup errors cannot mask L5/cancellation."""
    from .tracing import record_event
    healthy, seen = True, set()
    for tool in tools:
        for candidate in [tool, *getattr(tool, "recovery_fallbacks", [])]:
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            close = getattr(candidate, "close_session", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                healthy = False
                record_event("resource_cleanup_error", {"tool": candidate.name, "error": public_message(exc)})
    return healthy


def classify_error(value, *, model=False):
    if isinstance(value, RecoveryError):
        return value.failure
    status = getattr(value, "status_code", None)
    if status is None:
        status = getattr(value, "status", None)
    payload = value if isinstance(value, dict) else {}
    status = payload.get("status_code", status)
    try:
        status = int(status) if status is not None else None
    except (ValueError, TypeError):
        status = None
    message = public_message(payload.get("error", value))
    lower = (message + " " + type(value).__name__).lower()
    response = getattr(value, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = retry_after_seconds(payload.get("retry_after", headers.get("retry-after")))
    if status is None:
        match = re.search(r"(?:http|status(?:_code)?)[\s:=]*(\d{3})\b", lower)
        if match:
            status = int(match[1])
    quota = any(x in lower for x in ("insufficient balance", "insufficient_quota", "额度耗尽", "余额不足", "quota exceeded"))
    auth = any(x in lower for x in ("invalid api key", "unauthorized", "forbidden", "authentication", "缺少 api", "需要 api key", "权限不足"))
    if status in (401, 402, 403) or quota or auth or "llm_auth_or_quota_error" in lower:
        return Failure(5 if model else 3, "LLM_AUTH_OR_QUOTA_ERROR" if model else "TOOL_UNAVAILABLE",
                       message, "backend")
    if "context_length" in lower or "context length" in lower:
        return Failure(4, "CONTEXT_LENGTH_EXCEEDED", message, "task")
    if status in (404, 410) or any(x in lower for x in ("not found", "not in the fixed evidence", "access denied")):
        return Failure(3, "RESOURCE_UNAVAILABLE", message, "resource")
    if status in (408, 425, 429) or (status is not None and 500 <= status <= 599) or any(
        x in lower for x in ("timeout", "timed out", "connection", "connecterror", "network", "网络", "rate limit", "限流", "temporarily", "服务暂时")
    ) or isinstance(value, (ConnectionError, OSError)):
        return Failure(2, "TRANSIENT_ERROR", message, "backend", True, retry_after)
    if any(x in lower for x in ("syntaxerror", "indentationerror", "invalid value", "invalid url", "invalid path",
                                "empty expression", "unknown action", "division by zero", "unsupported ast",
                                "unknown function", "math domain error")):
        return Failure(1, "INVALID_ARGUMENTS", message)
    if any(x in lower for x in ("parse", "解析", "json", "malformed")):
        return Failure(2, "INVALID_OUTPUT", message, "call", True, retry_after)
    return Failure(3, "EXECUTION_ERROR", message, "call")


@dataclass
class RecoveryState:
    config: RecoveryConfig = field(default_factory=RecoveryConfig)
    tool_calls: int = 0
    model_calls: int = 0
    estimated_tokens: int = 0
    recovery_actions: int = 0
    circuits: set = field(default_factory=set)

    def action(self):
        if self.recovery_actions >= self.config.max_recovery_actions:
            return False
        self.recovery_actions += 1
        return True

    def tool(self):
        if self.tool_calls >= self.config.max_run_tool_calls:
            raise ResearchAbort(Failure(5, "TOOL_BUDGET_EXHAUSTED", "研究工具调用预算耗尽", "research"))
        self.tool_calls += 1

    def model(self, tokens):
        if (self.model_calls >= self.config.max_run_model_calls
                or self.estimated_tokens + tokens > self.config.max_estimated_tokens):
            raise ResearchAbort(Failure(5, "MODEL_BUDGET_EXHAUSTED", "研究模型调用或估算 token 预算耗尽", "research"))
        self.model_calls += 1
        self.estimated_tokens += tokens


_state = ContextVar("research_recovery_state", default=None)


def current_recovery():
    return _state.get()


@contextmanager
def recovery_scope(config=None):
    state = RecoveryState(RecoveryConfig.from_dict(config))
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)


async def wait_retry(failure, attempt, config):
    from .execution import bounded, check_deadline, remaining
    check_deadline()
    delay = min(config.max_delay, config.base_delay * 2 ** attempt)
    delay = min(config.max_delay, delay * random.uniform(1 - config.jitter, 1 + config.jitter))
    if failure.retry_after is not None:
        delay = max(delay, failure.retry_after)  # Never shorten Retry-After.
    left = remaining()
    if left is not None and delay >= left:
        return False  # Caller can degrade; do not start a request before Retry-After.
    await bounded(asyncio.sleep(delay))
    return True


def _validate(value, schema, path="arguments"):
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (float, int), "boolean": bool, "null": type(None)}
    if expected in types and (not isinstance(value, types[expected]) or
            (expected in ("integer", "number") and isinstance(value, bool))):
        raise ValueError(f"{path} must be {expected}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"{path}.{key} is not allowed")
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate(item, schema.get("items", {}), f"{path}[{i}]")
    for name, predicate in (("minimum", lambda x, y: x >= y), ("maximum", lambda x, y: x <= y)):
        if name in schema and isinstance(value, (float, int)) and not predicate(value, schema[name]):
            raise ValueError(f"{path} violates {name}={schema[name]}")
    if "anyOf" in schema:
        for branch in schema["anyOf"]:
            try:
                _validate(value, branch, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError(f"{path} must satisfy at least one anyOf alternative")


def validate_arguments(tool, args):
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    schema = tool.get_openai_tool_schema()["function"]["parameters"] if hasattr(tool, "get_openai_tool_schema") else {}
    _validate(args, schema)
    inspect.signature(tool.execute).bind(**args)
    for key in ("query", "url", "file_path", "code", "expression", "paper_id"):
        if key in args and (not isinstance(args[key], str) or not args[key].strip()):
            raise ValueError(f"arguments.{key} must be a nonempty string")
    if "url" in args and not args["url"].startswith(("https://", "http://")):
        raise ValueError("arguments.url must start with http:// or https://")
    for key in ("top_n", "max_results", "max_chars", "timeout", "max_entries"):
        if key in args and (type(args[key]) is not int or args[key] <= 0):
            if key == "top_n" and type(args[key]) is int and args[key] == 0:
                continue
            raise ValueError(f"arguments.{key} must be a positive integer")
    if tool.name == "notepad":
        allowed = {"write": {"content", "category", "source"}, "read": {"category", "max_entries"},
                   "clear": {"category"}, "search": {"keyword", "max_entries"}, "list_categories": set()}
        action = args.get("action")
        if action not in allowed or set(args) - {"action"} - allowed[action]:
            raise ValueError("arguments contain an unknown action or unsupported action parameters")
        required = {"write": "content", "search": "keyword"}.get(action)
        if required and not str(args.get(required, "")).strip():
            raise ValueError(f"arguments.{required} is required for {action}")


@dataclass
class ToolOutcome:
    status: str
    data: Any = None
    failure: Failure | None = None
    attempts: int = 0
    tool_name: str = ""
    backend: str = ""

    def to_dict(self):
        return {"status": self.status, "data": self.data,
                "error": asdict(self.failure) if self.failure else None,
                "attempts": self.attempts, "tool": self.tool_name, "backend": self.backend}


def normalize_result(tool_name, result):
    """Validate legacy tool returns without treating arbitrary article text as an error."""
    if isinstance(result, dict) and (result.get("error") or result.get("success") is False):
        payload = result if result.get("error") else {**result, "error": result.get("stderr", "Tool execution failed")}
        failure = classify_error(payload)
        if tool_name == "browser" and result.get("status_code"):
            failure.scope = "resource"  # A forbidden page does not disable browsing other sites.
            if failure.level == 3:
                failure.code = "RESOURCE_UNAVAILABLE"
        return ToolOutcome("error", failure=failure)
    if isinstance(result, str):
        match = re.match(r"^\s*(?:\[[\w ]+ (?:Error|Warning)\]|Error:)", result)
        if match:
            if "Warning" in match[0]:
                return ToolOutcome("empty", data=result)
            return ToolOutcome("error", failure=classify_error(result))
        if re.search(r"<html\b|<!doctype html", result[:500], re.I):
            return ToolOutcome("error", failure=Failure(2, "INVALID_OUTPUT", "Expected extracted text, received HTML", retryable=True))
    if result is None or result == "" or (isinstance(result, str) and not result.strip()):
        return ToolOutcome("empty", data=result)
    if tool_name in ("web_search", "arxiv_reader"):
        key = "results" if tool_name == "web_search" else "papers"
        if not isinstance(result, dict) or not isinstance(result.get(key), list):
            return ToolOutcome("error", failure=Failure(2, "INVALID_OUTPUT", f"Expected object with {key} array", retryable=True))
        if not result[key]:
            return ToolOutcome("empty", data=result)
        for row in result[key]:
            text_keys = ("snippet", "text") if key == "results" else ("summary", "abstract", "title")
            if not isinstance(row, dict) or not any(isinstance(row.get(k), str) and row[k].strip() for k in text_keys):
                return ToolOutcome("error", failure=Failure(2, "INVALID_OUTPUT", f"Invalid or content-free {key} item", retryable=True))
    if tool_name in ("browser", "file_reader", "calculator", "notepad") and not isinstance(result, str):
        return ToolOutcome("error", failure=Failure(2, "INVALID_OUTPUT", "Expected text result", retryable=True))
    if tool_name == "code_sandbox" and (not isinstance(result, dict) or result.get("success") is not True):
        return ToolOutcome("error", failure=Failure(2, "INVALID_OUTPUT", "Expected code execution status", retryable=True))
    return ToolOutcome("success", data=result)


class ToolExecutor:
    """Task-local L1/L2/L3 controller, with request-wide budgets and backend circuits."""
    def __init__(self, tools, config=None):
        self.tools = {t.name: t for t in tools}
        self.state = current_recovery() or RecoveryState(RecoveryConfig.from_dict(config))
        self.config = self.state.config
        self.calls = 0
        self.repairs = 0
        self.fallbacks = 0
        self.disabled = set()
        self.cache = {}
        self.history = []
        self.pending_failure = None

    def available(self):
        return [name for name, tool in self.tools.items() if name not in self.disabled and
                any(self._backend(t) not in self.state.circuits for t in [tool, *getattr(tool, "recovery_fallbacks", [])])]

    @staticmethod
    def _backend(tool):
        return tool.name, getattr(tool, "backend", type(tool).__name__)

    def _record(self, outcome):
        from .tracing import record_event
        self.history.append(outcome.to_dict())
        record_event("tool_recovery", outcome.to_dict())
        return outcome

    def repair_feedback(self):
        if self.repairs >= self.config.max_repairs or not self.state.action():
            return False
        self.repairs += 1
        return True

    async def run(self, name, arguments, call_id=None):
        from .execution import bounded, check_deadline, run_sync
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            tool = self.tools.get(name)
            if tool is None:
                raise ValueError(f"Unknown tool {name!r}; available tools: {self.available()}")
            validate_arguments(tool, args)
        except (ValueError, TypeError, KeyError) as exc:
            return self._record(ToolOutcome("error", failure=Failure(1, "INVALID_ARGUMENTS", public_message(exc)), tool_name=name))
        # Repeated read calls are valid; side effects with the same operation ID are not repeated.
        cache_key = (name, call_id)
        signature = json.dumps(args, sort_keys=True, ensure_ascii=False)
        read_only = name in ("web_search", "arxiv_reader", "browser", "file_reader", "calculator") or getattr(tool, "retry_safe", False)
        if call_id and not read_only and cache_key in self.cache:
            saved_signature, outcome = self.cache[cache_key]
            if saved_signature != signature:
                return self._record(ToolOutcome("error", failure=Failure(1, "CALL_ID_REUSED", "A side-effect call ID cannot be reused with different arguments"), tool_name=name))
            return outcome
        if name in self.disabled:
            return self._record(ToolOutcome("error", failure=Failure(3, "TOOL_DISABLED", "Tool disabled for this task; choose an available equivalent capability", "tool"), tool_name=name))
        families = ({"web_search", "arxiv_reader", "browser", "file_reader"}, {"calculator", "code_sandbox"})
        if self.pending_failure and name != self.pending_failure and any(
                name in group and self.pending_failure in group for group in families):
            if self.fallbacks >= self.config.max_fallbacks or not self.state.action():
                return self._record(ToolOutcome("error", failure=Failure(4, "FALLBACK_BUDGET_EXHAUSTED", "子任务替代路径预算耗尽，需要 Replan", "task"), tool_name=name))
            self.fallbacks += 1
            self.pending_failure = None
        candidates = [tool, *(getattr(tool, "recovery_fallbacks", []) if read_only else [])]
        final = ToolOutcome("error", failure=Failure(3, "CIRCUIT_OPEN", "Backend unavailable for this research", "backend"), tool_name=name)
        for candidate_index, candidate in enumerate(candidates):
            if self._backend(candidate) in self.state.circuits:
                continue
            if candidate_index:
                if self.fallbacks >= self.config.max_fallbacks or not self.state.action():
                    break
                self.fallbacks += 1
            for attempt in range(self.config.max_attempts if read_only else 1):
                check_deadline()
                if self.calls >= self.config.max_task_tool_calls:
                    return self._record(ToolOutcome("error", failure=Failure(4, "TASK_TOOL_BUDGET_EXHAUSTED", "子任务工具预算耗尽，需要 Replan", "task"), tool_name=name))
                self.state.tool()
                self.calls += 1
                try:
                    if inspect.iscoroutinefunction(candidate.execute):
                        raw = await bounded(candidate.execute(**args))
                    else:
                        raw = await run_sync(candidate.execute, **args)
                    final = normalize_result(name, raw)
                except asyncio.CancelledError:
                    raise
                except asyncio.TimeoutError:
                    # Expired enclosing deadlines never retry; an earlier native
                    # async transport timeout is an ordinary transient failure.
                    check_deadline()
                    if not inspect.iscoroutinefunction(candidate.execute):
                        raise
                    final = ToolOutcome("error", failure=Failure(2, "TRANSIENT_ERROR", "Tool transport timeout", "backend", True))
                except Exception as exc:
                    final = ToolOutcome("error", failure=classify_error(exc))
                final.attempts = attempt + 1
                final.tool_name = name
                final.backend = str(self._backend(candidate)[1])
                self._record(final)
                if final.status in ("success", "empty", "partial") or final.failure.level == 1:
                    break
                if not final.failure.retryable or attempt + 1 >= self.config.max_attempts or not self.state.action():
                    break
                if not await wait_retry(final.failure, attempt, self.config):
                    break
            if final.status in ("success", "empty", "partial") or final.failure.level in (1, 4):
                break
            if final.failure.scope == "backend":
                self.state.circuits.add(self._backend(candidate))
        if final.status == "error" and final.failure.level == 2:
            final.failure = Failure(3, "RETRIES_EXHAUSTED", final.failure.message, final.failure.scope)
            self._record(final)
        if final.status == "error" and final.failure.level == 3 and final.failure.scope != "resource":
            self.disabled.add(name)
        if final.status == "error" and final.failure.level == 3:
            self.pending_failure = name
        if call_id and not read_only:
            self.cache[cache_key] = signature, final
        return final
