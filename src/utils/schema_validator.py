"""
Schema 校验与 LLM 输出重试模块

设计原则：
- 每个 LLM 交互节点定义独立的校验函数
- 校验失败时构造结构化错误描述，回传给 LLM 重试
- 最大重试 2 次，仍失败则降级
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    MISSING_FIELD = "missing_field"
    INVALID_TYPE = "invalid_type"
    INVALID_ENUM = "invalid_enum"
    OUT_OF_RANGE = "out_of_range"
    EMPTY_VALUE = "empty_value"
    JSON_PARSE = "json_parse"


@dataclass
class SchemaError:
    field: str
    error_type: ErrorType
    expected: str
    got: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[SchemaError] = field(default_factory=list)

    def to_feedback(self) -> str:
        """将校验错误转换为 LLM 可理解的重试提示。"""
        if self.valid:
            return ""
        lines = ["⚠️ 你的上一次输出格式有误，请修正后重新输出："]
        for i, e in enumerate(self.errors, 1):
            lines.append(
                f"  {i}. 字段 '{e.field}': {e.error_type.value} "
                f"(期望 {e.expected}, 实际得到 {e.got})"
            )
        lines.append("\n请严格按照要求的 JSON 格式重新输出。")
        return "\n".join(lines)


# ============================================================================
# 各节点 Schema 定义与校验函数
# ============================================================================


def validate_red_agent_output(data: dict, sub_dim_keys: list[str]) -> ValidationResult:
    """校验 Red Agent 评分输出的 JSON 结构。

    期望格式: { sub_key: { "score": float(0-10), "issues": [...] }, ... }
    """
    errors: list[SchemaError] = []

    for key in sub_dim_keys:
        sub = data.get(key)
        if not isinstance(sub, dict):
            errors.append(SchemaError(
                field=key, error_type=ErrorType.MISSING_FIELD,
                expected="object with 'score' and 'issues'", got=str(type(sub).__name__)
            ))
            continue

        # score 校验
        score = sub.get("score")
        if score is None:
            errors.append(SchemaError(
                field=f"{key}.score", error_type=ErrorType.MISSING_FIELD,
                expected="float 0-10", got="None"
            ))
        elif not isinstance(score, (int, float)):
            errors.append(SchemaError(
                field=f"{key}.score", error_type=ErrorType.INVALID_TYPE,
                expected="float", got=str(type(score).__name__)
            ))
        elif score < 0 or score > 10:
            errors.append(SchemaError(
                field=f"{key}.score", error_type=ErrorType.OUT_OF_RANGE,
                expected="0-10", got=str(score)
            ))

        # issues 校验
        issues = sub.get("issues")
        if not isinstance(issues, list):
            errors.append(SchemaError(
                field=f"{key}.issues", error_type=ErrorType.INVALID_TYPE,
                expected="list", got=str(type(issues).__name__)
            ))
            continue

        for j, issue in enumerate(issues):
            if not isinstance(issue, dict):
                errors.append(SchemaError(
                    field=f"{key}.issues[{j}]", error_type=ErrorType.INVALID_TYPE,
                    expected="object", got=str(type(issue).__name__)
                ))
                continue
            # severity 校验
            sev = issue.get("severity", "")
            if sev not in ("critical", "major", "minor", "严重", "重要", "较大", "轻微", "一般"):
                errors.append(SchemaError(
                    field=f"{key}.issues[{j}].severity", error_type=ErrorType.INVALID_ENUM,
                    expected="critical|major|minor", got=str(sev)
                ))
            # description 校验
            desc = issue.get("description", "")
            if not desc or not isinstance(desc, str) or len(desc.strip()) < 3:
                errors.append(SchemaError(
                    field=f"{key}.issues[{j}].description", error_type=ErrorType.EMPTY_VALUE,
                    expected="non-empty string (min 3 chars)", got=str(desc)[:50]
                ))
            # fix_type 校验
            ft = issue.get("fix_type", "")
            if ft not in ("in_place", "search", "removal", "就地修复", "修正", "补充搜索", "删除", "移除"):
                errors.append(SchemaError(
                    field=f"{key}.issues[{j}].fix_type", error_type=ErrorType.INVALID_ENUM,
                    expected="in_place|search|removal", got=str(ft)
                ))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_blue_fix_output(data: dict, fix_type: str = "in_place") -> ValidationResult:
    """校验 Blue Agent 修复输出的 JSON 结构。

    期望格式: { "fixed_content": "<非空字符串>", "changes": [...] }
    或 removal: { "fixed_content": "...", "removed_segments": [...] }
    """
    errors: list[SchemaError] = []

    fc = data.get("fixed_content", "")
    if not fc or not isinstance(fc, str) or len(fc.strip()) < 10:
        errors.append(SchemaError(
            field="fixed_content", error_type=ErrorType.EMPTY_VALUE,
            expected="non-empty string (min 10 chars)", got=str(fc)[:50]
        ))

    # changes 或 removed_segments
    changes_key = "removed_segments" if fix_type == "removal" else "changes"
    changes = data.get(changes_key)
    if not isinstance(changes, list):
        errors.append(SchemaError(
            field=changes_key, error_type=ErrorType.INVALID_TYPE,
            expected="list", got=str(type(changes).__name__)
        ))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_planner_output(data: dict) -> ValidationResult:
    """校验 Planner 生成的 DAG JSON 结构。

    期望格式: { "sub_tasks": [{ "task_id": str, "task_type": search|analyze,
                "description": str, "dependencies": [...], ... }, ...] }
    """
    errors: list[SchemaError] = []

    sub_tasks = data.get("sub_tasks")
    if not isinstance(sub_tasks, list):
        errors.append(SchemaError(
            field="sub_tasks", error_type=ErrorType.INVALID_TYPE,
            expected="list", got=str(type(sub_tasks).__name__)
        ))
        return ValidationResult(valid=False, errors=errors)

    if not sub_tasks:
        errors.append(SchemaError(
            field="sub_tasks", error_type=ErrorType.OUT_OF_RANGE,
            expected="at least one sub-task", got=f"length={len(sub_tasks)}"
        ))

    valid_task_types = {"search", "analyze"}
    task_ids: set[str] = set()
    task_type_map: dict[str, str] = {}
    for i, st in enumerate(sub_tasks):
        if not isinstance(st, dict):
            errors.append(SchemaError(
                field=f"sub_tasks[{i}]", error_type=ErrorType.INVALID_TYPE,
                expected="object", got=str(type(st).__name__)
            ))
            continue

        tid = st.get("task_id", "")
        if not tid or not isinstance(tid, str):
            errors.append(SchemaError(
                field=f"sub_tasks[{i}].task_id", error_type=ErrorType.EMPTY_VALUE,
                expected="non-empty string", got=str(tid)
            ))
        else:
            if tid in task_ids:
                errors.append(SchemaError(f"sub_tasks[{i}].task_id", ErrorType.INVALID_TYPE, "unique task_id", tid))
            task_ids.add(tid)

        tt = st.get("task_type", "")
        if not isinstance(tt, str) or tt not in valid_task_types:
            errors.append(SchemaError(
                field=f"sub_tasks[{i}].task_type", error_type=ErrorType.INVALID_ENUM,
                expected="search|analyze", got=str(tt)
            ))
        # 收集 task_type 映射（用于后续依赖方向校验）
        if isinstance(tid, str) and tid:
            task_type_map[tid] = tt

        desc = st.get("description", "")
        if not desc or not isinstance(desc, str) or len(desc.strip()) < 5:
            errors.append(SchemaError(
                field=f"sub_tasks[{i}].description", error_type=ErrorType.EMPTY_VALUE,
                expected="non-empty string (min 5 chars)", got=str(desc)[:50]
            ))

        timeout = st.get("timeout_seconds", 300)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            errors.append(SchemaError(f"sub_tasks[{i}].timeout_seconds", ErrorType.OUT_OF_RANGE, "finite positive number", str(timeout)))
        if type(st.get("priority", 1)) is not int:
            errors.append(SchemaError(f"sub_tasks[{i}].priority", ErrorType.INVALID_TYPE, "integer", str(st.get("priority"))))
        for field_name in ("context_keys", "search_hints"):
            values = st.get(field_name, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                errors.append(SchemaError(f"sub_tasks[{i}].{field_name}", ErrorType.INVALID_TYPE, "list of strings", str(values)))
        if not isinstance(st.get("expected_type", "factual"), str):
            errors.append(SchemaError(f"sub_tasks[{i}].expected_type", ErrorType.INVALID_TYPE, "string", str(st.get("expected_type"))))
        deps = st.get("dependencies", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            errors.append(SchemaError(
                field=f"sub_tasks[{i}].dependencies", error_type=ErrorType.INVALID_TYPE,
                expected="list", got=str(type(deps).__name__)
            ))

    # 校验 dependencies 引用的 task_id 是否在 sub_tasks 中
    for i, st in enumerate(sub_tasks):
        if not isinstance(st, dict):
            continue
        deps = st.get("dependencies", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            continue
        for dep in deps:
            if dep not in task_ids:
                errors.append(SchemaError(
                    field=f"sub_tasks[{i}].dependencies", error_type=ErrorType.OUT_OF_RANGE,
                    expected=f"one of {sorted(task_ids)}", got=f"'{dep}' (not found in task_ids)"
                ))

    # 方案1: 依赖方向约束 —— search 不能依赖 analyze（信息流应为 search → analyze）
    # 方案3: 依赖数量约束 —— analyze 应有依赖（存在 search 任务时）
    has_search = "search" in task_type_map.values()
    for i, st in enumerate(sub_tasks):
        if not isinstance(st, dict):
            continue
        tt = st.get("task_type", "")
        deps = st.get("dependencies", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            continue

        # 方案1: search 依赖 analyze → 方向错误
        if tt == "search":
            for dep in deps:
                if task_type_map.get(dep) == "analyze":
                    errors.append(SchemaError(
                        field=f"sub_tasks[{i}].dependencies",
                        error_type=ErrorType.INVALID_ENUM,
                        expected="search 不能依赖 analyze（信息流应为 search → analyze）",
                        got=f"依赖了 analyze 类型任务 '{dep}'"
                    ))

        # 方案3: analyze 无依赖（且存在 search 任务）→ 退化成 search
        if tt == "analyze" and has_search and not deps:
            errors.append(SchemaError(
                field=f"sub_tasks[{i}].dependencies",
                error_type=ErrorType.MISSING_FIELD,
                expected="analyze 任务应至少依赖一个 search 任务",
                got="空列表（analyze 无依赖会退化成 search）"
            ))

    if not errors:
        graph = {st["task_id"]: set(st.get("dependencies", [])) for st in sub_tasks}
        completed = set()
        while graph:
            ready = {key for key, deps in graph.items() if deps <= completed}
            if not ready:
                errors.append(SchemaError("sub_tasks.dependencies", ErrorType.INVALID_TYPE, "acyclic graph", "cycle detected"))
                break
            completed.update(ready)
            graph = {key: deps for key, deps in graph.items() if key not in ready}
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# 通用重试包装器
# ============================================================================


async def retry_with_schema_feedback(
    policy_call: Callable,       # async func(messages) -> response
    messages: list[dict],        # 当前对话历史（会被原地修改）
    validate_fn: Callable[[dict], ValidationResult],
    max_retries: int = 2,
) -> dict | None:
    """调用 LLM → 解析 JSON → Schema 校验 → 失败则回传错误重试。

    Args:
        policy_call: 异步 LLM 调用函数
        messages: 对话历史（校验失败时会追加 user 消息）
        validate_fn: 校验函数，接受解析后的 dict，返回 ValidationResult
        max_retries: 最大重试次数

    Returns:
        校验通过的解析后 dict，或全部失败时返回 None
    """
    import json
    from .recovery import RecoveryError, current_recovery

    for attempt in range(max_retries + 1):
        state = current_recovery()
        if attempt and state is not None and not state.action():
            return None
        try:
            resp = await policy_call(messages)
            raw = resp.get("content", "") if isinstance(resp, dict) else getattr(resp, "content", "")
            raw = raw or ""
        except TimeoutError:
            raise
        except RecoveryError:
            # Transport retries belong exclusively to call_policy.
            return None
        except Exception as e:
            if "[LLM_AUTH_OR_QUOTA_ERROR]" in str(e):
                raise
            logger.warning(f"[SchemaValidator] LLM call failed (attempt {attempt+1}): {e}")
            return None

        # 尝试解析 JSON
        data = None
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 ```json 代码块
            import re
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    pass
            # 尝试提取 {...}
            if data is None:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass

        if not isinstance(data, dict):
            if attempt < max_retries:
                messages.append({
                    "role": "user",
                    "content": (
                        "⚠️ 你的上次输出不是合法的 JSON 格式。"
                        "请只输出 JSON 对象，不要包含其他文字。"
                    )
                })
                continue
            return None

        # Schema 校验
        result = validate_fn(data)
        if result.valid:
            return data

        logger.warning(
            f"[SchemaValidator] Validation failed (attempt {attempt+1}): "
            f"{len(result.errors)} error(s)"
        )
        if attempt < max_retries:
            messages.append({"role": "user", "content": result.to_feedback()})

    return None  # 所有重试均失败
