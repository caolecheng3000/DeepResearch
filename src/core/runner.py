#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
DeepResearch Agent 核心运行逻辑。

本模块包含初始化所有模块和执行完整研究流程的核心函数，
供 scripts/ 和 evaluation/ 统一调用，避免 evaluation/ 反向依赖 scripts/。

对外接口:
    - load_config(config_path) -> dict
    - initialize_modules(config) -> dict
    - run_research(query, config, modules) -> str
    - save_report(report, query, output_dir) -> str
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# 将项目根目录加入 sys.path，确保 src 包可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """配置全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    加载 YAML 配置文件。

    若未指定路径，默认加载 configs/default.yaml。
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


# ---------------------------------------------------------------------------
# 工具工厂
# ---------------------------------------------------------------------------
def _create_tools_factory(config: dict):
    """创建工具工厂函数，返回 Agent 可用的工具列表。"""
    tools_cfg = config.get("tools", {})
    if tools_cfg.get("snapshot_path"):
        from src.tools.snapshot import SnapshotSearchTool, SnapshotBrowserTool
        fixed_tools = [SnapshotSearchTool(tools_cfg["snapshot_path"]), SnapshotBrowserTool(tools_cfg["snapshot_path"])]
        if tools_cfg.get("snapshot_calculator", False):
            from src.tools.calculator import CalculatorTool
            fixed_tools.append(CalculatorTool())
        return fixed_tools
    mock_mode = tools_cfg.get("web_search", {}).get("mock_mode", True)

    from src.tools import (
        WebSearchTool,
        MockWebSearchTool,
        ArxivReaderTool,
        BrowserTool,
        MockBrowserTool,
        FileReaderTool,
        CodeSandboxTool,
        CalculatorTool,
        NotepadTool,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        tools["web_search"] = WebSearchTool()

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader（不限制目录）
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator
    tools["calculator"] = CalculatorTool()

    # 7. notepad
    tools["notepad"] = NotepadTool()

    # L3: only explicitly configured real backends are eligible; never enable mock
    # or live network fallback for fixed-snapshot evaluations.
    fallbacks = tools_cfg.get("fallback_backends", {})
    for name, allowed in (("web_search", {"serpapi", "bing", "bocha", "metaso"}),
                          ("arxiv_reader", {"arxiv", "semantic_scholar", "openalex"})):
        backends = fallbacks.get(name, [])
        if not isinstance(backends, list) or any(backend not in allowed for backend in backends):
            raise ValueError(f"tools.fallback_backends.{name} must be a list from {sorted(allowed)}")
        if not mock_mode:
            constructor = WebSearchTool if name == "web_search" else ArxivReaderTool
            tools[name].recovery_fallbacks = [constructor(backend=backend)
                for backend in dict.fromkeys(backends) if backend != tools[name].backend]

    # 返回列表形式（AgentPool 和 Agent 构造函数需要 list）
    return list(tools.values())


# ---------------------------------------------------------------------------
# 模块初始化
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """
    根据配置初始化所有核心模块。

    Args:
        config: 全局配置字典。
        session_id: 会话 ID，用于 memory store 的 session 隔离。

    返回一个包含各模块实例的字典。
    """
    logger = logging.getLogger("runner")
    logger.info("正在初始化核心模块...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 多后端 LLM 初始化（从 .env + configs/default.yaml 读取配置）
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "deepseek")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})

    # 辅助函数：根据模块名获取采样参数覆盖
    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        """合并后端全局默认 + 模块级覆盖参数。"""
        kwargs = {}
        # 1. 后端全局默认
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        # 2. 模块级覆盖（优先级更高）
        module_overrides = backend_sampling.get("modules", {}).get(module_name, {})
        kwargs.update(module_overrides)
        return kwargs

    # 默认后端模板；各模块与 Worker 使用独立 policy 副本
    default_kwargs = _get_sampling_kwargs("default", default_backend)
    default_policy = ModelRouter.create_backend(default_backend, **default_kwargs).clone()
    modules["default_policy"] = default_policy
    logger.info(f"[LLM] 默认后端已加载: {default_backend} ({default_kwargs})")

    # 多后端分工：不同模块用不同后端 + 不同采样参数
    # module_tag 传入模块名，确保每个模块持有独立 policy 实例，
    # 避免 set_tools / max_tokens 等运行时状态跨模块泄漏
    for module_name, backend_name in backend_mapping.items():
        kwargs = _get_sampling_kwargs(module_name, backend_name)
        modules[f"{module_name}_policy"] = ModelRouter.create_backend(
            backend_name, module_tag=module_name, **kwargs
        ).clone()
        logger.info(f"[LLM] {module_name} → 后端={backend_name}, 采样={kwargs}")

    # Even when no backend mapping is configured, roles do not share mutable policies.
    for role in ("solver", "planner", "summarizer", "compressor", "red_agent", "blue_agent"):
        modules.setdefault(f"{role}_policy", default_policy.clone())

    # 若未配置分工，模块使用默认配置的独立 policy 副本。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tools（真实工具或 Mock 工具）
    # 必须在 Planner 之前创建：Planner 的「规划前侦察」阶段需要白名单内的搜索工具
    # ------------------------------------------------------------------
    tools_list = _create_tools_factory(config)
    modules["tools"] = tools_list
    logger.info(f"Tools 模块已初始化（共 {len(tools_list)} 个工具）")

    # ------------------------------------------------------------------
    # M2: Adaptive Planner（Orchestrator 依赖 Planner，先初始化）
    # ------------------------------------------------------------------
    from src.planner.planner import Planner
    from src.planner.budget_tracker import BudgetTracker

    planner_policy = modules.get("planner_policy", default_policy)
    planner_cfg = config.get("planner", {})
    budget_tracker = BudgetTracker()
    planner = Planner(
        policy=planner_policy,
        budget_tracker=budget_tracker,
        tools=tools_list,
        search_calls=planner_cfg.get("search_calls", 1),
        search_tools=planner_cfg.get("search_tools"),
        max_sub_tasks=config.get("orchestrator", {}).get("max_sub_questions", 10),
    )
    modules["planner"] = planner
    logger.info(
        f"[M2] Planner 模块已初始化（规划前侦察搜索次数={planner.search_calls}）"
    )

    # M3: Context Compressor
    from src.compressor.compressor import ContextCompressor

    compressor_policy = modules.get("compressor_policy", default_policy)
    compressor_cfg = config.get("compressor", {})
    compressor = ContextCompressor(
        llm_policy=compressor_policy,
        budget=compressor_cfg.get("max_context_length", 16000),
        output_reserve=compressor_cfg.get("output_reserve_tokens", 2048),
    )
    if not compressor_cfg.get("enable_multilevel", True):
        compressor = None
    modules["compressor"] = compressor
    logger.info("[M3] Compressor 模块已初始化")

    # M4: Shared Memory Store
    from src.memory.memory_store import SharedMemoryStore

    memory_cfg = config.get("memory", {})
    memory_store = None
    if memory_cfg.get("enabled", True):
        memory_store = SharedMemoryStore(
            db_path=memory_cfg.get("db_path", "data/memory.db"),
            session_id=session_id,
            max_entries=memory_cfg.get("max_entries", 10000),
            similarity_threshold_dup=memory_cfg.get("similarity_threshold_dup", 0.92),
            similarity_threshold_conflict=memory_cfg.get("similarity_threshold_conflict", 0.65),
            evict_interval=memory_cfg.get("evict_interval", 100),
            enable_vector_index=memory_cfg.get("enable_vector_index", True),
            ttl_seconds=memory_cfg.get("ttl_seconds"),
            max_history_entries=memory_cfg.get("max_history_entries", 10000),
        )
    modules["memory_store"] = memory_store
    modules["session_id"] = memory_store.session_id if memory_store is not None else (session_id or uuid.uuid4().hex)
    logger.info("[M4] Memory Store enabled=%s", memory_store is not None)

    # M5: Red-Blue Adversarial Loop（先创建，再注入 Orchestrator）
    from src.adversarial.loop import AdversarialLoop
    from src.adversarial.red_agent import RedAgent
    from src.adversarial.blue_agent import BlueAgent

    red_policy = modules.get("red_agent_policy", default_policy)
    blue_policy = modules.get("blue_agent_policy", default_policy)
    adversarial_cfg = config.get("adversarial", {})

    red_agent = RedAgent(policy=red_policy)
    blue_agent = BlueAgent(policy=blue_policy, tools=tools_list,
        max_issues_per_round=adversarial_cfg.get("max_issues_per_round"))
    adversarial_loop = AdversarialLoop(
        red_agent=red_agent,
        blue_agent=blue_agent,
        policy=modules.get("judge_policy", default_policy),
        max_rounds=adversarial_cfg.get("max_rounds", 3),
        score_threshold=adversarial_cfg.get("score_threshold", 8.0),
        delta_threshold=adversarial_cfg.get("delta_threshold", 0.3),
    )
    modules["adversarial"] = adversarial_loop
    logger.info("[M5] Adversarial 模块已初始化")

    # M1: Multi-Agent Orchestrator
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.agent_pool import AgentPool

    orchestrator_cfg = config.get("orchestrator", {})
    researcher_max_turns = orchestrator_cfg.get("researcher_max_turns", 15)

    agent_pool = AgentPool(
        policy_factory=lambda: modules.get("solver_policy", default_policy).clone(),
        summarizer_policy_factory=lambda: modules.get("summarizer_policy", default_policy).clone(),
        tools_factory=lambda: _create_tools_factory(config),
        max_idle=3,
        max_turns=researcher_max_turns,
        report_min_chars=orchestrator_cfg.get("report_min_chars", 5000),
        recovery_config=config.get("recovery"),
    )
    modules["agent_pool"] = agent_pool

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        budget_tracker=budget_tracker,
        compressor=compressor,
        adversarial_loop=adversarial_loop,
        memory_store=memory_store,
        summarizer_policy=modules.get("summarizer_policy", default_policy),
        force_adversarial=config.get("adversarial", {}).get("force", False),
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] Orchestrator 模块已初始化")

    # M6: Self-Evolution Engine（预留，默认禁用）
    if config.get("evolution", {}).get("enabled", False):
        logger.info("[M6] Evolution 模块已启用（预留接口）")
    else:
        logger.info("[M6] Evolution 模块已禁用")

    return modules


# ---------------------------------------------------------------------------
# 研究流程主函数
# ---------------------------------------------------------------------------
async def run_research(query: str, config: dict, modules: dict[str, Any],
                       *, metadata: dict | None = None) -> str:
    """Compatible text API; structured telemetry is retained in modules['last_run']."""
    from src.utils.tracing import research_trace, config_fingerprint, record_event
    attrs = {"config_hash": config_fingerprint(config),
             "backend": config.get("model", {}).get("backend", "deepseek"),
             "tool_mode": "snapshot" if config.get("tools", {}).get("snapshot_path") else (
                 "mock" if config.get("tools", {}).get("web_search", {}).get("mock_mode", True) else "live"),
             "adversarial_enabled": config.get("adversarial", {}).get("enabled", True),
             **(metadata or {})}
    modules.pop("last_report", None)
    recorder = None
    try:
        with research_trace(attrs) as recorder:
            text = await _run_research(query, config, modules)
            report = modules.get("last_report")
            recorder.status = getattr(report, "status", "success")
            record_event("research_result", {"status": recorder.status,
                         "adversarial_rounds": getattr(report, "adversarial_rounds", 0),
                         "num_replan": getattr(report, "num_replan", 0),
                         "recovery": getattr(report, "recovery", {})})
            return text
    finally:
        if recorder:
            modules["last_run"] = recorder.snapshot()


async def _run_research(query: str, config: dict, modules: dict[str, Any]) -> str:
    """
    执行完整的研究流程。

    流程：
        1. Orchestrator 调用 Planner 拆解问题为子任务 DAG
        2. Orchestrator 调度 AgentPool 中的子 Agent 并行/串行执行
        3. 子 Agent 调用 Tools 检索信息并生成子报告
        4. Compressor 管理长上下文
        5. Memory 存储中间结果
        6. Adversarial Loop 对报告进行多轮对抗优化（若启用）
        7. 输出最终研究报告

    Args:
        query: 用户输入的研究问题。
        config: 全局配置字典。
        modules: 已初始化的模块实例字典。

    Returns:
        最终研究报告文本（Markdown 格式）。
    """
    import asyncio

    logger = logging.getLogger("runner")
    logger.info(f"开始研究，查询: {query[:80]}...")

    start_time = time.time()

    # Step 1-3: Orchestrator 内部完成规划、调度、收集、合成
    orchestrator = modules["orchestrator"]
    from src.orchestrator.schemas import RunConfig

    run_cfg = RunConfig(
        max_concurrent=config.get("orchestrator", {}).get("max_concurrent", 5),
        global_timeout_seconds=config.get("orchestrator", {}).get("global_timeout_seconds", 600),
        max_replan_rounds=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
        enable_adversarial=config.get("adversarial", {}).get("enabled", True),
        enable_evolution=config.get("evolution", {}).get("enabled", False),
    )

    try:
        report = await orchestrator.run(query, config=run_cfg)
        modules["last_report"] = report
    finally:
        # These are this module bundle's tools, never another run's global session.
        from ..utils.recovery import close_tools
        await close_tools(modules["tools"])
    logger.info(
        f"[Orchestrator] 报告生成完成 | 置信度={report.confidence:.2f} | "
        f"搜索轮数={report.num_searches} | 重规划={report.num_replan} | 对抗轮数={report.adversarial_rounds}"
    )

    # Step 4/5: 进化优化（如启用且已训练）
    if run_cfg.enable_evolution:
        logger.info("[Evolution] 进化优化已启用（预留接口）")
    else:
        logger.info("[Evolution] 进化优化已跳过")


    elapsed = time.time() - start_time
    logger.info(f"研究完成，耗时: {elapsed:.2f} 秒")

    # 组装最终输出
    final_report = _format_report(report, elapsed)
    return final_report


def _format_report(report, elapsed: float) -> str:
    """将 ResearchReport 格式化为 Markdown 文本。"""
    content = report.content or ""

    # 统一置信度：如果正文中有 LLM 自评的"整体置信度"，替换为实际计算值，避免不一致
    content = re.sub(
        r"(整体置信度|Overall Confidence|置信度)[:：]\s*0?\.\d+",
        f"\\1: {report.confidence:.2f}",
        content,
        flags=re.I,
    )

    lines = [
        f"# 研究报告：{report.query}",
        "",
        "---",
        "",
        content,
        "",
        "---",
        "",
        "## 元信息",
        "",
        f"- **置信度**: {report.confidence:.2f}",
        f"- **搜索轮数**: {report.num_searches}",
        f"- **重规划次数**: {report.num_replan}",
        f"- **对抗轮数**: {report.adversarial_rounds}",
        f"- **总耗时**: {elapsed:.2f} 秒",
        "",
    ]

    if report.sources:
        lines.append("## 参考来源")
        lines.append("")
        for i, src in enumerate(report.sources, 1):
            title = src.get("title", "未知标题")
            url = src.get("url", "")
            snippet = src.get("snippet", "")
            lines.append(f"{i}. [{title}]({url}) — {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告保存
# ---------------------------------------------------------------------------
def save_report(report: str, query: str, output_dir: str = "outputs/reports") -> str:
    """
    将研究报告保存到文件。

    文件名格式：report_YYYYMMDD_HHMMSS_<query前20字>.md
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"report_{timestamp}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath
