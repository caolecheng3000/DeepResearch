"""
Deep Research Agent — 核心编排器 (M1: Multi-Agent Orchestrator)

9 状态状态机驱动的异步任务编排引擎：
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
  失败时进入 REPLANNING，最终可进入 FAILED。

设计亮点:
  - 自研 asyncio + DAG executor，不依赖 LangGraph/AutoGen
  - 拓扑排序后按层并发执行，Semaphore 控制最大并发度
  - 三级降级策略：单任务超时→标记继续；>50%失败→re-plan；全局超时→强制合成
  - 状态机用字典映射实现，便于扩展新状态和转换逻辑
"""
from __future__ import annotations

import asyncio
import copy
import time
import uuid
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..planner.planner import Planner, PlanParseError
from ..planner.budget_tracker import BudgetTracker
from ..utils.tracing import trace_chain, trace_block, current_run_id, record_event
from ..utils.execution import deadline_scope, bounded, run_sync, check_deadline
from ..utils.recovery import recovery_scope, current_recovery, ResearchAbort, Failure, public_message

# M4: Memory Store 类型提示（延迟导入避免循环依赖）
SharedMemoryStore = Any


__all__ = ["Orchestrator"]


class Orchestrator:
    """Deep Research Agent 核心编排器。

    Attributes:
        planner: 自适应规划器，负责初始规划和增量重规划。
        agent_pool: Agent 对象池，管理 worker agent 生命周期。
        budget_tracker: Token 预算追踪器。
        memory_store: 全局共享内存，存储所有子任务结果和中间上下文。
        compressor: （预留）上下文压缩器接口。
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        budget_tracker: BudgetTracker | None = None,
        compressor: Any | None = None,
        adversarial_loop: Any | None = None,
        memory_store: Any | None = None,
        summarizer_policy: Any | None = None,
        force_adversarial: bool = False,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy
        self.force_adversarial = force_adversarial
        self.pre_adversarial_report = None
        if summarizer_policy is not None:
            self.agent_pool.summarizer_policy_factory = (
                summarizer_policy.clone if hasattr(summarizer_policy, "clone")
                else lambda: copy.deepcopy(summarizer_policy)
            )

        # 运行时状态（保留 dict 作为快速缓存，M4 提供持久化 + 语义检索）
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        self._replan_count: int = 0
        self._adversarial_count: int = 0
        self._running = False
        self._run_id = ""
        self._reusable_results = {}

        # 状态机处理器映射
        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.SYNTHESIZING: self._do_synthesizing,
            OrchestratorState.ADVERSARIAL: self._do_adversarial,
            OrchestratorState.REPLANNING: self._do_replanning,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @trace_chain(name="orchestrator.run", tags=["m1", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        """主入口：执行完整的研究流程。

        Args:
            query: 研究问题。
            config: 运行配置，默认使用 RunConfig()。

        Returns:
            ResearchReport: 最终研究报告。
        """
        if self._running:
            raise RuntimeError("An Orchestrator cannot run two queries concurrently; create a separate instance")
        self._running = True
        self._run_id = current_run_id() or uuid.uuid4().hex
        run_config = config or RunConfig()
        self._query = query
        self._config = run_config
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._memory_store.clear()
        self.pre_adversarial_report = None
        self._results.clear()
        self._dag = None
        self._task_map.clear()
        self._reusable_results.clear()
        self._current_state = OrchestratorState.IDLE

        try:
            with recovery_scope(getattr(self.agent_pool, "recovery_config", None)) as state:
                with deadline_scope(seconds=run_config.global_timeout_seconds):
                    try:
                        report = await bounded(self._run_once(query, run_config))
                    except asyncio.TimeoutError:
                        self._current_state = OrchestratorState.DONE
                        report = self._deadline_report()
                    except ResearchAbort as exc:
                        self._current_state = OrchestratorState.FAILED
                        report = self._abort_report(exc)
                    except Exception as exc:
                        self._current_state = OrchestratorState.FAILED
                        report = self._abort_report(ResearchAbort(Failure(5,
                            "UNHANDLED_RESEARCH_ERROR", public_message(exc), "research")))
                report.recovery.update({"tool_calls": state.tool_calls, "model_calls": state.model_calls,
                    "recovery_actions": state.recovery_actions, "estimated_reserved_tokens": state.estimated_tokens})
                return report
        except asyncio.CancelledError:
            self._current_state = OrchestratorState.FAILED
            record_event("research_termination", {"level": 5, "code": "CANCELLED"})
            raise
        finally:
            self._running = False

    def _abort_report(self, exc):
        """L5: no model calls; preserve already completed evidence and the explicit cause."""
        report = self._memory_store.get("final_report")
        had_report = bool(report and report.content)
        saved = {r.task_id: r for r in self._results if r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output}
        for item in self._memory_store.values():
            if isinstance(item, AgentResult) and item.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and item.output:
                saved[item.task_id] = item
        if report is None:
            content = "研究已停止，以下内容尚未完成最终核验。\n原因：" + public_message(exc)
            content += "".join(f"\n\n## {key}\n{item.output}" for key, item in saved.items())
            report = ResearchReport(query=self._query, content=content)
        if had_report:
            report.content += "\n\n## 研究终止说明\n" + public_message(exc)
        report.status = "partial" if had_report or saved else "failed"
        report.num_replan = self._replan_count
        report.recovery = {"level": 5, "code": exc.failure.code, "message": exc.failure.message}
        self._memory_store["final_report"] = report
        record_event("research_termination", report.recovery)
        return report

    def _deadline_report(self):
        """No new external work after deadline: preserve an existing report or completed evidence."""
        report = self._memory_store.get("final_report")
        if report is None:
            successes = {r.task_id: r for r in self._results if r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL)}
            for value in self._memory_store.values():
                if isinstance(value, AgentResult) and value.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL):
                    successes[value.task_id] = value
            content = "研究已达到总时间上限，以下是已完成任务的结果；尚未完成最终核验。"
            for task_id, result in successes.items():
                content += f"\n\n## {task_id}\n{result.output}"
            report = ResearchReport(query=self._query, content=content, confidence=0.0)
        report.num_replan = self._replan_count
        report.adversarial_rounds = self._adversarial_count
        report.status = "timeout"
        report.recovery.update({"level": 5, "code": "GLOBAL_TIMEOUT"})
        record_event("global_timeout", {"partial_report": bool(report.content)})
        self._memory_store["final_report"] = report
        return report

    async def _run_once(self, query, config):
        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            with trace_block(f"stage.{self._current_state.value}",
                             metadata={"replan_round": self._replan_count}):
                next_state = await bounded(handler())
            self._current_state = next_state

            print(f"[Orchestrator] State transition: {self._current_state.value}")

        # 返回结果
        if self._current_state == OrchestratorState.DONE:
            # 最终报告应在 memory 中
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count
            if report.status == "success" and any(r.status != AgentStatus.SUCCESS for r in self._results):
                report.status = "partial"

            # M4: 将最终报告存入 SharedMemoryStore
            if self.memory_store is not None and report.status == "success":
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"{self._run_id}:final_report",
                        claim=self._smart_truncate(str(report.content), max_chars=1200),
                        source="orchestrator",
                        confidence=report.confidence,
                        agent_id="orchestrator",
                        timestamp=time.time(),
                        evidence_type="primary",
                        embedding=[],
                        topic=query[:50],
                        metadata={
                            "num_searches": report.num_searches,
                            "num_replan": report.num_replan,
                            "adversarial_rounds": report.adversarial_rounds,
                        },
                    )
                    await run_sync(self.memory_store.put, entry)
                    print(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")

                    # M3+L3: M4 容量 >85% 时用 LLM 摘要压缩旧记忆
                    await self._maybe_compress_old_memories()
                except Exception as e:
                    print(f"[M4] Failed to store final report: {e}")

            return report

        # FAILED 状态
        return self._abort_report(ResearchAbort(Failure(5, "RESEARCH_FAILED",
            self._memory_store.get("failure_reason", "规划失败或没有足够的可用结果，研究无法继续"), "research")))

    # ------------------------------------------------------------------
    # 状态机处理器
    # ------------------------------------------------------------------

    async def _on_idle(self) -> OrchestratorState:
        """从 IDLE 自动进入 PLANNING。"""
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """调用 Planner 生成初始 DAG。

        失败时直接转入 FAILED（初始计划失败无法恢复）。
        """
        try:
            memory_ctx = await self._build_memory_context()
            # generate_plan 是协程：内部含「规划前侦察」的固定次数搜索循环
            self._dag = await self.planner.generate_plan(self._query, memory_ctx)
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
        except PlanParseError as e:
            self._memory_store["failure_reason"] = public_message(e)
            print(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            self._memory_store["failure_reason"] = public_message(e)
            print(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        print(f"[Planning] ✓ DAG 生成完成: {n_tasks} 个子任务, {n_layers} 个执行层")
        # 打印子任务描述以便诊断
        for tid, task in self._task_map.items():
            print(f"[Planning]   {tid}: {task.description}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """拓扑排序 + 并发调度 sub-agents。

        核心逻辑:
          1. 获取并行执行层 (parallel groups)
          2. 每层内用 asyncio.gather + Semaphore 并发执行
          3. 每个 sub-task 设置单独超时 (asyncio.wait_for)
          4. 收集结果到 self._results
        """
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results = {r.task_id: r for r in self._results
                       if r.status == AgentStatus.SUCCESS and r.task_id not in self._task_map}

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")

            # 构建本层的 coroutine 列表
            @trace_chain(name="worker.task")
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    if task_id in self._reusable_results:
                        return self._reusable_results[task_id]

                    record_event("task_started", {"task_id": task_id, "task_type": subtask.task_type.value,
                                 "dependencies": subtask.dependencies, "replan_round": self._replan_count})
                    unavailable = [dep for dep in subtask.dependencies
                                   if not isinstance(self._memory_store.get(f"result:{dep}"), AgentResult)
                                   or self._memory_store[f"result:{dep}"].status != AgentStatus.SUCCESS]
                    if unavailable:
                        return AgentResult(task_id=task_id, status=AgentStatus.SKIPPED,
                                           output=f"Skipped: unsuccessful or missing dependencies: {', '.join(unavailable)}")

                    # 准备上下文：先执行依赖任务的结果
                    context = self._build_task_context(subtask)

                    # 获取 Agent
                    agent = None
                    reusable = False
                    try:
                        agent = await self.agent_pool.get_agent(subtask.task_type)
                        with deadline_scope(seconds=subtask.timeout_seconds):
                            result = await bounded(agent.run(subtask, context))
                        reusable = result.status == AgentStatus.SUCCESS
                    except asyncio.TimeoutError:
                        result = getattr(agent, "partial_result", None) or AgentResult(
                            task_id=task_id,
                            status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {subtask.timeout_seconds}s",
                        )
                        result.status = AgentStatus.PARTIAL if getattr(agent, "partial_result", None) else AgentStatus.TIMEOUT
                        result.recovery.update({"level": 4, "needs_replan": True, "code": "TASK_TIMEOUT"})
                    except (ResearchAbort, asyncio.CancelledError):
                        partial = getattr(agent, "partial_result", None)
                        if partial:
                            self._memory_store[f"result:{task_id}"] = partial
                        raise
                    except Exception as e:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        await self.agent_pool.release_agent(agent, reusable=reusable)

                    # Publish each completion immediately, even if siblings are later cancelled.
                    self._memory_store[f"result:{task_id}"] = result
                    return result

            # 并发执行本层
            coros = [_run_one(tid) for tid in group]
            running = [asyncio.create_task(coro) for coro in coros]
            try:
                layer_results = await asyncio.gather(*running)
            except BaseException:
                # L5 and cancellation stop siblings immediately and drain their finally blocks.
                for pending in running:
                    if not pending.done():
                        pending.cancel()
                await asyncio.gather(*running, return_exceptions=True)
                raise

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # 将异常包装为 FAILED 结果
                    all_results["unknown"] = AgentResult(
                        task_id="unknown", status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}")
                else:
                    all_results[lr.task_id] = lr
                    # 每层结束后立即写入 _memory_store，使下层可见依赖结果
                    if isinstance(lr, AgentResult):
                        self._memory_store[f"result:{lr.task_id}"] = lr

        self._results = list(all_results.values())
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集结果，写入 memory，检查是否需要重规划。

        L4 的唯一 Replan 入口：任何显式验收失败都进入既有重规划流程，
        同时受 Replan 轮数与请求共享恢复预算限制。L5 在 run() 收敛。
        """
        # 将结果写入运行时 memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # M4: 将成功结果同步写入 SharedMemoryStore（持久化 + 向量索引）
        if self.memory_store is not None:
            for r in self._results:
                if r.status == AgentStatus.SUCCESS and r.output:
                    await run_sync(self._sync_result_to_memory_store, r)

        # M3: 对成功结果做提取式压缩（L2 TextRank），提升 Summarizer 输入质量
        if self.compressor is not None:
            success_indices = [i for i, r in enumerate(self._results) if r.status == AgentStatus.SUCCESS and r.output]
            if success_indices:
                success_outputs = [str(self._results[i].output) for i in success_indices]
                try:
                    compressed = await run_sync(self.compressor.compress_documents,
                        success_outputs, self._query, target_ratio=0.50
                    )
                    for j, idx in enumerate(success_indices):
                        if j < len(compressed):
                            self._memory_store[f"compressed:{self._results[idx].task_id}"] = compressed[j]
                    total_before = sum(len(o) for o in success_outputs)
                    total_after = sum(len(c) for c in compressed)
                    print(f"[M3] Results compressed: {total_before} → {total_after} chars "
                          f"({len(success_indices)} tasks, ratio={total_after/max(total_before,1):.2f})")
                except Exception as e:
                    print(f"[M3] Result compression failed: {e}")

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        status_icon = "✓" if success_count == total_count else "⚠"
        print(f"[Collect] {status_icon} 子任务完成: {success_count}/{total_count} 成功", end="")
        if fail_count > 0:
            print(f" ({fail_count} 失败)")
        else:
            print()

        # 检查是否需要重规划
        if self._should_replan(self._results):
            state = current_recovery()
            if (self._replan_count < self._config.max_replan_rounds
                    and (state is None or state.action())):
                self._replan_count += 1
                record_event("task_recovery", {"level": 4, "action": "replan", "round": self._replan_count})
                return OrchestratorState.REPLANNING
            else:
                print("[Collect] Max replan rounds reached, proceeding with partial results")
                # 超过最大重规划次数，继续合成（用已有结果）

        if not any(r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output for r in self._results):
            raise ResearchAbort(Failure(5, "NO_USABLE_EVIDENCE",
                "任务恢复与重规划后仍无可用结果，研究停止；未调用模型补写报告", "research"))

        return OrchestratorState.SYNTHESIZING

    def _sync_result_to_memory_store(self, result: AgentResult) -> None:
        """将 AgentResult 同步到 M4 SharedMemoryStore。

        提取 output 中的核心内容作为记忆条目，支持后续语义检索。
        改进点：
        - 按句子边界智能截断，避免在词语中间断裂
        - 从 trajectory 提取真实 URL 作为 source
        - topic 按子任务描述细分（而非所有 entry 共用 query 前 50 字）
        """
        try:
            # 延迟导入避免循环依赖
            from src.memory.long_term import MemoryEntry

            # 1. 智能截断 claim：在最后完整句子边界处切断
            claim_text = self._smart_truncate(str(result.output), max_chars=1200)

            # 2. 从 trajectory 提取真实来源 URL
            source = self._extract_source_from_trajectory(result)

            # 3. topic 细化为子任务描述（保留 query 前 50 字作为 fallback）
            subtask = self._task_map.get(result.task_id) if hasattr(self, '_task_map') else None
            if subtask and subtask.description:
                topic = subtask.description[:80]
            else:
                topic = self._query[:50]

            entry = MemoryEntry(
                entry_id=f"{self._run_id}:{self._replan_count}:{result.task_id}",
                claim=claim_text,
                source=source,
                confidence=getattr(result, "confidence", 0.5),
                agent_id=result.task_id,
                timestamp=time.time(),
                evidence_type="primary",
                embedding=[],  # SharedMemoryStore.put() 会自动生成 embedding
                topic=topic,
                metadata={
                    "status": result.status.value,
                    "token_usage": getattr(result, "token_usage", 0),
                    "task_description": subtask.description if subtask else "",
                },
            )
            self.memory_store.put(entry)
            print(f"[M4] Memory stored: {result.task_id} (claim={claim_text[:60]}...)")
        except Exception as e:
            print(f"[M4] Failed to store memory for {result.task_id}: {e}")

    @staticmethod
    def _smart_truncate(text: str, max_chars: int = 500) -> str:
        """在句子边界处智能截断，避免在词语中间断裂。

        策略：找到 [max_chars] 范围内最后一个完整句子边界，在该处截断。
        若无合适边界，回退到 max_chars 硬截断。
        """
        if len(text) <= max_chars:
            return text

        # 在 max_chars 范围内找最后一个句子边界（含多字符分隔符）
        window = text[:max_chars]
        # (分隔符, 分隔符长度) 按优先级排列
        seps = [('\n\n', 2), ('. ', 2), ('.\n', 2), ('! ', 2), ('?\n', 2),
                ('。', 1), ('！', 1), ('？', 1), ('\n', 1)]
        best_pos = -1
        best_sep_len = 1
        for sep, sep_len in seps:
            pos = window.rfind(sep)
            if pos > best_pos:
                best_pos = pos
                best_sep_len = sep_len

        # 若找到合适边界且不会截太短（保留至少 30%）
        if best_pos > max_chars * 0.3:
            return text[:best_pos + best_sep_len]

        # 回退：硬截断
        return text[:max_chars]

    @staticmethod
    def _extract_source_from_trajectory(result: AgentResult) -> str:
        """从 AgentResult 的 trajectory 中提取真实来源 URL/标题。

        优先级：tool 结果中的 url/paper title > trajectory 中的 url > task_id
        """
        urls: list[str] = []
        titles: list[str] = []

        for step in getattr(result, "trajectory", []) or []:
            if not isinstance(step, dict):
                continue

            # 从工具结果中提取 URL 和标题
            tool_result = step.get("result")
            if isinstance(tool_result, dict):
                # arxiv_reader 结果
                for paper in tool_result.get("papers", []):
                    if isinstance(paper, dict):
                        if paper.get("pdf_url"):
                            urls.append(paper["pdf_url"])
                        if paper.get("title"):
                            titles.append(paper["title"])

                # web_search / browser 结果
                for item in tool_result.get("results", []):
                    if isinstance(item, dict):
                        if item.get("url"):
                            urls.append(item["url"])
                        if item.get("title"):
                            titles.append(item["title"])

            # 从 tool_call 中提取
            tc = step.get("tool_calls", [])
            if isinstance(tc, list):
                for t in tc:
                    if isinstance(t, dict):
                        func = t.get("function", {})
                        args_str = func.get("arguments", "{}")
                        try:
                            import json
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            if isinstance(args, dict) and args.get("url"):
                                urls.append(args["url"])
                        except (json.JSONDecodeError, TypeError):
                            pass

        # 优先返回 URL，其次论文标题，最次 task_id
        if urls:
            return urls[0]
        if titles:
            return titles[0][:200]
        return f"task:{result.task_id}"

    async def _maybe_compress_old_memories(self) -> None:
        """M3 L3: M4 容量 >85% 时，用 LLM 摘要压缩最旧的记忆。

        策略：
        - 保留最近 50% 的记忆不变
        - 将较旧的 50% 记忆用 LLMSummarizer 聚合为一份摘要
        - 删除旧记忆，插入摘要条目
        - 避免 M4 无限膨胀导致语义检索质量下降
        """
        if self.memory_store is None or self.compressor is None:
            return

        max_entries = self.memory_store.max_entries
        current_count = len(self.memory_store)
        if current_count < max_entries * 0.85:
            return

        try:
            # 获取所有记忆，按时间排序
            all_entries = await run_sync(self.memory_store.list_entries)
            all_entries = [e for e in all_entries
                           if e.metadata.get("_memory", {}).get("status") != "disputed"
                           and not e.metadata.get("pinned", False)]
            if len(all_entries) < 20:  # 太少不值得压缩
                return
            all_entries.sort(key=lambda e: e.timestamp)

            # 保留最近 50%，压缩较旧 50%
            keep_count = max(len(all_entries) // 2, 10)
            old_entries = all_entries[: len(all_entries) - keep_count]

            if len(old_entries) < 5:  # 旧记忆太少，不值得压缩
                return

            # 收集旧记忆的 claim 文本
            old_texts = [e.claim for e in old_entries if e.claim]

            print(
                f"[M3] Compressing {len(old_texts)} old memories "
                f"(M4 usage {current_count}/{max_entries}, {current_count/max_entries:.0%})"
            )

            # L3: 用 LLM 将旧记忆聚合为一份摘要
            aggregate = await self.compressor.summarizer.asummarize_documents(
                old_texts,
                query=self._query,
                max_length=max(800, min(3000, len(old_texts) * 30)),
            )

            if not aggregate or len(aggregate) < 100:
                print("[M3] L3 compression produced empty result, skipping")
                return

            # 插入压缩摘要作为新记忆
            from src.memory.long_term import MemoryEntry
            import time as _time
            compressed_entry = MemoryEntry(
                entry_id=f"compressed:{uuid.uuid4().hex}",
                claim=aggregate[:1200],
                source="m3_compressor",
                confidence=0.6,
                agent_id="orchestrator",
                timestamp=_time.time(),
                evidence_type="secondary",
                embedding=[],
                topic=f"compressed_{self._query[:40]}",
                metadata={
                    "compressed_from": len(old_entries),
                    "original_timestamps": [e.timestamp for e in old_entries[:5]],
                },
            )
            committed = await run_sync(self.memory_store.replace_entries, old_entries, compressed_entry)
            if not committed:
                return  # Sources changed or summary rejected; original memories remain intact.

            print(
                f"[M3] Compressed {len(old_entries)} old memories into 1 summary entry "
                f"({len(aggregate)} chars). M4: {current_count} → {len(self.memory_store)} entries"
            )
        except Exception as e:
            print(f"[M3] Memory compression failed: {e}")

    async def _do_synthesizing(self) -> OrchestratorState:
        """调用 SummarizerAgent 合成研究报告。"""
        # 用 M3 压缩后的结果替换原始输出，提升合成质量
        # 保存原始输出以便后续恢复（如 adversarial 失败回退时需要完整内容）
        _original_outputs: dict[str, str] = {}
        for r in self._results:
            comp_key = f"compressed:{r.task_id}"
            if comp_key in self._memory_store and r.status == AgentStatus.SUCCESS:
                _original_outputs[r.task_id] = r.output
                r.output = self._memory_store[comp_key]

        # 创建合成任务
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        context = {
            "query": self._query,
            "results": self._results,
        }

        agent = await self.agent_pool.get_agent("synthesize")
        reusable = False
        try:
            with deadline_scope(seconds=synth_task.timeout_seconds):
                result = await bounded(agent.run(synth_task, context))
            reusable = result.status == AgentStatus.SUCCESS
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.TIMEOUT,
                output="Synthesis timed out",
            )
        except Exception as e:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.FAILED,
                output=f"Synthesis error: {type(e).__name__}: {e}",
            )
        finally:
            # 恢复原始输出（adversarial 和 M4 后续流程需要完整内容）
            for task_id, original in _original_outputs.items():
                for r in self._results:
                    if r.task_id == task_id:
                        r.output = original
                        break
            await self.agent_pool.release_agent(agent, reusable=reusable)

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            self._memory_store["final_report"] = result.output
        else:
            # 合成失败但已有结果，生成降级报告
            self._memory_store["final_report"] = ResearchReport(
                query=self._query,
                content=str(result.output) if result.output else "Synthesis failed.",
                status="timeout" if result.status == AgentStatus.TIMEOUT else "failed",
                confidence=0.0,
                num_searches=sum(
                    len([t for t in r.trajectory if t.get("role") == "tool"])
                    for r in self._results
                ),
            )

        if self._config.enable_adversarial:
            print("[Synthesize] ✓ 报告合成完成，进入对抗优化")
            return OrchestratorState.ADVERSARIAL
        print("[Synthesize] ✓ 报告合成完成")
        return OrchestratorState.DONE

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue 对抗降噪循环。

        调用 AdversarialLoop 对报告进行 challenge-verify 迭代优化。
        仅在报告置信度低于阈值时触发，避免资源浪费。
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE

        # 置信度足够高时跳过对抗
        self.pre_adversarial_report = copy.deepcopy(report)
        if report.confidence >= 0.8 and not self.force_adversarial:
            print("[Adversarial] ✓ 报告置信度已达标 (≥0.8)，跳过对抗优化")
            return OrchestratorState.DONE

        if self.adversarial_loop is None:
            print("[Adversarial] AdversarialLoop 未配置，跳过")
            return OrchestratorState.DONE

        # 保存对抗前的原始报告，用于对抗失败时回退
        pre_adversarial_report = copy.deepcopy(report)

        try:
            print(f"[Adversarial] ▶ 启动 Red-Blue 对抗优化 (当前置信度={report.confidence:.2f})")
            optimized_report, history = await self.adversarial_loop.run(report)
            self._adversarial_count += len(history)

            # 检查对抗后的报告是否比原始报告更差（内容为空/道歉）
            optimized_content = optimized_report.content or ""
            original_content = pre_adversarial_report.content or ""
            if self._is_report_degraded(optimized_content, original_content):
                print("[Adversarial] ⚠ 对抗后报告出现退化（API 错误/空内容），回退到对抗前版本")
                self._memory_store["final_report"] = pre_adversarial_report
                pre_adversarial_report.adversarial_rounds = 0
                return OrchestratorState.DONE

            self._memory_store["final_report"] = optimized_report
            print(f"[Adversarial] ✓ 对抗优化完成: {len(history)} 轮, 最终置信度={optimized_report.confidence:.2f}")
        except Exception as e:
            print(f"[Adversarial] ✗ 对抗优化失败: {e}，使用原始报告")
            pre_adversarial_report.status = "partial"
            self._memory_store["final_report"] = pre_adversarial_report

        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """触发增量重规划。

        复用未变更且通过验收的结果，变更任务及其后代重新执行。
        """
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        reason = self._build_failure_reason(self._results)
        print(f"[Replan] Round {self._replan_count}/{self._config.max_replan_rounds}. Failed tasks: {[t.task_id for t in failed_tasks]}")

        try:
            previous_tasks = dict(self._task_map)
            previous_results = {r.task_id: r for r in self._results
                if r.status == AgentStatus.SUCCESS and r.acceptance.get("passed", True)}
            new_dag = await self.planner.areplan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=self._results,
                reason=reason,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            # Reuse only unchanged accepted tasks. Invalidate all descendants of a changed task.
            changed = {key for key, task in self._task_map.items()
                       if key not in previous_results or task != previous_tasks.get(key)}
            while True:
                descendants = {key for key, task in self._task_map.items()
                               if set(task.dependencies).intersection(changed)}
                expanded = changed | descendants
                if expanded == changed:
                    break
                changed = expanded
            self._reusable_results = {key: result for key, result in previous_results.items()
                                      if key in self._task_map and key not in changed}
            obsolete = {key for key in previous_tasks if key not in self._task_map and key not in previous_results}
            for key in changed | obsolete:
                self._memory_store.pop(f"result:{key}", None)
                self._memory_store.pop(f"compressed:{key}", None)
            self._results = [r for r in self._results if r.status == AgentStatus.SUCCESS
                             and (r.task_id not in self._task_map or r.task_id in self._reusable_results)]
        except PlanParseError as e:
            print(f"[Replan] Failed: {e}")
            # 重规划失败，如果已有部分成功结果，尝试直接合成
            if any(r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Replan] Unexpected error: {e}")
            if any(r.status in (AgentStatus.SUCCESS, AgentStatus.PARTIAL) and r.output for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED

        return OrchestratorState.DISPATCHING

    async def _on_done(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        """终态，不应再转换。"""
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # 决策逻辑
    # ------------------------------------------------------------------

    def _should_replan(self, results: list[AgentResult]) -> bool:
        """Any rejected, failed, partial or skipped task requests the existing L4 Replan."""
        return any(r.status != AgentStatus.SUCCESS or r.recovery.get("needs_replan")
                   or r.acceptance.get("passed") is False for r in results)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """检查是否超过全局超时。"""
        elapsed = time.monotonic() - self._start_time
        return elapsed >= self._config.global_timeout_seconds

    async def _build_memory_context(self) -> str:
        """构建给 planner 的上下文摘要。

        优先使用 M4 SharedMemoryStore 的语义检索（如果已接入），
        否则回退到运行时 dict 遍历。
        """
        # M4: 语义检索相关记忆
        if self.memory_store is not None:
            try:
                ctx = await run_sync(self.memory_store.get_context_for_query,
                    self._query, max_tokens=3000  # 放宽限制，给 Compressor 更多素材
                )
                if ctx:
                    print(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    # M3串联: 对 M4 的粗粒度结果做细粒度语义压缩
                    if self.compressor is not None:
                        try:
                            compressed = await self.compressor.acompress(
                                texts=[ctx],
                                query=self._query,
                                level=2,  # L1+L2: 过滤无关记忆 + 提取关键句
                                system_prompt_tokens=500,  # Planner system prompt 估算
                            )
                            if compressed:
                                ctx = compressed[0]
                                print(f"[M3] Memory context compressed: {len(ctx)} chars")
                        except Exception as e:
                            print(f"[M3] Memory compression failed: {e}, using raw M4 context")
                    return ctx
            except Exception as e:
                print(f"[M4] Semantic memory query failed: {e}, falling back to dict")

        # 回退：运行时 dict 遍历
        parts = []
        for key, value in self._memory_store.items():
            if key.startswith("result:"):
                continue
            parts.append(f"{key}: {str(value)[:200]}")

        # M3: 如果上下文过长，启用压缩
        if self.compressor is not None and parts:
            total_chars = sum(len(p) for p in parts)
            if total_chars > 6000:  # 约 2000 tokens 的启发式阈值
                try:
                    compressed = await self.compressor.acompress(
                        texts=parts,
                        query=self._query,
                        system_prompt_tokens=0,
                    )
                    print(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    print(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """为单个 SubTask 构建执行上下文。"""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
        # 注入依赖任务的结果
        for dep_id in subtask.dependencies:
            dep_key = f"result:{dep_id}"
            if dep_key in self._memory_store:
                ctx[f"dep:{dep_id}"] = self._memory_store[dep_key]
        return ctx

    def _is_report_degraded(self, new_content: str, original_content: str) -> bool:
        """检查对抗后的报告是否比原始更差（被污染为道歉/空内容）。

        Returns:
            True 如果新报告内容退化（应该回退到原始版本）。
        """
        forbidden_patterns = [
            "由于搜索api", "由于api", "api密钥无效", "api密钥",
            "搜索api", "博查ai", "密钥无效", "无法获取任何搜索",
            "未能获取任何", "本报告内容为空", "无法补充来源",
            "由于搜索API错误", "报告已截断", "仅显示前",
        ]
        new_lower = (new_content or "").lower()
        orig_lower = (original_content or "").lower()

        # 新报告包含禁止内容而原始报告没有 → 退化
        for pat in forbidden_patterns:
            if pat in new_lower and pat not in orig_lower:
                return True

        # 新报告内容显著变短（<50% 原始长度，且原始 >4000 字） → 截断退化
        if len(original_content) > 4000 and len(new_content) < len(original_content) * 0.5:
            return True
        # 短报告也检查显著缩水（<30% 原始长度）
        if len(original_content) > 200 and len(new_content) < len(original_content) * 0.3:
            return True

        return False

    def _build_failure_reason(self, results: list[AgentResult]) -> str:
        """分析失败原因，生成给 replanner 的描述。"""
        reasons = []
        for result in results:
            if result.recovery.get("needs_replan") or result.acceptance.get("passed") is False:
                reasons.append(f"{result.task_id}: gaps={result.acceptance.get('gaps', [])}; "
                               f"recovery={{'last_error': {result.recovery.get('last_error')}, "
                               f"'tool_calls': {result.recovery.get('tool_calls')}, 'fallbacks': {result.recovery.get('fallbacks')}}}; "
                               f"partial_output={str(result.output)[:1200]}")
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        skipped_count = sum(1 for r in results if r.status == AgentStatus.SKIPPED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
        if skipped_count > 0:
            reasons.append(f"{skipped_count} tasks skipped because required dependencies did not succeed")
        return "; ".join(reasons) if reasons else "Unknown failure"

    def _rebuild_task_map_from_dag(self) -> dict[str, SubTask]:
        """从 DAG 重建 task_map（当缺少原始 SubTask 信息时使用占位符）。

        实际场景中，planner 应返回完整的 SubTask 列表；
        这里作为降级：为 DAG 中每个节点创建默认 SubTask。
        """
        if self._dag is None:
            return {}

        task_map: dict[str, SubTask] = {}
        for node_id in self._dag:
            deps = self._dag.get_dependencies(node_id)
            if node_id not in self._task_map:
                # 新建占位 SubTask
                task_map[node_id] = SubTask(
                    task_id=node_id,
                    task_type=TaskType.SEARCH,
                    description=f"Auto-generated task for {node_id}",
                    dependencies=deps,
                )
            else:
                # 保留已有信息，更新依赖
                old = self._task_map[node_id]
                task_map[node_id] = SubTask(
                    task_id=old.task_id,
                    task_type=old.task_type,
                    description=old.description,
                    dependencies=deps,
                    context_keys=old.context_keys,
                    timeout_seconds=old.timeout_seconds,
                    priority=old.priority,
                    expected_type=old.expected_type,
                    search_hints=old.search_hints,
                )
        return task_map
