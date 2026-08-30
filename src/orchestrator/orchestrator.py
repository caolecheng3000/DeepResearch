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
from ..utils.tracing import trace_chain

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
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy

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
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._memory_store.clear()
        self._results.clear()
        self._dag = None
        self._task_map.clear()
        self._current_state = OrchestratorState.IDLE

        # 状态机主循环
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            # 全局超时检查
            if self._is_global_timeout():
                if self._current_state in (
                    OrchestratorState.COLLECTING,
                    OrchestratorState.SYNTHESIZING,
                    OrchestratorState.ADVERSARIAL,
                ):
                    # 强制合成：用已有结果生成报告
                    self._current_state = OrchestratorState.SYNTHESIZING
                else:
                    self._current_state = OrchestratorState.FAILED
                break

            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            next_state = await handler()
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

            # M4: 将最终报告存入 SharedMemoryStore
            if self.memory_store is not None:
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"final_report:{int(time.time())}",
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
                    self.memory_store.put(entry)
                    print(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")

                    # M3+L3: M4 容量 >85% 时用 LLM 摘要压缩旧记忆
                    await self._maybe_compress_old_memories()
                except Exception as e:
                    print(f"[M4] Failed to store final report: {e}")

            return report

        # FAILED 状态
        return ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
        )

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
            memory_ctx = self._build_memory_context()
            self._dag = self.planner.generate_plan(self._query, memory_ctx)
            # 从 planner 获取完整的 SubTask 信息（包括 description、search_hints 等）
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # 降级：如果解析失败，使用占位符
                self._task_map = self._rebuild_task_map_from_dag()
        except PlanParseError as e:
            print(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
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
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (并行执行)")

            # 构建本层的 coroutine 列表
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    # 准备上下文：先执行依赖任务的结果
                    context = self._build_task_context(subtask)

                    # 获取 Agent
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    try:
                        # 设置单任务超时
                        result = await asyncio.wait_for(
                            agent.run(subtask, context),
                            timeout=subtask.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {subtask.timeout_seconds}s",
                        )
                    except Exception as e:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        await self.agent_pool.release_agent(agent)

                    return result

            # 并发执行本层
            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # 将异常包装为 FAILED 结果
                    all_results.append(AgentResult(
                        task_id="unknown",
                        status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}",
                    ))
                else:
                    all_results.append(lr)
                    # 每层结束后立即写入 _memory_store，使下层可见依赖结果
                    if isinstance(lr, AgentResult):
                        self._memory_store[f"result:{lr.task_id}"] = lr

        self._results = all_results
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """收集结果，写入 memory，检查是否需要重规划。

        三级降级策略检查点:
          - 单任务超时/失败：已在 dispatch 层处理（标记状态，继续执行）
          - >50% 失败：触发 REPLANNING
          - 全局超时：由外层 run() 的循环检查处理
        """
        # 将结果写入运行时 memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # M4: 将成功结果同步写入 SharedMemoryStore（持久化 + 向量索引）
        if self.memory_store is not None:
            for r in self._results:
                if r.status == AgentStatus.SUCCESS and r.output:
                    self._sync_result_to_memory_store(r)

        # M3: 对成功结果做提取式压缩（L2 TextRank），提升 Summarizer 输入质量
        if self.compressor is not None:
            success_indices = [i for i, r in enumerate(self._results) if r.status == AgentStatus.SUCCESS and r.output]
            if success_indices:
                success_outputs = [str(self._results[i].output) for i in success_indices]
                try:
                    compressed = self.compressor.compress_documents(
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
            if self._replan_count < self._config.max_replan_rounds:
                self._replan_count += 1
                return OrchestratorState.REPLANNING
            else:
                print("[Collect] Max replan rounds reached, proceeding with partial results")
                # 超过最大重规划次数，继续合成（用已有结果）

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
                entry_id=result.task_id,
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

        max_entries = 10000  # 与 config memory.max_entries 一致
        current_count = len(self.memory_store)
        if current_count < max_entries * 0.85:
            return

        try:
            # 获取所有记忆，按时间排序
            all_entries = self.memory_store.lt.get_all_entries()
            if len(all_entries) < 20:  # 太少不值得压缩
                return
            all_entries.sort(key=lambda e: e.timestamp)

            # 保留最近 50%，压缩较旧 50%
            keep_count = max(current_count // 2, 10)
            old_entries = all_entries[: current_count - keep_count]

            if len(old_entries) < 5:  # 旧记忆太少，不值得压缩
                return

            # 收集旧记忆的 claim 文本
            old_texts = [e.claim for e in old_entries if e.claim]

            print(
                f"[M3] Compressing {len(old_texts)} old memories "
                f"(M4 usage {current_count}/{max_entries}, {current_count/max_entries:.0%})"
            )

            # L3: 用 LLM 将旧记忆聚合为一份摘要
            aggregate = self.compressor.summarizer.summarize_documents(
                old_texts,
                query=self._query,
                max_length=max(800, min(3000, len(old_texts) * 30)),
            )

            if not aggregate or len(aggregate) < 100:
                print("[M3] L3 compression produced empty result, skipping")
                return

            # 删除旧记忆（使用同一个连接）
            conn = self.memory_store.lt._connect()
            try:
                for e in old_entries:
                    self.memory_store._remove_from_index(e.entry_id)
                    conn.execute(
                        "DELETE FROM entries WHERE entry_id = ?", (e.entry_id,)
                    )
                conn.commit()
            finally:
                conn.close()

            # 插入压缩摘要作为新记忆
            from src.memory.long_term import MemoryEntry
            import time as _time
            compressed_entry = MemoryEntry(
                entry_id=f"compressed:{int(_time.time())}",
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
            self.memory_store.put(compressed_entry)

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

        agent = await self.agent_pool.get_agent(TaskType.ANALYZE)
        # 需要 SummarizerAgent，但 agent_pool 可能返回 ResearcherAgent
        # 这里我们通过类型检查或强制创建 SummarizerAgent
        from ..agents.summarizer import SummarizerAgent
        if not isinstance(agent, SummarizerAgent):
            # 优先使用配置的 summarizer_policy（更大的 max_tokens），fallback 到 agent.policy
            policy = self.summarizer_policy or agent.policy
            agent = SummarizerAgent(name="summarizer", policy=policy, tools=agent.tools)

        try:
            result = await asyncio.wait_for(
                agent.run(synth_task, context),
                timeout=synth_task.timeout_seconds,
            )
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
            await self.agent_pool.release_agent(agent)

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            self._memory_store["final_report"] = result.output
        else:
            # 合成失败但已有结果，生成降级报告
            self._memory_store["final_report"] = ResearchReport(
                query=self._query,
                content=str(result.output) if result.output else "Synthesis failed.",
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
        if report.confidence >= 0.8:
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
            self._memory_store["final_report"] = pre_adversarial_report

        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """触发增量重规划。

        保留 confidence≥0.6 的成功结果，修改失败子问题。
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
            new_dag = self.planner.replan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=self._results,
                reason=reason,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            # 清空上一轮结果（保留在 memory 中，新任务可通过 context_keys 引用）
            self._results = []
        except PlanParseError as e:
            print(f"[Replan] Failed: {e}")
            # 重规划失败，如果已有部分成功结果，尝试直接合成
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Replan] Unexpected error: {e}")
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
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
        """判断是否需要重规划。

        策略:
          - 失败率 > 50% 时触发
          - 或存在任何 TIMEOUT 且成功结果不足 30%
        """
        if not results:
            return False
        total = len(results)
        failed = sum(1 for r in results if r.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT))
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)

        failure_rate = failed / total
        if failure_rate > 0.5:
            return True
        if success / total < 0.3 and failed > 0:
            return True
        return False

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """检查是否超过全局超时。"""
        elapsed = time.monotonic() - self._start_time
        return elapsed > self._config.global_timeout_seconds

    def _build_memory_context(self) -> str:
        """构建给 planner 的上下文摘要。

        优先使用 M4 SharedMemoryStore 的语义检索（如果已接入），
        否则回退到运行时 dict 遍历。
        """
        # M4: 语义检索相关记忆
        if self.memory_store is not None:
            try:
                ctx = self.memory_store.get_context_for_query(
                    self._query, max_tokens=3000  # 放宽限制，给 Compressor 更多素材
                )
                if ctx:
                    print(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    # M3串联: 对 M4 的粗粒度结果做细粒度语义压缩
                    if self.compressor is not None:
                        try:
                            compressed = self.compressor.compress(
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
                    compressed = self.compressor.compress(
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
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
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
