# DeepResearch Agent

### 从复杂 Query 到结构化深度研究报告，全链路自动化

基于多智能体协作的深度研究系统，覆盖规划、执行、记忆、压缩、对抗全链路，一键生成带引用的结构化研究报告。

---

## 系统架构

六大模块协同工作，实现从用户 Query 到最终研究报告的端到端自动化：

| 模块 | 职责 | 核心技术 |
|------|------|---------|
| M1 Orchestrator | 多智能体编排与调度 | asyncio + DAG 拓扑执行，9 状态状态机 |
| M2 Planner | 复杂问题拆解 | LLM 驱动 JSON DAG 动态规划，支持中英文 query |
| M3 Compressor | 上下文语义压缩 | L1 相关性过滤 → L2 TextRank 提取 → L3 LLM 摘要 |
| M4 Memory Store | 跨 Agent 共享记忆 | SQLite 持久化 + numpy 向量索引，写入前自动去重 |
| M5 Adversarial | Red-Blue 对抗降噪 | 三大组合维度并行攻击-修复，收敛与退化检测 |
| M6 Evolution | 在线自进化训练 | GRPO 强化学习 + 符号规则学习（预留接口） |

### 数据流

```
用户 Query → M2 Planner（拆解为 DAG 子任务图）
    → M1 Orchestrator（按拓扑排序并发调度）
    → Researcher Agent（调用 ArXiv / Browser / Web Search 等工具）
    → M4 Memory Store（写入中间结果，去重）
    → M3 Compressor（提取式压缩长输出）
    → Summarizer Agent（合成最终报告）
    → M5 Adversarial（Red-Blue 对抗优化）
    → 输出结构化 Markdown 研究报告（含引用来源）
```

---

## 核心特性

### 多智能体编排引擎
- 基于 `asyncio + Semaphore` 的 DAG 拓扑并发执行，不依赖 LangGraph/AutoGen
- 9 状态状态机：IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
- 三级降级策略：单任务超时标记继续 → >50% 失败触发 replan → 全局超时强制合成
- Planner 支持中英文 query 自动语言检测，SubTask 描述与 query 同语言

### Red-Blue 对抗降噪
- **Red Agent**：从"事实准确性+幻觉检测"、"逻辑一致性"、"来源可信度+覆盖完整度"三大组合维度并行攻击报告，`asyncio.gather` 并行执行
- **Blue Agent**：执行原地修正 / 补充搜索 / 移除三类修复，内置截断检测、防污染回退和 SUPPLEMENTARY 修复次数限制
- **收敛控制**：连续 3 轮分数未改善自动终止 + 历史最佳版本追踪

### 鲁棒报告生成
- 无商业搜索 API 环境下通过 ArXiv + Browser 工具链自动 fallback
- 4 层报告质量防护：输出清理 → 反道歉检测与重试 → Blue Agent 防污染验证 → 对抗后退化回退
- 强制中文输出（全链路 Prompt 语言约束）+ [Title](URL) Markdown 引用格式

### 上下文压缩与共享记忆
- **M3 Compressor**：PLANNING 阶段串联 M4→M3→Planner 做细粒度语义压缩；COLLECTING 阶段对 Researcher 长输出做独立提取式压缩（50% 保留率）
- **M4 Memory Store**：SQLite 持久化 + numpy 向量索引；写入前自动去重（cosine > 0.92）；句边界智能截断 + 真实 URL 提取 + 任务级 topic 标注

### 多后端 LLM 路由
- 支持 DeepSeek / vLLM / OpenAI 热切换，`.env` 驱动零源码修改
- 模块级采样参数集中管理（temperature / max_tokens / top_p），避免配置漂移

---

## 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone <repo-url>
cd deepresearch

# 2. 创建虚拟环境（推荐 Python 3.11）
conda create -n deepr python=3.11
conda activate deepr

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 配置 API Key
cp .env.template .env
# 编辑 .env：填入 DEEPSEEK_API_KEY
```

### 运行方式

**单条 Query（单次深度研究）**
```bash
python scripts/run_single.py \
    --query "2024-2025年大模型Agent技术趋势与落地案例研究" \
    --config configs/default.yaml
```

**交互式 REPL（支持 Session 继承与连续追问）**
```bash
python scripts/run_repl.py
```

**批量实验（全量评测）**
```bash
python scripts/run_all_experiments.py \
    --report_file outputs/reports/report_xxx.md \
    --report_query "你的研究问题"
```

---

## 项目结构

```
deepresearch/
├── configs/                      # YAML 配置中心
│   ├── default.yaml              # 全局默认配置（模型/编排/压缩/记忆/对抗/工具）
│   ├── agents/                   # Agent 行为配置
│   ├── evolution/                # 进化训练配置
│   ├── planner/                  # 规划器配置
│   └── tools/                    # 工具层配置
├── src/                          # 核心源码
│   ├── core/                     # 核心运行层（runner / judge / ablation）
│   ├── orchestrator/             # M1: 多智能体编排器
│   ├── planner/                  # M2: 自适应规划器
│   ├── compressor/               # M3: 上下文压缩器
│   ├── memory/                   # M4: 共享记忆存储
│   ├── adversarial/              # M5: 对抗降噪循环
│   ├── evolution/                # M6: 自进化引擎
│   ├── agents/                   # Agent 实现（Researcher / Summarizer）
│   ├── models/                   # 多后端 LLM 路由
│   ├── tools/                    # 工具层（ArXiv / Browser / Web Search / Calculator 等）
│   └── utils/                    # 工具函数（日志 / 追踪 / 环境变量）
├── evaluation/                   # 评测体系（Benchmark / Metrics / Stats）
├── scripts/                      # 可执行脚本（run_single / run_repl / run_eval 等）
├── tests/                        # 测试文件
├── requirements.txt              # 依赖清单
├── pyproject.toml                # 项目配置
└── .env.template                 # 环境变量模板
```

---

## 配置说明

编辑 `configs/default.yaml` 调整核心参数：

- `model.backend`: 默认 LLM 后端（deepseek / vllm / openai）
- `orchestrator.max_concurrent`: 最大并行子 Agent 数（默认 5）
- `orchestrator.global_timeout_seconds`: 全局超时（默认 600）
- `orchestrator.researcher_max_turns`: Researcher 最多工具调用轮数（默认 15）
- `adversarial.enabled`: 是否启用对抗降噪（默认 true）
- `adversarial.max_rounds`: 对抗最大轮数（默认 5）
- `tools.web_search.mock_mode`: Mock 模式开关（无搜索 API 时设为 true）

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| 异步框架 | asyncio |
| LLM 后端 | DeepSeek / vLLM / OpenAI |
| 嵌入模型 | sentence-transformers (all-MiniLM-L6-v2) |
| 持久化 | SQLite + numpy 向量索引 |
| 可观测性 | LangSmith（可选） |
