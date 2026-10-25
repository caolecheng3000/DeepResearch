# DeepResearch

面向文献综述、技术选型和开放问题分析的多 Agent 研究系统。系统将研究问题拆成有依赖关系的任务，通过检索、阅读和计算工具收集证据，整理上下文并生成报告，可选使用 Red/Blue 流程检查与修订。

## 主要功能

- **研究编排**：Asyncio + Semaphore 控制 DAG 分层并发，支持依赖检查、重规划、统一 deadline 和取消清理。
- **错误恢复**：参数修复、有限重试、后端或等价工具切换、重规划和终止，共享请求级调用预算。
- **上下文管理**：相关性过滤、TextRank 原句提取和模型摘要；持久记忆支持会话隔离、去重、规则冲突处理、LRU、TTL 和历史归档。
- **报告修订**：Red 多维审查，Blue 按问题执行修改、补搜或删除；默认最多 3 轮，每轮合计最多处理 5 项。
- **评测追踪**：固定材料工具、规则诊断、任务契约、五维 LLM Judge、本地 Trace 和可选 LangSmith SDK 集成。
- **模型接入**：配置 DeepSeek、MiMo、OpenAI 兼容接口或本地 vLLM，不在源码中保存账号凭证。

## 安装

推荐 Python 3.11，Torch 请按本机 CPU/CUDA 环境安装。

```bash
conda create -n deepr python=3.11 -y
conda activate deepr
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
cp .env.template .env.local
```

在 `.env.local` 填写所选模型和搜索服务的连接参数，实际模块配置见 `configs/default.yaml`。不使用平台追踪时设置 `LANGSMITH_TRACING=false`。模型名和服务可用性需根据自己的后端配置确认。

可选依赖：`pip install -e '.[observability]'` 用于 LangSmith，`.[serve]` 用于本地 vLLM，`.[train]` 用于训练入口。Python 分发名保留 `deep-research-agent`。

## 执行研究

以下命令会调用所配置的模型和工具，可能产生服务费用：

```bash
python -m scripts.run_single \
  --query "比较RAG与长上下文在文献研究中的适用条件，给出证据和验证方案" \
  --config configs/default.yaml
```

交互入口为 `python -m scripts.run_repl`。报告、日志和运行记忆由程序在本地生成，不随源码发布。

## 固定材料评测

`evaluation/data/` 包含版本化评测输入。文献研究草案 v5 包含 6 个领域的 24 个研究问题和 31 篇固定版本论文的选段；参考标注尚待独立人工审查，不作为经过认证的保留集或正式榜单。旧受控题集与 HotpotQA 改编子集用于独立诊断。

发布选段已遮盖论文联系邮箱，保留原始来源、选段位置和分页。重新构建会自动应用相同脱敏规则；来源校验区分原始档案哈希与脱敏材料哈希，详见[第三方来源说明](THIRD_PARTY_NOTICES.md)。

```bash
# 仅查看计划，不构造模型请求
python scripts/run_research_suite.py --pilot
```

实际运行需显式加入 `--execute`，`--judge` 会额外调用模型评审。固定材料搜索和阅读不访问实时网页，远程模型请求仍需网络。原始论文 PDF、全文提取文本和完整外部数据缓存不包含在本仓库中。

核对原始论文或重建选段时，可以恢复可选档案（还需安装系统 Poppler 的 `pdftotext`）：

```bash
python -m pip install requests beautifulsoup4
python scripts/fetch_research_sources.py --fetch
python scripts/build_research_dataset.py --check
```

## 离线验证

```bash
conda activate deepr
LANGSMITH_TRACING=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 python scripts/run_offline_checks.py
```

未恢复原始论文时，仅明确跳过依赖原始 PDF/全文档案的一项来源核验；其余测试照常执行。离线用例不等同于真实模型质量评测。

## 目录

```text
src/          编排、Agent、模型、工具、压缩、记忆和训练模块
configs/      运行及评测配置
evaluation/   评分与版本化题集
scripts/      研究、材料构建、实验和分析入口
tests/        离线回归与开发辅助脚本
.github/      CI 与协作模板
```

## 发布范围

公开内容为源码、配置模板、评测输入、测试及必要的使用和许可说明。个人总结、面试稿、内部分析文档、历史实验报告、Trace、数据库、下载档案和真实凭证不包含在发布树中。

这是研究原型。会话隔离不等同于多租户授权，受限表达式执行不等同于操作系统级沙箱。部署时应按实际场景配置工具权限和资源边界。

代码采用 [MIT License](LICENSE)。第三方论文及数据遵循其自身条款，见 [第三方说明](THIRD_PARTY_NOTICES.md)。参与方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全问题见 [SECURITY.md](SECURITY.md)。
