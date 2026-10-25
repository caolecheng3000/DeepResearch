# 参与 Deep-Research

欢迎通过Issue讨论可复现问题、研究任务和改进建议。Pull Request应说明问题、实际变更及验证结果。

## 开发与验证

从README建立conda环境deepr，安装依赖后使用可编辑安装。所有测试都在deepr中执行：

```bash
conda activate deepr
LANGSMITH_TRACING=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 python scripts/run_offline_checks.py
```

恢复原始论文档案后，可运行`python -m unittest discover -s tests -p 'test_*.py'`。真实模型实验应明确记录后端、配置、题集版本、预算和实际完成范围，不将计划或离线测试当成模型质量结果。

## 修改约定

- 运行逻辑变动同步更新对应文档。研究行为与发布/格式整理尽量分别提交，便于审查。
- 测评数据升级应使用新版本，不覆盖历史实验的正文、参考标签或原始评分。
- v5研究题目的维护源为`evaluation/research_tasks.py`，生成入口为`scripts/build_research_dataset.py`。参考要点需核对原文，允许有根据的不同建议。
- 不提交`.env`、Key、SQLite库、运行日志、模型缓存或完整实验输出。示例使用占位值；公开材料注明原始来源与许可。

提交前可运行`python scripts/prepare_public_release.py --check`检查拟发布文件。该检查针对当前文件，不会清除已有Git历史中的敏感记录。
