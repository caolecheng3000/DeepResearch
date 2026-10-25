# 可恢复的一手论文档案

原始PDF、完整提取文本和逐篇来源元数据保留在维护者本地，不随公开源码提供。运行用的固定选段已经包含在版本化JSON题集中。

从项目根目录恢复归档：

```bash
conda activate deepr
python -m pip install requests beautifulsoup4
# 另需系统pdftotext（Poppler）
python scripts/fetch_research_sources.py --fetch
python scripts/build_research_dataset.py --check
```

脚本下载固定v1论文并验证题名，不调用模型。先前下载的PDF和提取文本若已完整存在会被校验复用。

外部原档可能含论文作者的联系邮箱，仅保留在被忽略的本地归档目录。公开运行材料由构建脚本统一遮盖标准、组合及跨行邮箱，保留原始位置、分页和来源哈希。验证选段时比较原文经相同脱敏后的内容；生成材料不覆盖外部原档，也不将其纳入发布清单。
