# 第三方来源与许可

根目录MIT License适用于本项目的原创代码和原创说明，不把第三方论文、数据集、摘录及其衍生部分改许可为MIT。

## 文献研究材料

v5使用31篇论文的固定arXiv v1版本。来源题名、URL、下载记录、PDF与提取文本哈希、上游许可链接保存在[evaluation/data/research_v5/source_manifest.json](evaluation/data/research_v5/source_manifest.json)。

运行材料保留在版本化题集和materials档案中；原始PDF及完整提取文本属于可独立恢复的外部来源档案，不随公开源码快照提供。引用和再使用第三方内容时，应按清单核对各论文自己的许可与归属，不把arXiv可下载等同于任意再分发授权。

论文原始地址与所用选段记录在 evaluation/data/research_v5/materials/ 和来源清单中。发布材料中的联系邮箱已用等长字符█遮盖，包括共用域名的组合邮箱及跨行邮箱；保留必要的论文署名和来源归属。生成脚本先对完整来源文本脱敏，再按原始位置选段，以免选段边界泄露部分邮箱。选段字符位置和分页不变，校验比较的是原文经同一脱敏变换后的内容，不再宣称所有选段逐字等同于未处理原文。原始PDF和全文哈希仍用于核对未改动的本地档案，运行材料哈希则按脱敏后内容重新生成。重新构建也会应用相同规则；未脱敏的外部原档不随本仓库发布。

## HotpotQA

项目保留HotpotQA公开dev子集的适配题目及其Wikipedia历史材料，原始来源及许可记录在[evaluation/data/research_hotpotqa_v1.json](evaluation/data/research_hotpotqa_v1.json)的provenance中。

来源：Yang、Qi、Zhang等，*HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*，EMNLP 2018。[项目主页](https://hotpotqa.github.io/)。该子集及其衍生材料保留原数据的CC BY-SA 4.0归属与条款；中文报告指令、共享检索池和评测要求属于本项目的协议适配，不是官方榜单设置。

完整外部parquet缓存不随源码发布。固定子集本身已保留，正常快照评测不要求该缓存。

## 软件依赖

第三方Python包、Poppler及模型权重按各自许可分发。核心依赖见requirements.txt和pyproject.toml；本仓库不捆绑模型权重、训练框架源码或模型服务账号。
