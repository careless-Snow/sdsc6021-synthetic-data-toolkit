# SDSC6021 — Synthetic Data Toolkit

面向 20-shot 图像分类的合成数据增强工具：生成补充图片、选择样本、训练分类器，并评估真实测试效果。

> 当前：A 的 Python/CUDA、依赖完整性和模型库导入已通过检查；模型权重下载未完成，实际模型推理尚未验证。按用户要求暂停，G1 仅部分通过；没有正式实验结果。

## 阅读入口

- [agent.md](agent.md)：唯一项目规则，包含实验协议、数据接口、GitHub 共享范围、验收及报告框架。
- [progress.md](progress.md)：唯一实际进度，包含证据、待办、阻塞和成员分工。
- [团队分工](分工.md)：A–D 工作包及必须交出的产物；具体成员认领和实际完成情况仍以 progress.md 为准。

本文件只用于入门导航；详细协议只维护在 agent.md，运行状态只维护在 progress.md。

## 团队协作

- 本仓库公开，队友可以直接查看和克隆；直接推送仍需协作者权限，也可通过 fork 提交 PR。代码、配置和 Markdown 文档以本仓库为唯一共享来源。
- 各人在模块分支提交小范围修改，A 负责集成，D 汇总进度和报告；具体认领状态见 progress.md。
- 先按 agent.md 约定接口，用小样例并行开发，再进行完整试跑。GPU 作业统一排队。
- 每个模块作者编写对应报告内容；代码作者、实际运行者和写作者分别记录。

## 文件共享

- **提交仓库：**项目文档、代码、可执行配置与依赖清单（实现后）、报告/展示源稿、必要的小型结果与图表。
- **本机保存：**runs/ 原始产物、完整图片、特征、模型权重、环境和下载缓存；重要产物另做备份。
- 队友需要的大文件按 run_id 单独交接；共享位置和校验值登记在 shared/<run_id>/，不另外复制一套代码。
- 课程课件及要求摘录保留本地，不纳入上传范围。详细共享清单与交接责任见 agent.md。
- 文件目录、临时文件、备份及推送前后检查统一遵循 agent.md 的“文件管理与 GitHub 同步规则”，不另建管理规范。

## 开始工作

依次阅读 agent.md 和 progress.md。[环境检查摘要](shared/g1_environment/check.json) 保存已验证项与未验证项。当前 A/Codex 已暂停，恢复后先补齐权重并完成模型实测，再推进 G2。

已在本机独立环境验证的命令（默认只检查 CUDA，不下载模型）：

```powershell
.\.venv\Scripts\python.exe code/toolkit.py doctor
.\.venv\Scripts\python.exe -m unittest discover -s code -p test_a_protocol.py -v
```

环境使用 Python 3.12.14。重建时用 Python 3.12 创建 .venv，再依次安装 requirements-gpu.txt（官方 cu128 索引）和 requirements.txt（其余固定版本）；pip 命令加 --disable-pip-version-check。本机安装曾复用已有 CUDA 分发包，详细复制和校验记录保留在本地 runs/_cache/。

`doctor --models` 会下载并核验 SD1.5、CLIP 和 ResNet；本轮未完成该项。完整五条件 smoke、选择、特征和训练等阶段尚未集成，见 agent.md。

## B 的独立开发验证

B 已接入真实数据划分、四视图特征和真实基线训练。CPU 开发链路已实测；A 的生成阶段保持原有暂停状态，全组 G1/G2 仍未完整通过。使用与交接见 [B_handoff.md](report/B_handoff.md)，方法草稿见 [B_methods.md](report/B_methods.md)，实际结果见 progress.md。

```text
python code/toolkit.py smoke --component b --download --device cpu
```

本轮独立 CPU 环境使用 requirements-b-cpu.txt（Python 3.13.9）。它只用于 B 验证，不能覆盖 A 的 GPU 环境要求。开发集结果不是正式测试结果。
