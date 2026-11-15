# B 阶段使用与交接

当前交付只覆盖 B 可独立完成的开发工作：真实数据划分、冻结特征、真实基线、混合损失与 C 清单校验函数。A 的生成任务及 C 的选择、评估模块继续由对应成员完成。

## 从统一入口运行

在已安装匹配依赖的项目环境中运行：

```powershell
python code/toolkit.py prepare --smoke --download
python code/toolkit.py features --smoke --download --device cpu --threads 4 --batch-size 8
python code/toolkit.py train --smoke --device cpu --threads 4
```

也可以一次执行 B 的开发链路：

```powershell
python code/toolkit.py smoke --component b --download --device cpu
```

`--download` 允许下载公开 CIFAR-10 与官方 ResNet-18 权重；缓存齐全后可以省略。CPU 是默认且本轮实测的设备。使用 `--device cuda` 前由 A 安排 GPU 作业，并在集成环境验证实际推理。完整项目 smoke 不等于 `smoke --component b`。

开发产物位于 `runs/b_smoke/<配置摘要前12位>/`：

- `b_protocol.json`：完整配置、B 源码摘要、包版本和开发模式。
- `b_environment.json`：本机独立验证环境；不代表 A 的环境。
- `manifests/b_real.json`：图片、真实索引、三个 seed 的划分与原始训练批次校验值。
- `features/b_real.pt` 和 `.json`：599 张不同图片的四视图特征，以及行号、ID 和校验值。
- `models/real_only_seed{0,1,2}.pt` 和 `.json`：三个开发分类头、初始化、损失历史、重载验证。
- `metrics/dev_real_only_seed{0,1,2}.json`：开发集逐图预测；最终测试集预测另行产生。
- `metrics/b_summary.json`：本轮开发验证汇总；禁止作为正式实验结果。
- `logs/b_*.json`：分阶段运行与缓存记录。

同一配置可重复运行，成功产物经校验后复用。源码、包版本或配置变化会拒绝混用旧运行。修改实现时须改变配置中的 `classifier.implementation_version` 或 `run_id`，保留旧证据；不能手改成功状态绕过检查。

## 验证

```powershell
python -m unittest discover -s code -p "test_*.py" -v
python -m pip check
```

测试不下载数据或权重。它们检查样本划分、数据泄漏、损失权重及梯度、配对初始化、相同更新预算、模型重载、固定视图、缓存失效、选择数量及嵌套关系、未冻结协议的阻断。

## 给 A / C 的接口

`b_protocol.paired_splits` 产生三个 seed 的配对真实划分。开发模式每类 16 张训练、4 张开发；正式模式每类 20 张训练、不产生开发划分。两种模式选中的每类20张原图相同。所有类别使用 CIFAR-10 官方顺序。

A 的候选图需要提供 `image_id, class_id, class_name, relative_path, sha256, source=synthetic, status=success`。图片路径相对对应运行目录。

C 的选择清单采用 JSON 行对象列表，每行包含 `schema_version=1, pool_id, condition_id, image_id, class_id, rank`，并保留项目协议要求的评分或随机种子。`condition_id` 为 `random_50, clip_50, random_150, clip_150`。每类 `rank` 从1起；50张集合必须嵌套于对应150张集合。

`b_protocol.validate_selection(rows, candidates, pool_id=...)` 可检查数量、标签、重复、排名及嵌套性。它不代替 C 的 CLIP 打分和完整随机排名核验。`b_runtime.fit_head` 已支持成对的真实与合成特征张量及固定 50/50 损失；真实、合成张量形状均为 `[图片数, 视图数, 特征维度]`，标签每张原图一个。

当前统一入口的 `train` 只接通 `real_only`；混合组自动运行仍待候选图片、选择清单及集成。正式阶段命令不加 `--smoke` 时要求 G2 和冻结配置，不能为了运行而提前修改关卡状态。

## 下一步

1. A 完成模型实测，并提供有效候选池及图片清单。
2. C 提供选择清单与池标识，完成选择端实测。
3. B 接入合成图片特征和四个混合条件，与 A/C 完成完整五条件试跑。
4. 全组冻结配置，再跑正式15模型；由最终评估器统一处理官方测试标签。

本轮本地代码分支为 `feat/b-baseline`。未经实际推送，不得将本地完成记录写为 GitHub 已交付。
