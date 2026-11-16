# B → C G4 statistics cross-review

本目录为 2026-11-16 B 对 C PR #4 的独立统计交叉复核快照。基线 c707650；正式协议、缓存、模型及预测未变。

- `review.json`：16 万行、四表250个数值对照、两个负面对照、输入和受保护文件 SHA、环境、范围与限制。
- `summary.csv`、`per_class.csv`、`contrasts.csv`：从原始 G4 预测新算出的全精度数值，不覆盖 C 的原表。
- `per_run.json`：16 份预测的正确数及分类别计数。
- `development_note.json`：首轮工具把 CLIP 三列参考当作分类器六列记录而退出，修正适配后完整重跑通过；不把工具错误记作 C 数据缺陷。

阅读 [B 复核报告](../../report/B_C_G4_cross_review.md)；复跑使用 [标准库脚本](../../code/reviews/review_c_g4.py)和原始 [G4 交接包](../synthetic_v2/g4/README.md)。只验证已发布数据的统计一致性，不等于独立推理、三 seed 显著性证明、真人审计或全量跨平台特征复现。
