# TileScript 规格概览

TileScript 是面向 AI 辅助优化的跨硬件 Tile 级编程语言：把优化决策权从编译器启发式转移到"人 + AI agent"手中，编译器的职责是忠实执行显式意图并输出结构化诊断。三个核心价值主张：

1. **显式优化控制**——任何影响性能的决策用户必须能显式指定（MMA 形状、warp 特化、pipeline 手柄、PadPolicy）；未指定时使用固定、可预测、有文档的默认值，编译器不做搜索、不自动降级。
2. **源码级结构化诊断**——诊断 JSON Schema（v1，冻结，向后兼容）覆盖 resources、memory、async、compute、portability 和 suggestions 段，每条诊断可映射回源码行号。
3. **确定性编译**——同一源码 + 同一目标硬件产出一致的 `deterministic_hash`。

当前状态：项目处于启动期（阶段0 假设验证之前）。设计方案见 `veps/design.md`（非正式文档，不是规范来源）；进入实施的架构和行为契约以本目录下已归档的 specs 与 designs 为准。

## 范围

- `openspec/specs/` 承载归档后的稳定行为契约（语言原词语义、公共 API、错误码、诊断 JSON Schema、确定性承诺、跨硬件可移植性行为）。
- `openspec/designs/` 承载归档后的稳定设计事实。
- active change 的设计先写在 `openspec/changes/<change>/design.md`，归档后再提炼到稳定设计文档。
- `veps/` 只存放非正式交付文档（草稿、调研、过程记录），不是规范来源。

## 稳定基线

（尚无归档 change；首个 capability 基线建立后在此登记索引。）
