# TileScript

TileScript 是面向 AI 辅助优化的跨硬件 Tile 级编程语言：显式优化控制、源码级结构化诊断（JSON Schema）、确定性编译。设计方案见 `veps/design.md`（非正式文档）。

## 协作约定

- 与用户的文字交互使用中文。
- 非正式交付的文档（草稿、调研、过程记录）放入 `veps/`，不得散落仓库其他位置。
- `veps/` 不是规范来源；强制语义只存在于本文件和 `openspec/`。

## OpenSpec 强制开发流程（权威）

本仓库采用 OpenSpec 作为强制的软件开发流程。`openspec/` 是产品行为、公共契约和验收语义的权威来源。

- **规格先行**：新增或修改语言原词语义、公共 API、错误码、诊断 JSON Schema、确定性承诺、跨硬件可移植性行为前，必须先有 active OpenSpec change（`openspec/changes/<change>/`）；不得把未定义行为直接写进实现。
- **开发入口**：任何会修改生产代码或新增/实质修改 OpenSpec change 的任务，开始时使用 `tilescript-dev-workflow` skill（设计语义审查 → H1 人工确认 → 实施 → 代码语义审查 → H2 人工确认 → push → 归档）。
- **低风险快速通道**（不建 change、不建审查账本）：纯文档修改；或可追溯到既有 Requirement/Scenario、不改公共契约的低风险修复（须复现 + 回归测试 + 一次代码语义审查）。任一条件无法证明时退出快速通道。
- **实施阶段**默认只改 active change 和代码；specs/designs/overview 等长期基线文档在归档前按 change design 的"长期基线刷新计划"更新。
- **验证命令**：`npx openspec validate --all --strict`（pre-commit 与 pre-push 均强制执行，失败阻断；不得绕过或删除 hooks）。
- **审查账本**：`openspec/changes/<change>/reviews/` 下的 JSONL 是唯一原始审查记录；主智能体不代写或改写审查者记录。
- OpenSpec 项目级定制只写入 `openspec/config.yaml`；CLI 自带的 `openspec-*` skills（propose/apply/archive/explore 等）是通用工作流，不修改。
- 编写规则（中文写作、BCP 14 规范关键词、受约束自然语言、语义闭合、验收可推导）见 `openspec/config.yaml`，对全部 artifacts 生效。

## 工程约束（随代码引入逐步充实）

- 新增或修改可观察行为时先写表达目标行为的测试（TDD）；缺陷修复先复现失败。
- 行为测试断言公共可观察结果（编译输出、诊断 JSON、错误码、公共 API），不断言私有实现细节。
- 改动须追溯到人工请求、OpenSpec change、架构约束或直接强迫的联动；不得顺改无关代码。
- 当前尚无代码与质量脚本（build/test/lint）；引入后把对应检查挂入 `.husky/` 门禁并更新本文件。
