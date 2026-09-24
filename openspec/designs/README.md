# 稳定设计文档

本目录承载从已归档 change 提炼的长期设计事实，与 `openspec/specs/`（稳定行为契约）分工：

- **specs/**：系统黑盒行为——语言原词语义、公共 API、错误码、诊断 JSON Schema、可观察编译行为。
- **designs/**：系统设计事实——架构分层、IR 设计、pass 管线、HAL（硬件抽象层）、后端 lowering 策略、模块职责与技术取舍。

## 子目录约定

- `architecture/`：跨模块架构契约与设计（有内容时创建）。
- `adr/`：长期技术取舍记录（ADR，有内容时创建）。

active change 的设计先写在 `openspec/changes/<change>/design.md`；归档时按该 change design 的"长期基线刷新计划"提炼仍成立的事实到本目录，不机械复制历史过程。
