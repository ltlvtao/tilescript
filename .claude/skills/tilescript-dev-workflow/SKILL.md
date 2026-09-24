---
name: tilescript-dev-workflow
description: TileScript OpenSpec Change 和代码交付的端到端流程。实施前执行一轮设计语义审查并取得 H1 人工确认，实施后执行一轮代码语义审查并取得 H2 人工确认；每个审查 cycle 最多三轮。纯文档和低风险 issue 修复走快速通道，不使用本流程；只读解释、诊断或状态报告不使用。
---

# TileScript 开发工作流

本 Skill 只规定交付顺序和留痕。产品行为、契约与验证要求以仓库根 `CLAUDE.md` 和 `openspec/` 为准；OpenSpec 三个阶段（propose/apply/archive）的专业方法由 `openspec-*` Skill 承担。

智能体、Change 作者和审查者是同一流程中的可信协作者。门禁只防无意漏审、漏修问题和无限自动循环，不建设签名、身份 attestation 或逐提交批准链。

## 核心流程

```text
起草 Change（openspec-propose）；主智能体记录一次 baseSha 到 reviews/base.json
→ strict validation + 自查；先修复范围内确定缺陷
→ 设计语义审查：1 个独立审查 subagent 审 proposal/specs/design，向 reviews/design-review.jsonl 追加结论
→ 修订后影响复审；每个 cycle 最多三轮，全 PASS 提前退出，三轮不一致交作者
→ H1 Change 作者人工确认（自足决策材料 → reviews/h1.jsonl APPROVED）
→ 实施（TDD 节奏逐 task 闭环，实际验证勾选 tasks.md）+ 实际验证 → commit 最终候选代码
→ 代码语义审查：1 个独立审查 subagent 审候选 diff（正确性/质量/验证有效性合并），向 reviews/code-review.jsonl 追加结论
→ 修复后影响复审；同三轮规则
→ H2 Change 作者人工确认（reviews/h2.jsonl APPROVED）
→ pre-push 检查 → push / PR / owner approval
→ 归档（先按 design 的长期基线刷新计划同步 specs/designs/overview，再 openspec-archive）
```

H2 位于最终候选代码 commit 之后、push 之前。

## 1. 建立 Change 记录

创建 Change 时，主智能体创建 `openspec/changes/<change-id>/reviews/base.json`（只写一次）：

```json
{ "schemaVersion": 1, "changeId": "<id>", "baseSha": "<change 开始时的 HEAD>", "note": "<可选：范围备注>" }
```

`baseSha` 必须能解析且是当前 HEAD 的祖先。审查记录在 Change 的 `reviews/` 目录：

```text
reviews/base.json  reviews/design-review.jsonl  reviews/code-review.jsonl
reviews/h1.jsonl   reviews/h2.jsonl
```

JSONL 一行一条结论，格式：`{"cycle": 1, "round": 1, "decision": "PASS|CHANGES_REQUESTED|REJECTED", "findings": [{"severity": "...", "issue": "...", "status": "open|resolved|waived"}], "summary": "..."}`。角色 JSONL 是唯一原始记录；主智能体只编排、汇总与整体验证，不代写或改写审查者记录。

## 2. 设计审查与 H1

1. 需求澄清用第一性原理拆解：先问诉求背后的根本问题，再判断方案形态。只有无法从权威资料推导且会改变业务结果、范围、成本、风险或授权的事实才请求人工决定。
2. 调用 `openspec-propose` 起草 Change，运行 `npx openspec validate --all --strict`，通过后进入审查。
3. 派发 1 个独立审查 subagent（fork，不继承主对话结论），输入六项：职责/写域、审查范围（proposal + specs + design 全文）、权威章节（CLAUDE.md + openspec/config.yaml + 相关 stable specs）、未闭环问题、验证摘要。审查者独立形成语义判断，覆盖：业务目标与授权范围、行为契约闭合（Requirement/Scenario 可验证）、语言语义与兼容性（语法接受集、数值语义、确定性影响声明、错误码与诊断 Schema 的 BREAKING 影响已按 config.yaml 规则声明）、方案正确性（架构/owner/数据/失败路径）、tasks 可验收性。
4. 审查者返回 `CHANGES_REQUESTED` 时，主智能体汇总 finding 一次性修订、重新 strict validation，再派发影响复审（delta：相对前次已审内容的变化 + 未闭环项）。resolution 由原审查者在下一轮确认。全 `PASS` 提前结束；每 cycle 最多三轮，第三轮仍不一致时停止自动修订交作者。
5. 请求 H1 时必须先呈报**自足的决策材料**：Change 目标与范围摘要（含非目标）、每轮审查结论与最终状态、全部 finding 清单（每项含严重度、一句话问题、闭环状态：已修复+验证/待处理/已协商关闭）、strict validation 结果、剩余风险与延期项。材料呈报后等待 Change 作者向 `h1.jsonl` 追加带当前 cycle 的 `APPROVED`/`CHANGES_REQUESTED`/`REJECTED`。进入实施前 H1 最后一条必须为 `APPROVED`；自动审查有分歧而 H1 批准时，H1 必须逐项关闭未解决的分歧 finding。

## 3. 实施与验证

1. 取得 H1 后按 `openspec-apply-change` 实施：TDD 节奏逐 task 闭环（行为变更先写表达验收语义的失败测试，再实现到通过），以实际验证证据勾选 `tasks.md`，只改工作区不 commit。发现 design/task blocker 时回 Step 2 修订 artifact 后继续；环境 blocker 记录为 residual risk。
2. 实施收口后，主智能体读全部 diff（主智能体是实施结果的第一读者），按影响面运行 CLAUDE.md 要求的验证（build/test/OpenSpec strict validation），必需验证失败时修复后重跑。
3. commit 全部最终候选代码和 Change task 更新，再进入代码审查。

## 4. 代码审查与 H2

1. 派发 1 个独立审查 subagent 审查候选 diff，覆盖三个面（合并为一轮）：实现正确性（正常/异常/边界行为及副作用符合批准的规格与设计）、代码质量（架构、内聚、复杂度、并发、资源）、验证有效性（测试独立预期、有效断言、实际证据）。结论追加到 `reviews/code-review.jsonl`，同三轮规则；修复由实施流程统一执行，修复后重新验证、commit、影响复审。
2. 请求 H2 时呈报自足决策材料：当前 HEAD 与 diff 摘要（按关注面归纳）、每轮审查结论与最终状态、全部 finding 清单、实际执行的验证命令与结果、延期项与 residual risk。等待 Change 作者向 `h2.jsonl` 追加决定；最终必须 `APPROVED`。
3. commit H2 和审查记录，再执行 pre-push。

## 5. 机器门禁与 Push

- pre-commit 与 pre-push 都执行 `npx openspec validate --all --strict`（见 `.husky/`），失败阻断。
- 未完成 push/PR/owner approval 不得宣称已交付。
- 合并后按 design 的"长期基线刷新计划"同步长期基线并归档；`reviews/` 随 Change 一起进入 archive。

## 低风险快速通道（不建 Change、不建审查账本）

满足全部条件时走快速通道，不创建 OpenSpec change 和 reviews 账本：

- 纯文档修改：只改说明、示例、链接或 `veps/` 文档，不改变公共接口文档、功能 spec、架构规则或用户可观察承诺。
- 低风险 issue 修复：有可追溯 issue 或失败复现，只把实现恢复到既有 Requirement/Scenario；不新增/删除/改变功能 spec、公共 API、错误码、诊断 schema 或兼容性。

快速修复必须包含能复现缺陷并证明恢复的回归测试，执行一次代码语义审查和 owner approval；任一条件无法证明时退出快速通道建立正常 Change。

## 协作处理点

- 缺少审查记录、人工确认或 finding 闭环时，主智能体补齐或明确报告 WARN 状态，不得宣称已按标准流程完成。
- 审查三轮仍未一致时停止自动修改，交 Change 作者决定。
- 必需的基础质量验证（strict validation、build、test）失败时停止并修复。
