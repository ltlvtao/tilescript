# toolchain/cli Delta

本 change 对该 capability 的唯一 delta：「分段实现状态与诚实退出」Scenario 1 快照登记性更新（执行结构段接入后 `implemented_stages` 为四段）。正文零变更。

## MODIFIED Requirements

### Requirement: 分段实现状态与诚实退出

已实现的全部检查段对某输入零命中时，编译器 MUST NOT 宣称编译完成：stdout MUST 输出单个 JSON 对象，顶层字段为 `status`（值为 `"incomplete"`）、`implemented_stages`（字符串数组，已实现段名）与 `pending_stages`（字符串数组，未实现段名），退出码 MUST 为 `0`。段名封闭五值：`syntax`/`type-system`/`primitive-contract`/`execution-structure`/`numerics`，两数组 MUST 恰好合并为该五值且不重叠；本条辖域限定为「已实现检查段零命中」路径：该路径上 `pending_stages` 非空时 `status` MUST 为 `"incomplete"`（MUST NOT 输出 `passed`，MUST NOT 静默省略未实现段）；存在任一已实现段拒绝时 `status` 由「拒绝清单 JSON 序列化」定义（`"rejected"`，拒绝优先于 incomplete，本条不辖拒绝路径）。`status` 值域 v1 登记为封闭三值 `{rejected, incomplete, passed}`：`passed`（五段全过后）的输出形态（per-kernel 诊断 JSON 挂接）由 `diagnostics/json-schema` 承载，其 CLI 输出规则属后续段实现 change 的扩展；本 capability 在该扩展前 MUST NOT 产出 `passed`。

#### Scenario: 语法段全过的诚实退出

- **WHEN** 合法 TileScript 模块（全部已实现段零命中——本 change 后已实现段为语法段、类型系统段、原语契约段与执行结构段）
- **THEN** stdout 为 `status="incomplete"`、`implemented_stages=["syntax", "type-system", "primitive-contract", "execution-structure"]`、`pending_stages` 为 `["numerics"]`，退出码 `0`

#### Scenario: passed 在五段全实现前不可达

- **WHEN** 审查本 change 实现的全部输出路径
- **THEN** `status` 只可能为 `"rejected"` 或 `"incomplete"`（`"passed"` 无产出路径——未实现段 MUST NOT 被静默跳过）
