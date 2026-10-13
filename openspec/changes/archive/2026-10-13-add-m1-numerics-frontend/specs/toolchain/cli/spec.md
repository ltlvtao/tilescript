# toolchain/cli 变更

## MODIFIED Requirements

### Requirement: 分段实现状态与诚实退出

已实现的全部检查段对某输入零命中时，编译器输出由 `pending_stages` 是否为空裁决，且 MUST NOT 宣称超出已实现范围的编译完成。任一检查段未实现（`pending_stages` 非空）时：stdout MUST 输出单个 JSON 对象，顶层字段为 `status`（值为 `"incomplete"`）、`implemented_stages`（字符串数组，已实现段名）与 `pending_stages`（字符串数组，未实现段名），退出码 MUST 为 `0`，MUST NOT 输出 `passed`、MUST NOT 静默省略未实现段。全部检查段已实现（`pending_stages` 为空）且零命中时：stdout MUST 输出 `passed` 最小形态——单个 JSON 对象，字段集恰为 `status`（值为 `"passed"`），退出码 MUST 为 `0`；该形态为 v1 最小登记，per-kernel 诊断 JSON 挂接字段的扩展由 `diagnostics/json-schema` 的实现 change 经显式 change 承载（本 capability 登记该扩展点，不预先定义其字段名；演进遵循与「拒绝清单 JSON 序列化」一致的冻结规则：v1 已登记字段的删除、改名或语义变更 MUST NOT 发生，新增字段 MAY 经显式 change）。`passed` 输出对同一输入重复运行时 stdout MUST 逐字节一致。段名封闭五值：`syntax`/`type-system`/`primitive-contract`/`execution-structure`/`numerics`，两数组 MUST 恰好合并为该五值且不重叠；本条辖域限定为「已实现检查段零命中」路径：存在任一已实现段拒绝时 `status` 由「拒绝清单 JSON 序列化」定义（`"rejected"`，拒绝优先于 incomplete 与 passed，本条不辖拒绝路径）。`status` 值域 v1 登记为封闭三值 `{rejected, incomplete, passed}`。

#### Scenario: 语法段全过的诚实退出

- **WHEN** 合法 TileScript 模块（全部检查段零命中——本 change 后五段全部实现）
- **THEN** stdout 为单个 JSON 对象 `{"status": "passed"}`（字段集恰为 `status`），退出码为 `0`

#### Scenario: passed 在五段全实现前不可达

- **WHEN** 任一检查段未实现（`pending_stages` 非空——如未来经显式 change 新增检查段后的过渡期）且输入模块零命中
- **THEN** stdout 为 `status="incomplete"`（`implemented_stages`/`pending_stages` 按实现状态列名），退出码 `0`，MUST NOT 输出 `passed`、MUST NOT 静默省略未实现段

#### Scenario: 拒绝优先于 passed

- **WHEN** 五段全部实现状态下，模块存在任一检查段拒绝（如 `E0606`）
- **THEN** `status` 为 `"rejected"` 且退出码为 `1`（拒绝清单输出），MUST NOT 因五段全实现而输出 `passed`
