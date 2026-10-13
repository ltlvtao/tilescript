# Proposal: add-m1-numerics-frontend

## Why

M1 前端五检查段已实现四段（syntax 2026-10-10、type-system 2026-10-10、primitive-contract 2026-10-10、execution-structure 2026-10-11，均已归档）。数值语义段（`numerics/value-semantics`，spec 2026-10-08 冻结，`E0601`–`E0606`，7R/39S）是最后一段：实现后检查管线五段闭环，`toolchain/cli` 的 `passed` 状态可达（R3「其 CLI 输出规则属后续段实现 change 的扩展」——本 change 即该扩展的承载 change）。当前全量 467 测试通过、strict 9/9，四段行为稳定，具备收口条件。

## What Changes

- **数值语义段实现面**（`numerics/value-semantics` 6 个行为 Requirement 零 delta 实现）：算术运算家族/dtype/scope/shape 广播裁决（`E0601`–`E0604`）、一元负与具名常量 `inf`（`E0606` 位置绑定）、比较逻辑与布尔条件语境封闭（`E0605`）、数值常量与标量值到 dtype 位置的绑定（`E0606`）、comptime 折叠与 `tis.cdiv` 编译期求值（值域承载）、`tis.cast`/常量绑定数值转换的编译期可判定面（运行期位一致承诺间接承载）。
- **`numerics/value-semantics`「报告契约与 E06xx 段位」登记性 MODIFIED**（补洞，无 BREAKING）：补段间短路 Scenario——前序任一段（语法/类型/原语契约/执行结构）存在拒绝时数值语义段 MUST NOT 执行（与 type-system/memory-ops/execution 三段先例的段间短路补洞同构，4→5 Scenario）。
- **`toolchain/cli`「分段实现状态与诚实退出」MODIFIED**（登记性 + 扩展，无 BREAKING）：五段全过后 `passed` 可达——最小形态 `{"status": "passed"}`（单字段）+ 退出码 `0` + 逐字节一致；per-kernel 诊断 JSON 挂接字段仍为显式延期项（由 `diagnostics/json-schema` 的实现 change 经显式扩展承载，本 change 在 spec 中登记该扩展点）。
- **`diagnostics/json-schema`「输出形态分离与产出条件」登记性 MODIFIED**（无行为变化）：诊断 JSON 产出义务以「诊断实现里程碑」为生效条件显式化——里程碑前通过路径输出为 `toolchain/cli` 最小 `passed` 形态，不产出诊断 JSON 不构成违反；拒绝路径（存在拒绝 MUST NOT 产出）不受里程碑限定。消解本 change 使五段零命中首次可达而诊断产出面尚未实现之间的义务缺口（5→6 Scenario，原 Scenario 同名保留并补里程碑限定语）。
- **管线与 CLI 接入**：`pipeline.IMPLEMENTED_STAGES` 五段（`pending_stages` 恒空）、CLI 通过路径输出 `passed`。

## Impact

- **新增生产代码**：`tilescript/numerics/` 域（数值类型环境 + 各判定面 + 段编排）。
- **修改**：`tilescript/pipeline.py`（第五段接入）、`tilescript/cli.py`（通过路径 `passed` 输出；注释同步）。
- **不改**：`E01xx`–`E05xx` 既有行为、既有 specs 的其余 Requirement、`diagnostics/json-schema` 六段字段集与 schema v1 冻结规则（本 change 仅登记性 MODIFIED 其产出条件的实现里程碑限定，无行为变化）、HAL 能力描述、`tis.*` 公共 API、拒绝清单 JSON 五字段 v1 冻结形态（`rejected` 路径零变化）。
- **测试**：`tests/test_numerics_*.py` 新增（R1–R6 35S 零 delta 实现 + R7 5S 报告契约）；`tests/test_pipeline.py`/`tests/test_cli.py`/`tests/test_integration.py` 联动更新（五段快照、`passed` 输出、段间短路——`passed` 输出断言同时承载 diagnostics delta「诊断实现里程碑前的通过路径输出」Scenario）。
- **无 BREAKING**：本 change 不改任何既有错误码语义、不删不改既有拒绝路径；`incomplete` 输出路径在五段全实现后自然不可达（段名封闭五值不变，未来新段经显式 change 后该语义重新适用）。
