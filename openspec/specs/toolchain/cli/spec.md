# toolchain/cli Specification

## Purpose

TileScript 工具链命令行域的行为契约：编译入口形态与目标参数校验（工具链入口拒绝：stderr + 退出码 2，不触源文件、非 E0xxx）、源码锚定拒绝的机器可读 JSON 序列化（五字段冻结、逐字节一致、退出码 1），以及诚实退出承诺——未实现检查段以 `incomplete` 状态显式列名（退出码 0），`passed` 状态在全部检查段实现前不可达。`passed` 输出形态、`--profile` 类调优参数与批量编译为显式延期项，由后续 change 承载。

## Requirements

### Requirement: 编译入口与目标参数

编译器命令行入口 MUST 为 `python -m tilescript compile <源文件路径> --target <目标标识符>`（子命令与参数形态的扩展属后续 change，扩展 MUST 保持本条已登记形态的兼容）。`--target` MUST 为必选参数，其合法取值为且仅为 `hal/capability-descriptions` 登记表成员；调用方未提供 `--target`、或提供未登记标识符时，编译器 MUST 以工具链入口拒绝终止：stderr 输出一行拒绝说明（含登记表合法取值清单）、退出码 `2`、MUST NOT 读取或解析源文件、MUST NOT 产生任何 `E01xx`–`E06xx` 源码锚定拒绝（承接 `hal/capability-descriptions`「未登记标识符不构成合法编译输入」）。调用缺少源文件路径位置参数时 MUST 同样以工具链入口拒绝（stderr 一行说明缺少源文件路径、退出码 `2`；登记表清单说明仅辖 `--target` 相关拒绝）。源文件路径不存在或不可读时 MUST 同样以工具链入口拒绝（stderr 说明 + 退出码 `2`）。源文件扩展名不是 `.tis` 时 MUST NOT 拒绝（`language/syntax-acceptance-set`：扩展名为命名约定）。

#### Scenario: 未提供 target 被工具链入口拒绝

- **WHEN** 调用 `python -m tilescript compile kernel.tis`（缺 `--target`）
- **THEN** stderr 输出拒绝说明（含合法取值清单）、退出码为 `2`，源文件不被读取，无 `E0xxx` 输出

#### Scenario: 未登记标识符被工具链入口拒绝

- **WHEN** 调用 `python -m tilescript compile kernel.tis --target foo_bar`
- **THEN** stderr 输出拒绝说明（列出 `nvidia_h200`/`ascend_910b`）、退出码为 `2`，不产生源码锚定拒绝

#### Scenario: 源文件不存在被工具链入口拒绝

- **WHEN** 调用 `python -m tilescript compile no_such_file.tis --target nvidia_h200`
- **THEN** stderr 说明文件不可读、退出码为 `2`

#### Scenario: 缺少源文件路径参数被工具链入口拒绝

- **WHEN** 调用 `python -m tilescript compile --target nvidia_h200`（无位置参数）
- **THEN** stderr 一行说明缺少源文件路径、退出码为 `2`，不产生 `E0xxx` 输出

#### Scenario: 非 .tis 扩展名不拒绝

- **WHEN** 以 `kernel.py` 扩展名的合法 TileScript 源文件编译
- **THEN** 不因扩展名产生拒绝，照常进入语法检查

### Requirement: 拒绝清单 JSON 序列化

任一已实现检查段存在拒绝时，编译器 MUST 在 stdout 输出单个 JSON 对象：顶层字段 `status`（字符串，值为 `"rejected"`）与 `rejections`（数组）。`rejections` 每条条目的字段集 v1 冻结为：`code`（字符串，形如 `"E0105"`）、`line`（整数，1 起始）、`col`（整数，1 起始）、`category`（字符串，被拒绝类别的稳定机器可读标识——小写 ASCII 与连字符构成的 slug，如 `while-loop`、`list-comprehension`；slug 值域随实现登记，同一拒绝类别重复编译 MUST 取值一致）、`suggestion`（字符串，人类可读恢复建议，内容由所命中错误码的段规格定义）。条目顺序 MUST 为对应段规格定义的拒绝清单顺序（收集全部、位置升序、同位置 tiebreak），序列化 MUST NOT 改变该顺序。存在拒绝时退出码 MUST 为 `1`。字段集演进规则：v1 已登记字段的删除、改名或语义变更 MUST NOT 发生，新增字段 MAY 经显式 change。同一输入重复运行时 stdout 输出 MUST 逐字节一致（段规格确定性承诺的序列化面）。

#### Scenario: 多条拒绝的 JSON 输出与排序

- **WHEN** 模块第 12 行含 `while`、第 30 行含列表推导（两条语法拒绝）
- **THEN** stdout 为单个 JSON 对象（`status` 为 `"rejected"`、`rejections` 恰好两条），每条含 `code`/`line`/`col`/`category`/`suggestion` 五字段，顺序为第 12 行在前；退出码为 `1`

#### Scenario: 重复运行逐字节一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次 stdout 的 JSON 输出逐字节一致（含 `category` slug 与 `suggestion` 文本）

#### Scenario: v1 字段改名违规

- **WHEN** 后续版本把 `suggestion` 字段改名或删除
- **THEN** 违反本 capability 契约（破坏 v1 冻结演进规则）

### Requirement: 分段实现状态与诚实退出

已实现的全部检查段对某输入零命中时，编译器 MUST NOT 宣称编译完成：stdout MUST 输出单个 JSON 对象，顶层字段为 `status`（值为 `"incomplete"`）、`implemented_stages`（字符串数组，已实现段名）与 `pending_stages`（字符串数组，未实现段名），退出码 MUST 为 `0`。段名封闭五值：`syntax`/`type-system`/`primitive-contract`/`execution-structure`/`numerics`，两数组 MUST 恰好合并为该五值且不重叠；本条辖域限定为「已实现检查段零命中」路径：该路径上 `pending_stages` 非空时 `status` MUST 为 `"incomplete"`（MUST NOT 输出 `passed`，MUST NOT 静默省略未实现段）；存在任一已实现段拒绝时 `status` 由「拒绝清单 JSON 序列化」定义（`"rejected"`，拒绝优先于 incomplete，本条不辖拒绝路径）。`status` 值域 v1 登记为封闭三值 `{rejected, incomplete, passed}`：`passed`（五段全过后）的输出形态（per-kernel 诊断 JSON 挂接）由 `diagnostics/json-schema` 承载，其 CLI 输出规则属后续段实现 change 的扩展；本 capability 在该扩展前 MUST NOT 产出 `passed`。

#### Scenario: 语法段全过的诚实退出

- **WHEN** 合法 TileScript 模块（全部已实现段零命中——本 change 后已实现段为语法段、类型系统段与原语契约段）
- **THEN** stdout 为 `status="incomplete"`、`implemented_stages=["syntax", "type-system", "primitive-contract"]`、`pending_stages` 含其余两段名，退出码 `0`

#### Scenario: passed 在五段全实现前不可达

- **WHEN** 审查本 change 实现的全部输出路径
- **THEN** `status` 只可能为 `"rejected"` 或 `"incomplete"`（`"passed"` 无产出路径——未实现段 MUST NOT 被静默跳过）
