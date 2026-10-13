# diagnostics/json-schema 变更

## MODIFIED Requirements

### Requirement: 输出形态分离与产出条件

诊断 JSON 与拒绝清单为两种分离的编译输出形态，MUST NOT 混装。诊断 JSON 的产出义务以**诊断实现里程碑**为生效条件：在本 capability 的实现 change（诊断 JSON 检查与产出面的承载 change）归档之前，通过全部检查的编译输出为 `toolchain/cli`「分段实现状态与诚实退出」登记的通过路径输出（最小 `passed` 形态），不产出诊断 JSON 且不构成对本 capability 的违反；自诊断实现里程碑起，模块通过全部检查（`E01xx`–`E06xx` 零命中）时，编译器 MUST 为模块中每个入口装饰函数（`tis.kernel`/`tis.persistent_kernel`/`tis.fused_kernel`——其识别与数量由 `language/syntax-acceptance-set` 定义，不限于一）各产出一份诊断 JSON 文档（单入口模块即一份）；模块存在任一拒绝时，编译器 MUST NOT 产出任何诊断 JSON（拒绝输出为各段规格定义的拒绝清单，诊断 JSON 不携带拒绝条目，也不存在把拒绝嵌入诊断 JSON 的形态——本条拒绝路径不受实现里程碑限定，自本 capability 任何部分实现起即适用）。诊断 JSON 文档的顶层 MUST 含字段 `schema_version`，其值在本 schema v1 内 MUST 为字符串 `"1.0"`。schema v1 冻结承诺的演进规则为：v1 已登记字段的删除、改名或语义变更 MUST NOT 发生（向后兼容）；新增段或字段 MAY 发生，且 MUST NOT 改变既有字段的语义；消费方对未知字段的忽略策略属工具链约定，非本 capability 义务。诊断 JSON 每个条目级段（`memory`/`async`/`compute`/`portability`/`suggestions`）的每条条目 MUST 含恰好一个源码行定位：直接字段 `src_line`（整数，1 起始的源码行号），或——仅 `async` 段展开操作条目——`origin.src_line`；每条诊断由此可映射回源码行（语言级挂点义务的呈现侧）。

#### Scenario: 编译通过产出诊断 JSON

- **WHEN** 单入口模块通过全部检查（五段错误码零命中）编译，且诊断实现里程碑已到（本 capability 实现 change 已归档）
- **THEN** 编译器为其唯一的入口 kernel 产出一份诊断 JSON，顶层 `schema_version` 为 `"1.0"`

#### Scenario: 多入口模块逐 kernel 产出

- **WHEN** 模块含两个入口装饰函数（如 `@tis.kernel` 与 `@tis.persistent_kernel` 各一）且通过全部检查，诊断实现里程碑已到
- **THEN** 编译器产出两份诊断 JSON，各自 `kernel` 字段为对应入口函数名

#### Scenario: 存在拒绝时不产出诊断 JSON

- **WHEN** 模块存在任一 `E01xx`–`E06xx` 拒绝
- **THEN** 编译输出为拒绝清单，任何诊断 JSON 均不产出（无部分产出、无含拒绝条目的诊断 JSON）

#### Scenario: 诊断实现里程碑前的通过路径输出

- **WHEN** 模块通过全部检查（五段零命中），诊断实现里程碑未到（诊断 JSON 产出面尚未实现）
- **THEN** 编译输出为 `toolchain/cli` 登记的最小 `passed` 形态（`{"status": "passed"}`），不产出诊断 JSON 且不违反本 capability（诊断 JSON 产出义务未生效）

#### Scenario: 条目缺失源码行定位违规

- **WHEN** 诊断 JSON 任一条目级段的条目既无直接 `src_line` 字段、又不属带 `origin.src_line` 的展开操作条目
- **THEN** 违反本 capability 契约（该输出不是合法的诊断 JSON）

#### Scenario: v1 既有字段改名违规

- **WHEN** 后续版本把 v1 已登记字段（如 `registers_per_thread`）改名或删除
- **THEN** 违反 schema v1 冻结承诺（破坏向后兼容）
