# diagnostics/json-schema 规格增量

## ADDED Requirements

### Requirement: 输出形态分离与产出条件

诊断 JSON 与拒绝清单为两种分离的编译输出形态，MUST NOT 混装：模块通过全部检查（`E01xx`–`E06xx` 零命中）时，编译器 MUST 为模块中每个入口装饰函数（`tis.kernel`/`tis.persistent_kernel`/`tis.fused_kernel`——其识别与数量由 `language/syntax-acceptance-set` 定义，不限于一）各产出一份诊断 JSON 文档（单入口模块即一份）；模块存在任一拒绝时，编译器 MUST NOT 产出任何诊断 JSON（拒绝输出为各段规格定义的拒绝清单，诊断 JSON 不携带拒绝条目，也不存在把拒绝嵌入诊断 JSON 的形态）。诊断 JSON 文档的顶层 MUST 含字段 `schema_version`，其值在本 schema v1 内 MUST 为字符串 `"1.0"`。schema v1 冻结承诺的演进规则为：v1 已登记字段的删除、改名或语义变更 MUST NOT 发生（向后兼容）；新增段或字段 MAY 发生，且 MUST NOT 改变既有字段的语义；消费方对未知字段的忽略策略属工具链约定，非本 capability 义务。诊断 JSON 每个条目级段（`memory`/`async`/`compute`/`portability`/`suggestions`）的每条条目 MUST 含恰好一个源码行定位：直接字段 `src_line`（整数，1 起始的源码行号），或——仅 `async` 段展开操作条目——`origin.src_line`；每条诊断由此可映射回源码行（语言级挂点义务的呈现侧）。

#### Scenario: 编译通过产出诊断 JSON

- **WHEN** 单入口模块通过全部检查（五段错误码零命中）编译
- **THEN** 编译器为其唯一的入口 kernel 产出一份诊断 JSON，顶层 `schema_version` 为 `"1.0"`

#### Scenario: 多入口模块逐 kernel 产出

- **WHEN** 模块含两个入口装饰函数（如 `@tis.kernel` 与 `@tis.persistent_kernel` 各一）且通过全部检查
- **THEN** 编译器产出两份诊断 JSON，各自 `kernel` 字段为对应入口函数名

#### Scenario: 存在拒绝时不产出诊断 JSON

- **WHEN** 模块存在任一 `E01xx`–`E06xx` 拒绝
- **THEN** 编译输出为拒绝清单，任何诊断 JSON 均不产出（无部分产出、无含拒绝条目的诊断 JSON）

#### Scenario: 条目缺失源码行定位违规

- **WHEN** 诊断 JSON 任一条目级段的条目既无直接 `src_line` 字段、又不属带 `origin.src_line` 的展开操作条目
- **THEN** 违反本 capability 契约（该输出不是合法的诊断 JSON）

#### Scenario: v1 既有字段改名违规

- **WHEN** 后续版本把 v1 已登记字段（如 `registers_per_thread`）改名或删除
- **THEN** 违反 schema v1 冻结承诺（破坏向后兼容）

### Requirement: 顶层字段与确定性承诺

诊断 JSON 顶层字段集 v1 冻结为：`schema_version`（值恒 `"1.0"`）、`target`（字符串，编译目标 HAL 的稳定标识符；本份文档描述单一目标的编译——跨目标批量编译时每个目标各自产出其诊断 JSON 文档，批量组织形态属工具链，取值登记表归 HAL 域后续 change）、`kernel`（字符串，本份文档对应的入口 kernel 函数名）、`compile_time_ms`（数值，该 kernel 编译墙钟毫秒数）、`deterministic_hash`（字符串，确定性编译承诺的载体），加上六个段字段（`resources`/`memory`/`async`/`compute`/`portability`/`suggestions`）。非确定字段封闭清单为且仅为 `{compile_time_ms}`（墙钟时间天然非确定）；其余全部字段（含自由文本字段，如 `async` 段 `note`、`portability` 段 `issue`、`suggestions` 段 `msg`）参与确定性承诺：同一源码、同一目标硬件、同一入口 kernel 重复编译时，该 kernel 诊断 JSON 的 `deterministic_hash` 值 MUST 一致，且除 `compile_time_ms` 外逐字段一致（数值相等、字符串逐字符一致、数组条目数与顺序一致）。

#### Scenario: 同输入重复编译逐字段一致

- **WHEN** 同一源码以同一目标硬件对同一入口 kernel 重复编译两次
- **THEN** 两次该 kernel 诊断 JSON 的 `deterministic_hash` 一致，且除 `compile_time_ms` 外逐字段一致

#### Scenario: 非确定字段封闭清单

- **WHEN** 审查诊断 JSON 全部字段
- **THEN** 仅 `compile_time_ms` 允许重复编译时取值不同，其余任何字段取值不同即违反确定性承诺

#### Scenario: 源码或目标不同则 hash 不同

- **WHEN** 源码不同（或目标硬件不同）的两个编译输入各自产出诊断 JSON
- **THEN** 其 `deterministic_hash` 取值不受一致约束（同输入一致为承诺面；异输入取值关系不做承诺）

### Requirement: 字段精度等级与标注义务

诊断 JSON 数值字段的精度等级为封闭三值 `{exact, estimated, unknown}`（字符串与布尔字段不参与数值定级）：`exact` 为可静态精确判定的值；`estimated` 为静态估算值（不承诺与运行期实测一致）；`unknown` 为不可静态判定的值。逐字段的等级归属由各段 Requirement 登记（等级随字段固定，不随编译实例变化）。`estimated` 等级的数值字段 MUST 以名称前缀 `estimated_` 标注（结构性标注，机器可检查；v1 登记为 `async` 段 `estimated_gap_cycles` 与 `estimated_wait_coverage`；`resources.estimated_bound` 为字符串枚举字段，不属数值定级，见「resources 段字段集」）。`unknown` 为值级状态：等级可能为 `unknown` 的字段（v1 登记为 `memory` 段 `bank_conflict`）在不可静态判定时 MUST 取字面值 `"unknown"`，MUST NOT 省略该字段，MUST NOT 以估算值或猜测值代替。量化估算的权威载体为 `estimated_` 前缀数值字段；自由文本字段（`note`/`issue`/`msg`）MUST NOT 呈现任何 `estimated_` 字段未承载的量化估算（MAY 对已承载值作确定性文字说明）。静态诊断为编译期产物，不替代运行期 profiling（`--profile` 属工具层，非本 capability）。

#### Scenario: estimated 字段名前缀标注

- **WHEN** 检查诊断 JSON 的 estimated 等级字段
- **THEN** 其名称含 `estimated_` 前缀（如 `estimated_gap_cycles`），值不承诺与运行期实测一致

#### Scenario: 动态索引 bank conflict 标注 unknown

- **WHEN** 某张量的 bank conflict 因动态索引不可静态判定
- **THEN** 该条目的 `bank_conflict` 字段取字面值 `"unknown"`（不省略、不估算）

#### Scenario: 以猜测值代替 unknown 违规

- **WHEN** 不可静态判定场景下 `bank_conflict` 取了数值而非 `"unknown"`
- **THEN** 违反本 capability 契约（诚实性义务）

### Requirement: resources 段字段集

`resources` 段 MUST 为单个对象，字段集 v1 冻结为：`registers_per_thread`（整数，每线程寄存器数）、`register_spill_bytes`（整数，寄存器溢出字节数）、`shared_memory_bytes`（整数，静态 Shared 分配总字节数）、`occupancy`（对象：`active_warps_per_sm` 整数、`limiter` 字符串——占用率受限因素）、`estimated_bound`（字符串，取值封闭二值 `"compute"` 或 `"memory"`，静态估算的瓶颈方向）。精度等级：`registers_per_thread`、`register_spill_bytes`、`shared_memory_bytes` 与 `occupancy`（理论值）为 `exact`；`estimated_bound` 为字符串枚举字段、不参与数值精度定级，其名称保留 `estimated_` 前缀作为语义标注（值为静态粗估的方向提示，不承担性能预测，不构成对运行期表现的承诺）。

#### Scenario: 完整 resources 段

- **WHEN** 编译通过产出诊断 JSON
- **THEN** `resources` 段含全部五个字段，`estimated_bound` 取 `"compute"` 或 `"memory"` 二值之一

#### Scenario: 无 Shared 分配的 kernel

- **WHEN** kernel 无任何静态 Shared 分配
- **THEN** `shared_memory_bytes` 为 `0`（字段恒在，不省略）

### Requirement: memory 段字段集

`memory` 段 MUST 为数组，每条条目对应一次 Shared 静态分配（`tis.alloc_shared` 的产物——`bank_conflict` 分析以 Shared 存储为对象；Register 分配与视图构造（`zeros`/`make_tensor`）不入本段），字段集 v1 冻结为：`tensor`（字符串，张量绑定名）、`src_line`（整数，分配位置源码行）、`layout`（字符串，布局属性确定性描述，如 swizzle 掩码；同一输入重复编译逐字符一致）、`bank_conflict`（对象 `{read_way: 整数, write_way: 整数, worst_access_line: 整数}`，或字面值 `"unknown"`）。精度等级：`bank_conflict` 在静态可判定场景（已知布局与访问模式）为 `exact`；动态索引等不可静态判定场景 MUST 取 `"unknown"`（「字段精度等级与标注义务」的值级状态）。kernel 无 Shared 静态分配时 `memory` 段为空数组（段恒在）。

#### Scenario: swizzled 分配的 bank conflict 报告

- **WHEN** kernel 含 `tis.alloc_shared` 分配且访问模式静态可判定
- **THEN** 对应条目含 `tensor`/`src_line`/`layout`/`bank_conflict` 对象（read/write way 与最差访问行）

#### Scenario: 动态索引张量条目

- **WHEN** 某静态分配张量的访问含动态索引
- **THEN** 该条目 `bank_conflict` 为 `"unknown"`，其余字段照常报告

#### Scenario: 无静态分配

- **WHEN** kernel 无任何 Shared 静态分配
- **THEN** `memory` 段为空数组（不省略段）

### Requirement: async 段字段集与展开溯源

`async` 段 MUST 为数组，条目为两类（v1 冻结）。**pipeline 汇总条目**：`pipeline`（字符串，pipeline 绑定名）、`src_line`（整数，Pipeline 构造位置源码行）、`stages`（整数，级数）、`estimated_gap_cycles`（整数，估算空转周期）、`estimated_wait_coverage`（数值 0–1，估算覆盖率）、`note`（字符串，确定性文字说明）。**展开操作条目**（承接 `execution/pipeline-structure`「Async 完成保证与组管理不可见」的诊断字段形式让渡——编译器插入的每条底层操作 MUST 可溯源）：`op`（字符串，取值封闭二值 `"async_copy"` 或 `"wait"`）、`origin`（对象：`pipeline` 字符串、`stage` 字符串、`iteration_class` 取值封闭三值 `"prologue"`/`"steady"`/`"epilogue"`、`src_line` 整数——展开生成来源的源码行）。精度等级：`stages` 为 `exact`；`estimated_gap_cycles` 与 `estimated_wait_coverage` 为 `estimated`。kernel 无 Pipeline 时 `async` 段为空数组（段恒在）。

#### Scenario: pipeline 汇总条目

- **WHEN** kernel 含一个 `tis.Pipeline` 构造并 run
- **THEN** `async` 段含该 pipeline 的汇总条目（六字段齐全——`src_line` 定位 Pipeline 构造行，两个 estimated 字段带前缀）

#### Scenario: 展开操作条目溯源

- **WHEN** 编译器为 pipeline 展开插入异步拷贝与等待操作
- **THEN** 每条操作有条目：`op` 为 `"async_copy"` 或 `"wait"`，`origin` 四字段齐全且 `src_line` 定位生成来源源码行

#### Scenario: 无 pipeline 的 kernel

- **WHEN** kernel 不使用 Pipeline
- **THEN** `async` 段为空数组（不省略段）

### Requirement: compute 段字段集

`compute` 段 MUST 为数组，每条条目对应一次计算原语调用的 lowering，字段集 v1 冻结为两类。**通用必带字段**（全部六原语条目）：`op`（字符串，取值为 `primitives/compute-ops` 六原语名封闭集 `dot`/`reduce`/`maximum`/`exp`/`log`/`transpose`）、`src_line`（整数，调用位置源码行）、`instruction_count`（整数，`exact`）。**dot 专属字段**（`op` 为 `dot` 的条目必带）：`mma_shape`（字符串，实际生效的 MMA 形状确定性描述——显式实参值或 `Auto` 确定性选择结果，承接 `primitives/compute-ops` 的选择呈现让渡）、`tensor_core`（布尔）、`pad_policy`（字符串，取值封闭四值 `"Error"`/`"PadZero"`/`"Mask"`/`"Split"`）；策略关联字段按 `pad_policy` 取值条件必带：`"Mask"` 时 `tail_masked`（布尔）、`"PadZero"` 时 `pad_waste_ratio`（数值，`exact`——由 tile 形状与 MMA 形状静态可计算的浪费比例）、`"Split"` 时 `tail_fma_ratio`（数值，`exact`——尾部 FMA 占比）。精度等级除上述登记外无其他数值字段。

#### Scenario: dot 显式 mma 的 compute 条目

- **WHEN** 源码含 `tis.dot(A, B, C, mma=tis.MMA(16, 8, 16), pad=tis.PadPolicy.Error)`
- **THEN** `compute` 段对应条目：`mma_shape` 呈现该显式形状、`pad_policy` 为 `"Error"`、无策略关联字段（三条件字段均不适用）

#### Scenario: Auto 选择与 Mask 策略条目

- **WHEN** 源码 dot 省略 `mma`（取 `Auto`）且 `pad=tis.PadPolicy.Mask`
- **THEN** 条目 `mma_shape` 呈现该 HAL 目标确定性选择的形状（与编译文档一致），`pad_policy` 为 `"Mask"` 且 `tail_masked` 必带

#### Scenario: 非 dot 原语条目

- **WHEN** 源码含 `tis.exp(x)` 调用
- **THEN** 对应条目带 `op`/`src_line`/`instruction_count` 三通用字段，无 dot 专属字段

### Requirement: portability 段与 suggestions 建议域

**portability 段** MUST 为数组：当编译器对某 HAL 目标编译时静态判定同一源码在其他 HAL 目标上存在行为差异（如某维对齐要求不同、Placement 建议不同），`portability` 段 MUST 含对应条目；具体差异类目登记与 severity 赋值规则归 portability/HAL 域后续 change。每条条目字段集 v1 冻结为：`issue`（字符串，跨目标差异的确定性描述——同一源码在其他 HAL 目标上的行为差异提示）、`src_line`（整数）、`severity`（字符串，取值封闭二值 `"info"` 或 `"warning"`；不存在 `"error"`——错误属拒绝域，拒绝时不产出诊断 JSON）。portability 提示为建议性信息：提示 MUST NOT 改变编译行为（如 MUST NOT 自动替用户选择 `Placement.Fast`——建议由诊断给出，决策由用户/AI 做）。**suggestions 段**为数组，每条条目字段集 v1 冻结为：`code`（字符串，格式为字面 `S` 加四位十进制数字，如 `"S0201"`；同一诊断 JSON 内唯一）、`src_line`（整数）、`msg`（字符串，确定性建议文字）。建议非强制：编译器 MUST NOT 因建议未被采纳而拒绝编译或改变编译行为。两段在无内容时为空数组（段恒在）。

#### Scenario: 跨目标对齐差异报告

- **WHEN** 编译器判定源码在某维对齐要求上于两个 HAL 目标不同
- **THEN** `portability` 段含对应条目（`issue` 描述差异、`src_line` 定位、`severity` 取二值之一）

#### Scenario: portability 提示不改变编译行为

- **WHEN** portability 条目建议某 tensor 在某目标用 `Placement.Fast`
- **THEN** 编译行为不因该提示改变（源码未显式指定则不自动指定）

#### Scenario: 建议条目格式与唯一性

- **WHEN** 诊断 JSON 含多条建议
- **THEN** 每条 `code` 匹配 `S` 加四位十进制数字且互不重复，`src_line` 与 `msg` 齐全

#### Scenario: 无建议与无差异

- **WHEN** 编译通过且无跨目标差异、无建议
- **THEN** `portability` 与 `suggestions` 段均为空数组（段不省略）
