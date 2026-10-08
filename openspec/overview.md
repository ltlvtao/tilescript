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

- `language/syntax-acceptance-set`（2026-09-24，change `add-language-syntax-acceptance-set`）：M1 前端语法接受集——Python 3.10 载体（`tis.` 前缀、`.tis` 扩展名）、顶层/装饰器/签名/语句/表达式白名单与拒绝清单（`E0101`–`E0107`）、拒绝报告四要素契约。
- `language/type-system`（2026-09-28，change `add-language-type-system`）：类型系统行为契约——类型表达式结构与 dtype/scope 封闭集合（`E0302`）、三类标量种类与 shape 组件、类型等价（逐组件/符号维度/派生维度结构等价）、3×3 作用域转移矩阵（`E0301`，历史码语义不变）、状态类类型与字段约束（nominal 等价、`E0304`）、表达式结果类型规则（单类型不变量、切片/广播维变换、primitives/执行结构/数值语义显式让渡）、类型不匹配（`E0303`，适用集为带源码层形参注解的调用——豁免边界经两次扩展：`tis.*` 原语调用实参，`add-primitives-memory-ops` 于 2026-09-28；执行结构调用（`pipe.run`/`range` 等）实参，`add-execution-pipeline-structure` 于 2026-09-30）、E03xx 报告契约与段内 tiebreak。
- `primitives/memory-ops`（2026-09-28，change `add-primitives-memory-ops`）：核心存取原语行为契约——八原语（load/store/barrier/make_tensor/alloc_shared/zeros/full/cast）参数集与封闭性（`E0406`）、转移格承载映射（load 三格/store 三格/copy+move 无承载，`E0404`）、逐维长度相容判定（`E0405`，静态折叠支）、Sync 完成/Async 仅 Pipeline 语境（`E0407`）、分配与视图构造（swizzled 为分配物理属性不进类型组件）、cast 全六 dtype 开放、barrier Block 汇合可见性、E04xx 报告契约与段内 tiebreak；含 `language/type-system`「类型不匹配拒绝」BREAKING 修订（E0303 豁免边界扩展）。
- `execution/pipeline-structure`（2026-09-30，change `add-execution-pipeline-structure`）：执行结构行为契约——Pipeline 构造与 Pipeline 值/buffer 对象/range 可迭代值的合法使用位置封闭集（`E0502`/`E0503`）、produce/consume 嵌套函数签名与调用封闭、run 调用契约与迭代状态链（含空迭代）、buffer 命名空间类型推导、Async 完成保证与组管理不可见、warp_group 上下文契约（`E0501` 正式化为语境违规）、索引与整数内建（block_idx/cdiv/range，`E0505`）、E05xx 报告契约与段内 tiebreak；含 `language/type-system`「类型不匹配拒绝」BREAKING 修订（E0303 豁免边界第二次扩展至执行结构调用）。
- `numerics/value-semantics`（2026-10-08，change `add-numerics-value-semantics`）：数值语义行为契约——算术运算操作数与结果规则（int 家族种类推导、dtype 家族无隐式提升（`E0602`）、混家族拒绝含 int 编译期常量（`E0601`）、Tensor scope 一致与逐维广播（`E0603`）、`/` 浮点专属（`E0604`）、增强赋值等价展开）、一元负与数学具名常量 `inf`（IEEE 754 标准精度三 dtype；f8e4m3/整型/裸绑定 `E0606`；nan 不引入）、布尔值条件语境专用封闭集（`E0605`，不进类型世界、True/False 默认值取 1/0）、数值常量到 dtype 位置绑定（封闭枚举：float 字面量正确舍入、int 编译期常量限原语参数位、运行期 int 拒绝、dtype 标量须同 dtype）、comptime 折叠与 cdiv 编译期求值（ceil(a/b)）、cast 数值效果（家族规则矩阵 + OCP E4M3 特则 + 跨硬件位一致承诺）、E06xx 报告契约与段内 tiebreak（跨段管线自此五段）。纯 ADDED：承接 type-system/memory-ops/execution 五处让渡句。
