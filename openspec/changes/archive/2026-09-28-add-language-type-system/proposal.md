# Proposal

## Why

TileScript 的类型规则目前只存在于 `veps/design.md` §2.2 的文法定义与作用域转移矩阵中，没有任何权威规格。M1 验收要求"5 个 kernel 源码全部通过类型检查；故意注入的 20 个作用域错误全部被捕获"（§5.2），类型检查器是 M1 的核心交付之一——没有先行规格，类型边界将由实现隐式决定。已归档的 `language/syntax-acceptance-set` 只约束语法类别：`E0104` 只裁决"注解属于六类形式之一"，不裁决类型内部结构（`Tensor[...]` 内部参数是什么、shape 组件的类型规则、两个类型何时等价、哪些 scope 组合合法）。设计原则第 4 条（"类型系统承担安全，不承担性能"）与最早定义的错误码 `E0301` 都属于这一层，需要行为契约固化。

## What Changes

- 新增 capability `language/type-system`，定义类型系统的行为契约：
  - 类型表达式结构规则：dtype 封闭集合（6 种）、scope 封闭集合（3 域）、`Tensor`/`Pointer` 参数结构、`layout` 可选参数（`E0302`）
  - 标量种类全集：运行期 `int`、运行期 dtype 标量（6 种）、编译期 `comptime[int]`；shape 组件类型规则；comptime 单向兼容（反向 `E0303`）
  - 类型等价规则：逐组件等价、动态维度符号等价、派生维度结构等价、标量种类等价、状态类名义等价
  - 作用域转移矩阵：3×3 全格定义，非法组合以 `E0301` 拒绝（历史码，语义不变）
  - 状态类类型与字段约束：字段仅 `Tensor[dtype, shape, Register]`（`E0304`）；状态类类型的名义等价、构造绑定与可用位置
  - 表达式结果类型规则：变量单类型不变量、下标/切片/广播的维变换与派生、常量种类
  - 类型不匹配：赋值、实参、返回位置的绑定规则（`E0303`）；数据移动原语豁免（组合走 `E0301`）
  - `E03xx` 报告契约：四要素、收集式、跨段与段内 tiebreak、确定性；段位 `E0300`–`E0399` 分配
- **BREAKING**：无（新 capability；`E0301` 从 `veps/design.md` 事实升格为规格，无存量实现与受众）。

## Capabilities

### New Capabilities

- `language/type-system`：类型表达式结构、dtype/scope 封闭集合、类型等价、作用域转移矩阵（`E0301`）、状态类字段约束与类型不匹配的报告行为。

## 非目标

- 原语逐个的参数/返回类型契约（`primitives/*` 各 change 承载；本 change 只定义通用绑定规则与数据移动原语的豁免边界）。
- 模块属性（`tis.*`）、Pipeline buffer 属性（如 `buf.K`）与调用表达式的结果类型（由 `primitives/*` 与执行结构 capability 契约定义）。
- 数值类型提升与混合精度算术语义（数值语义 capability；如 `f32 + f16` 的结果类型、float 字面量到 dtype 标量位置的绑定、dtype 标量间转换）。
- `tis.cast` 的合法转换集（`primitives/*`）。
- `Distributed` scope：`veps/design.md` §2.2 引用"见 §2.6"，但 §2.6 未给出定义（设计悬空）；待设计补充后另立 change。
- 类型推断算法与 TSR 的实现策略（实现 change 承载；本 change 只定义推断结果必须满足的等价/绑定规则）。
- `copy`/`move` 操作的原语级承载：转移矩阵只定义类型层合法性与操作类别名；`tis.load`/`tis.store` 各承载哪些格子由 `primitives/*` 定义。
- 诊断 JSON Schema（`diagnostics/*`）：`E03xx` 报告要素沿用语法接受集的四要素惯例，整体 schema 延期。
- 布尔/枚举类型：`True`/`False`/`Sync`/`Async` 等在类型层未定义独立种类，作为原语参数值由 `primitives/*` 约束。

## Impact

- 为 M1 类型检查与作用域检查提供验收依据（含"20 个注入错误全部被捕获"的类型层来源）。
- 与 `language/syntax-acceptance-set` 的边界：语法层（`E0104`）裁决注解的**形式类别**，本 change 裁决类型表达式的**内部结构**与绑定语义；同一注解先过语法层再过类型层。
- `veps/design.md` §2.2 的文法与矩阵从本 change 起获得权威依据；示例与规格冲突时以规格为准。
- 无生产代码影响（纯规格先行 change）。
