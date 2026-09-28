# language/type-system 规格增量

## MODIFIED Requirements

### Requirement: 类型不匹配拒绝

类型绑定位置——赋值的目标与源、调用的实参与形参、`return` 值与返回注解——两侧类型 MUST 等价，或满足单向兼容（`comptime[int]` 值用于 `int` 位置）。不满足时，编译器 MUST 以 `E0303` 拒绝；报告 MUST 包含行列位置、两侧类型与绑定位置类别，恢复建议按差异组件给出（dtype 不同建议显式 cast 原语、scope 不同建议显式移动原语、shape 不同建议核对维度、种类不同建议核对标量种类）。

本 Requirement 的等价规则适用于赋值、返回绑定与非原语调用的实参绑定（状态类构造调用、执行结构方法与内建调用等）。`tis.*` 原语调用表达式的实参整体不适用等价规则：数据移动原语的实参 scope 组合由「作用域转移矩阵」裁决（`E0301`），全部原语实参的形态与值域约束由该原语在 `primitives/*` 中的参数契约承载（`E04xx`），均 MUST NOT 按本 Requirement 报告 `E0303`。原语的参数契约随各 primitives change 逐个定义，其权威集合以已归档的 `primitives/*` specs 为准。普通赋值语句 MUST NOT 承载跨 scope 数据移动：赋值两侧 scope 不同时按本 Requirement 以 `E0303` 拒绝（类型不等价），跨 scope 移动必须使用显式移动原语。

#### Scenario: dtype 不匹配的赋值被拒绝

- **WHEN** 将 `Tensor[f16, …]` 值赋给 `Tensor[f32, …]` 目标
- **THEN** 编译以 `E0303` 拒绝，报告包含两侧类型与绑定位置类别，恢复建议使用显式转换原语

#### Scenario: scope 不匹配的赋值被拒绝并指向显式原语

- **WHEN** 将 `Tensor[f16, …, Register]` 值赋给 `Tensor[f16, …, Shared]` 目标
- **THEN** 编译以 `E0303` 拒绝，恢复建议使用显式移动原语承载跨 scope 移动

#### Scenario: comptime 值绑定 int 位置被接受

- **WHEN** 以 `comptime[int]` 常量 `64` 为实参调用接受 `int` 形参的函数
- **THEN** 单向兼容规则接受该绑定（不产生 `E0303`）

#### Scenario: 状态构造实参类型不匹配被拒绝

- **WHEN** 以 `Tensor[f16, (BR, BC), Register]` 为 `init` 实参构造期望标量的状态类字段
- **THEN** 编译以 `E0303` 拒绝（状态类构造调用的实参绑定适用等价规则）
