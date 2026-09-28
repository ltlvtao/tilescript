# language/type-system Specification

## Purpose
定义 TileScript 类型系统的行为契约：类型表达式的内部结构与 dtype/scope 封闭集合、标量种类与 shape 组件规则、类型等价判定、设备代码作用域转移矩阵、状态类类型与字段约束、表达式结果类型规则、类型绑定位置的等价拒绝，以及 E03xx 诊断的报告契约与错误码段位。本 capability 承担安全不承担性能（设计原则 4）：与 `language/syntax-acceptance-set` 的分工是语法层裁决注解的形式类别（E0104 等），本层裁决类型内部结构与绑定语义；原语逐个的参数/返回类型契约（`primitives/*`）、执行结构与内建调用结果类型（执行结构 capability）、数值提升与字面量绑定（数值语义 capability）显式让渡至后续 capability。

## Requirements

### Requirement: 类型表达式结构规则

类型注解——无论出现在 kernel 签名（语法层经 `E0104` 检查形式类别）、`@tis.state` 字段（经 `E0107`）还是 Pipeline `produce`/`consume` 嵌套函数签名（语法层不检查注解类别）——其内部结构 MUST 满足：dtype 属于封闭集合 `f16`、`bf16`、`f32`、`f8e4m3`、`i8`、`i32`（六种，MUST NOT 出现自定义 dtype）；scope 属于封闭集合 `Global`、`Shared`、`Register`（三种）。`Tensor[dtype, shape, scope]` MUST 按该顺序携带恰好三个必选参数，MAY 携带可选第四参数 `layout`，`layout` 的唯一合法值为 `RowMajor`（默认值，固定且写进文档；未来扩展只能经显式 change 修改本 Requirement）。`Pointer[dtype, scope]` MUST 按该顺序携带恰好两个参数。类型注解结构不满足上述规则（参数数量、顺序错误，未知 dtype，未知 scope）时，编译器 MUST 以 `E0302` 拒绝；报告 MUST 包含行列位置、被违反的结构规则与恢复建议（给出最近的合法形式）。

#### Scenario: 合法类型表达式通过结构检查

- **WHEN** 注解为 `Tensor[f32, (BR, D), Register]`、`Pointer[f16, Global]` 或 `Tensor[f16, (seq_len, D), Global, RowMajor]`
- **THEN** 结构检查通过（dtype 与 scope 均在封闭集合内，参数顺序与数量正确）

#### Scenario: 未知 dtype 被拒绝

- **WHEN** 注解为 `Tensor[f64, (64, 64), Register]`
- **THEN** 编译以 `E0302` 拒绝，报告注明 `f64` 不在 dtype 封闭集合并列出六种合法 dtype

#### Scenario: 参数顺序错误被拒绝

- **WHEN** 注解为 `Tensor[Register, (64,), f32]`（scope 与 dtype 位置互换）
- **THEN** 编译以 `E0302` 拒绝，报告注明第一参数必须是 dtype 并给出正确形式

#### Scenario: 集合外 scope 被拒绝

- **WHEN** 注解使用 `Distributed` 作为 scope（如 `Tensor[f32, (64,), Distributed]`）
- **THEN** 编译以 `E0302` 拒绝，报告注明 scope 封闭集合为 Global/Shared/Register 三种

### Requirement: 标量类型与 shape 组件规则

类型系统提供三类标量种类：运行期整数标量 `int`（值在 kernel 启动时确定）；运行期 dtype 标量——六种 dtype 名各自可作为标量类型注解（值为宿主侧该 dtype 的标量，如 `scale: f32`）；编译期常量 `comptime[int]`（值在编译期确定）。

`comptime[int]` 与 `int` 的绑定规则：`comptime[int]` 的值 MAY 绑定到接受 `int` 标量的位置（单向兼容）；`int` 的值 MUST NOT 绑定到要求 `comptime[int]` 的位置，违反时编译器 MUST 以 `E0303` 拒绝（类型不匹配），恢复建议为改用整数字面量、`comptime` 参数或编译期可求值表达式。dtype 标量位置只接受同 dtype 标量种类的值；dtype 标量之间的转换与数值字面量到 dtype 标量位置的绑定规则由数值语义 capability 承载（本 capability 不定义）。

Tensor 类型 shape 的每个维度组件 MUST 是 `comptime[int]` 常量表达式或运行期 `int` 标量表达式（动态维度合法）；shape 组件出现浮点或其他非整数类型时，编译器 MUST 以 `E0302` 拒绝，恢复建议说明 shape 组件仅接受整数常量或整数标量。

#### Scenario: 动态维度 shape 被接受

- **WHEN** kernel 以 `seq_len: int` 为参数，某 Tensor 类型的 shape 维度组件为该运行期标量（如 `(seq_len, D)`）
- **THEN** shape 组件规则检查通过（动态维度合法）

#### Scenario: 浮点 shape 组件被拒绝

- **WHEN** shape 组件出现浮点字面量（如 `(2.5, D)`）
- **THEN** 编译以 `E0302` 拒绝，恢复建议说明 shape 组件仅接受整数常量或整数标量

#### Scenario: 运行期值用于 comptime 位置被拒绝

- **WHEN** 运行期 `int` 值被绑定到要求 `comptime[int]` 的位置
- **THEN** 编译以 `E0303` 拒绝（类型不匹配），恢复建议为改用整数字面量或编译期可求值表达式

#### Scenario: dtype 标量注解被接受

- **WHEN** kernel 签名参数注解为 `scale: f32`
- **THEN** 标量种类检查通过（`f32` 是合法的运行期 dtype 标量种类）

### Requirement: 类型等价规则

两个类型等价当且仅当逐组件等价：`Tensor` 的 dtype 相等、shape 逐维等价、scope 相等、layout 相等（默认 `RowMajor` 与显式 `RowMajor` 等价）；`Pointer` 的 dtype 与 scope 相等。标量种类等价：`int` 与 `int` 等价、同 dtype 的两个 dtype 标量等价、`comptime[int]` 与 `comptime[int]` 等价；不同种类之间、不同 dtype 的 dtype 标量之间 MUST NOT 判定等价（`comptime[int]` 值到 `int` 位置的单向兼容除外，见「标量类型与 shape 组件规则」）。状态类类型的等价见「状态类类型与字段约束」。

shape 维度等价判定：两边同为编译期常量且值相等；或两边为同一运行期符号；或两边为结构等价的派生维度（派生规则见「表达式结果类型规则」）。

#### Scenario: 逐组件相等判定等价

- **WHEN** 赋值两侧分别为 `Tensor[f32, (64, 64), Register]` 与 `Tensor[f32, (64, 64), Register, RowMajor]`
- **THEN** 两类型等价（显式默认 layout 与省略等价）

#### Scenario: 同符号动态维度判定等价

- **WHEN** 赋值两侧 Tensor 的某维度均为同一运行期符号 `seq_len`
- **THEN** 该维度等价，类型等价检查通过

#### Scenario: 常量与运行期符号判定不等价

- **WHEN** 一侧 shape 维度为编译期常量 `64`、另一侧为运行期符号 `seq_len`
- **THEN** 该维度不等价，类型等价检查失败

#### Scenario: 不同 dtype 标量判定不等价

- **WHEN** 绑定一侧为 `f32` 标量、另一侧为 `f16` 标量
- **THEN** 类型等价检查失败（不同 dtype 的 dtype 标量不等价）

### Requirement: 作用域转移矩阵

设备代码中任何数据移动操作的源 scope 与目标 scope 组合 MUST 落在合法转移矩阵内（行=源 scope，列=目标 scope）：

| 从 \ 到 | Global | Shared | Register |
|---|---|---|---|
| Global | 非法 | load | load |
| Shared | store | copy | load |
| Register | store | store | move |

标记为非法的组合（`Global → Global` 不提供同域复制操作）出现时，编译器 MUST 以 `E0301` 拒绝；报告 MUST 包含四要素，恢复建议列出该源 scope 的全部合法目标 scope 及对应操作类别。`E0301` 只裁决 scope 组合的合法性；承载各格子的具体原语（`tis.load`/`tis.store` 等）的参数契约由 `primitives/*` 定义。

#### Scenario: Global 到 Shared 的 load 合法

- **WHEN** 数据移动操作的源为 Global scope Tensor、目标为 Shared scope Tensor（如 `tis.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)`）
- **THEN** 组合落在矩阵 `Global→Shared`（load）格，转移检查通过

#### Scenario: Global 到 Global 被拒绝

- **WHEN** 数据移动操作的源与目标均为 Global scope
- **THEN** 编译以 `E0301` 拒绝（矩阵标"非法"），恢复建议列出 Global 的合法目标为 Shared（load）与 Register（load）

#### Scenario: 报告含合法目标清单

- **WHEN** 任一非法组合触发 `E0301`
- **THEN** 恢复建议按矩阵列出该源 scope 的全部合法目标 scope 与操作类别

### Requirement: 状态类类型与字段约束

`@tis.state` 类的每个字段类型 MUST 是 `Tensor[dtype, shape, Register]`——dtype 与 shape 任意合法，scope MUST 恰为 `Register`。字段类型为 `Pointer`、或 Tensor 的 scope 不为 `Register` 时，编译器 MUST 以 `E0304` 拒绝；报告 MUST 包含行列位置、违规字段名与违反的约束，恢复建议说明状态由 Pipeline 迭代间在 Register 携带，Shared/Global 数据应经移动操作进入计算后由状态承载结果。

每个 `@tis.state` 类定义引入一个**状态类类型**（以类名命名），其等价判定为名义等价：两个状态类类型等价当且仅当类名相同（同一类声明）。状态类类型的值由该类的构造调用创建（形如 `AttnState(O_acc=..., m=..., l=...)`）；构造调用的关键字实参与同名字段一一绑定，每个绑定受「类型不匹配拒绝」约束。状态类类型 MAY 用作 Pipeline `produce`/`consume` 嵌套函数的参数注解与返回注解，以及局部变量首次绑定的类型来源。状态类类型 MUST NOT 用作 kernel 签名参数注解（该注解形式不在 `E0104` 六类之内，语法层已拒绝）或出现在 `Tensor`/`Pointer` 的参数位置（`E0302` 结构违规）。

#### Scenario: 纯 Register 状态类被接受

- **WHEN** `@tis.state` 类的字段为 `O_acc: Tensor[f32, (BR, D), Register]`、`m: Tensor[f32, (BR,), Register]`、`l: Tensor[f32, (BR,), Register]`
- **THEN** 状态字段约束检查通过

#### Scenario: Shared scope 状态字段被拒绝

- **WHEN** 字段声明为 `k_s: Tensor[f16, (64, 64), Shared]`
- **THEN** 编译以 `E0304` 拒绝，报告注明 scope 必须为 Register

#### Scenario: Pointer 状态字段被拒绝

- **WHEN** 字段声明为 `q: Pointer[f16, Global]`
- **THEN** 编译以 `E0304` 拒绝，恢复建议说明状态字段只接受 Register scope 的 Tensor

#### Scenario: consume 签名的状态类类型被接受

- **WHEN** Pipeline `consume` 嵌套函数签名为 `def attend(j: int, buf, st: AttnState) -> AttnState`
- **THEN** 状态类类型用作参数注解与返回注解均合法

#### Scenario: 不同状态类类型互相赋值被拒绝

- **WHEN** 将类 `AttnState` 的值赋给类 `OtherState`（字段结构相同但类名不同）类型的目标
- **THEN** 编译以 `E0303` 拒绝（名义等价不成立），报告两侧类名

### Requirement: 表达式结果类型规则

类型检查的输入侧规则：设备代码变量在**首次绑定**（赋值、形参绑定、构造）时确定类型，此后单类型不变量——后续再绑定 MUST 与已确定类型等价，否则按「类型不匹配拒绝」报告 `E0303`；名称引用的结果类型为变量的绑定类型。对状态类类型值的字段访问，结果类型为该字段的声明类型。模块属性（`tis.*`）与 Pipeline buffer 属性（如 `buf.K`）访问及**原语调用表达式**的结果类型由 `primitives/*` 与执行结构 capability 的契约定义；执行结构方法调用（如 `pipe.run(...)`）与内建调用（如 `range(...)`）的结果类型由执行结构 capability 的契约定义；本 capability 对上述各类表达式的结果类型均不定义。状态类构造调用的类型规则由「状态类类型与字段约束」承载。

**下标与切片**：基对象 MUST 为 `Tensor`，其余任何类别（标量、状态类类型、`Pointer` 等）按 `E0303` 拒绝；结果为同 dtype、同 scope 的 `Tensor`，shape 逐维变换——整数下标消去该维；切片 `a:b` 保留该维（两边边界均为编译期常量时结果维度为编译期常量，否则为运行期派生维度）；`:` 全取保留该维；`None` 新增大小为 `1` 的编译期常量维度。两个运行期派生维度在其派生表达式结构等价（相同运算与逐操作数等价）时判定等价。

**常量**：int 字面量（四种进制形式）是 `comptime[int]` 的值（单向兼容到 `int` 位置）；布尔常量、`None` 与数学具名常量（如 `inf`）的类型地位由使用它们的原语契约承载；float 字面量到 dtype 标量位置的绑定规则由数值语义 capability 承载。**算术、比较与逻辑运算**（含 `and`/`or`/`not`）及 `if`/`elif` 条件的操作数与结果类型规则由数值语义 capability 承载（本 capability 非目标）。

#### Scenario: 单类型不变量违规被拒绝

- **WHEN** 变量首次绑定为 `int` 标量，随后被再绑定为 `Tensor[f32, (64,), Register]` 的值
- **THEN** 编译以 `E0303` 拒绝（再绑定与已确定类型不等价），报告定位再绑定

#### Scenario: 切片结果类型派生

- **WHEN** `Q` 为 `Tensor[f16, (运行期 seq_len, comptime D), Global]`，表达式为 `Q[bm*BR:(bm+1)*BR, :]`
- **THEN** 结果类型为 `Tensor[f16, (运行期派生, comptime D), Global]`（dtype 与 scope 不变；切片维运行期派生，全取维保留）

#### Scenario: 广播维度派生

- **WHEN** `m_new` 为 `Tensor[f32, (运行期 BR,), Register]`，表达式为 `m_new[:, None]`
- **THEN** 结果类型为 `Tensor[f32, (运行期 BR, comptime 1), Register]`（`None` 新增大小为 1 的编译期常量维度）

#### Scenario: 标量下标被拒绝

- **WHEN** 对 `int` 标量变量使用下标（如 `n[0]`）
- **THEN** 编译以 `E0303` 拒绝（下标基对象必须为 Tensor），恢复建议核对基对象类型

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

- **WHEN** `AttnState(O_acc=..., m=m_new, l=l_new)` 中某关键字实参的类型与字段声明类型不等价（如以 `Tensor[f16, (BR, BC), Register]` 绑定声明为 `Tensor[f32, (BR,), Register]` 的 `m` 字段）
- **THEN** 编译以 `E0303` 拒绝，报告定位该实参与两侧类型（状态类构造调用的实参绑定适用等价规则）

### Requirement: 类型检查报告契约与 E03xx 段位

类型系统产生的每条拒绝（`E03xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含两侧类型或违规结构）、恢复建议。同一模块存在多条类型拒绝时，编译器 MUST 收集并输出全部拒绝，MUST NOT 在第一条处停止；多条拒绝 MUST 按源码位置升序排列。

检查管线顺序为语法接受集（`E01xx`）在前、类型系统（`E03xx`）在后；同一源码位置同时命中两个阶段时 MUST 只报告 `E01xx` 一条。同一源码位置命中多个 `E03xx` 规则时 MUST 只报告一个错误码，按段内检查顺序取首个：`E0302`（结构）→ `E0304`（状态字段约束）→ `E0301`（转移组合）→ `E0303`（绑定等价）。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E0300`–`E0399` 归类型系统——`E0301` 为既有的作用域转移错误码（语义保持），`E0302` 类型表达式结构，`E0303` 类型不匹配，`E0304` 状态类字段约束，`E0305`–`E0399` 保留给类型系统后续扩展。

#### Scenario: 多条类型拒绝被收集并排序

- **WHEN** 同一模块第 8 行存在 `E0302`（未知 dtype）、第 20 行存在 `E0303`（赋值不匹配）
- **THEN** 输出恰好两条拒绝，按第 8 行在前、第 20 行在后排序

#### Scenario: 同位置语法与类型双命中只报语法错误

- **WHEN** 同一源码位置的表达式同时命中 `E0106`（语法拒绝）与 `E0303`（类型不匹配）
- **THEN** 该位置只报告管线更早的 `E0106` 一条拒绝

#### Scenario: 同位置多个类型规则命中只报段内首个

- **WHEN** 同一源码位置同时命中 `E0304`（状态字段约束）与 `E0303`（绑定等价）
- **THEN** 该位置只报告段内顺序更早的 `E0304` 一条拒绝

#### Scenario: 重复编译类型拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E03xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）
