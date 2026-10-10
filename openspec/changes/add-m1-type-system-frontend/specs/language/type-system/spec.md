# language/type-system Delta

## MODIFIED Requirements

### Requirement: 状态类类型与字段约束

`@tis.state` 类的每个字段类型 MUST 是 `Tensor[dtype, shape, Register]`——dtype 与 shape 任意合法，scope MUST 恰为 `Register`。字段类型为 `Pointer`、Tensor 的 scope 不为 `Register`、或字段类型既非 `Tensor` 也非 `Pointer`（标量种类注解 `int`/`comptime[int]`/dtype 名等）时，编译器 MUST 以 `E0304` 拒绝；报告 MUST 包含行列位置、违规字段名与违反的约束，恢复建议说明状态由 Pipeline 迭代间在 Register 携带，状态字段只接受 Register scope 的 Tensor，Shared/Global 数据应经移动操作进入计算后由状态承载结果。

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

#### Scenario: 标量状态字段被拒绝

- **WHEN** 字段声明为 `count: int`（标量种类注解——非 Tensor 非 Pointer）
- **THEN** 编译以 `E0304` 拒绝，报告注明状态字段只接受 Register scope 的 Tensor

#### Scenario: consume 签名的状态类类型被接受

- **WHEN** Pipeline `consume` 嵌套函数签名为 `def attend(j: int, buf, st: AttnState) -> AttnState`
- **THEN** 状态类类型用作参数注解与返回注解均合法

#### Scenario: 不同状态类类型互相赋值被拒绝

- **WHEN** 将类 `AttnState` 的值赋给类 `OtherState`（字段结构相同但类名不同）类型的目标
- **THEN** 编译以 `E0303` 拒绝（名义等价不成立），报告两侧类名

### Requirement: 类型检查报告契约与 E03xx 段位

类型系统产生的每条拒绝（`E03xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含两侧类型或违规结构）、恢复建议。同一模块存在多条类型拒绝时，编译器 MUST 收集并输出全部拒绝，MUST NOT 在第一条处停止；多条拒绝 MUST 按源码位置升序排列。

检查管线顺序为语法接受集（`E01xx`）在前、类型系统（`E03xx`）在后；同一源码位置同时命中两个阶段时 MUST 只报告 `E01xx` 一条。段间执行语义：语法段存在任一拒绝时，类型系统段 MUST NOT 执行（段间短路）——类型检查以干净语法为前提（语法拒绝的构造子树不再深入，其类型不可靠）；语法段零拒绝时类型系统段执行。段间短路使跨段同位置双命中不存在于输出，是前句管线顺序声明的执行语义。同一源码位置命中多个 `E03xx` 规则时 MUST 只报告一个错误码，按段内检查顺序取首个：`E0302`（结构）→ `E0304`（状态字段约束）→ `E0301`（转移组合）→ `E0303`（绑定等价）。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E0300`–`E0399` 归类型系统——`E0301` 为既有的作用域转移错误码（语义保持），`E0302` 类型表达式结构，`E0303` 类型不匹配，`E0304` 状态类字段约束，`E0305`–`E0399` 保留给类型系统后续扩展。

#### Scenario: 多条类型拒绝被收集并排序

- **WHEN** 同一模块第 8 行存在 `E0302`（未知 dtype）、第 20 行存在 `E0303`（赋值不匹配）
- **THEN** 输出恰好两条拒绝，按第 8 行在前、第 20 行在后排序

#### Scenario: 同位置语法与类型双命中只报语法错误

- **WHEN** 同一源码位置的表达式同时命中 `E0106`（语法拒绝）与 `E0303`（类型不匹配）
- **THEN** 该位置只报告管线更早的 `E0106` 一条拒绝

#### Scenario: 语法段存在拒绝时类型段不执行

- **WHEN** 同一模块第 6 行存在 `E0106`（语法拒绝），且若类型段执行将在第 15 行产生 `E0303`（类型不匹配）
- **THEN** 输出恰好一条拒绝（第 6 行的 `E0106`）；类型系统段未执行，第 15 行的潜在 `E0303` 不出现于输出

#### Scenario: 同位置多个类型规则命中只报段内首个

- **WHEN** 同一源码位置同时命中 `E0304`（状态字段约束）与 `E0303`（绑定等价）
- **THEN** 该位置只报告段内顺序更早的 `E0304` 一条拒绝

#### Scenario: 重复编译类型拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E03xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）
