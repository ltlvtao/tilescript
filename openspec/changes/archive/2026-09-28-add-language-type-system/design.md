# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| language/type-system | 新增类型系统行为契约（纯规格，无代码交付） | specs/language/type-system/spec.md | 第 1 节 |

## 1. language/type-system

### 目标与规范依据

目标 Requirements：`specs/language/type-system/spec.md` 的全部 ADDED Requirements——类型表达式结构、标量与 shape 组件、类型等价、作用域转移矩阵、状态类字段约束、类型不匹配、报告契约与段位。

规范依据：`veps/design.md` §2.1 原则 4（类型系统承担安全不承担性能）、§2.2（dtype/scope 文法、转移矩阵、`E0301`、`@tis.state` 字段 Register 约束）、§7（AttnState 与签名的类型用法）。

### 当前实现

仓库尚无编译器代码。类型规则只由 `veps/design.md` §2.2 的文法定义隐式表达：dtype 六种、scope 四种（含 `Distributed`，但 §2.6 未定义其语义——悬空引用）、`Tensor[dtype, shape, scope, layout=RowMajor]`、`Pointer[dtype, scope]`、`comptime[int]`、3×3 转移矩阵（8 格合法 1 格"-"）与 `E0301`。既有错误码事实：`E0301`（§2.2 转移矩阵）、`E0402`/`E0403`（MMA）、`E0501`（warp 角色）、`E0101`–`E0107`（已归档 `language/syntax-acceptance-set`）。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| 类型表达式结构规则 | 仅文法定义，无错误码 | `Tensor[f64, ...]` 等非法结构无拒绝契约 |
| 标量种类全集 | §7 用 `scale: f32` 但 §2.2 文法只定义 comptime[int] | dtype 标量注解的类型层地位未定义 |
| shape 组件类型规则 | 未定义 | `(seq_len, D)` 动态维度与 `(2.5, D)` 非法组件无区分规则 |
| 类型等价 | 未定义 | 逐组件等价、动态维度符号等价、layout 默认值等价、标量种类等价均无规则 |
| 转移矩阵 | §2.2 有矩阵与 `E0301` | 矩阵未规格化（"-"格语义、报告要素未定义）；矩阵格子的操作类别名（copy/move）无原语承载定义 |
| 状态类类型 | §7 用 `st: AttnState`/`-> AttnState`/构造调用 | 字段约束外，状态类名作为类型的地位与等价判定未定义 |
| 表达式类型输入侧 | §7 大量使用切片/广播/变量再绑定 | 变量单类型不变量、下标/切片/广播结果类型规则缺失 |
| 类型不匹配 | 未定义 | 绑定位置规则、与 `E0301` 的边界（移动原语豁免）、错误码均缺失 |
| `Distributed` scope | §2.2 引用 §2.6，§2.6 无定义 | 设计悬空，无法规格化 |
| E03xx 段位与报告 | 仅 `E0301` 存在 | 段位分配、四要素、跨段与段内 tiebreak 未定义 |

### 修改方案

本 change 为纯规格 change，无代码交付。关键裁决：

- **标量种类全集（设计审查 F1）**：标量种类钉为三类——运行期 `int`、运行期 dtype 标量（六种 dtype 名各自为标量注解，值为宿主侧标量，覆盖 `scale: f32`）、`comptime[int]`。dtype 标量等价按 dtype 名；dtype 标量间转换与数值字面量到 dtype 标量的绑定规则延期数值语义 capability。
- **comptime 双句重写（设计审查 F2）**：删除歧义的"运行期值被要求的位置"表述，改为精确两条——`comptime[int]` 值 MAY 绑定 `int` 位置（单向兼容）；`int` 值 MUST NOT 绑定 `comptime[int]` 位置，违反报 `E0303`（赋码入正文）。
- **移动原语豁免边界（设计审查 F3）**：`E0303` 等价规则适用于赋值、返回绑定与非移动调用实参；显式数据移动原语的实参 scope 组合不走等价规则、由转移矩阵（`E0301`）裁决。移动原语初始集合 `tis.load`/`tis.store`（`veps/design.md` §2.3 事实），权威集合由 `primitives/*` 定义——消除 `tis.load(Q[切片], Q_s)`（Global→Shared）被字面判为 `E0303` 的矛盾。
- **状态类类型（设计审查 F4）**：每个 `@tis.state` 类引入名义等价的状态类类型（类名同一性）；构造调用按关键字与字段一一绑定（受 `E0303` 约束）；可用位置 = produce/consume 嵌套函数参数与返回注解、局部变量首绑来源；禁用位置 = kernel 签名参数（语法层 `E0104` 已拒）与 Tensor/Pointer 参数位（`E0302`）。
- **表达式结果类型规则（设计审查 F5，新增 Requirement）**：单类型不变量（变量首绑定类型，再绑定须等价）；名称引用 = 绑定类型；状态字段访问 = 字段声明类型；下标/切片维变换规则（整数下标消维、切片保留维并按边界类别派生、`:` 全取保留、`None` 加 comptime 1 维）；派生维度等价 = 派生表达式结构等价；int 字面量 = `comptime[int]` 值。算术/比较提升、float 字面量绑定、`tis.*` 与 buffer 属性结果类型均显式延期（数值语义 / `primitives/*` / 执行结构）；执行结构方法调用（`pipe.run(...)`）与内建调用（`range(...)`）的结果类型同样让渡执行结构 capability（设计审查 cycle 1 round 3 N3 修订）；数学具名常量（`inf`）与布尔/`None` 同归原语契约、逻辑运算与 `if` 条件让渡数值语义、下标基对象闭合式拒绝（含 `Pointer`，`E0303`）（代码审查 round 1 修订）——非目标从"推断算法"修正为同时排除这些域。
- **段内 tiebreak（设计审查 F8）**：同位置多 `E03xx` 命中按 `E0302`→`E0304`→`E0301`→`E0303` 取首个（结构 → 状态约束 → 转移组合 → 绑定等价）。
- **与语法接受集的边界**：`E0104`（语法层）裁决注解的**形式类别**（六类之一）；本 change 裁决**内部结构**（参数顺序/数量、dtype 与 scope 封闭集合）与**绑定语义**。同一注解先过语法层，后过类型层；同位置双命中只报 `E01xx`（管线 tiebreak，对齐语法接受集报告契约的同位置规则）。
- **E03xx 段位分配**：`E0301` 为历史错误码，语义（"Invalid scope transfer"）与编号均保持不变；新增 `E0302`（类型表达式结构）、`E0303`（类型不匹配）、`E0304`（状态类字段约束）；`E0305`–`E0399` 保留。新码不按管线位置连续编排（`E0301` 已占 01），按子域小步分配，design 与 spec 双处记录。
- **转移矩阵裁决**：矩阵 3×3 全格规格化；`Global → Global` 标记非法（不提供同域复制操作——设备侧 Global 间复制无原语承载，也无意义）；格内名称（load/store/copy/move）是**操作类别名**，定义类型层合法性；各格子由哪个原语承载（`tis.load`/`tis.store` 的参数契约）延期至 `primitives/*`。
- **赋值不承载跨 scope 移动**：普通赋值两侧 scope 不同 = 类型不等价（`E0303`），恢复建议指向显式移动原语；跨 scope 移动必须显式原语调用（其 scope 组合由 `E0301` 裁决）。理由：显式优先原则（设计原则 1）——数据移动是影响同步语义的显式决策，不得由赋值语法隐式承载。
- **comptime 单向兼容**：`comptime[int]` 值可用于 `int` 位置（编译期值总是可在运行期使用）；反向拒绝（运行期值不可要求编译期求值）。类型位置两者不等价（不同种类）。
- **状态类字段裁决**：唯一合法形态 `Tensor[dtype, shape, Register]`；`Pointer` 字段拒绝——状态是 Pipeline 迭代间由 Register 携带的计算产物，不是内存视图；依据 §2.2"纯数据结构体，所有字段必须是 Register scope（状态不能藏在 Shared 里，否则同步语义不清）"。
- **shape 组件规则**：每维 = `comptime[int]` 常量表达式或运行期 `int` 标量表达式（动态维度合法，覆盖 §7 `(seq_len, D)`）；非整数组件（如浮点字面量）为结构错误 `E0302`。维度等价：常量按值、运行期按符号同一性（保守可判定）。
- **layout 裁决**：唯一合法值 `RowMajor`（默认，固定且写进文档——满足默认值确定性要求）；显式 `RowMajor` 与省略等价。swizzled 等布局是 `tis.alloc_shared` 的分配参数（§7 `layout=tis.Layout.swizzled(...)`），不是 Tensor 类型组件。
- **Distributed 非目标**：§2.2 引用"见 §2.6"但 §2.6 只定义 persistent/fused kernel，未定义 `Distributed` 的转移语义——设计悬空。scope 封闭集合钉为三域；`Distributed` 出现即 `E0302`（集合外 scope）。待设计补充后另立 change 扩展。
- **充分性标准**：覆盖 §7 FlashAttention 案例的全部类型用法（AttnState 三字段、五个 `Pointer` 签名参数、`int`/`f32`/`comptime[int]` 标量、`make_tensor` 动态 shape、`zeros`/`full` 的 Register Tensor、状态构造与 `pipe.run(init=...)` 实参绑定），核对任务见 tasks。
- **确定性影响**：本 change 无实现，不改变任何 `deterministic_hash`；"重复编译类型拒绝清单一致"Requirement 为将来实现固化确定性要求。类型规则全部为编译期静态检查，不涉及运行时反馈（对齐编译器定位：忠实执行显式意图 + 结构化诊断，类型拒绝是可静态化反馈）。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/language/type-system/spec.md`。
- designs：无（M1 类型检查实现设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `language/type-system` 索引。
