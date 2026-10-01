# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| numerics/value-semantics | 新增数值语义行为契约（纯规格，无代码交付） | specs/numerics/value-semantics/spec.md | 第 1 节 |

本 change 为纯 ADDED：三个既有 stable specs 的让渡句（type-system 两处、memory-ops 两处、execution 一处）在各自归档时即已显式指向"数值语义 capability"，本 change 纯承接闭合，无 MODIFIED、无 BREAKING。

## 1. numerics/value-semantics

### 目标与规范依据

目标 Requirements：`specs/numerics/value-semantics/spec.md` 的全部 ADDED Requirements——算术运算操作数与结果规则、一元负与数学具名常量、比较逻辑与条件语境布尔封闭、数值常量与标量值到 dtype 位置绑定、comptime 折叠与 cdiv 编译期求值、dtype 转换数值效果与跨硬件一致、报告契约与 E06xx 段位。

规范依据：`veps/design.md` §2.1 原则 1（显式优先——无隐式提升、无条件真值）与原则 4（类型系统承担安全——运算规则是类型检查的输入侧延伸）、§2.2 dtype 封闭集合与标量种类、§2.3 `tis.cast`/`tis.full` 签名、§7 FlashAttention 案例的全部运算符用法。承接已归档 specs 的五处显式让渡：type-system「标量类型与 shape 组件规则」（dtype 标量转换与数值字面量绑定）、「表达式结果类型规则」（float 字面量绑定、算术/比较/逻辑运算及 if/elif 条件规则）、memory-ops「分配与视图构造原语」（`full` value 数值转换）、「dtype 显式转换」（`cast` 数值效果与算术结果表达式类型）、execution「索引与整数内建」（`cdiv` 编译期折叠）。

### 当前实现

仓库尚无编译器代码。数值语义事实只存在于 `veps/design.md`：§2.2 dtype 六种封闭集合（f16/bf16/f32/f8e4m3/i8/i32）与三类标量种类；§2.3 `tis.cast(x, dtype)` 一行签名与 `tis.full(shape, value, dtype, scope)` 签名（舍入/饱和行为零描述）；§7 案例的运算符用法——索引算术（`bm*BR`/`(bm+1)*BR`/`j*BC`/`(j+1)*BC`，int 家族）、Tensor-标量（`S * scale`）、同 shape 算术（`st.m - m_new`、`alpha * st.l + reduce(...)`、`alpha[:, None] * st.O_acc + dot(...)`）、逐维广播（`S - m_new[:, None]`、`st.O_acc / st.l[:, None]`）、除法（该处 f32）、具名常量一元负（`-inf` 经 `tis.full`）。语法层（syntax-acceptance-set）已接受：算术 `+` `-` `*` `/`、六种比较（仅两操作数单比较）、`and`/`or`/`not`、一元负、int 字面量（四进制）与 float 字面量、布尔常量、`None`。三个 stable specs 的五处让渡句悬空；`E06xx` 段完全空白。既有用法映射中六处标注"形态级核对——类型依赖数值语义 capability"（cast 派生表达式、`init` 内构造实参、store 一维 src 等）。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| 算术运算规则 | §7 用法零规格（类型/shape/scope/家族规则均无） | 操作数家族、dtype 约束、scope 一致性、shape 广播、`/` 边界未定义 |
| 一元负与 inf | §7 `-inf` 一处用法；语法层 inf 不在常量白名单（以名称引用类别进入） | inf 的语言层地位、值语义、位置约束未定义 |
| 比较逻辑条件 | 语法层接受；§7 零用法 | 布尔值类型地位、条件规则、真值语义取舍未裁决 |
| 常量绑定 | type-system L36/L150 让渡悬空；§7 无 float 字面量用法 | float 字面量语境、int 常量到 dtype、运行期 int 边界未定义 |
| 折叠与 cdiv | execution 让渡悬空；memory-ops E0405 静态折叠支依赖运算求值 | 折叠语义、cdiv 折叠值未定义 |
| cast 数值效果 | memory-ops 让渡悬空；veps 零描述 | 六 dtype 转换矩阵、舍入/饱和/环绕、跨硬件一致性未定义 |
| E06xx 段位 | 完全空白 | 段位语义、tiebreak、与 E0303 边界未定义 |

### 修改方案

本 change 为纯规格 change，无代码交付。关键裁决：

- **无隐式 dtype 提升（`E0602`）**：家族内跨 dtype 运算一律拒绝、恢复建议显式 `tis.cast`——与 memory-ops「dtype 转换 MUST NOT 隐式发生」同构（绑定侧与运算侧贯彻同一原则）。§7 全部运算同 dtype，零误命中。
- **混家族拒绝（`E0601`）**：int 家族与 dtype 家族不混算——**int 编译期常量（含 int 字面量）同样不作算术操作数**（设计审查 round 1 M1 裁决：`Tensor * 2` 拒绝，恢复建议 `Tensor * 2.0`——float 字面量按操作数侧 dtype 定型；int 常量到 dtype 位置的绑定收窄为原语 dtype 标量参数位，消除与 R4 的双解）。§7 零事实（`scale` 由 host 以 `f32` 传入）；运行期标量跨家族转换无原语路径，保守封闭（扩展须显式 change）。恢复建议如实说明现状路径（宿主侧转换/Tensor cast）。
- **算术操作的 scope 一致性（`E0603` 扩，设计审查 round 1 M2 补全）**：Tensor 算术操作数 scope MUST 相同（结果 scope 唯一可推导），违反与 shape 广播不兼容同落 `E0603`（均为"操作数组合不兼容"类）；dtype 标量无 scope 组件。§7 全部 Tensor 运算同 scope（Register 内），零误命中。
- **`/` 浮点专属（`E0604`）**：int 家族整除由 `tis.cdiv` 独占承载（§7 事实即此分工）；整型 dtype Tensor 除法零承载零事实，一并拒绝。
- **布尔值条件语境封闭集（`E0605`）**：比较/逻辑结果为布尔值，不进 type-system 封闭类型世界（不新增种类），合法位置封闭集 = if/elif 条件、逻辑运算操作数、comptime 默认值（E0104 既有）——与执行结构三特殊值同构（先例）。比较限**标量**（Tensor 逐元素比较零事实，归计算原语域后续 change）；if 条件必须布尔（不采用 Python 真值语义——原则 1 显式优先，§7 案例已去除 `if warp_size > 0` 类分支）。§7 零比较/逻辑/if 用法，零误命中。
- **常量绑定统一按转换矩阵（`E0606`）**：适用位置为封闭枚举（设计审查 round 1 minor 3 收敛）——① `tis.*` 原语 dtype 标量参数位（`full` value）；② 算术操作数位（仅 float 字面量与 inf/-inf，int 常量经 M1 裁决排除）；③ 已绑定浮点 dtype 变量的再绑定赋值源。比较操作数位不接受待定型常量。float 字面量为**待定型常量**——只在浮点 dtype 语境合法（正确舍入绑定），裸绑定（`x = 1.5` 首绑）拒绝（无 `comptime[float]` 种类，type-system 三类标量不含 float 侧，不为其扩种类）；int 家族**编译期常量**可绑位置①（越界拒）；**运行期 `int` 不绑 dtype 位置**（运行期标量转换无原语，零事实封闭）；dtype 标量须同 dtype（E0602 同理）；`full` 的 value 按此统一裁决（承接 memory-ops 让渡）。
- **inf 预置名称**：`inf` 为语言预置具名常量（不经 import；语法层以名称引用类别进入——E0106 白名单不含具名常量字面形式，名称类别合法，其**地位**由本 capability 定义，type-system L150 让渡"由使用它们的原语契约承载"与本定义衔接：使用合法性归原语契约、值语义归本 capability）。值为 IEEE 754 **标准精度**浮点 dtype（f16/bf16/f32）正无穷；**f8e4m3 位置以 `E0606` 拒绝**（设计审查 round 1 M4 裁决：OCP E4M3 无无穷表示——与 cast 运行期溢出饱和 ±448 分工：cast 是行为承诺、常量绑定是表示能力缺失、编译期可判定）；裸绑定（`x = inf` 首绑）以 `E0606` 拒绝（M3 补全，与 float 字面量同构）；`nan` 零事实不引入。
- **comptime 折叠**：`comptime[int]` 算术编译期精确求值，**适用域为全 comptime 操作数的常量表达式**；服务于种类推导、E0405 静态折叠支的**常量子表达式**求值（含运行期符号的派生长度折叠判定机制仍由 memory-ops E0405 支 (d) 承载，本 change 不定义符号折叠——设计审查 round 1 M5 修正归属）与 cdiv 折叠（值 = `ceil(a/b)` 数学定义，b=0 已由 E0505 裁）。comptime 默认值的常量表达式形式**不放开**（E0104 保持仅单个字面量——该让渡属语法层，非本 change 承接面）。
- **cast 数值效果家族规则**：浮-浮 RN + 越值域饱和（IEEE 标准精度饱和 ±inf；f8e4m3 按 OCP FP8 E4M3 饱和 ±448、无 inf）；浮-整向零截断 + 越界饱和整型边界 + NaN→0；整-浮 RN；整-整补码环绕。跨硬件**位一致**为静态可承诺面（同一输入值任何 HAL 后端相同结果位模式）——运行期数值容差（rtol 类）属里程碑验收（veps），不进行为 spec。运行期转换失败是行为承诺（饱和/NaN 规则）非编译期拒绝；编译期可判定的位置违规（inf 到整型、字面量越界）才走 E0606。精度损失诊断字段归 diagnostics/*。
- **E06xx 段位与 tiebreak**：E0601（混家族）→E0602（跨 dtype）→E0603（shape 广播与 Tensor scope 一致）→E0604（int 除法）→E0605（布尔封闭）→E0606（常量位置）——先家族后 dtype 后 shape、先运算类别后值类后位置；同码多操作数合并报告；跨段管线扩为五段 E01xx→E03xx→E04xx→E05xx→E06xx（数值语义最后：运算规则以前段定型的类型信息为输入）；同位置运算违规（E06xx）与再绑定不等价（E0303）双命中只报 E0303（更早段）。既有 stable specs 的管线表述（memory-ops 三段、execution 四段）为各自视角的**前缀列举**，列举截止各自段位，在五段世界里仍为真，无需回改（round 1 minor 9 裁决，归档时不在既有 specs 上做 MODIFIED）。E0607–E0699 保留。
- **充分性标准**：覆盖 §7 全部数值用法类别（int 索引算术、Tensor-标量乘、Tensor-Tensor 同 shape 算术、逐维广播算术含除法、`-inf` 常量绑定、`cdiv` 折叠边界用法——六类，round 1 minor 8 补全），核对任务见 tasks。
- **确定性影响**：本 change 无实现，不改变任何 `deterministic_hash`；"重复编译 E06xx 清单一致"与"折叠值确定"为将来实现固化确定性要求；cast 位一致承诺与确定性编译主张一致（同一源码+同一目标硬件产出一致结果）。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载；跨硬件 rtol 数字属 veps 里程碑验收不写入）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/numerics/value-semantics/spec.md`（补写 Purpose——数值域职责与让渡分工）。无 MODIFIED，既有 specs 文本不变。
- designs：无（数值折叠与 cast 优化的实现设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `numerics/value-semantics` 索引（E06xx 段自此有归属）。
