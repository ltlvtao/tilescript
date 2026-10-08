# Proposal

## Why

已归档的三个 stable specs 共有**五处显式让渡**指向"数值语义 capability"：①`language/type-system`「标量类型与 shape 组件规则」——dtype 标量之间的转换与数值字面量到 dtype 标量位置的绑定规则（L36）；②`language/type-system`「表达式结果类型规则」——float 字面量到 dtype 标量位置的绑定、算术/比较/逻辑运算（含 `and`/`or`/`not`）及 `if`/`elif` 条件的操作数与结果类型规则（L150，数值提升与字面量绑定亦见于 Purpose）；③`primitives/memory-ops`「分配与视图构造原语」——`tis.full` 的 `value` 到 `dtype` 的数值转换规则（L111）；④`primitives/memory-ops`「dtype 显式转换」——`tis.cast` 的数值效果（舍入、饱和、精度损失及其诊断）与算术结果表达式作为 `x` 的结果类型（L145，该 capability 定义前仅可形态级核对）；⑤`execution/pipeline-structure`「索引与整数内建」——`tis.cdiv` 两 `comptime[int]` 实参的编译期数值折叠（L161）。

M1 FlashAttention 案例的数值层完全建立在运算符表达式上（§7：索引算术 `bm*BR`/`j*BC`/`j+1`、Tensor-标量 `S * scale`、同 shape 与逐维广播算术、除法 `st.O_acc / st.l[:, None]`、具名常量一元负 `-inf`），但它们没有任何权威规格：`veps/design.md` 对算术运算的类型/形状规则、字面量绑定、cast 舍入行为零描述（§2.2 只有类型语法、§2.3 cast 只有一行签名）。错误码方面 `E06xx` 段完全空白。此前两个 change 的用法映射中六处标注"形态级核对——类型依赖数值语义 capability"（cast 派生表达式实参、`init` 内构造实参、store 一维 src 等），没有先行规格，这些路径不可验收，`pipe.run` 状态链上的数值表达式（`st.m - m_new` 等）类型悬空。

## What Changes

- 新增 capability `numerics/value-semantics`，定义数值语义的行为契约：
  - 算术运算（`+` `-` `*` `/`，含增强算术赋值）的操作数与结果规则：int 家族（`comptime[int]`/`int`）种类推导与编译期折叠、dtype 家族同 dtype 要求（无隐式提升）、dtype 标量-Tensor 与 Tensor-Tensor 的逐维 shape 广播、混家族算术拒绝（`E0601`–`E0603`，`/` 对 int 家族拒绝建议 `tis.cdiv`，`E0604`）
  - 一元负与数学具名常量：`inf` 的语言层地位（预置具名常量，IEEE 754 标准精度浮点 dtype 的正无穷，`f8e4m3` 位置拒绝）、`-inf` 路径、整型 dtype 位置拒绝（`E0606`）
  - 比较与逻辑运算及条件语境：布尔值的类型层地位为"条件语境专用值"合法使用位置封闭集（`if`/`elif` 条件、逻辑运算操作数、`comptime` 默认值——语法层既有），不进 `language/type-system` 封闭类型世界（`E0605`）
  - 数值常量到 dtype 位置的绑定：float 字面量（正确舍入绑定）、int 家族字面量与值、具名常量；无 dtype 语境的 float 字面量裸绑定拒绝（`E0606`）；承接 `tis.full` 的 `value` 转换让渡
  - `comptime[int]` 算术与 `tis.cdiv` 的编译期折叠（折叠值确定性公式）
  - `tis.cast` 数值效果：六 dtype 转换矩阵的家族规则（浮-浮 RN 与值域行为、浮-整向零截断与饱和、整-浮 RN、整-整补码环绕、`f8e4m3` 按 OCP FP8 E4M3 特则）与跨硬件位一致承诺；运行期转换失败不做编译期拒绝（编译期可判定的具名常量位置违规除外）
  - `E06xx` 报告契约与段位：`E0601`–`E0606` 新增、四要素/收集式/跨段与段内 tiebreak/确定性
- **无 MODIFIED、无 BREAKING**：三个 stable specs 的让渡句在归档时即已显式指向"数值语义 capability"，本 change 为纯承接（ADDED），不修改任何既有 Requirement、错误码归属与诊断行为。

## Capabilities

### New Capabilities

- `numerics/value-semantics`：数值语义行为契约——算术/比较/逻辑运算的操作数与结果规则、shape 广播、数学具名常量与数值字面量绑定、comptime 折叠与 `cdiv` 编译期求值、`cast`/`full` 的数值转换效果与跨硬件一致性、E0601–E0606 报告行为。

### Modified Capabilities

- 无。

## 非目标

- 计算原语（`tis.dot` 的 MMA 契约、`tis.reduce`/`tis.maximum`/`tis.exp`/`tis.log`/`tis.transpose` 等元素级与归约函数）的参数与结果契约——独立计算原语 change 承载（`E0402`/`E0403` 为其既有占用事实）。本 change 只定义运算符表达式的规则；计算原语实参中的算术子表达式按本 capability 裁决。
- 诊断 JSON 字段（精度损失建议、`cast` 数值诊断段等）——诊断 JSON Schema 整体延期（`diagnostics/*`）；本 change 只承诺行为级数值规则。
- 跨硬件数值容差与 `portability` 报告（`--target=all` 的数值差异呈现）——运行期数值一致性的容差数字属里程碑验收（`veps`），不写入行为 spec；本 change 的跨硬件承诺仅限静态可定义的转换效果位一致。
- `nan` 具名常量、`comptime[float]` 种类、位运算与移码（语法层已拒）、三元条件与链式比较（语法层已拒）——零事实，保守不引入。
- 增强赋值之外的可变绑定语义、除 `/` 外的 int 家族除法语义（整除由 `tis.cdiv` 独占承载）。
- `E0607`–`E0699` 保留给数值语义域后续扩展。

## Impact

- 承接五处显式让渡：type-system 的字面量绑定与运算规则、memory-ops 的 `full` value 转换与 `cast` 数值效果、execution 的 `cdiv` 折叠自此闭合；此前两个 change 用法映射中六处"形态级核对"标注可升级为完整核对（后续 change 或核对记录更新时执行，不追溯改写已归档文档）。
- 与 `language/type-system` 的边界：运算符表达式的操作数/结果类型规则与 shape 广播自本 capability 定义（该 capability L150 显式让渡）；类型绑定等价（`E0303`）仍由该 capability 裁决——运算结果再绑定时，运算违规落 `E06xx`、再绑定不等价落 `E0303`，段间管线 `E01xx`→`E03xx`→`E04xx`→`E05xx`→`E06xx` 扩展为五段。布尔值不进其封闭类型世界（与执行结构三特殊值同构的封闭集模式）。
- 与 `primitives/memory-ops` 的边界：`cast`/`full` 的参数契约与错误码（`E0406`）不变；本 change 只定义其数值效果。`E0303` 豁免边界不变（运算符表达式非调用实参）。
- 与 `execution/pipeline-structure` 的边界：`cdiv` 的形态契约与 `E0505` 不变；本 change 定义其 `comptime` 折叠值。
- `veps/design.md` §7 的全部运算符用法自此获得权威依据；示例与规格冲突时以规格为准。
- 无生产代码影响（纯规格先行 change）。
