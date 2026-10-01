# M1 FlashAttention 数值语义用法映射核对记录

> 本文是 `openspec/changes/add-numerics-value-semantics` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/numerics/value-semantics/spec.md` 的 7 个 Requirement（纯 ADDED，无 MODIFIED）。
> 依据：`veps/design.md` §7（FlashAttention 完整案例）；让渡出处为三个 stable specs 的五处让渡句。
> 引用约定：R1–R7 依次指 numerics spec 的「算术运算的操作数与结果规则」「一元负与数学具名常量」「比较逻辑与条件语境的布尔封闭」「数值常量与标量值到 dtype 位置的绑定」「comptime 折叠与 cdiv 编译期求值」「dtype 转换数值效果与跨硬件一致」「报告契约与 E06xx 段位」。

核对方法：§7 源码逐行列出全部数值语义用法（运算符表达式、常量绑定、cdiv），每项按段内 tiebreak 顺序（E0601→E0602→E0603→E0604→E0605→E0606）过检查层并映射到 Requirement 条目；六类合法用法（design 充分性标准）必须全部覆盖、拒绝规则不得命中。计算原语（dot/reduce/maximum/exp/log/transpose）的结果类型不属本 change——其作为算术操作数时的**推导规则**由本 change 定义，操作数类型输入以计算原语域契约为前提（前提类型在 §7 事实下为 f32 家族），逐项如实标注。

## 1. §7 逐数值用法映射（六类全部用法）

| # | 源码（行号按 §7） | 用法 | 逐层核对（tiebreak 顺序） | 结果 |
|---|---|---|---|---|
| 1 | L485 `Q[bm*BR:(bm+1)*BR, :]`（同类切片另见 L514 `O[…]`、L515 `L[…]`） | int 索引算术 ×6 表达式（`bm*BR`×3、`(bm+1)*BR`×3） | R1 int 家族：`bm: int`（`tis.block_idx(0)` 结果，运行期）× `BR: comptime[int]` → `int`；`(bm+1)`：`int + comptime 字面量` → `int`；两侧均非全 comptime 不折叠（R5）；切片边界的运行期性由 type-system 切片规则承载（运行期派生维）✅ 零 E0601–E0606 命中 | ✅ |
| 2 | L492/L493 `K[j*BC:(j+1)*BC, :]`、`V[j*BC:(j+1)*BC, :]` | int 索引算术 ×4 表达式（`j*BC`×2、`(j+1)*BC`×2） | 同 #1（`j: int` produce 形参 × `BC: comptime[int]` → `int`）✅ | ✅ |
| 3 | L499 `S = S * scale` | Tensor-标量 ×1 | R1 dtype 家族：`S` 为 dot 结果 `Tensor[f32, (BR, BC), Register]`（dot 结果类型归计算原语域，前提类型）、`scale: f32` 标量——dtype 同 ✅（E0602 不命中）、标量无 scope 组件结果取 Register、标量广播 shape 不变 ✅；float 字面量不涉及；再绑定：S 首绑来自 dot、再绑 `S*scale` 结果 f32 (BR,BC) 等价（E0303 由 type-system 承载，不命中）✅ | ✅ |
| 4 | L502 `st.m - m_new` | Tensor-Tensor 同 shape ×1 | R1：两侧 `Tensor[f32, (BR,), Register]`（m_new 为 maximum 结果——计算原语域前提类型）——dtype 同 ✅、scope 同（Register）✅、shape 同 ✅ → `Tensor[f32, (BR,), Register]`；作 `tis.exp` 实参（计算原语域） | ✅ |
| 5 | L503 `S - m_new[:, None]` | 逐维广播 ×1 | R1：`(BR, BC)` − `(BR, 1)`——dim0 相等保留、dim1 的 1 扩展为 BC → `Tensor[f32, (BR, BC), Register]`（全 comptime 维合流为 comptime 维）✅；作 exp 实参 | ✅ |
| 6 | L504 `alpha * st.l` | Tensor-Tensor 同 shape ×1 | R1：`(BR,) × (BR,)` 同 shape → `Tensor[f32, (BR,), Register]`（alpha 为 exp 结果——计算原语域前提类型）✅ | ✅ |
| 7 | L504 `alpha * st.l + tis.reduce(P, axis=1, op=Sum)` | Tensor-Tensor 同 shape ×1（复合） | R1：左侧算术结果 `(BR,)` + reduce 结果 `(BR,)`（计算原语域前提类型）——dtype/scope/shape 同 ✅ → `Tensor[f32, (BR,), Register]` | ✅ |
| 8 | L505 `alpha[:, None] * st.O_acc` | 逐维广播 ×1 | R1：`(BR, 1) × (BR, D)`——dim1 扩展 → `Tensor[f32, (BR, D), Register]` ✅ | ✅ |
| 9 | L505 `alpha[:, None] * st.O_acc + tis.dot(...)` | Tensor-Tensor 同 shape ×1（复合） | R1：#8 结果 `(BR, D)` + dot 结果 `(BR, D)`（E0402/E0403 既有事实段的 MMA 计算，结果类型归计算原语域前提类型）✅ → `Tensor[f32, (BR, D), Register]` | ✅ |
| 10 | L514 `st.O_acc / st.l[:, None]` | 除法（含广播）×1 | R1：f32 为浮点 dtype ✅（E0604 不命中——`/` 浮点合法）；`(BR, D) / (BR, 1)` 广播 → `Tensor[f32, (BR, D), Register]`；作 `tis.cast` 第一实参（cast 参数契约 E0406 由 memory-ops 承载，实参整体豁免 E0303） | ✅ |
| 11 | L511 `tis.full((BR,), -inf, f32, Register)` | 具名常量绑定 ×1 | R2/R4：`-inf` 为具名常量 `inf` 经一元负（预置名称，名称引用类别语法合法）；绑定位置为 `tis.full` 的 value——R4 位置①原语 dtype 标量参数位 ✅；`f32` 属 IEEE 754 标准精度 ✅（M4 裁决：f8e4m3 才拒绝）；绑定值为 f32 负无穷 ✅（E0606 不命中）；value 标量形态由 memory-ops E0406 先行裁决（已裁接受） | ✅ |
| 12 | L509 `tis.cdiv(seq_len, BC)` | cdiv 折叠边界 ×1 | R5：`seq_len: int` 运行期、`BC: comptime[int]`——非全 comptime **不折叠**，结果为运行期 `int`（种类推导由 execution 承载）✅；形态契约 E0505 已裁（BC≠0） | ✅ |
| 13 | L515 `tis.log(st.l) + st.m` | Tensor-Tensor 同 shape ×1（代码审查 round 1 F2 补入——原仅在 §2 #20 推导，未入逐项清单） | R1：log 结果（计算原语域前提类型，§7 事实下 f32 (BR,)）+ `st.m`（f32 (BR,)）——dtype 同 ✅、scope 同（Register）✅、shape 同 ✅ → `Tensor[f32, (BR,), Register]`；作 `tis.store` 一维 src（长度相容由 memory-ops E0405 承载） | ✅ |

六类充分性标准覆盖核对：int 索引算术（#1/2——乘法表达式 10 个：`bm*BR`×3、`(bm+1)*BR`×3、`j*BC`×2、`(j+1)*BC`×2；其内层加法子表达式 `(bm+1)`×3、`(j+1)`×2 共 5 个另计并同样经 R1 int 家族推导核对，见 #1/2 推导列）、Tensor-标量乘（#3）、Tensor-Tensor 同 shape 算术（#4/6/7/9/13）、逐维广播算术含除法（#5/8/10）、`-inf` 常量绑定（#11）、`cdiv` 折叠边界（#12）——**全部覆盖，全部通过，零拒绝规则命中（E0601–E0606 无一触发）**。§7 无比较/逻辑/if 用法（R3 零事实核对：案例已去除 `if warp_size > 0` 类分支）、无 float 字面量、无 int 常量算术位/裸绑定用法——封闭条款零误命中。

## 2. 六处"形态级核对"标注的升级闭合（此前两个映射文档的遗留）

本 change 定义运算符表达式规则后，此前标注"形态级核对——依赖数值语义"的路径按新规则升级：

| 原标注处 | 表达式 | 升级后状态 |
|---|---|---|
| m1-primitive-usage-mapping #17 | L505 `tis.cast(P, f16)`（P 首绑自 exp 结果） | **规则闭合**：cast 实参 `S - m_new[:, None]`（exp 的实参运算）类型由 R1 完整推导为 `Tensor[f32, (BR, BC), Register]`（#5）；P 自身类型仍依赖 exp 返回类型（计算原语域，前提类型）——依赖从"运算规则未定义"收敛为"仅 exp 契约" |
| m1-primitive-usage-mapping #18 | L514 `tis.cast(st.O_acc / st.l[:, None], f16)` | **完整闭合**：除法表达式类型完整推导（#10）+ cast 数值效果（f32→f16 RN，R6）已定义——该路径零残留 |
| m1-primitive-usage-mapping #20 | L515 `tis.store(tis.log(st.l) + st.m, L[…])` | **规则闭合**：`+` 同 shape 规则由 R1 定义（f32 (BR,)+f32 (BR,)→f32 (BR,)）；log 返回类型仍为计算原语域前提——src 类型从"运算+原语双依赖"收敛为"仅 log 契约" |
| m1-execution-usage-mapping §2 O_acc | L505/L507 `O_new = alpha[:,None] * st.O_acc + dot(…)` → `AttnState(O_acc=O_new)` | **规则闭合**：算术推导规则完整（#8/9，结果 f32 (BR,D) Register 与字段声明等价——E0303 路径现可完整判定）；dot 结果类型为计算原语域前提 |
| m1-execution-usage-mapping §2 m | L501/L507 `m_new = tis.maximum(st.m, m_ij)` | **部分闭合（如实）**：maximum 返回类型归计算原语域（形态级保留）；其操作数 `st.m` 类型已定、`m_ij`（reduce 结果）同域——本 change 未新增该路径的闭合面 |
| m1-execution-usage-mapping §2 l | L504/L507 `l_new = alpha * st.l + reduce(…)` | **规则闭合**：`alpha * st.l` 与 `+` 推导规则完整（#6/7，结果 f32 (BR,) Register 与字段声明等价）；reduce 结果类型为计算原语域前提 |

总结：六处中 #18 完整闭合、四处规则闭合（残留依赖收敛为计算原语域单一前提）、一处（m ← maximum）如实保留形态级（该路径本就不含运算符表达式，不属本 change 承接面——原标注将其计入"依赖数值语义"略宽，本记录据实收敛）。已归档的两个映射文档不追溯改写，本节为升级闭合的权威记录。

## 3. 五处让渡闭合核对（task 3 前半）

| # | 让渡句（stable specs） | 承接条款 | 闭合核对 |
|---|---|---|---|
| ① | type-system「标量类型与 shape 组件规则」L36："dtype 标量之间的转换与数值字面量到 dtype 标量位置的绑定规则由数值语义 capability 承载（本 capability 不定义）" | R4（位置封闭枚举 + 各值类绑定规则：float 字面量/int 编译期常量/运行期 int/dtype 标量/inf） | ✅ dtype 标量间转换（同 dtype 才绑，跨 dtype E0606）与字面量绑定（int 常量值域、float 正确舍入）均闭合 |
| ② | type-system「表达式结果类型规则」L150："float 字面量到 dtype 标量位置的绑定规则……算术、比较与逻辑运算（含 and/or/not）及 if/elif 条件的操作数与结果类型规则由数值语义 capability 承载" | R4（float 字面量）+ R1（算术）+ R3（比较逻辑条件） | ✅ 三类规则均定义；§7 用法（#3–#10）经 R1 完整裁决 |
| ③ | memory-ops「分配与视图构造原语」L111："full 的 value……其到 dtype 的数值转换规则由数值语义 capability 承载" | R4（value 为位置①；-inf 绑定 #11） | ✅ 闭合 |
| ④ | memory-ops「dtype 显式转换」L145："转换的数值效果（舍入、饱和、精度损失及其诊断）由数值语义 capability 承载……算术结果表达式作为 x 传入时，其结果类型同样由数值语义 capability 承载（该 capability 定义前，此路径仅可做形态级核对）" | R6（cast 家族规则）+ R1（算术结果表达式类型） | ✅ 数值效果闭合（RN/向零/饱和/环绕/OCP E4M3）；算术实参路径 #10 完整核对——"定义前形态级"限定解除 |
| ⑤ | execution「索引与整数内建」L161："（cdiv）两实参均为 comptime[int] 时结果为 comptime[int]（值的编译期折叠归数值语义 capability）" | R5（折叠值 = ceil(a/b)） | ✅ 闭合；§7 用法 #12 为非折叠边界（运行期 int） |

## 4. 错误码段位核对（task 3 后半）

| 段 | 错误码 | 归属 | 冲突核对 |
|---|---|---|---|
| 语法段 | E0101–E0107 | `language/syntax-acceptance-set`（已归档） | 与 E06xx 无重叠；运算符集/字面量形式/比较限两操作数在语法层先决（E0106），E06xx 裁语义契约——互补无双解（round 2 审查维度 C 复核） |
| 类型段 | E0301–E0304 | `language/type-system`（已归档） | 与 E06xx 无重叠；同位置运算违规（E06xx）与再绑定不等价（E0303）双命中只报 E0303（R7 跨段 tiebreak）；E0301 限数据移动原语、算术 scope 违规落 E0603（round 2 核验无双解） |
| 原语契约段 | E0402–E0403 / E0404–E0407 | 计算原语既有占用 / `primitives/memory-ops`（已归档） | 与 E06xx 无重叠；cast/full 参数契约（E0406）先行、数值效果（R6）与常量绑定（R4）后置——先后分工经 round 1 审查确认无双解 |
| 执行结构段 | E0501–E0505 | `execution/pipeline-structure`（已归档） | 与 E06xx 无重叠；cdiv 形态契约（E0505）先行、折叠值（R5）后置 |
| 数值语义段 | E0601–E0606 | `numerics/value-semantics`（本 change） | E06xx 段此前空白（grep 核实零命中）；E0607–E0699 保留 |

- 跨段管线五段 E01xx→E03xx→E04xx→E05xx→E06xx 与段内 tiebreak E0601→E0602→E0603→E0604→E0605→E0606：spec R7 与 design「E06xx 段位与 tiebreak」条目一致 ✅；既有 specs 的三段/四段表述为前缀列举、五段世界仍为真（design minor 9 裁决）✅。
- §7 全部数值用法（§1 十三项——代码审查 round 1 F2 补入原漏列的 L515 加法一项后）经逐层核对零命中任何 E06xx 拒绝——合法路径全部覆盖 ✅。

**Scenario 计数勘误（代码审查 round 1 F1）**：本 change spec 实际 **39 个 Scenario**（R1=10/R2=5/R3=4/R4=7/R5=3/R6=6/R7=4，grep `^#### Scenario` 复核）。实施 commit `7a4b355` 的 message 与 h1.jsonl summary 所记「34 Scenario」为起草时计数未随设计审查 round 1 五项 major 修复（跨 scope、`S*2` 拒绝、inf 裸绑、f8e4m3 拒、折叠适用域——各新增 Scenario 共 5 个）同步所致。按 execution change F4 先例：历史 commit message 与 h1.jsonl 凭证性保留不改写，由修复 commit 登记正确计数（修正链留痕）。
