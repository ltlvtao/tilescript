# M1 FlashAttention 类型用法映射核对记录

> 本文是 `openspec/changes/add-language-type-system` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/language/type-system/spec.md` 的 8 个 Requirement 与让渡/延期声明。
> 依据：`veps/design.md` §7（FlashAttention 完整案例）、§2.2（类型系统文法与转移矩阵）。
> 引用约定：R1–R8 依次指 spec 的「类型表达式结构」「标量类型与 shape 组件」「类型等价」「作用域转移矩阵」「状态类类型与字段约束」「表达式结果类型」「类型不匹配拒绝」「报告契约与段位」。

核对方法：§7 源码逐构造列出全部**类型用法**（注解、绑定、切片/下标、字段访问、scope 组合、常量），每项映射到 Requirement 条目或显式让渡/延期声明；白名单必须全部覆盖，拒绝规则（E0301/E0302/E0303/E0304）不得命中。

## 1. §7 逐构造类型用法映射

| # | 源码构造（行号按 §7） | 类型用法 | 承载方 | 结果 |
|---|---|---|---|---|
| 1 | L463-465 `AttnState` 三字段 | `Tensor[f32, (BR, D), Register]`、`Tensor[f32, (BR,), Register]`×2 | R1（f32∈dtype 封闭集、Register∈scope 封闭集、三参顺序）；R2（shape 组件 BR/D 为 comptime[int]）；R5（字段恰为 Register Tensor，不触发 E0304） | ✅ |
| 2 | L469-471 签名 | `Pointer[f16, Global]`×4、`Pointer[f32, Global]` | R1（Pointer 两参结构，dtype/scope 均在封闭集合） | ✅ |
| 3 | L472 | `seq_len: int` | R2（运行期 int 标量种类） | ✅ |
| 4 | L472 | `scale: f32` | R2（运行期 dtype 标量种类——六种 dtype 名之一，cycle 1 F1 闭合点） | ✅ |
| 5 | L473-474 | `D/BR/BC/STAGES: comptime[int] = 64/64/64/2` | R2（comptime[int] 种类）；R6 常量段（int 字面量是 comptime[int] 的值）；R3（同种类等价，默认值绑定成立） | ✅ |
| 6 | L476 | `bm = tis.block_idx(0)` | R6 首绑（赋值确定类型）；结果类型让渡（`tis.*` 原语调用 → `primitives/*`）；`0` 为 comptime[int] 值，实参绑定走 R7 + R2 单向兼容 | ✅ |
| 7 | L477-479 | `Q = tis.make_tensor(Q_ptr, (seq_len, D))` ×5 | R6 首绑 + 让渡（结果类型 `primitives/*`）；shape 实参 `(seq_len, D)`：seq_len 运行期 int 标量 + D comptime → R2 shape 组件规则（动态维度合法） | ✅ |
| 8 | L481-483 | `Q_s = tis.alloc_shared((BR, D), f16, layout=…swizzled)` ×3 | R6 首绑 + 让渡；`layout=tis.Layout.swizzled` 是分配参数而非 Tensor 类型组件（design 裁决：R1 只认 `RowMajor`） | ✅ |
| 9 | L485 | `tis.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)` | R7 移动原语实参**整体**豁免；scope 组合 Global→Shared 落 R4 矩阵 load 格（合法）；`Q[切片]` 结果类型走 R6（dtype/scope 不变、切片维运行期派生、`:` 全取维保留）；其余维度归 `primitives/*` 契约 | ✅ |
| 10 | L486 | `tis.barrier()` | 无类型层用法；原语调用让渡 | ✅ |
| 11 | L488 | `pipe = tis.Pipeline(stages=STAGES, buffers={…})` | R6 首绑 + 让渡（结果类型 `primitives/*`）；`STAGES` 为 comptime[int] 实参（R7/R2） | ✅ |
| 12 | L491 | `fetch(j: int, buf)` | `j: int` R2；`buf` 无注解——首绑类型与 `buf.K`/`buf.V` 属性类型由执行结构 capability 契约承载（R6 让渡句「Pipeline buffer 属性」） | ✅ |
| 13 | L492-493 | `tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)` ×2 | 同 #9：R4 Global→Shared load 合法；切片 R6；豁免 R7 | ✅ |
| 14 | L496 | `attend(j: int, buf, st: AttnState) -> AttnState` | R5（状态类类型可用位置：嵌套函数参数注解 + 返回注解）；`j: int` R2；`buf` 同 #12 | ✅ |
| 15 | L497-498 | `S = tis.dot(Q_s, tis.transpose(buf.K), tis.zeros((BR, BC), f32, Register), …)` | R6 首绑 + 让渡（dot/transpose 结果类型 `primitives/*`）；zeros 的 Register∈R1 封闭集、shape 组件 R2；`buf.K` 属性让渡 | ✅ |
| 16 | L499 | `S = S * scale` | R6 单类型不变量（再绑定须与已确定类型等价）；右侧提升类型由数值语义 capability 显式延期（R6/proposal 非目标），延期域承载 §7 合法性 | ✅（显式延期） |
| 17 | L500 | `m_ij = tis.reduce(S, axis=1, op=Max)` | R6 首绑 + 让渡 | ✅ |
| 18 | L501 | `m_new = tis.maximum(st.m, m_ij)` | R6 首绑 + 让渡；`st.m` 字段访问 = 字段声明类型 `Tensor[f32, (BR,), Register]`（R6） | ✅ |
| 19 | L502 | `alpha = tis.exp(st.m - m_new)` | R6 首绑 + 让渡；字段访问 R6；`-` 提升延期数值语义 | ✅ |
| 20 | L503 | `P = tis.exp(S - m_new[:, None])` | `[:, None]` 维变换走 R6（`None` 新增 comptime 1 维）；其余让渡/延期同 #19 | ✅ |
| 21 | L504 | `l_new = alpha * st.l + tis.reduce(P, axis=1, op=Sum)` | R6 首绑 + 字段访问（`st.l`）+ 让渡（reduce）；`* +` 提升延期 | ✅ |
| 22 | L505-506 | `O_new = alpha[:, None] * st.O_acc + tis.dot(tis.cast(P, f16), buf.V, tis.zeros((BR, D), f32, Register))` | R6 广播维（`alpha[:, None]`）+ 字段访问（`st.O_acc`）+ 让渡（`tis.cast` 结果类型 `primitives/*`、`buf.V` 属性、dot/zeros 同 #15） | ✅ |
| 23 | L507 | `return AttnState(O_acc=O_new, m=m_new, l=l_new)` | R5（状态类构造调用：关键字实参与同名字段一一绑定）+ R7（构造绑定受等价约束，返回值与返回注解 AttnState 同类名 nominal 等价）；三个实参的类型来源属让渡/延期域，绑定**义务**由 R5+R7 承载 | ✅ |
| 24 | L509-512 | `st = pipe.run(range(tis.cdiv(seq_len, BC)), init=AttnState(…zeros…/full((BR,), -inf, f32, Register)/zeros…))` | `st` 首绑类型 = `pipe.run` 结果类型 → R6 让渡句**执行结构方法调用**分句（cycle 2 N3 闭合点）；`range(...)` → 让渡句**内建调用**分句（同）；`tis.cdiv` → `tis.*` 原语让渡；构造 `AttnState(…)` R5；`init=` 实参绑定 R7（非移动调用）+ 形参类型执行结构契约；`-inf`/`full` 值契约归 `primitives/*`；zeros 同 #15 | ✅ |
| 25 | L514 | `tis.store(tis.cast(st.O_acc / st.l[:, None], f16), O[bm*BR:(bm+1)*BR, :])` | R4 Register→Global store 格（合法）；R7 豁免（实参整体）；字段访问与 `[:, None]` R6；`/` 提升延期；`tis.cast` 让渡；`O[切片]` R6 | ✅ |
| 26 | L515 | `tis.store(tis.log(st.l) + st.m, L[bm*BR:(bm+1)*BR])` | R4 Register→Global store（合法）；字段访问 R6；`+` 延期；`tis.log` 让渡；`L[切片]` R6（切片维运行期派生） | ✅ |

## 2. 拒绝规则命中核对

- **E0302**：§7 全部 dtype（f16/f32）与 scope（Global/Shared/Register）在封闭集合内；Tensor 均三参（无第四参）；Pointer 均两参；shape 组件全为 comptime[int] 或运行期 int 标量（`(seq_len, D)`、`(seq_len,)`）——零命中。
- **E0304**：AttnState 三字段均为 `Tensor[dtype, shape, Register]`，无 Pointer、无非 Register scope——零命中。
- **E0301**：全部数据移动 scope 组合为 Global→Shared（load，#9/#13）与 Register→Global（store，#25/#26），均落矩阵合法格——零命中。
- **E0303**：绑定侧——comptime 默认值同种类等价（#5）；`0` 字面量单向兼容（#6）；构造绑定与再绑定（#16/#23）的等价判定输入部分来自显式延期域（数值语义/primitives），本 change 承载绑定**义务**、延期域承载类型**来源**，不构成规格命中；无 scope 不同的普通赋值（跨 scope 移动全部走显式原语，#9/#13/#25/#26）——零命中。

## 3. 其余四 kernel 类型需求推导（M1 尚无参考源码，保守推导）

| Kernel | 超出 §7 的类型用法 | 承载方 | 结果 |
|---|---|---|---|
| MatMul | comptime 默认值 `= 4096` 类大字面量（R2/R6/R3 同 #5）；嵌套 for 的循环变量（R6 首绑，range 结果类型让渡）；边界 tile `if` 比较 `<`（操作数提升延期数值语义） | 均在 §7 已核对形态或显式延期域内 | ✅ |
| LayerNorm | float 常量 eps（float 字面量到 dtype 标量位置的绑定规则——R6 常量段显式延期数值语义）；`tis.sqrt` 类原语（让渡） | 显式延期/让渡 | ✅ |
| Softmax | `tis.reduce(op=Max)`、`tis.exp`（让渡，§7 #17/#19 同类） | 让渡 | ✅ |
| Transpose | 变量整数下标 `A[i, j]`（R6 整数下标消维——基对象 Tensor、结果消去该维）；`T[j, i]` 赋值（R6 首绑/再绑定） | R6 已定义条目 | ✅ |

四 kernel 无需要扩展本 capability 的新类型用法类别。

## 4. 错误码段位核对（task 3）

| 本 change | 既有事实（veps/design.md §2.2/§2.3 与已归档 specs） | 冲突 |
|---|---|---|
| `E0302`（类型表达式结构）/`E0303`（类型不匹配）/`E0304`（状态字段约束） | `E0301`（§2.2 作用域转移，本 change 升格为规格、语义与编号不变） | 无：E0301 未占用 02–04，语义域同段 |
| `E0301`–`E0304` | `E0101`–`E0107`（已归档 language/syntax-acceptance-set，语法段） | 无：E01xx 与 E03xx 不重叠 |
| `E0301`–`E0304` | `E0402`/`E0403`（MMA）、`E0501`（warp 角色） | 无：E04xx/E05xx 与 E03xx 不重叠 |
| `E0305`–`E0399` 保留 | — | 与 design「修改方案」段位分配一致（按子域小步分配，非按管线位置连续） |

## 5. 结论

- **充分性**：§7 全部类型用法（26 组构造，含 12 类以上形态）每项映射到 R1–R8 条目或显式让渡/延期声明，白名单全覆盖；四 kernel 推导无新增类别——task 2 通过。
- **拒绝规则**：E0301/E0302/E0303/E0304 对 §7 零命中——task 2 通过。
- **段位冲突**：无重叠、分配原则与 design 一致——task 3 通过。
- **strict validation**（task 1）：`npx openspec validate add-language-type-system --strict` 与 `--all --strict` 通过（2026-09-28，主智能体与设计审查 subagent 双通道复跑）。
