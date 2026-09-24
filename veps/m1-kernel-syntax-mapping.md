# M1 五 kernel 语法映射核对记录

> 本文是 `openspec/changes/add-language-syntax-acceptance-set` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/language/syntax-acceptance-set/spec.md` 的白名单与拒绝清单。
> 依据：`veps/design.md` §7（FlashAttention 完整案例）、§5.2（M1 验收：五 kernel 全部通过类型检查）、§摘要第 21 行（五 kernel 名单）。

## 1. FlashAttention（§7 完整案例逐构造映射）

§7 源码剔除 `module flash_attention:` 伪代码行后逐构造核对（前缀按命名裁决映射 `ts.` → `tis.`）：

| 源码构造 | 所属 Requirement | 白名单条目 | 结果 |
|---|---|---|---|
| `@tis.state class AttnState`（顶层） | R2 顶层 / R7 类体 | 顶层白名单第 2 类；类体=带注解无右值字段（`O_acc`/`m`/`l` 三条均合规） | ✅ |
| `@tis.kernel def flash_attn_fwd(...)`（顶层） | R2 / R3 | 顶层白名单第 3 类；入口装饰器+顶层函数 | ✅ |
| 签名 `Pointer[f16, Global]`×5、`int`、`f32` | R4 | Pointer[...] / int / dtype 名 | ✅ |
| 签名 `D: comptime[int] = 64` 等 4 个 | R4 | comptime[int] + int 字面量默认值 | ✅ |
| `bm = tis.block_idx(0)` 等 | R5 | 单目标赋值（无注解）+ 调用 | ✅ |
| `Q = tis.make_tensor(Q_ptr, (seq_len, D))`（分号连接，AST 为多条 Assign） | R5/R6 | 赋值；元组在调用实参位置 | ✅ |
| `tis.load(..., mode=Sync)` / `tis.barrier()` | R5/R6 | 表达式语句；关键字实参值为名称 | ✅ |
| `tis.alloc_shared((BR, D), f16, layout=tis.Layout.swizzled(xor=0b11100))` | R6 | 元组实参；`xor=0b11100` 二进制 int 字面量；属性访问 | ✅ |
| `pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})` | R6 | 字典字面量在关键字实参值位置、键为字符串常量 | ✅ |
| `@pipe.produce def fetch(...)` / `@pipe.consume def attend(...)` | R3/R5 | 按属性名识别的嵌套函数装饰器；嵌套函数定义白名单 | ✅ |
| `fetch(j: int, buf)` 中 `buf` 无注解 | R4 | E0104 仅约束设备入口函数，不约束 produce/consume 嵌套函数 | ✅（不触发） |
| `Q[bm*BR:(bm+1)*BR, :]`、`m_new[:, None]` | R6 | 切片下标（含元组形态 `[:, None]`）；算术 `*` `+` | ✅ |
| `S = tis.dot(..., mma=tis.MMA(16, 8, 16), pad=tis.PadPolicy.Error)` | R5/R6 | 赋值；调用链与属性访问；元组实参 `ts.zeros((BR, BC), f32, Register)` | ✅ |
| `S = S * scale`、`alpha = ts.exp(st.m - m_new)`、`l_new = alpha * st.l + tis.reduce(...)`、`O_new = alpha[:, None] * st.O_acc + tis.dot(...)` | R5/R6 | 赋值；算术 `* + -`；调用；属性；切片 | ✅ |
| `return AttnState(O_acc=O_new, m=m_new, l=l_new)` | R5/R6 | return；名称调用与关键字实参 | ✅ |
| `st = pipe.run(range(tis.cdiv(seq_len, BC)), init=AttnState(..., m=tis.full((BR,), -inf, f32, Register), ...))` | R5/R6 | 赋值；调用实参位置的 `range(...)`；元组实参 `(BR,)`；`-inf` = 一元负+名称引用 | ✅ |
| `ts.store(ts.cast(st.O_acc / st.l[:, None], f16), ...)`、`ts.store(ts.log(st.l) + st.m, ...)` | R5/R6 | 表达式语句；除法 `/`；切片；调用 | ✅ |

**结论**：拒绝清单（while/match/break/continue/try/raise/assert/del/global/nonlocal/AnnAssign/设备内 import/设备内类定义；lambda/推导/f-string/yield/await/星号解包/海象/三元/链式比较/白名单外运算符/字符串越位）**零命中**。

## 2. 其余四 kernel（按算法语义推导的语法需求映射）

M1 时尚无四 kernel 的参考源码，按标准 Tile 实现形态保守推导所需语法类别：

| Kernel | 推导所需语法（超出 FA 案例的部分加粗） | 映射结果 |
|---|---|---|
| **MatMul**（M2-M3 验收"无 pipeline"形态） | `@tis.kernel` 签名（`Pointer[f16, Global]`×3、`comptime[int]` 含 `= 4096` 类字面量默认值）；赋值；嵌套 `for range`（bm/bn/bk tile 循环）；边界 tile `if`（尾块判断，比较 `<`）；`tis.make_tensor`/`alloc_shared`/`load`/`store`/`barrier`/`dot`（零初始化 `tis.zeros((BR,BC), f32, Register)`）/`block_idx` 调用；切片；算术 `* +` | 全部 ∈ FA 已核对集合：`if` 与比较 `<` ∈ R5 白名单 / R6 运算符白名单 ✅；无 produce/consume 需求（M1 无 pipeline）✅ |
| **LayerNorm** | 行循环 `for range`；归约 `tis.reduce(axis=1, op=Sum)` 得 mean/var；中心化与归一化算术 `- * / +`；`tis.sqrt` 类原语调用；float 常量（eps）；`tis.store` | `- * / +` ∈ R6 算术白名单；float 常量 ∈ 数值常量 ✅；名称调用原语 ✅ |
| **Softmax** | `tis.reduce(op=Max)`、`tis.exp`、减 max（`-`）、归一化（`/`）、行循环 `for range`、load/store | 全部 ∈ FA 已核对集合（Max/exp 均为名称调用原语，§7 `op=Max`/`ts.exp` 已锚定）✅ |
| **Transpose** | 嵌套 `for range`（i/j）；整数下标 `A[i, j]` → `T[j, i]`（下标含常量与变量混合）；赋值；`tis.load`/`store` 或 `make_tensor` + copy | 下标 ∈ R6 白名单（§7 已含切片与 `[:, None]`，整数下标为其子形态）✅ |

**结论**：四 kernel 语法需求为 §7 案例已核对集合的**子集**，未出现任何需要扩展白名单的新类别；拒绝清单同样零命中。

## 3. 错误码段位冲突核对（task 3）

| 本 change | 既有事实（veps/design.md） | 冲突 |
|---|---|---|
| `E0101`–`E0107`（载体/顶层/装饰器/签名/语句/表达式/状态类体） | `E0301`（§2.2 作用域转移）、`E0402`/`E0403`（§2.3 MMA）、`E0501`（§2.3 warp 角色） | 无：E01xx 与 E03xx/E04xx/E05xx 不重叠 |
| 段位规则：E01xx 按检查管线顺序（载体→顶层→装饰器→签名→语句→表达式→类体），`E0108`–`E0199` 保留 | 既有段位按检查阶段分配（E03xx 类型/作用域、E04xx 计算原语、E05xx 执行结构） | 一致：同为"段位=检查阶段"的分配原则 |

## 4. 总结

- 白名单充分性：五 kernel 全部语法需求被白名单覆盖，拒绝清单零命中——task 2 通过。
- 段位冲突：无重叠、分配原则一致——task 3 通过。
- strict validation（task 1）：`npx openspec validate add-language-syntax-acceptance-set --strict` 与 `--all --strict` 通过（2026-09-24，主智能体与设计审查 subagent 多次独立复跑）。
