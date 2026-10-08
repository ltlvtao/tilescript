# M1 FlashAttention 计算原语用法映射核对记录

> 本文是 `openspec/changes/add-primitives-compute-ops` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/primitives/compute-ops/spec.md` 的 7 个 Requirement（ADDED）+ `specs/primitives/memory-ops/spec.md` 的 1 个 MODIFIED（登记性联动）。
> 依据：`veps/design.md` §7（FlashAttention 完整案例，签名默认值 D=BR=BC=64、STAGES=2）与 §2.3/§2.4（dot/reduce 签名草案、E0402/E0403 事实、PadPolicy 表）。
> 引用约定：CR1–CR7 依次指 compute-ops spec 的「原语集合与调用结构」「dot 的操作数与 MMA 契约」「reduce 的归约契约」「元素级与成对计算原语」「transpose 的数据重排契约」「计算原语类别清单」「报告契约与 E04xx 段位补充」。

核对方法：§7 源码逐行列出全部计算原语调用（六类），每项过调用结构（CR1）→ 操作数/参数契约（CR2–CR5）→ 报告层（CR7：E0408→E0402→E0403 段内序），合法路径全覆盖、拒绝规则零误命中；§7 全部计算原语均在 consumer（`pipe.consume` 嵌套函数）内或 kernel 顶层，无 producer 体内用法（E0501 零命中，类别清单 CR6 核对）。计算原语实参中的算术子表达式按 `numerics/value-semantics` 裁决（其类型标注沿用 `veps/m1-numerics-usage-mapping.md` 的推导结果），本记录核对原语侧契约。

## 1. §7 逐计算原语调用映射（六类全部用法）

| # | 源码（行号按 §7） | 用法 | 逐层核对 | 结果 |
|---|---|---|---|---|
| 1 | L497-498 `S = tis.dot(Q_s, tis.transpose(buf.K), tis.zeros((BR, BC), f32, Register), mma=tis.MMA(16, 8, 16), pad=tis.PadPolicy.Error)` | dot ×1（显式 mma/pad） | CR1：A/B/C 三位置实参 + `mma`/`pad` 关键字传递 ✓（可位置参数上限 3 未超）。CR2：A=`Q_s` 为 `Tensor[f16, (64, 64), Shared]`（alloc_shared 产物，comptime 维）✓；B=#2 transpose 结果 `Tensor[f16, (64, 64), Shared]` ✓（A/B 同 f16）；C=`zeros` 结果 `Tensor[f32, (64, 64), Register]` ✓；K 相容（A dim1=D=64 = B dim0）✓；`tis.MMA(16, 8, 16)` 三 comptime 字面量 ✓；M=64/16、N=64/8、K=64/16 整除 ✓（零 E0403；E0402 零命中——§7 语境该形状在 HAL 支持列表）；`PadPolicy.Error` 值在 mma=/pad= 封闭位置 ✓（CR2 值类别条款零违规）。结果 `Tensor[f32, (64, 64), Register]`（S 首绑） | ✅ |
| 2 | L497 内 `tis.transpose(buf.K)` | transpose ×1 | CR5：`buf.K` 为 `Tensor[f16, (BC=64, D=64), Shared]`（buffer 属性访问，类型由 execution buffer 命名空间推导）——二维 ✓、dtype 六集合之一 ✓、scope Shared 在允许集 ✓ → 结果 `Tensor[f16, (D=64, BC=64), Shared]`（dtype/scope 不变，两维互换）；作 dot 的 B 实参（#1） | ✅ |
| 3 | L500 `m_ij = tis.reduce(S, axis=1, op=Max)` | reduce ×1（Max） | CR1：三位置实参（可位置参数 4 未超）。CR3：x=S `Tensor[f32, (64, 64), Register]` Register ✓、dtype 六集合 ✓；axis=1 为 comptime 字面量 ∈ [0, 2) ✓；op=Max 封闭二值 ✓；scope 默认 Auto（确定性选择承诺）✓ → 结果 `Tensor[f32, (64,), Register]`（去 axis=1 维，dtype 不变） | ✅ |
| 4 | L501 `m_new = tis.maximum(st.m, m_ij)` | maximum ×1 | CR4：a=`st.m` 与 b=`m_ij` 均 `Tensor[f32, (64,), Register]`——同 dtype ✓、严格同 shape ✓、同 Register scope ✓ → 结果 `Tensor[f32, (64,), Register]`（同 dtype/shape/scope） | ✅ |
| 5 | L502 `alpha = tis.exp(st.m - m_new)` | exp ×1 | CR1：单位置实参。CR4：x 为算术子表达式（numerics 裁决 `Tensor[f32, (64,), Register]`）——浮点 dtype ✓、Register ✓ → 结果同 dtype/shape/scope `Tensor[f32, (64,), Register]`（alpha 首绑）；子表达式位置与调用位置各自报告（CR7） | ✅ |
| 6 | L503 `P = tis.exp(S - m_new[:, None])` | exp ×1 | CR4：x 为 numerics 裁决的广播结果 `Tensor[f32, (64, 64), Register]` ✓ → 结果 `Tensor[f32, (64, 64), Register]`（P 首绑）；作 cast 实参（L505）与 reduce 实参（#7） | ✅ |
| 7 | L504 `l_new = alpha * st.l + tis.reduce(P, axis=1, op=Sum)` 内 `tis.reduce(P, axis=1, op=Sum)` | reduce ×1（Sum） | CR3：x=P `Tensor[f32, (64, 64), Register]` ✓；axis=1 ∈ [0,2) ✓；op=Sum 封闭二值 ✓ → 结果 `Tensor[f32, (64,), Register]`；作算术 `+` 右操作数（numerics #7 已核） | ✅ |
| 8 | L505-506 `tis.dot(tis.cast(P, f16), buf.V, tis.zeros((BR, D), f32, Register))` | dot ×1（全默认） | CR1：三位置实参、省略 mma/pad ✓（mma 取 Auto、pad 取 Error）。CR2：A=`tis.cast(P, f16)` 结果 `Tensor[f16, (64, 64), Register]`（Register 在 A 允许集 ✓——cast 实参先经 memory-ops/numerics 裁决定型再进 dot 契约，管线分工）；B=`buf.V` `Tensor[f16, (BC=64, D=64), Shared]` ✓（同 f16）；C=`Tensor[f32, (64, 64), Register]` ✓；K 相容（A dim1=BC = B dim0）✓；mma=Auto 确定性选择（§7 语境候选形状整除：M=64/16、N=64/8、K=64/16——NVIDIA H200 m16n8k16 与 Ascend 910B m16n16k16 均整除，零 E0403）；结果 `Tensor[f32, (64, 64), Register]`（O_new 算术右操作数，numerics #9 已核） | ✅ |
| 9 | L515 `tis.log(st.l)` | log ×1 | CR4：x=`st.l` `Tensor[f32, (64,), Register]` 浮点 ✓ → 结果 `Tensor[f32, (64,), Register]`；作算术 `+` 左操作数（numerics 映射 #13）与 store src | ✅ |

六类充分性标准覆盖核对：dot ×2（#1 显式 mma/pad、#8 全默认——A 分别为 Shared 与 Register，两 scope 路径均覆盖）、transpose ×1（#2）、reduce ×2（#3 Max、#7 Sum）、maximum ×1（#4）、exp ×2（#5/6）、log ×1（#9）——**全部覆盖、全部通过，零 E0402/E0403/E0408 命中**。类别清单（CR6）核对：§7 全部计算原语调用位于 `pipe.consume` 嵌套函数体内（consumer 不设类别限制）与 kernel 顶层 store 路径，无 producer warp_group 体内用法——E0501 零命中。无值类别违规（`tis.MMA`/`tis.PadPolicy.*` 仅出现在 #1 的 mma=/pad= 位置）。

## 2. numerics 映射"计算原语域前提类型"逐处落实

`veps/m1-numerics-usage-mapping.md` 中以计算原语结果类型为前提的标注，自本 change 定义后逐处落实：

| numerics 映射标注处 | 前提类型 | 落实（本记录 #） |
|---|---|---|
| #3 `S = S * scale`——S 为 dot 结果（前提类型） | dot 返回 `Tensor[f32, (M, N), Register]` | #1：S 首绑 = dot 结果 f32 (64,64) Register ✓ |
| #4 `st.m - m_new`——m_new 为 maximum 结果（前提类型） | maximum 返回同 dtype/shape/scope | #4：f32 (64,) Register ✓ |
| #6 `alpha * st.l`——alpha 为 exp 结果（前提类型） | exp 返回同 dtype/shape/scope | #5：f32 (64,) Register ✓ |
| #7 `+ tis.reduce(P, axis=1, op=Sum)`——reduce 结果（前提类型） | reduce 去维、dtype 不变、Register | #7：f32 (64,) Register ✓ |
| #9 `+ tis.dot(...)`——dot 结果（前提类型） | dot 返回 f32 (M, N) Register | #8：f32 (64,64) Register ✓ |
| §2 #17 `tis.cast(P, f16)`——P 依赖 exp 返回类型（"仅 exp 契约"残留） | exp 返回同 dtype/shape/scope | #6：P = exp 结果 f32 (64,64) Register——**残留兑现，该路径零依赖** |
| §2 #18 完整闭合（无前提） | — | 维持 ✓ |
| §2 #20 store src——log 返回类型（"仅 log 契约"残留） | log 返回同 dtype/shape/scope | #9：f32 (64,) Register——**残留兑现，该路径零依赖** |
| §2 O_acc 行——dot 结果（前提类型） | #8 同 | ✓ |
| §2 m 行——"部分闭合（如实）：maximum 返回类型归计算原语域（形态级保留）" | maximum 返回同 dtype/shape/scope | #4：`m_new` f32 (64,) Register 与 `AttnState.m` 字段声明 `Tensor[f32, (BR,), Register]` 等价——E0303 路径现可完整判定，**形态级残留最终闭合** |
| §2 l 行——reduce 结果（前提类型） | #7 同 | ✓ |

至此 numerics 映射 §2 六处"形态级核对"标注全部完整闭合（原"部分闭合"的 m 行因 maximum 返回类型定义而闭合）；已归档文档不追溯改写，本节为落实与闭合的权威记录。

## 3. 承接面闭合核对（task 3 前半）

| # | 承接面 | 既有文本 | 本 change 承接 | 闭合核对 |
|---|---|---|---|---|
| ① | memory-ops E0402/E0403 既有占用 | L185"E0402/E0403 为计算原语（MMA/pad）既有占用，语义不变" | CR2 正式化 + CR7 段位补充 | ✅ 逐条对照 veps §2.3：E0402 = 显式 mma 不在 HAL 支持列表、报告列出全部支持形状（L154）；E0403 = 不整除且 Error、报告最近合法形状、MUST NOT 自动回退 SIMD（L155/L184）；mma=Auto 取 HAL 列表第一项确定性（L153）；Split 尾部 FMA 显式选择非自动回退（L155/L187） |
| ② | memory-ops 保留区间 | L185"E0408–E0499 保留给原语域后续扩展" | MODIFIED 联动（保留区间改 E0409–E0499、E0408 归属指明） | ✅ diff 逐字对照：delta 重述与 stable L181-205 唯一差异为段位句（E0408 归属 + E0409 起），正文与全部 4 Scenario 逐字相同 |
| ③ | numerics R3 两处指向 | "逐元素比较归计算原语域后续 change" | proposal 非目标落位裁决：仍不引入 | ✅ E0605 恢复建议文字保持有效（"后续 change"不限定为本 change）；CR1 封闭集 + 扩展须显式 change |
| ④ | execution E0501 类别引用 | "出现计算原语（`tis.dot` 等）以 E0501 拒绝" | CR6 六原语封闭清单 | ✅ "等"闭合；拒绝行为与错误码仍归 execution（本 change 不改其 Requirement，spec CR6 明文）；§7 零 producer 体内用法（§1 已核） |
| ⑤ | numerics Purpose 让渡 | "计算原语域为后续 change" | 本 change 即该 capability | ✅ 前提类型全部落实（§2） |
| ⑥ | type-system L146 让渡 | "模块属性（tis.*）访问及原语调用表达式的结果类型由 primitives/* 契约定义" | 六原语调用结果类型（CR2–CR5）+ MMA/PadPolicy 值类别（CR2 值类别条款） | ✅ `tis.MMA(...)` 构造与 `tis.PadPolicy.*` 属性访问的结果侧闭合（dot 实参语境专用值，他处 E0408） |

## 4. 错误码段位核对（task 3 后半）

| 段 | 错误码 | 归属 | 冲突核对 |
|---|---|---|---|
| 语法段 | E0101–E0107 | `language/syntax-acceptance-set` | 与本域无重叠；调用/属性访问/关键字实参/元组实参为语法层通用类别（E0106），计算原语调用形式先决、语义契约后置——互补无双解 |
| 类型段 | E0301–E0304 | `language/type-system` | 与本域无重叠；E0303 豁免（tis.* 原语实参）覆盖计算原语实参（type-system L176 "tis.* 原语"不含 capability 限定）；计算原语结果的再绑定不等价落 E0303 |
| 原语契约段 | E0402/E0403/E0408 / E0404–E0407 / E0409–E0499 | 本 change（正式化/新增）/ memory-ops / 保留 | E0408 与全部既有码零重叠（grep 核实既有 specs 中 E0408 除 memory-ops 段位句外零出现）；同位置无跨集双命中（原语名唯一决定适用集——CR7）；memory-ops 段位句经 MODIFIED 联动一致（§3 ②） |
| 执行结构段 | E0501–E0505 | `execution/pipeline-structure` | 与本域无重叠；E0501 的类别判定面由 CR6 供给，拒绝行为归 execution 不变 |
| 数值语义段 | E0601–E0606 | `numerics/value-semantics` | 与本域无重叠；计算原语实参中的算术子表达式落 E06xx（子表达式位置与调用位置各自报告——§7 事实 #5/#6/#7/#8/#9 均经此分工，先 numerics 定型再进原语契约） |

- 段内 tiebreak E0408→E0402→E0403（先形态契约后硬件支持面后对齐策略）：spec CR7 与 design「修改方案」一致 ✅；与 memory-ops 段内序（E0404→E0405→E0406→E0407）为不同原语各自适用，无交叉。
- 跨段五段管线 E01xx→E03xx→E04xx→E05xx→E06xx 不变（CR7 明文；既有 specs 表述无需回改）✅。
- §7 全部计算原语调用（§1 九项）经逐层核对零命中 E0402/E0403/E0408——合法路径全部覆盖 ✅。
