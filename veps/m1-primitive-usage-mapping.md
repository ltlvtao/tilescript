# M1 FlashAttention 原语用法映射核对记录

> 本文是 `openspec/changes/add-primitives-memory-ops` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/primitives/memory-ops/spec.md` 的 8 个 Requirement，及 `specs/language/type-system/spec.md` 的 MODIFIED delta（「类型不匹配拒绝」豁免边界扩展）。
> 依据：`veps/design.md` §7（FlashAttention 完整案例）。
> 引用约定：R1–R8 依次指 memory-ops spec 的「原语集合与调用结构」「转移格承载映射」「load 与 store 的数据维度契约」「加载模式与完成语义」「分配与视图构造原语」「dtype 显式转换」「屏障语义」「报告契约与 E04xx 段位」。

核对方法：§7 源码逐行列出全部原语用法，每项按段内 tiebreak 顺序（E0404→E0405→E0406→E0407）过四层检查并映射到 Requirement 条目；十类合法用法（tasks 充分性标准）必须全部覆盖、拒绝规则不得命中；非本 change 原语显式标注归属。cast 派生表达式实参按 design 裁决做形态级核对（数值语义 capability 定义前的依赖顺序限制）。

## 1. §7 逐原语用法映射（八原语全部用法）

| # | 源码（行号按 §7） | 原语用法 | 逐层核对（tiebreak 顺序） | 结果 |
|---|---|---|---|---|
| 1 | L477 `tis.make_tensor(Q_ptr, (seq_len, D))` | 视图构造 ×1/5 | R1：两位置实参 = 必选数 ✅；R5：ptr 为 `Pointer[f16, Global]`（Global ✅）、shape 元组（运行期 `seq_len` int 标量 + `comptime D`，均整数组件 ✅）→ 返回 `Tensor[f16, (运行期 seq_len, comptime D), Global]`（零拷贝、类型 layout 组件默认 RowMajor） | ✅ |
| 2 | L477 `tis.make_tensor(K_ptr, (seq_len, D))` | 视图构造 ×2/5 | 同 #1 | ✅ |
| 3 | L478 `tis.make_tensor(V_ptr, (seq_len, D))` | 视图构造 ×3/5 | 同 #1 | ✅ |
| 4 | L478 `tis.make_tensor(O_ptr, (seq_len, D))` | 视图构造 ×4/5 | 同 #1 | ✅ |
| 5 | L479 `tis.make_tensor(L_ptr, (seq_len,))` | 视图构造 ×5/5（一维） | 同 #1，shape 为单元素元组 `(seq_len,)` → 返回 `Tensor[f32, (运行期 seq_len,), Global]` | ✅ |
| 6 | L481 `tis.alloc_shared((BR, D), f16, layout=tis.Layout.swizzled(xor=0b11100))` | 分配 ×1/3（swizzled） | R1：两位置 + layout 关键字 ✅；R5：f16∈六 dtype 封闭集 ✅、shape 全 comptime ✅、`xor=0b11100`（=28）为 comptime[int] 非负 ✅ → 返回 `Tensor[f16, (comptime BR, comptime D), Shared]`（类型 layout 组件 RowMajor，物理属性 `swizzled(xor=28)` 记录于资源诊断 memory 段，不参与类型等价） | ✅ |
| 7 | L482 `tis.alloc_shared((BC, D), f16, layout=…swizzled)` | 分配 ×2/3 | 同 #6（BC 维） | ✅ |
| 8 | L483 `tis.alloc_shared((BC, D), f16, layout=…swizzled)` | 分配 ×3/3 | 同 #6 | ✅ |
| 9 | L485 `tis.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)` | load Sync 模式 | R2：Global→Shared = load 格、原语匹配 ✅；R3：dtype f16/f16 ✅、逐维长度相容——dim0 切片 `(bm+1)*BR − bm*BR` 静态折叠 `BR` vs `comptime BR` 支 (d) ✅、dim1 `:` 全取保留维 vs `comptime D` 支 (a) ✅；R4：`Sync`∈值域 ✅、kernel 顶层语境合法 ✅、表达式语句 ✅ | ✅ |
| 10 | L486 `tis.barrier()` | 屏障（无参） | R1：scope 省略取默认 `Block` ✅；R7：Block 汇合与跨线程可见性（承载 #9 的 Sync 可见性补充）；表达式语句 ✅ | ✅ |
| 11 | L492 `tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)` | load Async ×1/2 | R2：Global→Shared load 格 ✅；R3：折叠 `BC` vs buffer 维 `comptime BC` 支 (d) ✅（`buf.K` 类型由执行结构 capability 契约承载，本核对以其为 `Tensor[f16, (comptime BC, comptime D), Shared]` 为前提）；R4：`Async`∈值域 ✅、produce 嵌套函数体内（E0407 语境合法）✅、完成保证让渡 Pipeline 契约 | ✅ |
| 12 | L493 `tis.load(V[j*BC:(j+1)*BC, :], buf.V, mode=Async)` | load Async ×2/2 | 同 #11（buf.V） | ✅ |
| 13 | L497 `tis.zeros((BR, BC), f32, Register)` | zeros ×1/3（三位置） | R1：三位置实参 = 参数集总长 3（scope 按声明顺序位置传递）✅；R5：f32∈封闭集 ✅、shape comptime ✅、scope=`Register`∈{Register, Shared} ✅ → 返回 `Tensor[f32, (comptime BR, comptime BC), Register]`；作为 `tis.dot` 实参——dot 契约延期（计算原语 change），zeros 返回类型已定义 | ✅ |
| 14 | L506 `tis.zeros((BR, D), f32, Register)` | zeros ×2/3 | 同 #13 | ✅ |
| 15 | L510 `tis.zeros((BR, D), f32, Register)` | zeros ×3/3（init 内） | 同 #13；init 绑定本身走 type-system R7（非原语调用，E0303 适用）——zeros 返回类型满足 | ✅ |
| 16 | L511 `tis.full((BR,), -inf, f32, Register)` | full（四位置 + -inf） | R1：四位置实参 = 参数集总长 4 ✅；R5：value=`-inf` 为数学具名常量 `inf` 经一元负的表达式（R5 正文显式纳入）✅、f32 ✅、Register ✅ → 返回 `Tensor[f32, (comptime BR,), Register]`；`-inf`→f32 的数值表示延期数值语义 capability | ✅ |
| 17 | L505 `tis.cast(P, f16)` | cast ×1/2（Tensor） | R1：两位置 ✅；R6：x=P 为 Tensor（P 首绑自 `tis.exp` 结果——结果类型依赖数值语义，**形态级核对**通过）✅、f16∈封闭集 ✅ → 返回 dtype 换 f16、shape/scope 不变（形态级） | ✅（形态级） |
| 18 | L514 `tis.cast(st.O_acc / st.l[:, None], f16)` | cast ×2/2（派生表达式） | R1：两位置 ✅；R6：x 为除法表达式——结果类型依赖数值语义 capability，**形态级核对**通过（design/tasks 注明的依赖顺序限制）；f16 ✅ | ✅（形态级） |
| 19 | L514 `tis.store(tis.cast(…), O[bm*BR:(bm+1)*BR, :])` | store ×1/2 | R1：两位置 ✅；R2：Register→Global = store 格 ✅（src 为 cast 结果、scope Register——形态级）；R3：dtype f16/f16 ✅、折叠 `BR` vs `comptime BR` 支 (d) ✅、`D` vs `D` 支 (a) ✅；R4：表达式语句 ✅ | ✅ |
| 20 | L515 `tis.store(tis.log(st.l) + st.m, L[bm*BR:(bm+1)*BR])` | store ×2/2（一维） | R1：两位置 ✅；R2：Register→Global store 格 ✅；R3：一维——src 为算术表达式（形态级：`st.l` 为 `Tensor[f32, (BR,), Register]`，运算结果类型延期数值语义）、dst 切片折叠 `BR` vs L 的运行期 `seq_len` 维——src 形态级 `BR` 维与折叠后 `BR` 支 (a)/(d) 相容 ✅；dtype f32/f32 ✅；R4：表达式语句 ✅ | ✅（形态级） |

十类充分性标准覆盖核对：load Sync（#9）、load Async（#11/12）、store×2（#19/20）、barrier（#10）、make_tensor×5（#1–5）、alloc_shared×3 含 swizzled（#6–8）、zeros×3（#13–15）、full 的 -inf 值（#16）、cast×2 含派生表达式（#17/18）、Layout.swizzled 参数（#6–8）——**全部覆盖，全部通过，零拒绝规则命中**。

## 2. §7 出现的非本 change 原语（归属标注，映射完整性）

| 源码 | 原语 | 归属（proposal 非目标显式声明） |
|---|---|---|
| L476 `tis.block_idx(0)` | 线程索引 | 执行结构/内建函数域 change（`block_idx`/`cdiv` 延期项） |
| L488/L490/L495/L509 `tis.Pipeline`/`@pipe.produce`/`@pipe.consume`/`pipe.run`/`range` | Pipeline 机制 | 执行结构 capability |
| L509 `tis.cdiv(seq_len, BC)` | 整除工具 | 执行结构/内建函数域 change |
| L497/L505 `tis.dot`（含 `mma=`/`pad=`） | MMA 计算 | 计算原语 change（E0402/E0403 既有事实段） |
| L497/L500 `tis.transpose`/`tis.reduce`、L501 `tis.maximum`、L502/503 `tis.exp`、L515 `tis.log` | 计算/归约原语 | 计算原语与数值语义 change |
| L499 等 算术与比较运算 | 数值提升 | 数值语义 capability |

## 3. MODIFIED delta 影响核对（E0303 豁免边界扩展）

- #9/11/12（load）与 #19/20（store）移动原语实参：MODIFIED 前后均在豁免内（初始集合即 load/store），scope 组合走 E0301、维度走 E0405，无行为变化。
- #1–8/13–18（make_tensor/alloc_shared/zeros/full/cast）构造与转换原语实参：MODIFIED **前**存在 E0303/E0406 双解（type-system 豁免只点名数据移动原语）；MODIFIED **后**实参整体豁免 E0303，违规一律落 E04xx——§7 全部构造原语用法按 R5/R6 契约裁决通过，不触发双解。
- 非原语调用不受影响：L507 `AttnState(...)` 构造、L509 `init=` 绑定仍走 type-system R7 等价规则（E0303 适用，MODIFIED 明确保留）。
- 普通赋值不承载跨 scope 移动：§7 无此类赋值（所有跨 scope 移动均显式 load/store）。

## 4. 错误码段位核对（task 3）

| 段 | 错误码 | 归属 | 冲突核对 |
|---|---|---|---|
| 语法段 | E0101–E0107 | `language/syntax-acceptance-set`（已归档） | 与 E04xx 无重叠 |
| 类型段 | E0301–E0304 | `language/type-system`（已归档；E0303 适用集经本 change MODIFIED 收敛） | 与 E04xx 无重叠 |
| 原语契约段 | E0402/E0403 | MMA/pad 既有事实（veps，语义不变） | 本段既有占用，未动 |
| 原语契约段 | **E0404/E0405/E0406/E0407** | 本 change 新增（格类别/维度/参数值域形态/语境） | 与全部既有码无重叠；E0408–E0499 保留 |
| 执行结构段 | E0501 | warp（veps 既有事实） | 与 E04xx 无重叠 |

管线顺序 E01xx→E03xx→E04xx 与段内 tiebreak E0404→E0405→E0406→E0407 均与 design「修改方案」一致（R8 正文逐句核对相符）。

## 5. 结论

§7 十类原语用法全部映射通过（两项 cast/store 派生表达式按声明的形态级限制核对）；拒绝规则（E0404/E0405/E0406/E0407）零误命中；MODIFIED 消除构造原语实参双解且非原语调用行为不变；错误码段位无冲突。`primitives/memory-ops` 对 M1 案例的原语边界充分性成立。
