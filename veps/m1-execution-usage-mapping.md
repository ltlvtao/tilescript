# M1 FlashAttention 执行结构用法映射核对记录

> 本文是 `openspec/changes/add-execution-pipeline-structure` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/execution/pipeline-structure/spec.md` 的 8 个 Requirement，及 `specs/language/type-system/spec.md` 的 MODIFIED delta（「类型不匹配拒绝」豁免边界第二次扩展）。
> 依据：`veps/design.md` §7（FlashAttention 完整案例）；§2.3（warp 特化）、§2.5（Pipeline 抽象）作事实核对。
> 引用约定：R1–R8 依次指 execution spec 的「Pipeline 构造契约」「produce 与 consume 嵌套函数契约」「run 调用契约」「buffer 命名空间」「Async 完成保证与组管理不可见」「warp_group 上下文契约」「索引与整数内建」「报告契约与 E05xx 段位」。

核对方法：§7 源码逐行列出全部执行结构用法，每项按段内 tiebreak 顺序（E0501→E0502→E0503→E0504→E0505）过检查层并映射到 Requirement 条目；八类合法用法（design 充分性标准）必须全部覆盖、拒绝规则不得命中；非本 change 的语义显式标注归属。`init=` 两层分开核对（外层执行结构调用实参经 MODIFIED 豁免 E0303，构造内部关键字实参仍走 type-system 等价规则）。

## 1. §7 逐执行结构用法映射（八类全部用法）

| # | 源码（行号按 §7） | 用法 | 逐层核对（tiebreak 顺序） | 结果 |
|---|---|---|---|---|
| 1 | L488 `pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})` | Pipeline 构造 | R1：`STAGES: comptime[int] = 2`（L474 默认值）≥1 ✅；buffers 非空字典字面量、键 `"K"`/`"V"` 为字符串常量且内容形如标识符、互不相同 ✅、值 K_s/V_s 均为 `alloc_shared` 产物 `Tensor[f16, (comptime BC, comptime D), Shared]`（swizzled 为分配物理属性不进类型，见 m1-primitive-usage-mapping #7/8）✅；恰两关键字实参、无位置实参 ✅；位于 kernel 顶层函数体 ✅、每 kernel 至多一个实例（全案例仅此一处）✅；首次绑定 `pipe =` ✅ | ✅ |
| 2 | L490–493 `@pipe.produce def fetch(j: int, buf):` | produce 嵌套函数 | R2：装饰器接收者 `pipe` 为 Pipeline 值 ✅（R1 封闭集：装饰器接收者）；恰两形参——第一形参 `j: int` 带 `int` 注解 ✅、第二形参 `buf` 无注解 ✅；每实例至多一个 produce（仅一处）✅；体内 L492/493 两条 `tis.load`（合法语句，E0105 白名单 + memory-ops 契约）；`fetch` 名称未被显式调用、未作值使用 ✅（R2 禁止条款不命中） | ✅ |
| 3 | L495–507 `@pipe.consume def attend(j: int, buf, st: AttnState) -> AttnState:` | consume 嵌套函数 | R2：恰三形参——`j: int` ✅、`buf` 无注解 ✅、`st: AttnState` 状态类注解 ✅；返回注解 `AttnState` 与第三形参**同一类声明** ✅（状态链同 S）；每实例至多一个 consume ✅；体内 `buf.K`/`buf.V` 属性访问（#5/6）、计算原语不限（consumer 侧无限制；produce/consume 区别见 R6——此处非 warp_group 语境，无原语类别限制）；`attend` 未被显式调用、未作值使用 ✅ | ✅ |
| 4 | L509–512 `st = pipe.run(range(tis.cdiv(seq_len, BC)), init=AttnState(…))` | run 调用 | R3：恰两实参——位置 0 为 `range(...)` 内建调用 ✅（#7）、`init=` 为状态类构造值 ✅；init 状态类 `AttnState` 与 consume 第三形参注解同类 ✅；kernel 顶层 ✅、每实例恰一次 ✅、调用前已装饰 produce+consume（L490/L495 先于 L509）✅；接收者 `pipe` 为 Pipeline 值 ✅（R1 封闭集：run 接收者）；结果 `st` 类型 = `AttnState`（L514/515 `st.O_acc`/`st.l`/`st.m` 字段访问由 type-system 状态类规则裁决）✅。**init 两层核对见 §2** | ✅ |
| 5 | L492 `buf.K`（load dst）、L497 `tis.transpose(buf.K)`（dot 实参） | buf.K 属性 ×2 | R4：`K` 为已注册名 ✅；类型推导 = `buffers["K"]` 值类型 `Tensor[f16, (comptime BC, comptime D), Shared]` ✅；使用位置——L492 属性访问结果作 `tis.load` dst（R4：Async 拷贝写入目标 ✅）、L497 作 `tis.transpose` 实参（属性访问产生的 Tensor 值正常使用 ✅）；`buf` 整对象未逃逸 ✅ | ✅ |
| 6 | L493 `buf.V`（load dst）、L505 `buf.V`（dot 实参） | buf.V 属性 ×2 | 同 #5（`V` 注册名、类型 `Tensor[f16, (comptime BC, comptime D), Shared]`） | ✅ |
| 7 | L509 `range(tis.cdiv(seq_len, BC))` | range 可迭代值 | R7：恰一实参、实参 `tis.cdiv(seq_len, BC)` 结果为 `int` 种类（见 #8）✅；结果为可迭代值，使用位置 = run 第一实参 ✅（R7 封闭集：for 可迭代表达式 + run 第一实参）；`seq_len` 为运行期值——`n = cdiv(seq_len, BC)` 运行期求值，迭代序列长度 n；n≤0 时空迭代结果为 init（R3） | ✅ |
| 8 | L509 `tis.cdiv(seq_len, BC)` | cdiv ×1 | R7：两实参——`seq_len: int`（L472，运行期标量）✅、`BC: comptime[int]` ✅；除数 BC≠0 ✅；两实参非全 comptime → 结果种类 `int`（值折叠归数值语义 capability）✅ | ✅ |
| 9 | L476 `bm = tis.block_idx(0)` | block_idx ×1 | R7：dim=0 为 `comptime[int]` 非负 ✅；结果为运行期 `int` 标量（后续 `Q[bm*BR:(bm+1)*BR, :]` 切片索引使用由 type-system/make_tensor 契约承载）✅ | ✅ |

八类充分性标准覆盖核对：Pipeline 构造（#1）、produce 签名（#2）、consume 签名（#3）、run 调用（#4）、buf.K/buf.V 属性 ×4（#5/6）、range（#7）、cdiv（#8）、block_idx（#9）——**全部覆盖，全部通过，零拒绝规则命中（E0501–E0505 无一触发）**。

## 2. init= 两层分开核对（tasks task 2 注意项）

- **外层**（执行结构调用实参）：`init=AttnState(…)` 作为 `pipe.run` 关键字实参——经本 change MODIFIED，执行结构调用实参整体豁免 E0303；形态与契约约束由 R3 裁决：实参为状态类构造值 ✅、状态类与 consume 注解同类 ✅。即使违规也落 `E0502`（非 E0303）。
- **内层**（构造调用自身）：`AttnState(O_acc=O_new, m=m_new, l=l_new)`（L507 与 L510–512 两处）是**带源码层形参注解的调用**（`@tis.state` 类字段注解 L462–465），其关键字实参绑定仍走 type-system 等价规则（E0303 适用，MODIFIED 明确保留）：
  - `O_acc` 声明 `Tensor[f32, (BR, D), Register]` ← `O_new`（L505 算术+dot 表达式）——运算结果类型依赖数值语义 capability，**形态级核对**通过（design 注明的依赖顺序限制）
  - `m` 声明 `Tensor[f32, (BR,), Register]` ← `m_new`（L501 `tis.maximum` 结果）——同上，形态级
  - `l` 声明 `Tensor[f32, (BR,), Register]` ← `l_new`（L504 算术表达式）——同上，形态级
  - L510–512 init 内三处 `tis.zeros`/`tis.full` 返回类型已由 memory-ops 定义（m1-primitive-usage-mapping #15/16），与字段声明等价 ✅
- **range 侧内层**：`cdiv(seq_len, BC)` 为 tis.* 内建调用，实参契约由本 change R7 承载（E0505）；`range(...)` 实参契约同 R7——均不触发 E0303。

## 3. §2.3/§2.5 事实核对（E0501 语义与 warp_group 契约）

- **§2.3 示例**：`with tis.warp_group(role="producer", warps=2) as pg:` / `role="consumer", warps=6` / `tis.warp_group_sync(pg, cg, barrier_id=0)`——role∈{producer, consumer} ✅、warps comptime ≥1 ✅、sync 两**不同**绑定 ✅、barrier_id=0 comptime 非负 ✅、表达式语句位于两 with 之后 ✅。示例在 R6 契约下全部合法。
- **E0501 原文语义**："只允许 load/commit/barrier，出现 dot 则编译错误 E0501"——R6 正式化为"warp_group 语境违规"段内首码：producer 体内计算原语（dot）以 E0501 拒绝，**原场景语义不变**；原文"load/commit/barrier"中的 commit 对用户不可见（memory-ops 已封闭手写路径），R6 封闭为 load/store/barrier + 普通赋值 + 语句控制流，语义涵盖原文（load/barrier 保留，commit 折叠进编译器插入的组管理）。新增涵盖 `barrier(scope=WarpGroup)` 语境违规（承接 memory-ops 让渡）。
- **§2.5 示例不一致裁决**：§2.5 `buf: tis.BufferSlot` 注解 vs §7 无注解 `buf`——design 裁决不引入 `tis.BufferSlot`（不进类型世界），无注解为 canonical（§7 事实）；§2.5 的 `fetch(j: int, buf: tis.BufferSlot)` 若照抄将落 type-system 类型表达式封闭集拒绝（判据式让渡，具体路径以该 capability 权威文本为准）。
- **§2.5 签名与 R2 一致**：produce 两形参（j: int + buf）、consume 三形参 + 同类返回注解（§2.5 `compute` 返回 `AttnState` 与 st 注解同类）✅。
- **§2.5"半自动"原则承接**：用户不能手写 `wait_group`——R5 组管理不可见条款承接（编译器插入 commit/wait，用户不可手写）；Z3 形式化验证承诺 → R5 写为 MUST 行为承诺（完成保证、槽复用安全），验证机制延期实现 design ✅。

## 4. MODIFIED delta 影响核对（E0303 豁免二次扩展，task 3 前半）

- **L509 `pipe.run` 两实参**：MODIFIED 前（现行 stable）「类型不匹配拒绝」适用集表述含"执行结构方法与内建调用的实参绑定"——但 `pipe.run` 无源码层形参注解，等价规则无从判定（design GAP 所指双解源）；MODIFIED 后执行结构调用实参整体豁免 E0303，实参违规一律落 E0502——§7 用法（init= 状态类值、range() 可迭代值）合规，无 E0303 命中。
- **`range(...)` 实参**：同上豁免；实参契约由 R7 承载（E0505）。
- **保留面不受影响**：L507/L510 `AttnState(...)` 构造内部关键字实参（带注解调用）仍走 E0303（§2 已核）；L509 `st = pipe.run(...)` 赋值绑定——run 结果类型由 R3 定义（= init 状态类 AttnState），赋值两侧等价 ✅（type-system「表达式结果类型规则」的执行结构方法调用让渡由 R3 承接闭合）；L514/515 `st.O_acc`/`st.l` 字段访问与普通赋值由 type-system 裁决，不在豁免内。
- **无存量受众**：仓库无编译器代码，E0303 适用集收敛无迁移成本（proposal BREAKING 条目）。

## 5. 错误码段位核对（task 3 后半）

| 段 | 错误码 | 归属 | 冲突核对 |
|---|---|---|---|
| 语法段 | E0101–E0107 | `language/syntax-acceptance-set`（已归档） | 与 E05xx 无重叠；E0105 语句白名单（with 仅 `tis.warp_group`、for 仅 range|tile_iter）与 E0103 装饰器白名单（含 pipe.produce/consume）在**形态层**先决，E05xx 在**语义契约层**裁决——互补封闭无双解（round 1 审查维度 C 核验） |
| 类型段 | E0301–E0304 | `language/type-system`（已归档；E0303 适用集经本 change MODIFIED 二次收敛） | 与 E05xx 无重叠；pipe.run/range 实参移出 E0303 适用集 |
| 原语契约段 | E0402–E0407 | `primitives/memory-ops`（已归档） | 与 E05xx 无重叠；同位置跨段命中只报最早段（L492 Async load 违规时 E0407 先于 E05xx，R8 tiebreak） |
| 执行结构段 | E0501–E0505 | `execution/pipeline-structure`（本 change） | E0501 正式化（原 veps 事实语义不变）；E0502–E0505 新增；E0506–E0599 保留 |

- 跨段管线 E01xx→E03xx→E04xx→E05xx 与段内 tiebreak E0501→E0502→E0503→E0504→E0505：spec R8 与 design「E05xx 段位与 tiebreak」条目一致 ✅。
- §7 全部执行结构用法（§1 九项）经逐层核对零命中任何 E05xx 拒绝——合法路径全部覆盖 ✅。
