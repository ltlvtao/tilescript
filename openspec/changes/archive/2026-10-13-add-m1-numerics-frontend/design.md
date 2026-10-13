# Design: add-m1-numerics-frontend

## 设计范围导航

- 行为契约：`openspec/specs/numerics/value-semantics/spec.md`（7R/39S；本 change delta MODIFIED R7「报告契约与 E06xx 段位」4→5 Scenario）+ `openspec/specs/toolchain/cli/spec.md`（R3 MODIFIED：五段快照 + `passed` 最小形态）+ `openspec/specs/diagnostics/json-schema/spec.md`（R1 登记性 MODIFIED：诊断 JSON 产出义务以诊断实现里程碑为生效条件——消解五段零命中首次可达与诊断产出面未实现之间的义务缺口，5→6 Scenario）。其余 6 个数值语义 Requirement 零 delta 实现。
- 既有实现面：`tilescript/typecheck/`（类型环境基座——`types.py` 类型对象与 Dim 三形态、`infer.py` Inferencer 让渡面、`bindings.py` 遍历器、`symbols.py` 形参表）、`tilescript/primitives/results.py`（原语结果类型构造器）、`tilescript/primitives/fold.py`（维长多项式等价）、`tilescript/execution/`（形态环境与段编排先例）、`tilescript/pipeline.py`/`tilescript/cli.py`（段接入与诚实退出）。
- 本 design 的 D 编号仅辖本 change；`veps/design.md` 为非正式总设计，不构成约束。

## 当前实现（基座事实）

1. 类型段对算术/比较/逻辑/一元/float 字面量/`True`/`False` 全让渡（`infer.py` L54-67：Constant float→UNKNOWN、BinOp/Compare/BoolOp/UnaryOp→UNKNOWN 不递归）——数值语义域在类型段零承载，本 change 全新实现面。
2. 类型段 env：kernel 形参注解类型（`symbols.kernel_params`）+ 语句流绑定；算术赋值绑定 UNKNOWN（`x = S * scale` 后 x 为 UNKNOWN）。comptime 参数只有名集合（`kernel_comptime_names`），无值表——形参默认值（E0104 保证单 int 字面量）在 AST `arg.default` 可读，是 comptime 值的唯一静态来源。
3. 原语段在合法调用下构造结果 `TensorType`（`results.py` 构造器表 + `infer_shape_component` shape 组件归约）；`traverse` 维护同构 env。
4. 执行结构段形态环境（`execution/shape_env.py`）为粗形态（无 dtype/dims），只辖 E05xx 结构判定面——数值段需要完整类型（dtype/dims/scope），不复用该层。
5. 语法段封闭了数值段可见表达式域：算术 `+ - * /`、单比较、`and`/`or`、一元 `-`/`not`、int/float 字面量、`True`/`False`/`None`（E0106 拒三元/链式比较/`//`/`%` 等）；语句无 `while`/注解式局部赋值（E0105）。数值段遍历器无需处理这些形态。
6. pipeline 四段短路编排（`IMPLEMENTED_STAGES` 四元组）；CLI 拒绝优先 → incomplete（`pending_stages` 非空期间）。
7. 段间短路下数值段输入的前置保证：语法/类型/原语/执行四段零拒绝——模块内 `tis.*` 调用参数形态合法（原语段已裁）、comptime 名集合与状态类 registry 可确定性重放（执行段先例 `execution/__init__.py` L38-39）。

## GAP 分析

| spec Requirement | 现状 | GAP |
|---|---|---|
| R1 算术操作数与结果（10S） | 零承载（类型段让渡） | 全新：家族/dtype/scope/shape 裁决 + 结果类型派生 |
| R2 一元负与具名常量 inf（5S） | 零承载 | 全新：待定型常量形态 + 位置绑定 |
| R3 比较逻辑与布尔封闭（4S） | 零承载 | 全新：布尔值条件语境专用地位 + 使用封闭集 |
| R4 常量与标量值绑定（7S） | 零承载（原语段只裁 value 非标量形态） | 全新：full value 位/算术操作数位/再绑定位三类 |
| R5 comptime 折叠与 cdiv 求值（3S） | 形态面在执行段（E0505 种类）；值面零承载 | 值环境 + 折叠 + cdiv 值 |
| R6 cast 数值效果（6S） | 零承载（运行期承诺） | 间接承载（返回类型派生 + 零拒绝负例） |
| R7 报告契约（4S+delta 1S） | 段不存在 | 全新段编排 + pipeline/CLI 接入 |

## 修改方案

### D1 段编排与模块布局

`tilescript/numerics/` 新域，四模块（划分原则：判定域内聚，与 execution 域同粒度）：

- `__init__.py`：段编排 `check_module(tree)`——comptime 名与值表 + 状态类 registry 确定性重放（与执行段同模式，输入为零拒绝模块）→ 遍历 kernel 与 produce/consume 体（与 `typecheck/bindings.check_device_functions` 同范围——设备代码的全部语句面）→ `report.finalize` 收口。段名 `numerics`，段内 order：E0601=1 → E0606=6（spec R7 段内顺序）。**签名不带 target**：数值语义属语言层，跨 HAL 行为不变（hal spec「跨 HAL 行为不变面」）；`pipeline.compile_stages` 调用点 `numerics.check_module(tree)`。
- `value_env.py`：数值类型环境（D2）与表达式推断总入口（数值域节点自载 + 其余委托）。
- `binary_ops.py`：算术（R1）与比较/逻辑（R3）裁决 + 结果派生（D4/D6）。
- `constants.py`：待定型常量与常量绑定（R2/R4，D5）+ 折叠值环境（D7）。

### D2 数值类型环境：复用类型对象 + 委托推断 + 类型域拒绝零重放（核心架构决策）

数值段 env：`dict[名, 类型]`，值复用 `typecheck/types.py` 全部类型对象（`ScalarType`/`TensorType`/`StateType`/`PointerType`/`UNKNOWN`/Dim 三形态）——**不建平行表示**。env 起点 = `symbols.kernel_params(tree, comptime_syms)`；produce/consume 嵌套体按 `bindings._walk_nested` 同构快照恢复（外层快照 + 自身形参 + 体后恢复）。

表达式推断分派：

- **数值域节点自载**：`Constant`（int→`ScalarType(COMPTIME_INT)` 带值、float→待定型、`True`/`False`→布尔、`None`/`str`→让渡）、`Name('inf')` 预置名（env 未遮蔽时→待定型 inf；env 命中优先——residual 5）、`BinOp`/`Compare`/`BoolOp`/`UnaryOp(USub/Not)`（binary_ops 裁决，子表达式递归自载入口）、`tis.*` 调用（D3 返回类型 + full value 位绑定裁决）。
- **其余节点委托** `typecheck/infer.Inferencer`（共享 env 引用）：`Name`（非 inf）、`Attribute`（状态类字段）、`Subscript`（切片维派生——复用避免重实现）、非 `tis` `Call`（状态类构造→`StateType`、带注解设备函数→返回注解类型）。

**类型域拒绝零重放**：委托 Inferencer 会触发其 `_reject`（E0303 构造实参/调用实参检查）——数值段 env 比类型段精确（算术结果有类型），重放可能产生类型段当时因 UNKNOWN 让渡而未报的 E0303。类型段「UNKNOWN 参与绑定不产生 E0303」是 type-system spec 冻结行为，数值段不得以更精确环境改变其输出面。实现：数值段持 `_SilentInferencer(Inferencer)` 覆写 `_reject` 为 no-op——**只取类型推断纯函数面，类型域错误码不在委托路径出现**。

**算术赋值再绑定等价检查由数值段显式承载**（R7 明文授权面——「算术表达式运算违规落 `E06xx`，其赋值再绑定不等价落 `E0303`——同位置双命中时只报 `E0303`」；stable R7 Scenario「同位置类型与数值双命中只报类型错误」的唯一可满足路径）：`Assign`/`AugAssign` 的源为数值域算术表达式（`BinOp` 算术/一元负）且目标已绑定非 UNKNOWN 时，数值段以算术结果类型做再绑定等价判定——运算违规（E06xx）时结果类型按类别近似（Tensor 类别取左操作数 dtype/scope/广播 shape、标量取左操作数家族/dtype），等价判定复用 type-system 语义（`equivalent` + `compatible` 单向兼容）；不等价 → 构造 `E0303`（`stage="type-system"`、order 取类型段 E0303 口径、`category="assign"`、建议文案复用类型段 mismatch 格式）。**同位置双命中只构造 E0303 一条**（构造时择一——E06xx 不追加；不经 finalize 的 order 竞争，规避各段 order 为段内序的跨段比较语义错位）；单命中各自照报。非算术源（委托路径）的再绑定等价维持让渡（residual 1 收窄面）。

数值段 env 的赋值绑定：`Assign` 首绑定记录推断类型（含算术结果）；再绑定（目标已绑定）时数值段裁待定型定型规则（D5 位置③）、数值域违规（如布尔绑定 E0605）与上述算术再绑定等价（E0303）；`AugAssign` 按等价展开（运算面本段 E06xx，展开再绑定等价同上承载）。`For` 循环变量：`range(...)` 可迭代→`ScalarType(INT)`；`tis.tile_iter()`→UNKNOWN（元素类型未定义，residual 6）；`With` 绑定名→UNKNOWN（warp_group 语境归执行段，已裁）。

### D3 `tis.*` 调用的数值段返回类型

短路前置保证：数值段所见的 `tis.*` 调用全部通过原语段参数契约（E0406/E0408 形态）与执行段结构检查（E05xx）——参数形态可读。返回类型表（复用 `primitives/results.py` 构造器，不重实现）：

- `alloc_shared`→`TensorType(dtype, dims, "Shared")`；`zeros`/`full`→`alloc_result`（dtype/scope 读关键字实参，shape 组件经 `infer_shape_component` 归约——复用，env 传数值段 env）；`make_tensor`→承接 `PointerType` 实参；`cast`→`cast_result(x, dtype)`（x 取数值段类型）。
- 六计算原语→`dot_result`/`reduce_result`/`elementwise_result`/`transpose_result`（compute-ops 结果规则）。
- 执行结构域：`range`→可迭代值（循环变量 int）；`block_idx`→`ScalarType(INT)`；`cdiv`→D7（种类 + 折叠值）；`Pipeline`/`produce`/`consume`/`run`/`warp_group`/`warp_group_sync`/`barrier`/`load`/`store`→让渡（UNKNOWN——结构判定归执行段/原语段，无数值域消费面）。
- `tis.full` 调用点额外承载 value×dtype 常量绑定裁决（R4 位置①，D5）——原语段已裁 value 非标量形态（E0406），数值段只裁标量值绑定规则。

### D4 算术裁决矩阵与广播派生（R1；E0601→E0604 判定序）

操作数分类：int 家族（`ScalarType` kind ∈ {`comptime_int`, `int`}）、dtype 标量（kind ∈ DTYPES）、`TensorType`（dtype 家族）、待定型（D5）、布尔（D6）、UNKNOWN/其他（让渡）。

判定序（报告契约段内顺序，同表达式多码只报首个）：

1. 布尔操作数 → E0605（算术操作数位不在布尔封闭集）。
2. 一侧 int 家族、一侧 dtype 家族（标量或 Tensor）→ **E0601**（int 字面量与 comptime 常量不豁免——spec R1 明文）；一侧 int 家族一侧待定型浮点 → E0601（无 dtype 语境不可定型，恢复建议提供浮点语境——design 决策，spec 未明文的兜底）。
3. 两侧 dtype 家族 dtype 不同 → **E0602**（同表达式多操作数同类违规合并一条列出全部——报告契约）。
4. `/`：两侧 int 家族或 dtype 为整型 → **E0604**（建议 `tis.cdiv`）。
5. 两侧 Tensor scope 不同 → **E0603**（报告两侧 scope）。
6. 两侧 Tensor shape 逐维广播：右对齐比较（长侧前缀维直接保留）——维长相等保留（取左维）、一侧为 1 扩展（取另一侧维）、否则 **E0603**（报告两侧 shape）。
7. 一侧 UNKNOWN → 全部让渡（不报，结果 UNKNOWN）。

结果派生：int 家族 `+ - *`：全 comptime → `COMPTIME_INT`（带 D7 折叠值），否则 `int`；dtype 不变；标量⊗Tensor → Tensor 的 shape/scope；Tensor⊗Tensor → 广播 shape、同 scope；待定型⊗dtype 家族 → 按该侧定型（dtype/shape/scope 承接，值按 R6 舍入承诺内部承载）。

**广播维相等判定 = `primitives/fold.py` 多项式等价**（两维 fold 结果相等 ⇔ 维长相等）：语言内已有的维长等价机制（E0405 相容判定同语义），数值段复用而非重定义——`ConstDim(64)` 与 `DerivedDim('*', (32, 2))` 不误报；fold 任一侧 None（不可折叠）→ 让渡该维判定（结果维 UnknownDim 传播）。相等时结果维取左操作数维（comptime 性由该维承载；同名 SymbolDim 的 comptime 性全局一致，无跨性冲突现实形态）。

增强赋值（`+=` 等）按等价展开裁决：运算面本段 E06xx；`S *= 2`（int 常量×Tensor）按展开 `S = S * 2` 命中 E0601（int 编译期常量不作算术操作数——spec R1 同条）。

### D5 待定型常量与 dtype 位置绑定（R2/R4；E0606）

待定型值类（数值段内部形态，带值）：float 字面量（含一元负 `-1.5` 形态，值取负）、`inf`、`-inf`（具名常量经一元负，符号承载）。`nan` 不在具名常量集（spec 冻结）。

定型位置（封闭，spec R4 枚举）：

- **位置②算术操作数位**：待定型与已定型 dtype 家族值运算 → 按该 dtype 定型（合法；`S * 2.0` 结果承接 S）；与 int 家族 → E0601（D4 第 2 条）。
- **位置①原语 dtype 标量参数位**（当前 `tis.full` 的 `value`）：按 dtype 实参裁决——float 字面量→浮点 dtype 合法（roundTiesToEven 正确舍入，值内部承载）/整型 dtype E0606；int 字面量与 comptime 表达式→整型 dtype 值域判定（i8 −128–127 / i32 ±2³¹，越界 E0606 报值域与实际值）、浮点 dtype 按整到浮舍入合法；`inf`/`-inf`→`f16`/`bf16`/`f32` 合法、`f8e4m3`（无无穷表示）与整型 E0606；运行期 `int` 名 E0606（无原语路径）；`comptime[int]` 名按其值（D7 值表）同字面量判定；dtype 标量名须同 dtype（跨 dtype E0606，无隐式转换）。
- **位置③再绑定赋值源**：目标已绑定浮点 dtype 标量 → float 字面量/`inf`/`-inf` 按其 dtype 定型（合法）；目标已绑定 int 标量 ← float 字面量 → E0606（float 字面量只在浮点 dtype 语境合法）；目标已绑定浮点标量 ← int 字面量 → E0606（int 常量 dtype 绑定限位置①原语参数位——spec R4 明文限定）；目标为 Tensor/UNKNOWN/其他 → 让渡（residual 1）。

裸绑定（首次绑定无语境）：float 字面量与 `inf` → **E0606**（恢复建议 `tis.full` 构造或与已定型值运算）。裸绑定 int 字面量合法（`ScalarType(COMPTIME_INT)`——int 家族自归类型，非待定型）。

`inf` 的位置绑定与一元负：`-inf` 绑定浮点位置合法（Scenario R2-1）；`inf` 出现在整型/f8e4m3/int 家族位置 E0606；一元负操作数布尔 → E0605；一元负 int 家族（comptime 值取负折叠）/dtype 家族（同 dtype/shape）合法。

### D6 布尔封闭（R3；E0605）

布尔值来源（数值段内部形态）：合法 `Compare` 结果、合法 `BoolOp` 结果、`not` 结果、`True`/`False` 字面量。**不进类型段类型世界**（spec 冻结——数值段内部枚举，序列化不出现）。

- 合法位置封闭集：`if`/`elif` 条件（`If.test`——elif 为嵌套 If 同构覆盖）、`and`/`or` 操作数、`not` 操作数、comptime 参数默认值（语法层 E0104 已裁形态限定 `True`/`False` 字面量；其 1/0 整数值经 D7 值表进入 R4 位置①值域判定——静态消费面，见 D7）。
- 封闭集外任一使用 → **E0605**：赋值绑定（`flag = a > b`）、原语/调用实参、返回值、下标、算术操作数、比较操作数、一元负操作数。实现：布尔值作为推断结果传播，到达上述位置节点即报。
- 非布尔条件 → **E0605**（`if seq_len:`——不支持真值语义；建议显式比较）。UNKNOWN 条件让渡。
- 比较裁决：操作数须同家族标量——int 家族两标量（comptime/int 单向兼容混用）或 dtype 家族同 dtype 两标量；Tensor 参与比较 E0605（建议归计算原语后续 change）；待定型/布尔/混家族作比较操作数 E0605；UNKNOWN 让渡。结果布尔。
- `and`/`or` 操作数须布尔（E0605）；结果布尔。

### D7 comptime 折叠与 cdiv 值（R5）

comptime 值表：kernel 形参中注解 `comptime[int]` 且带默认值者（E0104 允许单 int 字面量与 `True`/`False`——布尔默认值按 R3 取 1/0 整数值）`{名: 值}`（`arg.default` 读取）；int 字面量值直接可得。折叠域：全 `comptime[int]` 操作数的 `+`/`-`/`*`/一元负（整数精确算术）与 `tis.cdiv(a, b)`（`b` 值为 0 时让渡不报——E0505 已拒，本段不重复）。折叠值确定（同表达式重复折叠同值——报告契约确定性承载）。

值的可观察面（M1）：**R4 位置①值域判定消费**（`tis.full((4,), BC * 4 + 68, i8, ...)` 折叠 324 超 i8 值域 → E0606——R5×R4 连接验收）；合法折叠零拒绝（`BC * 2 - 1` 折叠 127 用于 value 位 i32 合法）；种类面（全 comptime → `COMPTIME_INT`）与执行段 E0505 口径一致（执行段管原语位形态、数值段管算术表达式与折叠值——分工无重叠）。cdiv 两实参 comptime → `COMPTIME_INT` 带折叠值 `ceil(a/b)`；否则 `int`。

`comptime` 默认值常量表达式形式仍由 E0104 拒（语法层单字面量限定）——数值段折叠不改变该约束（spec R5 明文）。

### D8 cast 与运行期承诺（R6 间接承载）

M1 编译期可判定拒绝面为零：cast 参数契约归原语段（E0406）、运行期数值效果（舍入/饱和/环绕/NaN 规则与跨后端位一致）是行为承诺非拒绝路径（spec R6「运行期转换失败 MUST NOT 产生编译期拒绝」）。数值段承载：`cast` 返回类型派生（D3——`cast_result`）+ 合法 cast 零拒绝负例 + 常量绑定值的舍入（float 字面量 roundTiesToEven、整到浮）内部值计算（诊断/后端消费，M1 无静态输出）。测试面：R6 六 Scenario 以间接承载映射（合法调用零拒绝 × 类型派生 + 运行期承诺以 spec 引用注记，不上卡）。

### D9 管线接入与 CLI passed（R7/R3 delta）

- `pipeline.py`：`IMPLEMENTED_STAGES` 追加 `numerics.STAGE_NAME`（五元组）；`compile_stages` 执行段非空短路后 `return numerics.check_module(tree)`。
- CLI：`pending_stages` 为空（五段全实现后恒空）→ 输出 `{"status": "passed"}`（单字段）+ 退出码 0；`rejected` 路径零变化（拒绝优先）。`passed` 输出无自由文本，逐字节一致天然承载（delta Scenario 2）。
- 段间短路（delta 新 Scenario）：四段任一非空 → 数值段不执行（与既有三处短路条款同构）。

### D10 测试映射

| spec 面 | 测试载体 |
|---|---|
| R1 10S | `tests/test_numerics_arith.py`（家族/dtype/scope/广播/种类/增强赋值全矩阵 + E0601–E0604 类别与文案） |
| R2 5S / R4 7S / R5 3S | `tests/test_numerics_const.py`（待定型/位置绑定/值域/折叠值连接） |
| R3 4S | `tests/test_numerics_bool.py`（条件/比较/封闭集外使用） |
| R6 6S | `test_numerics_const.py` 间接承载组（cast 类型派生 + 零拒绝负例） |
| R7 5S（含 delta） | `tests/test_pipeline.py`（五段接入/段内 order/同码合并/重复一致/段间短路）+ `tests/test_numerics_env.py`（同位置 E0303×E06xx 双命中只报 E0303——R7 Scenario 2 经 D2 算术再绑定承载面直测） |
| CLI R3 delta 3S | `tests/test_cli.py`/`tests/test_integration.py`（passed 输出/逐字节一致/拒绝优先/FLASH 五段零拒绝→passed） |

既有测试联动：段清单断言四→五、incomplete 断言翻转为 passed、类型段载体测试若有算术面复测（预期无——类型段让渡面不动）。

## 让渡面与 residual

1. **非算术源的再绑定类型等价**（E0303 域）：类型段对算术结果 UNKNOWN 让渡（冻结行为）、数值段零重放（D2）——算术赋值（`BinOp` 算术/一元负源）的再绑定等价已由数值段显式承载（D2/R7 授权面）；委托路径来源（`Name`/`Attribute`/`Subscript`/带注解设备函数调用）类型段本就精确推断，其再绑定等价无缺口；数值域自载的非算术源（`tis.*` 原语调用——D3 返回类型，类型段按 UNKNOWN 让渡）的再绑定等价 M1 让渡，后续 change 显式承载。
2. `StateType`/`PointerType` 参与算术/比较：M1 让渡（不报，结果 UNKNOWN）——spec 未定义违规形态。
3. `inf` 名遮蔽（用户绑定 `inf = ...`）：env 命中优先于预置名，不额外拒（spec 未定义）。
4. `tile_iter` 循环变量元素类型：UNKNOWN（spec 未定义其元素类型面）。
5. 广播维含 UnknownDim/fold None：该维判定让渡（UnknownDim 传播）。
6. float 字面量舍入值/`inf` 位模式/comptime 布尔 1/0：无静态输出面（诊断 JSON 归 `diagnostics/*` 实现里程碑）。
7. cast 运行期位一致与数值效果：运行时承诺（测试不上卡；跨硬件 rtol 属 `veps` 里程碑验收）。
8. import 别名/from-import：延续既有 residual（字面 `tis`/`inf`/`range` 识别）。
9. `nan` 具名常量：spec 冻结不引入。

## 长期基线刷新计划（归档时执行）

1. `openspec archive`：`numerics/value-semantics`「报告契约与 E06xx 段位」MODIFIED（4→5 Scenario 补段间短路）、`toolchain/cli` R3 MODIFIED（五段快照 + passed 最小形态）与 `diagnostics/json-schema` R1 登记性 MODIFIED（产出条件限定诊断实现里程碑，5→6 Scenario）并入 stable。
2. `overview.md`：当前状态改五段全实现、`passed` 可达（M1 前端检查段闭环，诊断 JSON 挂接仍延期）；`numerics/value-semantics` 基线行追加首个实现注记（39S 零 delta 实现 + R5×R4 值域连接注记）；`toolchain/cli` 基线行追加 passed 联动注记（第四次登记性 MODIFIED 并述五段快照与 passed 最小形态）；`diagnostics/json-schema` 基线行追加登记性 MODIFIED 注记（产出条件实现里程碑限定与 CLI 扩展点对仗）。
3. CLAUDE.md 无需更新（M1 前端检查段收口不改变协作约定；质量脚本清单不变）。
4. `veps/` 不回改。
