# Design: add-m1-type-system-frontend

## 设计范围导航

- 权威行为契约：`openspec/specs/language/type-system/spec.md`（8R/32S + 本 change 两条 delta：R5 补标量字段路径、R8 补段间短路——验收面 34S）。
- 强依赖只读复用：`primitives/memory-ops`（load/store 参数集——E0301 的 src/dst 位置）、`execution/pipeline-structure`（produce/consume 签名形态——返回注解检查辖域；显式调用禁止归 E0502 不在辖内）。
- 既有实现基座：`tilescript/frontend`（carrier.parse、top_level.iter_entry_functions、_has_tis_decorator、report.Rejection/finalize）、`tilescript/cli`（诚实退出）。
- 本文档只裁实现结构与既有 spec 的映射；不重述 spec 语义。

## 当前实现与 GAP

- syntax 段（`E01xx`）全量已交付（`add-m1-syntax-frontend`）：AST 载体、入口函数发现、状态类形态识别、报告 finalize 均可复用；但**没有任何类型表示、符号表、表达式类型推断**——type-system 段从零建立这三层。
- `Rejection` 无 stage 区分（现值恒 `"syntax"`）；CLI 管线是单段直连（`frontend.check_module`），无多段编排点；既有 CLI 测试断言单段 incomplete（随 `toolchain/cli` delta 联动更新）。
- 两处 spec 豁口（proposal Why 3）：R8 段间执行语义 undefined、R5 标量状态字段 `E0304` 路径 undefined——本 change delta 补洞。

## 修改方案（裁决记录）

### D1 新包 `tilescript/typecheck`（模块边界 = 错误码边界，延续 frontend 惯例）

```
tilescript/typecheck/
  __init__.py      # check_module(tree) -> list[Rejection]：类型段管线（E0302→E0304→E0301→E0303 收集 + finalize）
  annotations.py   # E0302：注解解析器（结构规则）——三类检查位置的共享入口
  state_fields.py  # E0304：@tis.state 字段约束（Pointer/scope/标量三路径）+ 状态类注册表构建
  transfer.py      # E0301：load/store 非法格检查
  bindings.py      # E0303：绑定等价（赋值/return/构造/带注解调用实参）+ 单类型不变量 + 下标基对象
  types.py         # 类型表示 + 等价判定（R3 全量）+ 类型→描述文本（建议与报告用）
  infer.py         # 表达式类型推断（R6 实现面 + 让渡面 UNKNOWN）
  symbols.py       # 符号表：kernel 形参、函数体流式作用域、嵌套函数签名
```

依赖方向：`typecheck` → `frontend`（AST 原语与 Rejection）；`frontend` 不反向依赖；`pipeline` 编排两段，`cli` 只依赖 `pipeline`。状态类注册表归 `state_fields`（类体注解解析天然在此），`symbols`/`annotations` 消费——任务依赖序：注册表（3.x）先于 produce/consume 状态类名注解解析（2.3）。

### D2 类型表示（`types.py`）

- dtype/scope 封闭集冻结为常量元组（与 spec 六值/三值一致）；layout 唯一合法值 `RowMajor`（显式与省略等价——表示层不存省略态，等价判定恒真）。
- `TensorType(dtype, dims: tuple[Dim, ...], scope, layout)`；`ScalarType(kind)`：kind ∈ {`int`, `comptime_int`, dtype 标量（六值）}；`StateType(class_name, fields)`（fields 为有序 `(名, 类型)` 列表，名义等价仅比类名）；`UNKNOWN` 单例。
- `Dim` 统一三形态（常量/运行期由 `comptime` 标志承载，避免值域装不下符号与派生）：
  - `ConstDim(value: int)` —— int 字面量（四进制在 AST 层同 Constant(int)）；
  - `SymbolDim(name: str, comptime: bool)` —— 参数符号（`comptime[int]` 形参→True、`int` 形参→False）；
  - `DerivedDim(op: str, operands: tuple[Dim, ...], comptime: bool)` —— 算术派生（操作数全 comptime→True，否则 False）。
- 维度等价（R3）：`ConstDim`×`ConstDim` 值相等；`SymbolDim`×`SymbolDim` 同名且 comptime 同；`DerivedDim`×`DerivedDim` 同 op、comptime 同、逐操作数等价（R6「相同运算与逐操作数等价」）；**跨形态不等**（无数值折叠：`BR*2` 与 `128` 不判等——折叠归 numerics 段，登记 residual）。含 UNKNOWN 操作数的派生见 D4。
- 等价判定入口 `equivalent(a, b)`：Tensor 逐组件（dtype/dims 逐维/scope/layout）、Scalar 同种类（`comptime_int`≠`int`）、StateType 同类名、UNKNOWN 恒不参与（见 D4）。

### D3 符号表（`symbols.py`）

- 模块级：状态类注册表（`top_level` 层 `@tis.state` ClassDef → `StateType`；字段类型经 annotations 解析）；同名状态类——spec 未定义，按首次注册（登记 residual，不发明拒绝）。
- kernel 形参：注解解析为类型（`int`→`int`、dtype 名→dtype 标量、`comptime[int]`→`comptime_int`、`Tensor`/`Pointer`→结构解析）。
- 函数体流式作用域：语句序遍历，`Assign(Name)` 首次绑定定类型；再绑定按单类型不变量（D6）。for 循环变量（range/tile_iter）→ UNKNOWN（元素类型归 execution/numerics）。嵌套函数（produce/consume）体的作用域 = 外层快照 + 自身形参（`j: int`、buffer 形参 UNKNOWN、`st: 状态类`、返回注解）。
- produce/consume 签名形态（两/三形参、注解位置）的**契约违规**归 `E0502`（execution 段）——本段只消费可解析部分：形参注解可解析则入表（含状态类名查注册表），不可解析（无注解/形态不合）该形参为 UNKNOWN，不报 `E03xx`。

### D4 未知类型策略（R6 让渡面的实现语义）

让渡清单（结果类型 = UNKNOWN）：`tis.*` 原语调用（含 make_tensor/zeros/dot 等已规格化返回类型者——实现归 primitive-contract 段）、算术/比较/逻辑/一元运算、`pipe.run`/`range` 等执行结构调用、Pipeline 值与 buffer 属性访问（`pipe.*`/`buf.*`）、warp_group with 绑定名（`g`）、float/bool/`None`/`inf` 常量、for-range/for-tile_iter 循环变量、UNKNOWN 变量引用、UNKNOWN 基对象的下标/切片/属性访问。
- UNKNOWN 参与**任何**绑定不产生 `E0303`（源未知或目标类型未知均跳过）。
- **UNKNOWN 组件传播**：切片/下标边界含 UNKNOWN 操作数（如 `bm*BR` 中 `bm` 让渡）→ 该派生维为 UNKNOWN 维 → 含 UNKNOWN 维的 TensorType 整体按 UNKNOWN 参与绑定（不做部分等价判定——等价是全有或全无）。
- 变量首次绑定表达式为 UNKNOWN → 变量类型 UNKNOWN，此后保持（已知值再绑定也不升级、不回溯报告——后续段全类型推断时该变量类型由该段推断，本段不预支）。
- 已知类型基对象的派生（切片/下标/字段访问，且派生不含 UNKNOWN 组件）为已知；已知×UNKNOWN 的算术为 UNKNOWN。

### D5 E0302：注解解析器 = 结构检查器（`annotations.py`）

- 解析 AST 注解表达式 → 类型对象或 `E0302` Rejection（带被违反规则类别与建议：未知 dtype 列六值、未知 scope 列三值、参数数量/顺序错给最近合法形式、shape 组件非整数类别）。
- 三类检查位置：kernel 签名参数注解（E0104 形式内的内部结构——E0104 已裁形式类别，E0302 裁内部：dtype/scope 值域、参数序、shape 组件；E0104 拒绝的形式不再到达本段——段间短路保证）、`@tis.state` 字段注解、produce/consume 形参与返回注解（含状态类名注解——解析为 StateType，查注册表）。
- `Pointer` 出现在 kernel 签名合法（E0104 形式集内）；`Tensor[...]` 第四参数 layout 仅 `RowMajor`。
- shape 组件类别判定（R2）：int 字面量 → `ConstDim`；已知 comptime 符号 → `SymbolDim(name, comptime=True)`；int 符号 → `SymbolDim(name, comptime=False)`；算术表达式 → `DerivedDim`（操作数 comptime 性递归判定：全 comptime→True）；float/bool/其他表达式 → `E0302`。
- 同一注解一次解析产出一个结果（结构失败即 Rejection，不产类型）——供 E0304 与符号表复用；同位置 E0302 优先于 E0304（R8 段内顺序，order 字段承载）。

### D6 E0304：状态类字段约束（`state_fields.py`）

- 字段注解经 D5 解析成功后查约束：类型为 `Tensor` 且 scope==Register → 合法；`Tensor` 但 scope≠Register → E0304（scope 类别）；`Pointer` → E0304（pointer 类别）；**标量种类（`int`/`comptime[int]`/dtype 名）或其余非 Tensor 注解 → E0304（scalar 类别，R5 delta 补全路径）**。
- 解析失败（E0302）的字段不产 E0304（同位置段内 E0302 优先）。
- 建议：状态字段只接受 Register scope 的 Tensor；Shared/Global 数据经移动操作进入计算后由状态承载（R5 建议文本冻结）。

### D7 E0303：绑定检查（`bindings.py` + `infer.py`）

- 绑定位置四类：
  1. 赋值：`Assign` 目标 Name → 单类型不变量；目标 Subscript → 目标侧下标派生类型×源类型；目标 Attribute → 字段声明类型×源类型（基对象为已知状态类时才查）；
  2. `Return` × 所在函数返回注解（consume 形态）；无返回注解不查（kernel 顶层与 produce 无返回注解形态——proposal 非目标）；
  3. 状态类构造调用：`Name` 调用且名字在注册表 → 关键字实参×同名字段类型；位置实参与未注册名不查（proposal 非目标）；
  4. **带源码层形参注解的用户函数调用实参**：`Name` 调用且名字解析为设备代码内 produce/consume 嵌套函数 → 位置实参×按序形参注解类型、关键字实参×同名形参注解类型（R7 适用面「带源码层形参注解的调用的实参绑定」；调用本身的合法性归 `E0502` 不在辖内——管线顺序上本段先于 execution 段，实参绑定裁决先行）；无注解形参位（buffer 形参）实参不查（UNKNOWN）。
- 单向兼容：源 `comptime_int` × 目标 `int` 接受；反向拒绝（建议改整数字面量/`comptime` 参数/编译期可求值表达式）——R7「comptime 值绑定 int 位置」Scenario 的验收构造落在第 4 类（显式调用 `j: int` 形参的 produce/consume 函数传字面量）。
- 跨 scope 赋值（两侧已知 Tensor 且 scope 不同）→ 不等价拒绝，建议显式移动原语（R7 末句）。
- 差异组件建议映射：dtype→`tis.cast`、scope→显式移动原语、shape→核对维度、种类→核对标量种类（R7 恢复建议规则）。
- 单类型不变量遍历与语句序一致（流式）；`AugAssign` 目标类型×结果类型——算术结果 UNKNOWN，实际仅 UNKNOWN 跳过（保留遍历占位）。
- 下标基对象已知非 Tensor（标量/状态类）→ `E0303`（R6 下标规则，建议核对基对象类型）。

### D8 E0301：转移非法格（`transfer.py`）

- 识别 `tis.load`/`tis.store` 调用（`Attribute(value=Name 'tis', attr∈{load,store})`）；实参位置 0=src、1=dst（`primitives/memory-ops` R1 参数集只读复用；位置不足/多余实参归 `E0406`，本段不裁）。
- 两实参类型均已知为 Tensor 且 scope 组合 = Global→Global（矩阵唯一非法格）→ `E0301`，建议按矩阵列 Global 的合法目标：Shared（load）与 Register（load）。
- 其余情形不报：合法格（load/store 格与原语匹配性属 `E0404` 段外知识，本段只裁非法格）、copy/move 格误用（`E0404`，primitive-contract 段）、任一侧 UNKNOWN 或非 Tensor（`E0406`/后续段）。

### D9 报告、管线编排与 CLI 联动

- `Rejection` 增加 stage 值（构造参数，默认 `"syntax"` 保持 frontend 兼容；typecheck 传 `"type-system"`）；stage 不进 JSON 五字段（v1 冻结不变），仅内部归属。
- typecheck 段内 order：E0302=1、E0304=2、E0301=3、E0303=4；复用 `report.finalize`（同位置取 order 最小条目集合——段内短路由「每位置每检查器只产一条」+ finalize 联合保证；跨段比较不存在，段间已短路）。
- 建议文本冻结表（D6/D7/D8 的 slug 与中文建议常量化），确定性：不依赖 dict 迭代序（注册表按声明序、字段按类体序）。
- `tilescript/pipeline.py compile_stages(source) -> (rejections, implemented)`：`frontend.check_module`（syntax 段，含 E0101 特例）→ **非空即返回**（段间短路，R8 delta）→ `carrier.parse` 重解析（frontend.check_module 消费 source 文本；typecheck 需 AST，二次 parse 确定性无虞）→ typecheck.check_module → 合并输出。
- `cli` 改调 pipeline；incomplete 实例化 `implemented_stages=["syntax","type-system"]`、`pending_stages=["primitive-contract","execution-structure","numerics"]`（`frontend.ALL_STAGES` 为唯一段名来源，两处列表由其推导——R3「两数组合并恰为五值」由构造保证）。**既有 CLI 测试的单段 incomplete 断言随 `toolchain/cli` delta 联动更新为两段**（tests/test_cli.py 内更新，属规格联动非顺改）。
- `frontend.check_module` 公共 API 语义不变（单段入口，测试与既有用例不动）。

### D10 测试映射（stable 32S + delta 新增 2S = 验收面 34S）

- R1×4/R2×4：注解解析器直测（合法三形态、f64、顺序互换、Distributed、2.5 组件、dtype 标量注解、动态维度）。
- R3×4：types.equivalent 直测（layout 省略等价、同符号、常量×符号不等、f32×f16 不等）。
- R4×3：transfer（Global→Shared 合法、Global→Global 拒绝、建议含清单）。
- R5×6（含 delta 新增）：state_fields（纯 Register 类通过、Shared 字段、Pointer 字段、**标量字段（delta）**、consume 签名状态类注解、异类互赋 E0303）。
- R6×4：infer/bindings（int→Tensor 再绑定、切片派生逐维断言、`[:, None]` 广播、`n[0]` 基对象）。
- R7×4：bindings（f16→f32、Register→Shared、**comptime 64→显式调用 int 形参接受（第 4 类绑定）**、构造实参不匹配）。
- R8×5（含 delta 新增）：收集排序、同位置语法优先（段间短路下输出无 E03xx）、**语法拒绝时类型段不执行（delta 新 Scenario）**、E0304>E0303 段内优先、重复编译一致（pipeline 两次调用断言相等）。
- `toolchain/cli` R3 Scenario 1：既有 CLI incomplete 断言更新两段（登记性联动）。
- 集成：FLASH_ATTENTION 过 pipeline 零拒绝（syntax+type 两段）；注入已知面 E0303 走 CLI rejected JSON。让渡面负例（`x = tis.zeros(...)` 后 `x = 5` 不报、UNKNOWN 边界切片不报）入回归。

### D11 确定性与 residual risk

- 确定性：类型/建议表示全为冻结常量与结构化对象；无时间/随机/环境输入；二次 parse 与单次等价。
- residual（登记不阻塞）：
  1. 无数值折叠（`BR*2` vs `128`、`SymbolDim("BR")` vs `ConstDim(64)` 不判等）——折叠能力归 numerics 段（spec R3 等价规则不要求折叠，常量值相等判定在可表示范围内执行）。
  2. 同名状态类注册冲突 spec 未定义——按首次注册，后续补洞。
  3. 状态构造位置实参绑定规则未定义——不检查（proposal 非目标）。
  4. M1 让渡面的 E0303 漏报（如 `x = tis.zeros(...)` 后 `x = 5` 不报）——设计使然，后续段收紧。
  5. produce/consume 显式调用在 M1 不被 E0502 拒（execution 段 pending）且其实参绑定按第 4 类被 E0303 检查——execution 段实现后同位置跨段顺序 E03 优先语义不变（既有管线句）。

## 长期基线刷新计划（归档时执行）

1. `openspec archive`：两条 `language/type-system` MODIFIED（R5 6S、R8 5S）与 `toolchain/cli` R3 登记性 MODIFIED（Scenario 1 快照）并入 stable。
2. `overview.md`：双轨注记更新（type-system 段已实现，`implemented_stages` 两段）；`language/type-system` 基线行追加实现注记与本 change 两处补洞；`toolchain/cli` 基线行追加快照联动注记。
3. CLAUDE.md 无需更新（质量脚本门禁已登记；无新命令引入）。
4. `veps/` 不回改。

## Scenario 计数

- `language/type-system` MODIFIED：**2 Requirement**——R5（5→6 Scenario，补标量字段路径）、R8（4→5 Scenario，补段间短路）。
- `toolchain/cli` MODIFIED：**1 Requirement / 2 Scenario**（登记性联动，Scenario 1 快照更新，计数不变）。
- 实现映射（其余零 delta）：stable 8R/32S 全量（R1×4、R2×4、R3×4、R4×3、R5×5、R6×4、R7×4、R8×4——grep 实数核对）+ delta 新增 2S = **验收面 34S**。
- 无 BREAKING（三条 delta 均为 undefined→定义或登记性联动）；错误码、诊断 JSON Schema、`tis.*` API 零变更。
