# Design: add-m1-primitive-contract-frontend

## 设计范围导航

- 行为契约：`openspec/specs/primitives/memory-ops`（8R/33S，delta 补 2S）、`openspec/specs/primitives/compute-ops`（7R/34S，零 delta）、`openspec/specs/hal/capability-descriptions`（HAL 依赖检查的数据来源，零 delta）、`openspec/specs/toolchain/cli`（R3 快照登记性 delta）。
- 类型基座：`openspec/specs/language/type-system`（零 delta）——本段复用其符号环境与表达式推断；`E0301`（矩阵非法格）已由类型段实现，本段不重复。
- 本文件裁决实现结构（模块边界、推断复用、折叠算法、target 传递、order 空间）；不重述 spec 行为。

## 当前实现与 GAP

- 已有：`frontend`（语法段）、`typecheck`（类型段，8 模块——symbols 环境构建 / Inferencer 表达式推断 / bindings 设备函数遍历骨架 / transfer E0301）、`pipeline`（两段编排 + 段间短路）、`hal`（仅标识符集，字段加载按前设计 D2 明示「延期至首个消费段 change」）、`cli`。
- GAP：E0402–E0408 全部行为、HAL `mma_shapes`/`reduce_scopes` 字段值、原语调用结果类型、第三段管线接入、`compile_stages` 无 target 参数。

## 修改方案（裁决记录）

### D1 新包 `tilescript/primitives`（模块边界 = 错误码域边界，延续 frontend/typecheck 惯例）

```
tilescript/primitives/
  __init__.py     STAGE_NAME="primitive-contract"；check_module(tree, target)
  hal_data.py     ——不并入：HAL 登记值放 tilescript/hal.py 扩展（见 D2）
  results.py      原语调用结果类型规则（八存取 + 六计算；违规调用结果 UNKNOWN）
  fold.py         Dim → 整数系数符号多项式规范化（E0405 折叠支 (d) 专用）
  traverse.py     设备函数遍历骨架（env 构建 + 嵌套跟踪；复用 typecheck.symbols/Inferencer）
  memory_ops.py   E0404/E0405/E0406（存取原语参数与维度）/E0407（语境）
  compute_ops.py  E0408/E0402/E0403（计算原语）
```

`cli` 只依赖 `pipeline`（既有 D1 纪律不变）；`pipeline` 依赖 `frontend`/`typecheck`/`primitives`。

### D2 HAL 登记值载体：Python 冻结常量（扩展 `tilescript/hal.py`）

`hal/capability-descriptions` 裁定「描述的文件载体形态属工具链实现」。M1 选择冻结常量（`frozen dataclass` + 两目标登记值，含 `mma_shapes`/`reduce_scopes` 顺序）：静态数据、同目标逐字段一致、无 IO 无解析、无设备在环——契约四要点全满足；YAML 文件载体保持延期（新目标登记经显式 change 时一并裁决）。`Auto` 选择 = 列表第一项；`E0402` 报告面 = 全列表按登记顺序；`reduce_scopes` 小写值 ↔ 语言值 `Warp`/`Block` 映射表亦在此。

### D3 环境与推断复用：复用 `typecheck` 公开设施，不改其行为

- env 构建：复用 `typecheck.symbols.kernel_params` / `annotated_functions` / `_param_type`（kernel 形参、状态类注册表 `state_fields.check`、produce/consume 形参表）。
- 表达式推断：复用 `Inferencer.infer`（局部变量、下标/切片派生、状态类字段访问）——但其 `tis.*` 调用返回 UNKNOWN（类型段让渡，正确），本段在 `traverse` 层对 `tis.*` Call 节点改走 `results.py` 的结果类型规则（合法时给类型、违规时给 UNKNOWN 并记录 E04xx）。
- 遍历骨架：`traverse.py` 自写（赋值绑定、循环变量 UNKNOWN、嵌套 produce/consume 进入/退出跟踪）——结构与 `typecheck.bindings._walk` 同构但**不做 E0303 检查**（类型段职责）。不重构 `bindings`（已过 H2 的代码不动，避免跨段回归面）。
- 复用安全性依据：段间短路（类型段任一拒绝 → 本段不执行）保证本段只在类型段零拒绝时运行；同构遍历下 `Inferencer` 内嵌的 `E0301`/构造实参/调用实参检查同样零命中，不会重复报告。
- **防御性收尾断言**：`traverse` 遍历结束处断言 `Inferencer.rejections` 为空——非空即遍历与类型段 `bindings._walk` 发生漂移（同构性破坏，属实现缺陷），在开发期即暴露而非混入错段的 `E03xx` 拒绝。

### D4 原语结果类型规则（`results.py`）

| 原语 | 合法时结果 | 违规时 |
|---|---|---|
| `make_tensor(ptr, shape)` | `Tensor[d(ptr), shape 维, Global]`（运行期维成 SymbolDim(comptime=False)） | UNKNOWN + 已记录 E0406 |
| `alloc_shared(shape, dtype, layout)` | `Tensor[dtype, shape 维, Shared]` | 同上 |
| `zeros(shape, dtype, scope)` / `full(...)` | `Tensor[dtype, shape 维, scope]` | 同上 |
| `cast(x, dtype)` | `Tensor[dtype, shape(x), scope(x)]` | 同上 |
| `dot(A, B, C, ...)` | `Tensor[f32, (M, N), Register]` | 同上（E0408/E0402/E0403） |
| `reduce(x, axis, op, scope)` | `Tensor[dtype(x), shape 去 axis 维, Register]` | 同上（E0408） |
| `maximum/exp/log(x)` | 与操作数同 dtype/shape/scope | 同上（E0408） |
| `transpose(x)` | 二维互换 | 同上（E0408） |
| `load/store/barrier` | 不产生值（值位置使用 → E0407，见 D6） | —— |

shape 实参元组的组件推断：int 字面量 → ConstDim；comptime 符号（kernel comptime 集）→ SymbolDim(True)；int 符号 → SymbolDim(False)；算术 → DerivedDim；浮点/非整数 → E0406（`make_tensor`/`alloc_shared`/`zeros`/`full` 的 shape 约束）。

### D5 E0405 折叠支 (d) 与 E0403 整除的可判定域

`fold.py` 把 Dim 规范化为多项式 `dict[单项式, 系数]`（单项式 = 排序后的符号名元组；ConstDim → 常数项；SymbolDim → 单符号单项式；DerivedDim 按记录的 op/operands 递归——`+`/`-`/`*`/一元负做多项式代数；`//`/`%` 在**双 ConstDim 纯常量域**做整数求值（除数为 0 → 不可折叠 None），含符号操作数 → None；`/` → None；`slice` 维取 `upper - lower` 差分）。`bm*BR:(bm+1)*BR` → `(bm·BR + BR) - (bm·BR) = BR` ✓；`(128//8,)` → 16 ✓。相容判定四支按 spec 逐支实现；折叠**仅用于 E0405 相容判定**，不改类型段等价（D2「无数值折叠」承诺不动、不产生类型层等价结论——spec 明文）。

**E0403 整除可判定域**：M/N/K 各维仅 ConstDim（数值常量）参与整除判定；含 comptime 符号维（编译期可知但无数值，如 `BR`）→ 整除不可判定 → 不触发 `E0403`（compute-ops R2 Scenario「dot 默认参数 Register 操作数被接受」隐含此读法——`BR`/`D` 为 comptime 常量且被所选形状整除的 WHEN 条件在无数值下不构成拒绝）。`E0403` 的最近合法对齐形状建议算法：每维向上取整到所选 MMA 形状对应分量的倍数（`ceil(d/s)*s`，逐维独立）。

### D6 E0407 语境判定

- Async：`traverse` 维护「当前函数 = kernel 顶层 / produce-consume 嵌套」状态；`load` 的 `mode` 解析为 `Async` 且非嵌套语境 → E0407。
- 值位置：封闭集 = spec 明文三位置——赋值右侧（`Assign.value` / `AugAssign.value` 表达式树内任一层）、调用实参（其他 Call 的任一实参表达式树内）、`return` 值（`Return.value` 表达式树内）。命中 → E0407。`If`/`While` 条件等 spec 未定义位置**不裁**（登记 residual——undefined，非本 change 定义；后续补洞 change 裁决）。`dot` 等有值原语不受限。

### D7 MMA/PadPolicy 实参语境专用（E0408）

形态识别：`tis.MMA(...)`（Call，func=Attribute(tis, MMA)）、`tis.PadPolicy.<名>`（Attribute(tis, PadPolicy) 的属性）。合法位置封闭集：前者仅 `dot` 的 `mma=` 关键字实参、后者仅 `dot` 的 `pad=` 关键字实参。其余任何出现（赋值右侧、其他原语/调用实参、BinOp/UnaryOp/Compare/IfExp 操作数、return 值）→ E0408，定位该表达式节点。实现为遍历中的独立扫描（与 D6 同层）。

### D8 管线与 target 传递

`pipeline.compile_stages(source, *, target)`：target 必选关键字（登记表成员；CLI `--target` 透传）。语法段非空短路；类型段非空短路（本 change delta R8）；否则原语段 `check_module(tree, target)`。HAL 依赖检查（E0402/E0403/reduce scope）以 target 的能力描述为唯一数据源。`IMPLEMENTED_STAGES` 三段；CLI 断言联动。

### D9 order 空间与 stage

统一段内 order（同段两包共用全序，跨域同位置双命中按 spec 不可能）：`E0404`=1、`E0405`=2、`E0406`=3、`E0407`=4、`E0408`=5、`E0402`=6、`E0403`=7（memory-ops 段内序 + compute-ops 段内序 E0408→E0402→E0403 拼接）；`stage="primitive-contract"`。报告契约复用 `frontend.report.finalize`（位置升序、同位置 order tiebreak、收集全部、重复一致）。

### D10 测试映射（验收面 67S，构成见 D11 计数）

- `test_hal.py`：登记值两目标、`Auto` 第一项、映射表（hal R「列表顺序即优先序」3S 直测数据层）。
- `test_memory_ops.py`：R1 5S（参数集含缺失必选）、R2 5S（E0404 格承载；GG 让 E0301 直调验证不报）、R3 4S（E0405 四支含折叠 (d)；其中「dtype 一致的 load」Scenario 字面用 `buf.K`（执行结构段 M1 未知）——以 kernel 形参 Tensor 的等价切片构造承载同一折叠支语义）、R5 6S（分配与视图值域）、R6 3S（cast）、R7 3S（barrier）、R4 4S（E0407；其中 Sync 完成语义与 barrier 跨线程可见 2S 为运行时语义，静态面以「调用被接受」间接承载）。
- `test_compute_ops.py`：R1 5S、R2 10S（含 E0402/E0403 两目标差异）、R3 7S（reduce 含 ascend 支持面）、R4 5S、R5 2S、R7 3S（段内序与 Auto 确定性）。
- `test_prim_pipeline.py`：R8 5S（delta 短路 Scenario 直测公共管线）。compute R6 类别清单 2S 归 execution 段（E0501 拒绝行为 M1 不实现，登记让渡——类别判定面供给后续段）。
- `test_cli.py`/`test_integration.py`：快照三段；FLASH 样例升级三段回归（load 折叠支、dot 合法、buf.* 让渡）。
- 每类拒绝断言四要素（码/位置/类别/建议关键句）。

### D11 确定性与 residual risk

确定性来源：HAL 冻结常量 + 列表第一项选择 + 多项式折叠纯结构 + finalize 稳定排序——同输入同目标输出逐条一致。

residual（登记，非本 change 缺陷）：
1. `buf.*` 与执行结构对象 UNKNOWN 让渡——依赖它的 E0405/dot/reduce 检查跳过（后续段收紧）。
2. 算术子表达式 UNKNOWN 让渡（numerics 段定型后收紧；「先经其定型再进本域契约」的 M1 现实为跳过）。
3. copy/move 格无承载原语（spec 显式延期；同 scope 赋值走类型段等价）。
4. `full` 的 value 只做标量形态级核对（数值转换归 numerics）。
5. `alloc_shared` 的 `swizzled` 物理属性不进任何类型组件（诊断 memory 段 M2-M3 才呈现；类型组件恒 RowMajor）。
6. dot/reduce 的 `Auto` 选择已确定（第一项），但其诊断呈现（compute 段字段）归 diagnostics——拒绝清单路径下选择只影响 E0403 的整除检查输入，无可见输出差异。
7. `E0407` 值位置封闭集外（`If`/`While` 条件等）为 spec undefined，M1 不裁（D6 收窄；后续补洞 change 裁决）。
8. `E0403` 整除在 comptime 符号维（无数值）下不可判定 → 不触发（D5 裁决的既定读法，非缺陷——compute-ops R2 Scenario 2 隐含）。

## 长期基线刷新计划（归档时执行）

1. `openspec archive`：两条 `primitives/memory-ops` MODIFIED（R1 5S、R8 5S）与 `toolchain/cli` R3 登记性 MODIFIED（Scenario 1 快照）并入 stable。
2. `overview.md`：双轨注记更新（三段已实现）；`primitives/memory-ops` 基线行追加两处补洞注记；`primitives/compute-ops` 基线行追加实现注记；`hal/capability-descriptions` 基线行追加首个消费段落地注记；`toolchain/cli` 基线行追加快照联动注记。
3. CLAUDE.md 无需更新（质量脚本门禁已登记；无新命令引入）。
4. `veps/` 不回改。

## Scenario 计数

- `primitives/memory-ops` MODIFIED：**2 Requirement**——R1（4→5 Scenario，补必选实参缺失）、R8（4→5 Scenario，补段间短路）。
- `toolchain/cli` MODIFIED：**1 Requirement / 2 Scenario**（登记性联动，Scenario 1 快照更新，计数不变）。
- 实现映射（其余零 delta）：stable 总量 memory-ops 33S + compute-ops 34S + delta 新增 2S = 69S；其中 compute-ops R6「类别清单」2S 的拒绝行为归 execution 段（E0501，判定面供给）让渡，memory-ops 运行时语义 2S（Sync 完成、barrier 可见）以静态接受面间接承载——**本段验收面 67S**（静态直测 65S + 间接 2S）。
- 无 BREAKING（三条 delta 均为 undefined→定义或登记性联动）；错误码、诊断 JSON Schema、`tis.*` API 零变更。
