# Tasks: add-m1-type-system-frontend

约定：每个 task 先写表达验收语义的失败测试（TDD），实现到通过后以实际验证证据勾选；只改 active change 与生产代码。依赖序：3.1/3.2（注册表）先于 2.3（状态类名注解查表）。

## 1. 类型表示与等价（R3）

- [x] 1.1 `typecheck/types.py`：dtype/scope 封闭集常量、`TensorType`/`ScalarType`/`StateType`/`UNKNOWN`、`Dim` 三形态（ConstDim/SymbolDim(comptime 标志)/DerivedDim(op, operands, comptime)）与等价判定（值相等/同名同标志/同 op 逐操作数、跨形态不等）；`equivalent()` 全量直测（layout 省略等价、同符号、常量×符号不等、异 dtype 标量不等、状态类名义等价——R3×4 Scenario）
- [x] 1.2 类型→描述文本（`desc()`，报告「两侧类型」与建议用）；冻结文本直测

## 2. 注解解析与 E0302（R1/R2）

- [x] 2.1 `typecheck/annotations.py` 解析器：合法三形态通过（Tensor 三参/Pointer 两参/Tensor 四参 layout）；未知 dtype（列六值）/参数顺序/集合外 scope/参数数量 → E0302 类别与建议（R1×4 Scenario）
- [x] 2.2 shape 组件类别判定：int 字面量→ConstDim、comptime 符号→SymbolDim(True)、int 符号→SymbolDim(False)、算术→DerivedDim（操作数 comptime 性递归）；float（2.5）→ E0302（R2 浮点组件 Scenario）；`comptime[int]` 与 dtype 标量注解解析（R2 动态维度/dtype 标量两 Scenario）
- [x] 2.3 三类检查位置接入：kernel 签名、`@tis.state` 字段、produce/consume 形参与返回注解（含状态类名注解→StateType，查 3.2 注册表）；同位置一注解一结果（结构失败不产类型）——依赖 3.1/3.2 先行（cycle 1 审查后补齐嵌套 `@pipe.*` 签名辖域：probe `Tensor[f64,...]` 返回注解 → E0302；验证 `pytest tests/test_annotations.py tests/test_bindings.py` 189 passed）

## 3. 状态类 E0304 与注册表（R5）

- [x] 3.1 `typecheck/state_fields.py`：纯 Register 字段类通过；Shared scope 字段 / Pointer 字段 / **标量种类字段（R5 delta 补全路径）** → E0304（类别+建议，R5×3 Scenario）；同位置 E0302 优先（结构坏的字段不产 E0304）
- [x] 3.2 `typecheck/symbols.py` 状态类注册表（类名→有序字段类型）；consume 签名状态类注解被接受（R5 Scenario）；异类状态互赋 → E0303 报告两侧类名（R5 Scenario，E0303 检查器落地于 5.x，验收证据在 5.1 后补记于此）

## 4. 表达式类型推断（R6）

- [x] 4.1 `typecheck/infer.py`：变量引用（符号表）、int 字面量→comptime_int、状态类字段访问→声明类型、下标/切片/`None` 广播派生（整数下标消维、切片保留、常量边界常量维/否则派生维、`:` 保留、`None` 增常量 1 维——R6 切片/广播两 Scenario 逐维断言）
- [x] 4.2 让渡面 → UNKNOWN（tis.* 调用、算术/比较/逻辑/一元、pipe.*/buf.* 属性、warp_group with 绑定名、for-range/tile_iter 循环变量、float/bool/None/inf、UNKNOWN 基对象派生）；UNKNOWN 绑定不报、**UNKNOWN 组件传播**（切片边界含 UNKNOWN → 派生维 UNKNOWN → 含 UNKNOWN 维类型整体按 UNKNOWN 参与绑定，负例回归）
- [x] 4.3 下标基对象已知非 Tensor（int 标量/状态类）→ E0303（R6 Scenario）

## 5. 绑定检查 E0303（R7/R2/R5 构造）

- [x] 5.1 `typecheck/bindings.py` 赋值：Name 目标单类型不变量（int 后绑 Tensor 拒——R6 Scenario）、Subscript/Attribute 目标侧派生×源、跨 scope 赋值拒绝（建议显式移动原语——R7 Scenario）
- [x] 5.2 dtype 不匹配（建议 tis.cast——R7 Scenario）；comptime→int 单向兼容接受（R2/R7 Scenario——显式调用 `j: int` 形参的 produce/consume 函数传 comptime 常量）；差异组件建议映射（dtype/scope/shape/种类）
- [x] 5.3 return×返回注解（consume 形态）；无返回注解不查
- [x] 5.4 状态类构造调用关键字实参×字段类型（R5/R7 构造实参 Scenario）；位置实参与未注册名不查（负例）
- [x] 5.5 带源码层形参注解的用户函数调用实参绑定（design D7 第 4 类）：produce/consume 显式调用的位置/关键字实参×形参注解类型（无注解形参位不查）；调用合法性归 E0502 不裁（负例：M1 下显式调用本身不报）

## 6. 转移检查 E0301（R4）

- [x] 6.1 `typecheck/transfer.py`：load/store 识别与 src/dst 位置（memory-ops 参数集只读复用）；Global→Shared 合法（R4 Scenario）；Global→Global → E0301 且建议列合法目标清单（R4×2 Scenario）；UNKNOWN/非 Tensor 实参不报（负例）

## 7. 段管线、报告与 CLI（R8 delta + toolchain/cli R3 联动）

- [x] 7.1 `Rejection` stage 参数化（默认 "syntax"）；typecheck 段内 order（E0302=1/E0304=2/E0301=3/E0303=4）；`typecheck/__init__.check_module(tree)` 收集+finalize
- [x] 7.2 `tilescript/pipeline.py compile_stages(source)`：syntax → 非空短路返回 → type-system；段间短路新 Scenario（E0106 在前则潜在 E0303 不出现——R8 delta 新增 Scenario）+ 同位置语法优先既有 Scenario 在管线层复测
- [x] 7.3 CLI 改调 pipeline：incomplete 两段实例化（implemented=[syntax, type-system]、pending 三段，由 ALL_STAGES 推导）；**既有 CLI 测试单段 incomplete 断言随 toolchain/cli delta 联动更新为两段**；E03xx 条目五字段 JSON；重复编译一致（两次调用断言相等——R8 Scenario）
- [x] 7.4 收集排序（E0302 在前 E0303 在后恰两条——R8 Scenario）、E0304>E0303 同位置段内优先（R8 Scenario）

## 8. 集成与收口

- [x] 8.1 FLASH_ATTENTION 样例过 `compile_stages` 零拒绝（两段）；CLI 端到端（exit 0 + incomplete 两段 JSON）
- [x] 8.2 注入已知面 E0303（状态构造字段 dtype 不匹配）走 CLI → rejected JSON exit 1；让渡面负例（zeros 后再绑 int 不报、UNKNOWN 边界切片不报）入回归
- [x] 8.3 `.venv/bin/python -m pytest -q` 全量通过；`npx openspec validate --all --strict` 通过；主智能体读全部 diff → commit 候选代码

## 9. 流程闭环

- [x] 9.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）
- [x] 9.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design「长期基线刷新计划」）（cycle 1 两轮：round1 CHANGES_REQUESTED 6 findings → round2 PASS 6/6 resolved；H2 APPROVED 2026-10-10；push 323e5e3 含 pre-push strict 10/10 + pytest 189 passed）
