# Proposal: add-m1-type-system-frontend

## Why

1. **M1 第二检查段**（用户指令「开始 M1 下一检查段」）：五段检查管线顺序为 syntax → type-system → primitive-contract → execution-structure → numerics（`toolchain/cli` R3）。第一段（语法段，`E01xx` 全量）已随 `add-m1-syntax-frontend`（2026-10-10 归档）交付；本 change 实现第二段——类型系统段。
2. **规格就绪而实现为零**：`language/type-system`（8 Requirement / 32 Scenario）自 2026-09-28 冻结（`E0303` 豁免边界经 2026-09-28 / 2026-09-30 两次扩展），至今无一行实现；类型检查器是 TSR 构造与后续三段的基座。
3. **三处规格豁口须先闭合**（规格先行）：
   - **段间执行语义**（R8）：定义了跨段同位置 tiebreak，未定义「语法段存在拒绝时类型段是否执行」——短路读法与并行收集读法输出不同，属公共可观察行为；
   - **标量状态字段错误码路径**（R5）：`E0107` 已把字段类型规则让渡至本 capability，R5 的 `E0304` 触发列举仅含 Pointer 与 Tensor-scope 两路径，「字段类型 MUST 是 Tensor」对标量字段（`count: int`）的违反 undefined；
   - **诚实退出时点快照**（`toolchain/cli` R3 Scenario 1）：WHEN「仅语法段已实现」在本 change 后永不可满足，须按仓库先例做登记性 MODIFIED 联动更新。
4. **诚实退出实例化**：`toolchain/cli` R3 的五段列表本 change 后 `implemented_stages` 扩为两段——正文零变化（本就泛化），仅 Scenario 快照联动。

## What Changes

- **MODIFIED** `language/type-system`「类型检查报告契约与 E03xx 段位」（4→5 Scenario）：补段间短路执行语义——语法段存在任一拒绝时类型系统段 MUST NOT 执行；语法段零拒绝时类型系统段执行。原「同位置双命中只报 `E01xx`」保持独立规范句地位，短路句为其执行语义。原 undefined 行为被定义，**非 BREAKING**；新增 1 个 Scenario 验证「语法拒绝在前则类型拒绝不出现于输出」。
- **MODIFIED** `language/type-system`「状态类类型与字段约束」（5→6 Scenario）：`E0304` 触发路径补全——字段类型既非 `Tensor` 也非 `Pointer`（标量种类注解）时同样 MUST 拒绝（原 undefined→定义，**非 BREAKING**，与既有恢复建议「状态字段只接受 Register scope 的 Tensor」一致）；新增 1 个 Scenario。
- **MODIFIED** `toolchain/cli`「分段实现状态与诚实退出」（2 Scenario 不变，登记性联动）：Scenario 1 快照更新——已实现段为语法段与类型系统段两段（正文泛化句不动；与 memory-ops/compute-ops/execution 段位句联动先例同型）。**非 BREAKING**。
- **实现 `language/type-system` 全量**（其余语义零 delta，既有契约为准）：
  - `E0302`（类型表达式结构）：dtype 六值 / scope 三值封闭集、`Tensor` 三必选一可选参数序、`Pointer` 两参数序、shape 组件类别规则（浮点组件拒绝）——检查位置为 kernel 签名注解、`@tis.state` 字段注解、produce/consume 嵌套函数签名与返回注解。
  - `E0304`（状态类字段约束）：字段必须为 `Tensor[..., Register]`；Pointer 字段 / 非 Register scope / 标量种类字段拒绝（delta 补全路径）；状态类注册表（类名 → 有序字段类型，名义等价）。
  - `E0301`（作用域转移矩阵）：识别 `tis.load`/`tis.store` 调用（参数位置 src=0/dst=1 按 `primitives/memory-ops` 参数集只读复用），双实参类型已知为 Tensor 且 scope 组合为矩阵非法格（Global→Global）时拒绝；copy/move 格误用归 `E0404`、非 Tensor / 未知实参归 `E0406` 与后续段，本段不报。
  - `E0303`（类型不匹配）：四类绑定位置——赋值（Name 目标单类型不变量 / Subscript / Attribute 目标）、`return`×返回注解、状态类构造调用关键字实参、**带源码层形参注解的用户函数调用实参**（produce/consume 嵌套函数显式调用——其调用本身合法性归 `E0502` 不在辖内，本段只裁实参绑定等价；无注解形参位不查）；`comptime[int]`→`int` 单向兼容；下标 / 切片 / `None` 广播的类型派生与基对象非 Tensor 拒绝；跨 scope 普通赋值按不等价拒绝。
  - 报告契约：收集全部、位置升序、同位置段内顺序 `E0302`→`E0304`→`E0301`→`E0303` 取首个、重复编译逐条一致。
- **让渡面的诚实处理**（依 R6 既有让渡句，非本 change 定义）：`tis.*` 原语调用、算术 / 比较 / 逻辑运算、执行结构方法与内建调用（`pipe.run`/`range`）、float/bool/`None`/`inf` 常量、Pipeline/buffer 属性、warp_group with 绑定名、for-range/tile_iter 循环变量的类型在 M1 为**未知**；未知类型参与任何绑定不产生 `E0303`，含未知组件的派生类型整体按未知参与绑定——后续段实现后检查自然收紧。
- **管线编排**：新增 `tilescript/pipeline.py`（多段检查编排：syntax → 段间短路 → type-system；后续段的扩展点）；CLI 诚实退出实例化 `implemented_stages=["syntax","type-system"]`；**既有 CLI 测试中单段 incomplete 断言随 delta 联动更新为两段**。
- **无 BREAKING**：三条 delta 均为 undefined→定义或登记性联动；错误码、诊断 JSON Schema、拒绝清单五字段形态均不变（`E03xx` 条目首次进入清单值域，属既有 v1 冻结演进规则内的错误码使用，非 schema 变更）。

## 非目标

- **`E04xx`/`E05xx`/`E06xx` 各段实现**：原语契约（含 `E0404` copy/move 格误用、`E0405` 长度相容、`E0406` 参数值域、`E0407` 语境）、执行结构（`E0502` Pipeline 构造与签名契约——含 produce/consume **显式调用禁止**本身、`E0503` buffer 命名空间、`E0505` 内建）、数值语义（`E06xx`）——CLI 以 `pending_stages` 如实列出。
- **`tis.*` 原语结果类型推断**：`make_tensor`/`zeros`/`dot` 等的返回类型虽已在 `primitives/*` specs 定义，但其实现归 primitive-contract 段；本段按 R6 让渡处理为未知。同理 `pipe.run`/`range` 结果与 `buf.<名>` 属性类型、循环变量元素类型归 execution-structure 段。
- **算术/比较/逻辑运算结果类型与 float 字面量绑定**：数值语义段（`numerics/value-semantics`）。
- **状态类构造调用位置实参的绑定规则**：R5 只定义关键字实参绑定，位置实参绑定规则 spec 未定义——本 change 不实现不发明（登记为已知缺口，后续补洞 change 裁决）。
- **kernel 顶层函数返回注解的形式与绑定检查**：语法层 `E0104` 注解封闭集辖参数注解，kernel 返回注解形式未规格化；本 change 的 return×返回注解检查辖 produce/consume 嵌套函数（consume 返回注解由 `execution/pipeline-structure` 定义）。
- **TSR 构造与诊断 JSON 输出**：随后续段 / M2-M3 诊断 v0。
- **常量折叠数值求值**：`cdiv` 折叠、comptime 表达式求值归 numerics；shape 组件的 comptime/运行期类别判定按结构类别（字面量、参数符号、算术派生形态）裁决，不做数值折叠（`BR*2` 与 `128` 不判等）。

## Impact

- 五段管线第二段落地：类型表示 / 符号表 / 推断器成为 primitive-contract 段（第三段）的直接基座——原语实参类型检查将复用本段推断结果。
- 拒绝清单 JSON 首次出现 `E03xx` 条目（五字段形态不变，v1 冻结兼容）；`{"status":"incomplete"}` 的 `implemented_stages` 扩为两段（`toolchain/cli` delta 联动；既有 CLI 测试断言同步更新）。
- 集成样例（veps §7 FlashAttention）在类型段零拒绝：已知面（签名注解、状态字段、字段访问、构造绑定、return 绑定）全部等价，让渡面按未知跳过——该样例继续作为段级回归。
- `language/type-system` R8 补洞后，「段间短路」成为后续各段（E04xx/E05xx/E06xx 实现时）复用的管线语义模板；R5 补洞使状态字段的全部违规路径闭合。
