# Design: add-hal-capability-descriptions

## 设计范围导航

| 文件 | 操作 | 内容 |
|---|---|---|
| `specs/hal/capability-descriptions/spec.md` | ADDED | 7 Requirement / 19 Scenario：存在性与确定性、目标标识符登记表、字段集 v1 冻结、列表顺序即优先序、入口 HAL 支持面（E0506）、资源与建议域消费面、跨 HAL 行为不变面 |
| `specs/execution/pipeline-structure/spec.md` | MODIFIED | 「报告契约与 E05xx 段位」完整重述——唯一差异为段位句（`E0506` 起用归属 + 保留区间联动 `E0507`–`E0599`）；正文其余句与全部 4 个 Scenario 逐字不变（sed 抽取原文块后单点编辑，保真） |
| `specs/primitives/compute-ops/spec.md` | MODIFIED | 「reduce 的归约契约」完整重述——差异为 scope 值域句扩展（显式 `Warp`/`Block` 须在目标 `reduce_scopes` 支持列表内，`E0408`）+ 新增 1 个 Scenario；其余正文与原 6 个 Scenario 逐字不变 |
| `proposal.md` / `tasks.md` / `reviews/` | — | 五件套其余（proposal 已含 Why 六项、非目标八项、Impact） |

## 当前实现（事实基线）

仓库无生产代码。行为事实来源：`veps/design.md` §4.1（两目标静态 YAML——字段集与登记值的设计事实来源）、§4.2（Placement/UB——延期对象）、§3.2（async gap 引「HAL 带宽」——`§4.1` 无此字段的 veps 内部不一致，估算参数裁决的依据）。

既有 specs 的 HAL 消费面（grep 核实位置）：

1. `primitives/compute-ops` dot 契约：「`Auto` 时编译器 MUST 依 HAL 报告的支持形状列表确定性地选择第一项」「显式 `mma` 不在支持列表以 `E0402` 拒绝，报告列出该 HAL 支持的全部 MMA 形状」——列表载体悬空。
2. `primitives/compute-ops` reduce 契约：「`scope=Auto` 时编译器 MUST 依 HAL 能力描述确定性地选择 `Warp` 或 `Block` 之一——同一 HAL 目标的选择 MUST 唯一且写进文档」——描述结构悬空。
3. `primitives/compute-ops` Purpose：「HAL 能力字段」显式延期项。
4. `diagnostics/json-schema` DR2：`target`「取值登记表归 HAL 域后续 change」。
5. `diagnostics/json-schema` DR8 + Purpose：portability「差异类目登记与 severity 赋值规则归 portability/HAL 域后续 change」、HAL 能力描述字段延期项。
6. `primitives/memory-ops`：「掩码与行宽的可满足性属资源诊断 `memory` 段（建议域），不做编译期拒绝」——资源量消费面先例。
7. `execution/pipeline-structure` 段位句：「`E0506`–`E0599` 保留给执行结构域后续扩展」——E0506 起用的字面冲突点（同 E0408 起用时的 memory-ops 段位句）。

## GAP 分析（proposal Why ↔ 落点）

| Why 项 | 落点 |
|---|---|
| 1 让渡句悬空（3 处） | HR2（target 承接）、HR7（portability 数据来源与对齐判定面）；D11 纯让渡论证（无 MODIFIED diagnostics） |
| 2 Auto 确定性选择数据语义悬空 | HR1（存在性与确定性=数据前提）+ HR4（顺序即优先序=选择规则） |
| 3 E0402 报告面数据来源 | HR4（报告面=全列表原顺序） |
| 4 支持面行为豁口（2 处） | HR5（`persistent_kernel=false` → E0506）+ compute-ops MODIFIED（reduce 显式 scope → E0408 扩值域） |
| 5 字段语义未冻结 | HR3（十字段必选性/类型/值域 + 两目标登记值 + 演进规则） |
| 6 跨 HAL 不变面未总括 | HR7（影响面封闭四类 + 数值边界总括） |

## 修改方案（裁决记录）

**D1 域形态：ADDED 为主 + 两条 MODIFIED（选定方案）**。弃选纯 ADDED（只写字段语义）：`persistent_kernel=false` 拒绝与 reduce 显式 scope 两处行为豁口留存，违背「不得把未定义行为留给实现」；弃选连带 Placement/UB 全扩展：`tis.alloc_shared(shape, dtype, layout=RowMajor)` 冻结参数集无 `hint` 参数，语言层无承载面，扩展须先 BREAKING 改 memory-ops 参数集——随该机制整体延期（D5）。

**D2 E0506 起用模式：复用 E0408 先例**。段位句字面「`E0506`–`E0599` 保留」起用后为假 → execution 登记性 MODIFIED；保留区间联动 `E0507`–`E0599`；新句措辞逐字对照 memory-ops 段位句先例（「由 `hal/capability-descriptions` 定义——经该 change 对本条的 MODIFIED 起用」）。行为定义归起用方（本 capability HR5），execution 只改段位句——与 E0408（行为在 compute-ops、memory-ops 只改段位句）同构。

**D3 reduce 显式 scope 支持面：compute-ops 行为性 MODIFIED**。非新码——`E0408` 已占用于 reduce 参数违规族，本 change 将值域相对化（显式值合法与否取决于目标 `reduce_scopes`，报告注明目标支持清单）。此前未定义行为，非 BREAKING。最小 diff：仅扩 scope 值域句 + 1 新 Scenario；`Auto` 句与其余 6 Scenario 逐字不动（选择规则经 HR4 数据闭合，不改原句）。

**D4 `registers_per_sm` 与 `warp_group.roles` 定为可选**。veps §4.1 仅 `nvidia_h200` 登记 `registers_per_sm`（65536）、`ascend_910b` 无值——不臆造硬件数字；资源诊断估算数据（占用率查表、带宽）本就不限于字段集（occupancy 需要 `max_threads_per_sm` 等未登记参数），字段定位为实现参考参数。`roles` 同理（910B 有 `[vector, cube]`、H200 无）。

**D5 `ub_bytes`/Placement/带宽整体延期**。`ub_bytes` 的语言消费面（`hint=tis.Placement.Fast`）未进 memory-ops 冻结参数集；带宽在 §4.1 无字段（§3.2 引用悬空）——估算参数归实现内部登记数据（proposal 非目标 #1/#6，HR3 `registers_per_sm` 条内声明数据来源边界）。

**D6 资源量 vs 支持面分界**。`smem_bytes`/`smem_banks`/`warp_group.max_roles` 超限=容量约束→建议域不拒绝（承 memory-ops swizzle 先例，HR6）；`persistent_kernel=false`=二值能力判定→拒绝（E0506，HR5）。分界作为两 Requirement 的合并裁决句写入 HR6 正文。

**D7 顺序即优先序正式化**。`mma_shapes` 第一项=dot `Auto` 选择（compute-ops「第一项」句的数据闭合）；`reduce_scopes` 第一项=reduce `Auto` 选择（veps「NVIDIA 选 Warp/Ascend 选 Block」事实=各自列表第一项的正式化）；小写字面 `warp`/`block` ↔ 语言值 `Warp`/`Block` 映射显式化；`E0402` 报告面=全列表按登记顺序。

**D8 跨 HAL 影响面封闭四类**。①`Auto` 选择结果；②支持面拒绝（`E0402`/`E0403`/reduce scope `E0408`/`E0506`）；③诊断内容；④实现路径与指令生成（不可观察面，禁止反向影响前三类）。portability 差异判定数据来源=登记目标能力描述比较，v1 对齐差异判定面=`mma_shapes`（其余类目延期 portability 专门 change）。

**D9 `warp_size=0` 为架构事实、非支持面判据**。HR3 反约束句：支持面判定 MUST NOT 以 `warp_size` 为单独依据（reduce 支持面以 `reduce_scopes` 为准）——防实现由 `warp_size=0` 自行推导拒绝。

**D10 未登记标识符=工具链入口拒绝**。无源码位置可锚定 → 不属 `E01xx`–`E06xx` 源码锚定体系，报告形态属工具链约定（HR2）——与五段体系分界，不引入无位置错误码。

**D11 登记表封闭 + diagnostics 纯让渡承接**。diagnostics `target`/延期项为「归 HAL 域」域级指向，本 change 落地后指向有实体、原句字面依然为真——无需 MODIFIED diagnostics（对照：E0408/E0506 起用时段位句字面为假，故须 MODIFIED；本域无同类字面冲突）。

**D12 E0506 段内顺位最后 + 语境不相交声明**。execution 段内顺序句作用域为「本 capability 规则」，E0506 非 execution capability 规则，原句字面保持为真、不改；HR5 声明顺位（`E0501`→…→`E0506`）与语境不相交（装饰器行 vs 函数体语境）作完备性承诺。

**D13 HR1 排他性作用域限定（设计审查 round 1 M1 修复）**。「MUST 且仅 MUST」限定于 `Auto` 选择与 HAL 支持面判定（判定类——规则与登记字段一一对应，可穷举）；资源与建议域估算以描述登记字段为权威、实现内部静态参数为辅助（同目标唯一）——消除与 HR3 `registers_per_sm` 条数据来源边界声明的字面冲突（occupancy 估算需 `max_threads_per_sm` 等未登记参数，排他清单无法满足）。

**D14 `warp_group.supported=false` 行为显式延期（设计审查 round 1 m1 修复）**。v1 两目标均登记 `true`，`false` 分支无 veps 行为事实——不臆造拒绝或接受语义；HR3 条内声明「登记 `false` 目标前 MUST 经显式 change 定义」+ proposal 非目标登记（与 `ub_bytes` 延期同纪律：无事实不定义）。

## 长期基线刷新计划（归档时执行）

1. `openspec archive` 自动：新建 stable `hal/capability-descriptions/spec.md`（ADDED 应用）+ 两条 MODIFIED 应用到 execution/compute-ops stable。
2. 补写 stable hal spec 的 Purpose（域定位、两条补洞、显式延期项清单）。
3. `overview.md`：稳定基线登记 `hal/capability-descriptions` 行；execution 行追加段位句联动注记、compute-ops 行追加 reduce MODIFIED 注记（对照 memory-ops 行 E0408 联动注记格式）。
4. `veps/` 不回改（§4.1 为设计事实来源，spec 冻结后以 spec 为准；映射核对记录入 `veps/m1-hal-capability-mapping.md`）。

## Scenario 计数

- hal ADDED：**7 Requirement / 19 Scenario**（HR1×2、HR2×3、HR3×3、HR4×3、HR5×3、HR6×2、HR7×3）
- execution MODIFIED：1 Requirement / 4 Scenario（全部原样保留）
- compute-ops MODIFIED：1 Requirement / 7 Scenario（原 6 + 新增 1）
- delta 合计：9 Requirement / 30 Scenario；新增错误码 `E0506` 一枚
