# execution/pipeline-structure Delta

本 change 对该 capability 的唯一 delta：「报告契约与 E05xx 段位」Requirement 补段间短路句与对应 Scenario（E0501–E0506 行为契约本体已于 2026-09-30 change 定义，本 change 为首个实现面，零行为 delta）。

## MODIFIED Requirements

### Requirement: 报告契约与 E05xx 段位

本 capability 产生的每条拒绝（`E05xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含违规形态、名称或位置类别）、恢复建议。同一模块存在多条 `E05xx` 拒绝时 MUST 收集输出全部并按源码位置升序排列。检查管线顺序为语法（`E01xx`）→ 类型（`E03xx`）→ 原语契约（`E04xx`）→ 执行结构（`E05xx`）；前序段存在任一拒绝时执行结构段 MUST NOT 执行（段间短路：语法段、类型段或原语契约段任一非空即短路，执行结构拒绝不产生——与 `language/type-system`「语法段存在拒绝时类型段不执行」、`primitives/memory-ops`「类型段存在拒绝时原语段不执行」同型）；同一源码位置跨段命中时 MUST 只报告最早段一条。执行结构调用（`pipe.run`、`range` 等无源码层形参注解的调用）的实参不适用 `language/type-system` 绑定等价规则（`E0303`，经本 change 对该 capability 的 MODIFIED 裁决），实参违规一律落 `E05xx`。同一源码位置命中多个本 capability 规则时 MUST 只报告一个错误码，按段内顺序取首个：`E0501`（warp_group 语境）→ `E0502`（Pipeline 构造、签名与调用契约，及 Pipeline 值与 range 可迭代值的封闭使用位置）→ `E0503`（buffer 命名空间，含 buffer 对象的封闭使用违规）→ `E0504`（warp_group 参数契约，含 warp_group 调用位置与 with 绑定名的封闭使用）→ `E0505`（索引与整数内建参数）。同一调用命中多个实参的同类契约违规时 MUST 合并为一条报告并列出全部违规实参。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E05xx` 为执行结构段——`E0501` 为 warp_group 语境违规（`veps` 既有事实"producer 体内出现 dot"的正式化，语义涵盖 producer 体内计算原语与 `barrier(scope=WarpGroup)` 语境违规，原场景语义不变）；`E0502`/`E0503`/`E0504`/`E0505` 为本 change 新增；`E0506` 为入口 HAL 支持面拒绝（由 `hal/capability-descriptions` 定义——经该 change 对本条的 MODIFIED 起用）；`E0507`–`E0599` 保留给执行结构域后续扩展。

#### Scenario: 多条 E05xx 拒绝被收集并排序

- **WHEN** 同一模块第 8 行存在 `E0502`（buffers 形态违规）、第 21 行存在 `E0503`（未注册 buffer 名）
- **THEN** 输出恰好两条拒绝，按第 8 行在前、第 21 行在后排序

#### Scenario: 同位置原语与执行结构双命中只报原语错误

- **WHEN** 同一 `tis.load(x, y, mode=Async)` 调用位置同时命中 `E0407`（Async 语境）与某 `E05xx`
- **THEN** 该位置只报告管线更早的 `E0407` 一条（原语契约段先于执行结构段）

#### Scenario: 原语契约段存在拒绝时执行结构段不执行

- **WHEN** 同一模块既存在原语契约段拒绝（如第 5 行 `E0406`）、又存在执行结构段命中（如第 12 行本应触发的 `E0502`）
- **THEN** 输出只含原语契约段的拒绝（执行结构段未执行，第 12 行的 `E05xx` 不出现——段间短路；与 `primitives/memory-ops` Scenario「类型段存在拒绝时原语段不执行」同型）

#### Scenario: 同位置多个执行结构规则命中只报段内首个

- **WHEN** 同一 `pipe.run(Tensor_obj, init=t)` 调用同时命中 `E0502`（第一实参形态与 init 形态）
- **THEN** 该位置只报告 `E0502` 一条（同码合并，报告全部违规实参）

#### Scenario: 重复编译执行结构拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E05xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）
