# Proposal

## Why

已归档的 `language/type-system` 与 `primitives/memory-ops` 共有三处显式让渡指向"执行结构 capability"：①type-system「表达式结果类型规则」——执行结构方法调用（`pipe.run(...)`）与内建调用（`range(...)`）的结果类型、Pipeline buffer 属性（`buf.K`）访问的类型；②memory-ops「加载模式与完成语义」——`mode=Async` 的完成保证（发起后何时可读 `dst`）由 Pipeline 契约承载；③memory-ops「屏障语义」——`barrier(scope=WarpGroup)` 的上下文约束与汇合/可见性语义。M1 FlashAttention 的循环主体完全建立在 `tis.Pipeline`/`@pipe.produce`/`@pipe.consume`/`pipe.run`/`buf.K`/`range`/`tis.cdiv`/`tis.block_idx` 上（§7 共九类用法），但它们没有任何权威规格：`veps/design.md` §2.5 只有一段示例与"半自动"原则（用户不能手动写 `wait_group`），构造参数约束、嵌套函数签名契约、迭代语义、完成保证、buffer 类型推导均未定义。错误码方面 E05xx 段只有 `E0501`（warp 组内 `dot`）一个 veps 事实。没有先行规格，M1 案例在执行结构边界处不可验收，Async 完成语义与 WarpGroup 汇合语义悬空。

## What Changes

- 新增 capability `execution/pipeline-structure`，定义执行结构的行为契约：
  - Pipeline 构造：`tis.Pipeline(stages=…, buffers=…)` 的参数契约（`stages` 为正 `comptime[int]`、`buffers` 为非空字典字面量且值为 Shared scope Tensor）与 Pipeline 值的合法使用位置封闭集（`E0502`）
  - produce/consume 嵌套函数契约：装饰器配对唯一性、签名契约（produce 恰两形参 `j: int` + `buf`；consume 恰三形参 + 返回注解同一状态类）、`buf` 形参无注解且类型由 `buffers` 推导（`E0502`）
  - `pipe.run` 调用契约：恰两实参（`range(...)` 位置实参 + `init=` 状态类关键字实参）、结果类型 = `init` 类型、逐迭代链式状态传递语义（`E0502`）
  - buffer 命名空间：`buf.<名>` 属性类型 = `buffers["<名>"]` 值类型、未注册名拒绝、`buf` 合法使用位置封闭（`E0503`）
  - Async 完成保证：produce 发起的 Async 拷贝在 consume 同迭代访问前完成且可见；组管理（commit/wait）由编译器插入、用户不可手写；物理槽复用不发生 overwrite-before-consumed
  - warp_group：`with tis.warp_group(role=…, warps=…)` 参数契约、producer 体内原语类别限制（`E0501`）、`tis.warp_group_sync` 契约、`barrier(scope=WarpGroup)` 的合法位置与汇合语义（`E0504`）
  - 索引与整数内建：`tis.block_idx`/`tis.cdiv`/`range` 的参数契约与结果类型（`E0505`）
  - `E05xx` 报告契约与段位：`E0501` 既有事实正式化（精化为 warp_group 语境违规）、`E0502`–`E0505` 新增、四要素/收集式/跨段与段内 tiebreak/确定性
- **MODIFIED** `language/type-system`「类型不匹配拒绝」：等价豁免边界第二次扩展——从「`tis.*` 原语调用表达式的实参整体」扩展为「`tis.*` 原语调用与执行结构调用（`pipe.run`、`range` 等无源码层形参注解的调用）的实参整体」，原适用集相应表述为「带源码层形参注解的调用（状态类构造调用等）」。理由与原语豁免同构：执行结构调用无源码层形参注解，实参约束是调用契约（形态+值域）而非类型化绑定，等价规则无从落地。
- **BREAKING**：一项——上述 MODIFIED 使 `E0303` 的适用集再次收敛（执行结构调用的实参不再可能命中 `E0303`，其违规统一落 `E05xx`）。影响面：`E0303` 报告行为的适用范围；迁移路径：无存量实现与受众（仓库无编译器代码），无需迁移。其余无（新 capability；`E0501` 从 veps 事实升格为 stable 行为契约，语义精化为"warp_group 语境违规"且涵盖原文"producer 体内出现 dot"场景，无存量实现与受众）。

## Capabilities

### New Capabilities

- `execution/pipeline-structure`：执行结构行为契约——Pipeline 构造与 produce/consume/run 调用结构、buffer 命名空间类型推导、Async 完成保证与不可见组管理、warp_group 上下文与 WarpGroup 汇合、索引内建（block_idx/cdiv/range）与 E0501–E0505 报告行为。

### Modified Capabilities

- `language/type-system`（`openspec/specs/language/type-system/` 既有路径）：「类型不匹配拒绝」Requirement 的等价豁免边界第二次扩展——执行结构调用（`pipe.run`/`range` 等无源码层形参注解的调用）实参整体豁免 `E0303`，违规归 `execution/*` 调用契约（`E05xx`）（详见 What Changes 的 MODIFIED 条目；delta 见 `specs/language/type-system/spec.md`）。

## 非目标

- `@tis.persistent_kernel`/`@tis.fused_kernel`/`tis.tile_iter()` 的行为契约（语法层已识别；`veps/design.md` §2.6 明确 1.0 仅 NVIDIA、M1 五 kernel 零使用——独立 change 承载；本 change 内 `tile_iter` 出现在 `for` 可迭代表达式位置时按本段未定义处理，不产生 `E05xx` 拒绝）。
- 异步组管理原语 `tis.commit_group`/`tis.wait_group` 与 `GroupHandle`（memory-ops 已显式延期至独立 async change；本 change 只承诺编译器插入的组管理对用户不可见——手写形态已在 memory-ops 按未知原语/参数拒绝）。
- Pipeline 展开 pass 的实现机制（多缓冲分配算法、prologue/epilogue 展开策略、Z3 形式化验证工具链）——本 change 承诺展开后的**可观察行为**（完成保证、槽复用安全、诊断可追溯性），验证机制归实现 design。
- 诊断 JSON 字段（`origin` 标签、`pipeline` 段等）——诊断 JSON Schema 整体延期（`diagnostics/*`）；本 change 只承诺"每条展开产物可定位源码行列"的行为要求。
- 数值折叠（`cdiv` 两 `comptime` 实参的编译期求值）——归数值语义 capability；本 change 只定形态与结果种类推导。
- `Distributed` scope、跨 kernel 融合语义、HAL 能力字段（`warp_group.max_roles` 等）。
- warp 级归约 `tis.reduce` 的 `scope=Warp|Block|Auto` 行为（计算原语与数值语义 change）。
- 多 Pipeline 实例与嵌套 `run`（M1 零事实，保守封闭为单实例单 `run`，扩展须经显式 change）。

## Impact

- 承接三处显式让渡：type-system「表达式结果类型规则」的执行结构方法/内建调用结果类型与 Pipeline buffer 属性类型自此有定义；memory-ops 的 Async 完成保证与 WarpGroup 汇合语义自此闭合。
- 与 `language/type-system` 的边界：类型层封闭世界不含 Pipeline 值、buffer 命名空间对象与 range 可迭代值——三者的类型地位由本 capability 定义为"合法使用位置封闭集"（违反按 `E0502`/`E0503`），不进入 type-system 等价规则；嵌套函数签名的注解形式合法性（如出现 `tis.BufferSlot` 类注解）仍由 type-system 以 `E0302` 裁决；`pipe.run`/`range` 实参经本 change MODIFIED 移出 `E0303` 适用集（消除与 `E05xx` 的双解），赋值、返回与状态类构造绑定仍由 type-system 裁决。
- 与 `primitives/memory-ops` 的边界：`mode=Async` 的合法语境（E0407）与参数契约（E0406）已由该 capability 裁决；本 change 承接其完成保证让渡——承诺 consume 可读即 produce 已完成，不改变其语境规则。
- `veps/design.md` §2.5 的 Pipeline 示例与 §2.3 的 warp_group 签名自此获得权威依据；示例与规格冲突时以规格为准。
- 无生产代码影响（纯规格先行 change）。
