# Proposal

## Why

已归档的 `language/type-system` 在两处显式让渡给 `primitives/*`：数据移动原语的实参"其余维度（dtype、shape 等）由该原语在 `primitives/*` 中的参数契约承载"，以及 `tis.*` 原语调用表达式的结果类型。M1 五 kernel 的数据路径完全建立在这批原语上（§7 FlashAttention 使用 `make_tensor`/`alloc_shared`/`load`/`barrier`/`zeros`/`full`/`cast`/`store` 共十类用法），但它们没有任何权威规格：`veps/design.md` §2.3 的内存操作清单只有六条单行签名（且与 §7 用例存在缺口——`make_tensor`/`alloc_shared`/`zeros`/`full`/`cast` 根本不在清单内），参数约束、返回类型、scope 组合承载、async 完成语义均由实现隐式决定。错误码方面 E04xx 段只有 MMA 的 `E0402`/`E0403` 两个事实，存取原语的违规无码可报。没有先行规格，M1"5 kernel 通过类型检查"在原语边界处不可验收。

## What Changes

- 新增 capability `primitives/memory-ops`，定义八个核心存取原语的行为契约：
  - 原语集合与调用结构：`tis.load`/`tis.store`/`tis.barrier`/`tis.make_tensor`/`tis.alloc_shared`/`tis.zeros`/`tis.full`/`tis.cast` 的参数集、默认值与封闭性
  - 转移格承载映射：`load` 承载矩阵全部三个 `load` 格、`store` 承载全部三个 `store` 格；`copy`/`move` 格无承载原语（显式声明）；原语与格子操作类别不匹配以 `E0404` 拒绝
  - 数据维度契约：`load`/`store` 实参 dtype 相同、shape 逐维长度相容（`E0405`，本 capability 专用判定，区别于 `language/type-system` 的维度等价）
  - 加载模式与完成语义：`mode=Sync|Async`；`Sync` 语句处完成；`Async` 仅 Pipeline 嵌套函数语境合法（`E0407`），完成保证让渡执行结构 capability
  - 分配与构造：`make_tensor`（Global Pointer 的零拷贝视图）、`alloc_shared`（含 `Layout.swizzled`）、`zeros`/`full`（scope 限 Register/Shared，默认 Register）（`E0406`）
  - dtype 显式转换：`tis.cast` 返回同 shape/scope 新 dtype Tensor；转换 MUST NOT 隐式发生；数值效果延期数值语义 capability
  - 屏障：`tis.barrier(scope=Block|WarpGroup)` 的汇合与可见性语义
  - `E04xx` 报告契约与段位：四要素、收集式、跨段与段内 tiebreak、确定性；`E0404`–`E0407` 分配
- **MODIFIED** `language/type-system`「类型不匹配拒绝」：等价豁免边界从「显式数据移动原语（初始集合 `tis.load`/`tis.store`）」扩展为「`tis.*` 原语调用表达式的实参整体」——原语实参的形态与值域约束统一归 `primitives/*` 参数契约（`E04xx`），消除构造/转换原语实参的 `E0303`/`E0406` 双解（原语只有参数契约、无类型化形参注解，等价规则无从落地）。
- **BREAKING**：一项——上述 `language/type-system` MODIFIED 使 `E0303` 的适用集收敛（`tis.*` 原语调用实参不再可能命中 `E0303`，其违规统一落 `E04xx`）。影响面：`E0303` 报告行为的适用范围；迁移路径：无存量实现与受众（仓库无编译器代码），无需迁移。其余无（新 capability；`E0402`/`E0403`（MMA）、`E0501`（warp）等既有事实不动；E04xx 段位语义从"计算原语"精化为"原语参数契约"系扩展声明，无存量实现与受众）。

## Capabilities

### New Capabilities

- `primitives/memory-ops`：核心存取原语（load/store/barrier/make_tensor/alloc_shared/zeros/full/cast）的参数契约、转移格承载映射、完成语义与 E0404–E0407 报告行为。

### Modified Capabilities

- `language/type-system`（`openspec/specs/language/type-system/` 既有路径）：「类型不匹配拒绝」Requirement 的等价豁免边界扩展——从数据移动原语（初始集合 `tis.load`/`tis.store`）扩展为 `tis.*` 原语调用表达式的实参整体（详见 What Changes 的 MODIFIED 条目；delta 见 `specs/language/type-system/spec.md`）。

## 非目标

- 异步组管理：`tis.commit_group`/`tis.wait_group` 与 `tis.load` 的 `group=` 参数（`GroupHandle` 的类型层地位、组生命周期与手动完成规则——`veps/design.md` §2.5 明确 Pipeline 语境用户不能手动写 `wait_group`，独立 async change 承载；本 change 内传 `group=` 按未知参数拒绝）。
- `tis.atomic_add`（原子操作子域，M1 五 kernel 零使用）。
- 索引与整数内建：`tis.block_idx`/`tis.cdiv`（线程索引与整除工具，归执行结构/内建函数域 change）。
- 计算原语：`tis.dot`（`E0402`/`E0403` 既有事实）、`tis.reduce`/`tis.transpose`/`tis.exp`/`tis.log`/`tis.maximum` 及数值提升（数值语义与计算原语 change）。
- `Layout` 完整代数：本 change 只定义 `RowMajor` 与 `Layout.swizzled(xor=...)`；`Padded` 等其他布局值延期（`veps/design.md` 仅在诊断建议文本中提及）。
- 数值转换语义：`tis.cast` 的舍入/饱和/精度损失规则、`tis.full` 值到 dtype 的数值转换合法性（数值语义 capability；本 change 只约束参数形态与返回类型结构）。
- Pipeline 机制：多缓冲分配、prologue/epilogue 展开、commit/wait 插入与形式化验证（执行结构 capability；本 change 只裁 `mode=Async` 的合法语境与完成保证的让渡指向）。
- `warp_group` 上下文规则（`E0501` 等执行结构域）；`barrier(scope=WarpGroup)` 的上下文约束让渡执行结构。
- Shared 容量/占用等资源检查与 `memory`/`async` 诊断段字段（HAL 与 `diagnostics/*`）。
- 诊断 JSON Schema（`diagnostics/*`）整体延期。

## Impact

- 承接 `language/type-system` 的两处让渡：移动原语实参的其余维度（dtype/shape）自此有权威裁决（`E0405`）；本批 `tis.*` 调用表达式的结果类型自此有定义（返回类型规则）。
- 与 `language/type-system` 的边界：scope 组合合法性（含矩阵非法格 `E0301`）与通用绑定等价（`E0303`，适用于赋值/返回/非原语调用）仍由该 capability 裁决；本 change 裁原语与格子的匹配、原语参数契约与语境，并经 MODIFIED 将原语调用实参整体移出 `E0303` 适用集（消除构造/转换原语实参的双解）。
- `veps/design.md` §2.3 的六原语签名与 §7 用例原语自此获得权威依据；示例与规格冲突时以规格为准。
- 无生产代码影响（纯规格先行 change）。
