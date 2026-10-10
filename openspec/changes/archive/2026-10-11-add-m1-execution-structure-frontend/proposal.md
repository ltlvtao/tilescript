# Proposal: add-m1-execution-structure-frontend

## Why

M1 前端三段（syntax/type-system/primitive-contract）已实现并归档，`incomplete` 快照中执行结构段（`execution-structure`）仍以 pending 显式延期。`execution/pipeline-structure` stable spec（2026-09-30，8 Requirement / 33 Scenario）已完整定义 E0501–E0505 行为契约，`hal/capability-descriptions` 已定义 E0506（入口 HAL 支持面，段位经登记性 MODIFIED 起用），两者均无实现。本 change 落地第四检查段实现面：Pipeline 构造/produce/consume/run 调用契约、buffer 命名空间、warp_group 上下文、索引与整数内建、入口 HAL 支持面，接入管线第四段并联动 CLI 快照。类目同类先例：`add-m1-syntax-frontend`（首段）、`add-m1-type-system-frontend`（第二段）、`add-m1-primitive-contract-frontend`（第三段）。

## What Changes

- 新增 `tilescript/execution/` 检查段实现（纯结构扫描，不引入类型推断重放）：E0502（Pipeline 构造、produce/consume 签名与调用封闭、run 调用契约、Pipeline 值与 range 可迭代值封闭使用位置）、E0503（buffer 命名空间：未注册名、buffer 形参逃逸）、E0501（warp_group 语境：producer 体内计算原语——类别判定面消费 `primitives/compute-ops` 封闭六原语清单、`barrier(scope=WarpGroup)` 语境）、E0504（warp_group 参数与位置契约、with 绑定名封闭使用、warp_group_sync 契约）、E0505（block_idx/cdiv/range 参数契约）、E0506（入口 HAL 支持面：`persistent_kernel=false` 目标拒绝 `@tis.persistent_kernel` 入口）。
- `tilescript/hal.py` 能力描述加载 `persistent_kernel` 字段（spec v1 冻结字段集内既登记字段，实现面扩展，零 spec delta）。
- `tilescript/pipeline.py` 接入第四段：`IMPLEMENTED_STAGES` 四段；原语契约段存在任一拒绝时执行结构段不执行（与既有段间短路同型）。
- specs delta 两条（均登记性，无行为变更、非 BREAKING）：
  - `execution/pipeline-structure`「报告契约与 E05xx 段位」MODIFIED：补段间短路句（原语契约段存在任一拒绝时执行结构段 MUST NOT 执行），与 `language/type-system`/`primitives/memory-ops` 既有短路句同型。
  - `toolchain/cli`「诚实退出」MODIFIED：Scenario 1 快照 `implemented_stages` 更新为四段（pending 仅剩 numerics）。
- 集成回归：FLASH_ATTENTION 样例升级四段零拒绝回归（Pipeline 构造/produce/consume/run/cdiv/range/block_idx 全套合法面、`buf.*` 与 `st.*` 让渡链维持）；两目标语言层一致性 + E0506 按目标分化（ascend 拒 / h200 接受）。

## Impact

- **影响 specs**：`execution/pipeline-structure`（MODIFIED 1 Requirement——段间短路句，Scenario 计数不变）、`toolchain/cli`（MODIFIED 1 Requirement——快照登记性联动，Scenario 计数不变）。`primitives/compute-ops`「计算原语类别清单」为本 change 的 E0501 判定面供给，零 delta（消费其封闭六原语清单）；`hal/capability-descriptions`「入口 HAL 支持面」零 delta（行为已定义，本 change 为首个实现面）。
- **影响代码**：新增 `tilescript/execution/` 包与 `tests/test_execution*.py`；`hal.py` 加载 `persistent_kernel`；`pipeline.py` 第四段接入；`cli.py` 注释面联动；既有测试 `implemented_stages` 断言联动更新。
- **兼容性**：无 BREAKING——不新增/改变错误码语义、诊断 JSON Schema 五字段冻结不动、`tis.*` API 零变更；两条 delta 均为 undefined→定义（段间短路）或登记性联动（快照）。
- **非目标**：numerics 段（M1 第五段，后续 change）；Pipeline 展开与后端产物（Async 完成保证为语言级运行时承诺，静态段只承载调用契约面）；`diagnostics/*` 展开产物定位字段；HAL YAML 载体；未定义 `tis.*` 符号（`tis.wait_group` 等）的拒绝错误码（归属原语总集 capability 正式化）。
