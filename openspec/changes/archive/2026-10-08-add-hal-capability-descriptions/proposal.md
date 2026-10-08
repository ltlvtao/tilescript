# Proposal: add-hal-capability-descriptions

## Why

HAL 能力描述是 TileScript「跨硬件」承诺的数据基础，目前六处悬空：

1. **让渡句悬空**：`diagnostics/json-schema` 两处（`target` 取值登记表、portability 差异类目归「portability/HAL 域后续 change」；Purpose 显式延期项「HAL 能力描述字段」）、`primitives/compute-ops` Purpose 显式延期项「HAL 能力描述字段」——均指向尚无实体的域。
2. **Auto 确定性选择的数据语义悬空**：compute-ops 两处「依 HAL 报告/能力描述**确定性地**选择」（dot `mma=Auto` 选支持形状列表第一项、reduce `scope=Auto` 选 Warp/Block 之一）依赖的能力描述结构未定义——「同一 HAL 目标的选择 MUST 唯一」失去前提（描述不唯一则选择不唯一）。
3. **报告面数据来源未定义**：`E0402` MUST 列出「该 HAL 支持的全部 MMA 形状」——列表本身的载体与语义未规格化。
4. **HAL 支持面行为豁口（未定义行为）**：登记目标 `ascend_910b` 的 `persistent_kernel: false`（`veps` §4.1 事实）下 `@tis.persistent_kernel` 入口的编译行为无任何 spec 定义；reduce 显式 `scope=Warp` 在无 warp 概念目标（`warp_size: 0`、`reduce_scopes: [block]`）上的行为同样未定义——违背「不得把未定义行为留给实现」的流程原则。
5. **字段语义未冻结**：`veps/design.md` §4.1 的静态 YAML（两目标示例、字段集）是 veps 层事实，字段名/值域/必选性无行为契约。
6. **跨 HAL 不变面未总括**：numerics cast「任何 HAL 后端相同位模式」与 reduce「归约顺序依硬件不承诺位一致」的边界、语言语义对 HAL 目标的独立性（HAL 影响面封闭清单）散落各 capability，缺总括声明。

## What Changes

- **ADDED** capability `hal/capability-descriptions`：能力描述的存在性与确定性（静态、编译期读取、同一目标唯一——Auto 确定性选择的前提）、标识符登记表（v1 登记 `nvidia_h200`/`ascend_910b`，新目标 MUST 经显式 change；承接 diagnostics `target` 让渡）、字段集 v1 冻结（name/warp_size/smem_bytes/smem_banks/registers_per_sm/async_copy/mma_shapes/warp_group/reduce_scopes/persistent_kernel——必选性/类型/值域/非空约束）、顺序即优先序语义（`mma_shapes` 与 `reduce_scopes` 列表顺序即 Auto 默认选择优先序、第一项即选择；`E0402` 报告面=`mma_shapes` 全列表）、资源与建议域消费面（`smem_bytes`/`smem_banks` 与 `warp_group.max_roles` 超限不属编译期拒绝——承接 memory-ops swizzle 可满足性先例）、跨 HAL 行为不变面（语言语义不依赖目标；HAL 影响面封闭清单）。
- **MODIFIED** `execution/pipeline-structure`「报告契约与 E05xx 段位」：登记性联动——`E0506` 起用为入口 HAL 支持面拒绝（由本 capability 定义），保留区间联动为 `E0507`–`E0599`；正文与全部 Scenario 逐字不变。
- **MODIFIED** `primitives/compute-ops`「reduce 的归约契约」：行为性补洞——reduce 显式 `scope`（`Warp`/`Block`）MUST 在该 HAL 能力描述 `reduce_scopes` 支持列表内，不支持以 `E0408` 拒绝（参数实际值域相对目标；此前为未定义行为，非兼容性破坏）；`Auto` 选择与其余条款不变。
- 新增错误码 `E0506`（入口 HAL 支持面；归 execution E05xx 段位，经登记性 MODIFIED 起用——与 `E0408` 起用同模式）。
- 无 BREAKING：两条 MODIFIED 分别为登记性联动（无行为变化）与未定义行为补洞（此前无兼容性承诺）。

## 非目标

- **Placement 与 UB 层级映射**：`veps` §4.2 的 `hint=tis.Placement.Fast`（Ascend 映射 UB）未进入 `primitives/memory-ops` 冻结参数集（`tis.alloc_shared(shape, dtype, layout=RowMajor)` 无 `hint` 参数）——语言层无承载面；`ub_bytes` 字段与 Placement/portability 提示类目随该机制整体延期（后续 change 显式扩展参数集时一并登记）。
- **portability 差异类目完整清单与 severity 赋值规则**：本 change 只定义差异判定数据来源（登记目标能力描述比较）与对齐差异判定面（`mma_shapes`）；完整类目登记延期 portability 专门 change。
- **诊断 JSON 字段变更**：本 change 不新增/修改诊断 JSON 任何字段（`target` 取值语义经登记表承接，schema v1 冻结不动）。
- **HAL 描述文件载体形态**（YAML 文件布局/加载机制）：属工具链实现；本 change 规格化描述的结构与语义（字段名跨工具一致性冻结于字段集）。
- **后端实现策略**（lowering 路径、`cp_async`/`tma`/`dma` 机制选择、CANN/ptxas 细节）：`async_copy` 等字段值为后端实现词汇，供实现与诊断参考，本 change 不冻结其枚举值集。
- **带宽等估算参数**：`veps` §3.2 async gap 估算公式引用「HAL 带宽」但 §4.1 字段集无此字段——估算参数属实现内部数据，不登记入 v1 字段集。
- **warp_group 分组资源约束与 `supported=false` 行为**：`max_roles` 超限等分组资源问题归资源诊断建议域（同 smem 超限），不做编译期拒绝；`roles` 值为后端词汇；`supported=false` 分支的行为（v1 两目标均登记 `true`）随此类目标的登记需求经显式 change 定义。
- **新增 HAL 目标**：登记表封闭（v1 两项），新目标登记属显式 change。

## Impact

- 承接让渡：diagnostics `target` 取值登记表与 Purpose 延期项、compute-ops Purpose 延期项「HAL 能力描述字段」——域级指向落地后依然成立，无需 MODIFIED diagnostics。
- execution E05xx 段位句经登记性 MODIFIED 联动（`E0506` 起用）；compute-ops reduce 契约经行为性 MODIFIED 补洞（`E0408` 值域扩展为相对目标）。
- Auto 确定性选择（dot mma / reduce scope）自此具有可验证前提（描述唯一 + 顺序语义）。
- 检查管线五段与各段错误码归属不变（`E0506` 为 execution 段内新增码）。
