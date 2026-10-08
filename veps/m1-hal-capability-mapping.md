# M1 HAL 能力描述字段与消费面映射核对记录

> 本文是 `openspec/changes/add-hal-capability-descriptions` tasks.md task 1.2/2.1/2.2/3.1/3.2/4.1/4.2 的验证证据（非正式过程记录）。
> 核对对象：`specs/hal/capability-descriptions/spec.md`（ADDED，7 Requirement）与两条 MODIFIED delta。
> 依据：`veps/design.md` §4.1（L328-348 两目标静态 YAML）、§4.2（L352-363 Placement/UB）、§3.2（L300 带宽引用）、§3.1 L153/L176（Auto 选择事实）；既有 stable specs 的 HAL 引用（grep 定位）。
> 引用约定：HR1–HR7 依次指 hal spec 的「能力描述的存在性与确定性」「目标标识符登记表」「能力描述字段集」「列表顺序即优先序」「入口 HAL 支持面」「资源与建议域消费面」「跨 HAL 行为不变面」。

## 1. veps §4.1 字段零缺漏映射（task 2.1）

| veps §4.1 字段（nvidia_h200） | spec 承载 | 核对 |
|---|---|---|
| `name: nvidia_h200` | HR3 必选=登记表键；HR2 封闭二成员 | ✅ |
| `warp_size: 32` | HR3 登记值 32（非负整数；0=无 warp 概念的反约束句 D9） | ✅ |
| `smem_bytes: 232448` | HR3 登记值；消费面 HR6 | ✅ |
| `smem_banks: 32` | HR3 登记值；bank_conflict 参数（HR6） | ✅ |
| `registers_per_sm: 65536` | HR3 可选字段、登记 65536（D4：910B 无值不臆造；数据来源边界句） | ✅ |
| `async_copy: [cp_async, tma]` | HR3 登记值；后端词汇不冻结枚举（非目标 #5） | ✅ |
| `mma_shapes: [[16,8,16],[16,8,32]]` | HR3 登记值 + 非空互异；顺序语义 HR4（注释「顺序即 Auto 的默认选择顺序」正式化） | ✅ |
| `warp_group: {supported: true, max_roles: 2}` | HR3 登记值；max_roles→HR6、supported=false 显式延期（D14，round 1 m1） | ✅ |
| `reduce_scopes: [warp, block]` | HR3 登记值（封闭小写二值域）；HR4 第一项=Warp | ✅ |
| `persistent_kernel: true` | HR3 登记值；HR5 true 分支接受 | ✅ |

| veps §4.1 字段（ascend_910b） | spec 承载 | 核对 |
|---|---|---|
| `name: ascend_910b` | HR2/HR3 同上 | ✅ |
| `warp_size: 0`（注释「无 warp 概念」） | HR3 登记值 0 + 语义句「0 表示无 warp 概念」 | ✅ |
| `smem_bytes: 524288`（注释「L1 Buffer」） | HR3 登记值；Shared 层级=NVIDIA shared memory / Ascend L1 Buffer 的措辞承载注释事实 | ✅ |
| `ub_bytes: 262144` | **不入 v1 字段集**（D5：语言消费面 hint=Placement.Fast 未进 memory-ops 冻结参数集 `tis.alloc_shared(shape, dtype, layout=RowMajor)`——grep 核实无 hint；proposal 非目标 #1 显式延期） | ✅ 裁决性缺席 |
| `smem_banks: 16` | HR3 登记值 | ✅ |
| （无 `registers_per_sm`） | HR3 定可选（D4：veps 无值不臆造硬件数字；缺失合法） | ✅ 裁决性缺席 |
| `async_copy: [dma]` | HR3 登记值 | ✅ |
| `mma_shapes: [[16,16,16]]` | HR3 登记值；HR4 Scenario 3（E0402 报告面）用例 | ✅ |
| `warp_group: {supported: true, max_roles: 2, roles: [vector, cube]}` | HR3 登记值（roles 可选成员的登记实例） | ✅ |
| `reduce_scopes: [block]` | HR3 登记值；HR4 单值列表→Auto=Block（veps L176「Ascend 选 Block」正式化=第一项规则） | ✅ |
| `persistent_kernel: false`（注释「1.1」） | HR3 登记值；HR5 false 分支 E0506 拒绝（行为豁口补洞） | ✅ |

**结论：§4.1 两目标全部字段行逐项承载，零缺漏；两处缺席（`ub_bytes`、910B `registers_per_sm`）均为显式裁决（D5/D4）非遗漏。§3.2 引用的「HAL 带宽」在 §4.1 本无字段——估算参数归实现内部（proposal 非目标 #6，HR3 `registers_per_sm` 条边界句）。**

## 2. 既有 specs HAL 消费面逐处落地（task 2.2，7 处）

| # | 既有 spec 文本（位置） | 落点 Requirement | 核对 |
|---|---|---|---|
| 1 | compute-ops dot 契约（L40）：「`Auto` 时……确定性地选择第一项」「`E0402`……报告列出该 HAL 支持的全部 MMA 形状」 | HR4（第一项规则 + 报告面=全列表原顺序——数据闭合，原句不改） | ✅ |
| 2 | compute-ops reduce 契约（L94）：「`scope=Auto` 时……依 HAL 能力描述确定性地选择 `Warp` 或 `Block` 之一——同一 HAL 目标的选择 MUST 唯一」 | HR1（描述存在且同目标唯一=数据前提）+ HR4（选择规则=第一项）+ compute-ops MODIFIED（显式值支持面） | ✅ |
| 3 | compute-ops Purpose：「HAL 能力字段」显式延期项 | HR3（域落地——指向有实体，延期项使命终结） | ✅ |
| 4 | diagnostics DR2（L40）：`target`「取值登记表归 HAL 域后续 change」 | HR2（合法取值=登记表成员——域级让渡承接，D11 论证无 MODIFIED） | ✅ |
| 5 | diagnostics DR8（L149）+ Purpose：「差异类目登记与 severity 赋值规则归 portability/HAL 域后续 change」 | HR7（数据来源=登记表比较、v1 对齐判定面=`mma_shapes`；类目完整清单仍延期 portability 专门 change） | ✅ |
| 6 | memory-ops（L111）：「掩码与行宽的可满足性属资源诊断 `memory` 段（建议域），不做编译期拒绝」 | HR6（建议域先例承接为域内规则：smem/banks/max_roles 超限不拒绝） | ✅ |
| 7 | execution 段位句（L187 末句）：「`E0506`–`E0599` 保留给执行结构域后续扩展」 | execution MODIFIED（起用后字面为假 → 登记性联动，E0408 先例同模式） | ✅ |

## 3. MODIFIED 逐字对照（task 3.1/3.2，diff 证据）

- **3.1 execution**：`diff stable(L183-207) delta` 输出唯一 hunk `5c5`——仅段位句一行；新句 = 原句在「`E0506`–`E0599` 保留」前插入「`E0506` 为入口 HAL 支持面拒绝（由 `hal/capability-descriptions` 定义——经该 change 对本条的 MODIFIED 起用）；」并将保留区间改 `E0507`–`E0599`；措辞与 memory-ops L185 的 E0408 起用句逐字同构。正文其余句与全部 4 Scenario 零差异。
- **3.2 compute-ops**：`diff stable(L92-125) delta` 输出两个 hunk——`3c3`（scope 值域句原句号改逗号接「且显式 `Warp` 或 `Block` MUST 在……`reduce_scopes` 支持列表内……报告该目标支持的 scope 清单」）+ `23a24,28`（新增 1 个 Scenario「reduce 显式 scope 不在目标支持列表被拒绝」）；`Auto` 句、位一致对照句与原 6 Scenario 零差异。

## 4. 管线与错误码无冲突核对（task 4.1/4.2）

| # | 核对面 | 核对结果 |
|---|---|---|
| ① | E0506 段位归属 | execution E05xx 段位句经登记性 MODIFIED 起用（§3 证据）；五段管线顺序句零改动（delta 中无管线句 hunks）✅ |
| ② | E0506 段内顺位 | execution 段内顺序句作用域「本 capability 规则」——E0506 非 execution capability 规则，原句字面保持为真（D12）；HR5 声明顺位最后；既有 E0501–E0505 触发面均为函数体/表达式语境（warp_group 体内、Pipeline 构造与调用、buffer 使用、warp_group 调用与绑定、索引内建参数），与装饰器行语境不相交 ✅ |
| ③ | E0506 与最早段规则 | 装饰器行同命中 E01xx（syntax 入口装饰器识别与位置规则）→ 依管线最早段只报一条（HR5 Scenario 3 承载，管线句不变）✅ |
| ④ | E0506 确定性 | HR5 末句「同一输入重复编译 MUST 产生一致的 E0506 拒绝清单」与 execution 报告契约末句同构；数据前提 HR1 ✅ |
| ⑤ | E0408 值域扩展与既有命中面 | 既有命中（scope 词法值域外如 `Cluster`、跨 scope 输入、axis 类型/越界、op 二值外）与新增命中（显式 `Warp`/`Block` 不在目标 `reduce_scopes`）互斥：前者为词法/类型/常量判定，后者为登记表成员判定；同码不同规则经报告内容区分（新增命中报告注明该目标支持的 scope 清单）✅ |
| ⑥ | compute-ops 段内 tiebreak | 「报告契约与 E04xx 段位补充」Requirement 本 change 零触及（delta 无该条）；tiebreak `E0408`→`E0402`→`E0403` 语义不变——reduce 支持面命中位置（reduce 调用）与 E0402/E0403 命中位置（dot 调用）为不同原语调用位置，无同位置竞争 ✅ |
| ⑦ | Auto 路径不受扩展影响 | `scope=Auto` 恒定选择 `reduce_scopes` 第一项（HR4），第一项必在列表内——永不触发支持面拒绝；扩展仅约束显式值 ✅ |

## 5. 计数核对（task 1.2）

grep 实数（`grep -c '^### Requirement:'` / `'^#### Scenario:'`）：hal 7R/19S、execution delta 1R/4S、compute-ops delta 1R/7S；与 design「Scenario 计数」（7/19、1/4、1/7、合计 9R/30S）及 proposal What Changes 各处口径一致。strict validation：`npx openspec validate add-hal-capability-descriptions --strict` 与 `--all --strict` 均 passed（8/8，修复前后各两轮）。
