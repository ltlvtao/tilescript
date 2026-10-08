## ADDED Requirements

### Requirement: 能力描述的存在性与确定性

每个登记目标（见「目标标识符登记表」）MUST 恰好有一份能力描述：编译以某登记目标为编译目标时，`Auto` 选择与 HAL 支持面判定 MUST 且仅 MUST 以该目标的能力描述为数据来源（判定类——规则与登记字段一一对应）；资源与建议域分析以能力描述的登记字段为权威数据，估算所需的其余静态参数（占用率查表、带宽等）属实现内部登记数据、同一目标内 MUST 保持唯一（数据来源边界见「能力描述字段集」）。能力描述 MUST 为静态数据：内容在编译开始前已确定，编译过程 MUST NOT 依赖目标设备的在环探测或运行期环境查询（交叉编译保证——无设备环境 MUST 能完成全部静态判定与拒绝输出）。同一登记目标的能力描述在任意两次编译中 MUST 逐字段一致（与设备在环状态、宿主环境、编译时刻无关）——本条与「列表顺序即优先序」共同构成既有 specs「依 HAL 能力描述确定性地选择……同一 HAL 目标的选择 MUST 唯一」承诺的数据前提。描述的文件载体形态（布局、加载机制）属工具链实现；字段名跨工具一致性由「能力描述字段集」冻结。

#### Scenario: 同目标描述跨编译逐字段一致

- **WHEN** 同一源码以同一登记目标重复编译两次
- **THEN** 两次编译读取的能力描述逐字段一致（如 `mma_shapes` 列表条目与顺序相同），由此 `Auto` 选择结果一致

#### Scenario: 交叉编译不依赖设备在环

- **WHEN** 在无目标设备的主机上以某登记目标编译含 `tis.dot`（`mma=Auto`）的模块
- **THEN** 编译完成全部静态判定（`Auto` 选择、HAL 支持面检查），无设备探测参与

### Requirement: 目标标识符登记表

登记表 v1 冻结为封闭二成员 `{nvidia_h200, ascend_910b}`（登记值见「能力描述字段集」）。能力描述的 `name` 字段 MUST 等于其登记表键；两个标识符为稳定标识符，MUST NOT 因后端版本、驱动或工具链变化而改值。新增登记目标 MUST 经显式 change 扩展本登记表（MUST NOT 经实现内建旁路）。诊断 JSON 顶层 `target` 字段的合法取值为且仅为登记表成员（承接 `diagnostics/json-schema` 的域级让渡——该让渡指向 HAL 域，本 capability 为域内首个实体；schema v1 冻结不动）。未登记标识符 MUST NOT 构成合法编译输入：以未登记标识符为目标的编译在工具链入口拒绝，该拒绝不属源码锚定的 `E01xx`–`E06xx` 体系（无源码位置可锚定），报告形态属工具链约定。

#### Scenario: 登记表封闭二成员

- **WHEN** 审查 v1 登记表
- **THEN** 恰好含 `nvidia_h200` 与 `ascend_910b` 两项，各自能力描述 `name` 等于其键

#### Scenario: 诊断 target 承接

- **WHEN** 编译通过产出诊断 JSON
- **THEN** 顶层 `target` 取值为登记表成员之一（如 `"nvidia_h200"`）

#### Scenario: 未登记标识符不构成合法编译输入

- **WHEN** 以 `"foo_bar"` 为编译目标调用编译器
- **THEN** 工具链入口拒绝（无合法能力描述可用，不产生源码锚定的 `E01xx`–`E06xx` 拒绝）

### Requirement: 能力描述字段集

字段集 v1 冻结为十字段（九必选一可选），逐字段的必选性/类型/值域如下：

- `name`：必选，字符串，等于登记表键。
- `warp_size`：必选，非负整数；`0` 表示该目标无 warp 概念。该值为架构事实描述，语言行为的支持面判定 MUST NOT 以 `warp_size` 为单独依据（如 reduce 的 scope 支持面以 `reduce_scopes` 为准）。
- `smem_bytes`：必选，正整数；目标 Shared 层级（NVIDIA shared memory / Ascend L1 Buffer）的可用字节数，消费面见「资源与建议域消费面」。
- `smem_banks`：必选，正整数；`bank_conflict` 静态分析的 bank 数参数。
- `registers_per_sm`：可选，正整数；实现参考参数。资源诊断的静态估算数据来源 MUST NOT 解释为限于本字段集——其余估算参数（占用率查表、带宽等）属实现内部登记数据，同一目标内 MUST 保持唯一。
- `async_copy`：必选，非空字符串列表；值为后端实现词汇（如 `cp_async`/`tma`/`dma`），供实现与诊断参考，本 capability MUST NOT 冻结其枚举值集。
- `mma_shapes`：必选，非空列表；每项为三个正整数 `[m, n, k]`，项间 MUST 互异。语义见「列表顺序即优先序」。
- `warp_group`：必选，对象；成员 `supported`（布尔）与 `max_roles`（正整数）必有，`roles`（字符串列表，后端实现词汇）可选。`max_roles` 消费面见「资源与建议域消费面」；`supported` v1 两目标均登记 `true`，`false` 分支的行为（无 warp_group 支持目标上的 warp_group 语境）为显式延期项——登记 `false` 目标前 MUST 经显式 change 定义。
- `reduce_scopes`：必选，非空列表；值为封闭小写字面 `warp`/`block`，项间 MUST 互异。语义与映射见「列表顺序即优先序」。
- `persistent_kernel`：必选，布尔。消费面见「入口 HAL 支持面」。

v1 登记值：`nvidia_h200`——`warp_size=32`、`smem_bytes=232448`、`smem_banks=32`、`registers_per_sm=65536`、`async_copy=["cp_async","tma"]`、`mma_shapes=[[16,8,16],[16,8,32]]`、`warp_group={supported=true, max_roles=2}`、`reduce_scopes=["warp","block"]`、`persistent_kernel=true`；`ascend_910b`——`warp_size=0`、`smem_bytes=524288`、`smem_banks=16`、`async_copy=["dma"]`、`mma_shapes=[[16,16,16]]`、`warp_group={supported=true, max_roles=2, roles=["vector","cube"]}`、`reduce_scopes=["block"]`、`persistent_kernel=false`（`registers_per_sm` 未登记——可选字段缺失合法；`ub_bytes` 不属 v1 字段集，随 Placement/UB 机制整体延期）。字段集演进规则：v1 已登记字段的删除、改名或语义变更 MUST NOT 发生，新增字段 MAY 经显式 change。

#### Scenario: nvidia_h200 登记值

- **WHEN** 读取 `nvidia_h200` 能力描述
- **THEN** 九个必选字段齐全且值如正文登记（`mma_shapes` 两项、`reduce_scopes` 两项、`persistent_kernel` 为 `true`、`registers_per_sm` 登记 `65536`）

#### Scenario: ascend_910b 登记值

- **WHEN** 读取 `ascend_910b` 能力描述
- **THEN** 值如正文登记：`warp_size=0`（无 warp 概念）、`reduce_scopes=["block"]`、`persistent_kernel=false`、`warp_group.roles=["vector","cube"]`；`registers_per_sm` 未登记（可选字段缺失合法）

#### Scenario: 非空与互异约束

- **WHEN** 某目标能力描述的 `mma_shapes`（或 `reduce_scopes`）为空列表，或含重复项
- **THEN** 该描述违反 v1 契约（「列表顺序即优先序」的第一项规则无定义——空列表与重复项均非合法登记值）

### Requirement: 列表顺序即优先序

`mma_shapes` 与 `reduce_scopes` 列表的条目顺序承载语义（顺序为登记值的一部分，参与逐字段一致性）：

- `mma_shapes` 顺序即 `tis.dot` `mma=Auto` 的默认选择优先序：编译器 MUST 选择第一项（`primitives/compute-ops`「依 HAL 报告的支持形状列表确定性地选择第一项」的数据闭合）；`E0402` 报告面 MUST 为该列表全部条目、按登记顺序原样列出。
- `reduce_scopes` 顺序即 `tis.reduce` `scope=Auto` 的默认选择优先序：编译器 MUST 选择第一项；列表值为小写字面 `warp`/`block`，分别映射语言层 scope 值 `Warp`/`Block`。显式 scope 的支持面判定用同一列表：语言值 `Warp` 合法当且仅当 `warp` 在列表内，`Block` 同理（拒绝行为经 `primitives/compute-ops`「reduce 的归约契约」的 MODIFIED 定义，`E0408`）。

#### Scenario: dot Auto 选择第一项

- **WHEN** `nvidia_h200` 目标上 `tis.dot(A, B, C)` 省略 `mma`
- **THEN** 选择 `mma_shapes` 第一项 `[16,8,16]`，诊断 JSON `compute` 段 `mma_shape` 呈现该选择

#### Scenario: reduce Auto 选择因目标不同

- **WHEN** 同一 `tis.reduce(x, axis=0, op=Sum)` 分别以 `nvidia_h200` 与 `ascend_910b` 为目标编译
- **THEN** 前者 `Auto` 选择 `Warp`（列表 `["warp","block"]` 第一项的映射），后者选择 `Block`（单值列表）——各自目标内唯一确定

#### Scenario: E0402 报告列全部支持形状

- **WHEN** 显式 `mma=tis.MMA(16,8,32)` 出现在 `ascend_910b` 目标编译中
- **THEN** `E0402` 报告列出的支持形状为该目标 `mma_shapes` 全部条目按登记顺序（`[16,16,16]`）

### Requirement: 入口 HAL 支持面

登记目标 `persistent_kernel=false` 时，`@tis.persistent_kernel` 装饰的入口函数 MUST 以 `E0506` 拒绝；`persistent_kernel=true` 的目标上同一入口 MUST 正常参与后续编译（此前两分支均无 spec 定义——本条为行为补洞，非兼容性破坏）。`E0506` 为入口 HAL 支持面拒绝：段位归 execution `E05xx`（经对该 capability「报告契约与 E05xx 段位」段位句的登记性 MODIFIED 起用），行为定义归本 capability。`E0506` 报告 MUST 含四要素：错误码、源码位置（入口装饰器行）、被违反的规则（目标 `persistent_kernel` 支持状态为 `false`）、恢复建议（改用 `tis.kernel` 入口或更换支持该特性的目标）。段内 tiebreak：`E0506` 在 `E05xx` 段内顺位最后（`E0501`→`E0502`→`E0503`→`E0504`→`E0505`→`E0506`——装饰器行与既有 `E05xx` 规则的触发语境不相交，顺位声明为完备性承诺）；装饰器行同时命中更早段拒绝（如语法段装饰器位置违规）时，依管线最早段规则该行 MUST 只报告一条。同一输入重复编译 MUST 产生一致的 `E0506` 拒绝清单。

#### Scenario: ascend_910b 拒绝 persistent_kernel 入口

- **WHEN** 模块含 `@tis.persistent_kernel` 装饰的入口函数，以 `ascend_910b` 为目标编译
- **THEN** 编译以 `E0506` 拒绝，报告四要素齐全（位置为装饰器行，恢复建议含 `tis.kernel` 替代路径）

#### Scenario: nvidia_h200 接受同一入口

- **WHEN** 同一模块以 `nvidia_h200` 为目标编译（`persistent_kernel=true`）
- **THEN** 该入口正常参与后续编译（无 `E0506` 命中）

#### Scenario: 装饰器行跨段命中报最早段

- **WHEN** `@tis.persistent_kernel` 装饰器行同时命中语法段拒绝（装饰器位置违反语法接受集）且目标 `persistent_kernel=false`
- **THEN** 该行依管线最早段规则只报告语法段一条（`E0506` 判定存在但不改变管线输出）

### Requirement: 资源与建议域消费面

`smem_bytes`、`smem_banks` 与 `warp_group.max_roles` 的消费面为诊断与建议域，MUST NOT 作为编译期拒绝依据：Shared 静态分配总字节超出 `smem_bytes`、swizzle 掩码相对 `smem_banks` 的可满足性、warp_group 分组角色数超出 `max_roles`——均 MUST NOT 触发 `E01xx`–`E06xx` 拒绝（信息经诊断 JSON `resources`/`memory`/`suggestions` 段或 portability 提示呈现；承接 `primitives/memory-ops`「掩码与行宽的可满足性属资源诊断 `memory` 段（建议域），不做编译期拒绝」先例）。`smem_banks` 为 `memory` 段 `bank_conflict` 静态分析的 bank 数参数。资源量与入口支持状态不同类，分界为本域合并裁决：资源量为容量约束，MUST 呈现而不拒绝；支持状态（`persistent_kernel`）为二值能力判定，MUST 拒绝（见「入口 HAL 支持面」）。

#### Scenario: Shared 分配超 smem_bytes 不拒绝

- **WHEN** kernel 的静态 Shared 分配总量超出目标 `smem_bytes`
- **THEN** 编译不被拒绝（无 `E04xx`/`E05xx` 命中），超限信息经诊断 JSON `resources` 段与建议呈现

#### Scenario: warp_group 角色数超 max_roles 不拒绝

- **WHEN** 某 `warp_group.max_roles=1` 的登记目标上，warp_group 分组使用 `"producer"` 与 `"consumer"` 两种角色（超出该目标上限——role 值域封闭二值，v1 两目标登记值 2，本例为未来登记形态）
- **THEN** 编译不被拒绝，约束冲突经建议域呈现（资源量非支持面）

### Requirement: 跨 HAL 行为不变面

语言行为 MUST NOT 依赖编译目标：语法接受集与拒绝错误码（`language/syntax-acceptance-set`）、类型规则与作用域转移（`language/type-system`）、原语契约的参数与结果规则（`primitives/*`）、执行结构契约（`execution/pipeline-structure`）、数值语义含 cast 静态位一致承诺（`numerics/value-semantics`）在任意登记目标上 MUST 逐条相同。HAL 目标的影响面为且仅为封闭四类：①`Auto` 选择结果（`dot` `mma`、`reduce` `scope`——经「列表顺序即优先序」）；②HAL 支持面判定触发的拒绝（`E0402`/`E0403` 形状类、reduce 显式 `scope` 支持面 `E0408`、`E0506`）；③诊断内容（`target` 取值、资源数字、portability 差异条目、`Auto` 选择的呈现）；④实现路径与指令生成（不可观察面，MUST NOT 反向影响前三类）。portability 差异判定的数据来源为登记目标能力描述的比较，v1 对齐差异判定面为 `mma_shapes`（其余差异类目完整登记延期 portability 专门 change——`diagnostics/json-schema` 的该延期项承接后依然成立）。跨 HAL 数值边界总括：cast 静态位一致为跨目标承诺（`numerics/value-semantics`），归约结果位模式不承诺跨目标一致（`primitives/compute-ops`——归约顺序依实现）——二者合成「类型与结构规则跨硬件一致、数值位模式逐字段声明」的总括。

#### Scenario: 同源码两目标语言层拒绝一致

- **WHEN** 同一模块仅含语言层违规（如类型不匹配 `E0303`，不触发任何 HAL 支持面），分别以两个登记目标编译
- **THEN** 两目标的拒绝清单逐条一致（错误码、位置、规则、建议——目标不是语言层判定的输入）

#### Scenario: 支持面差异以拒绝呈现

- **WHEN** 同一含显式 `mma=tis.MMA(16,8,32)` 的模块分别以 `nvidia_h200`（支持列表含该项）与 `ascend_910b`（不含）为目标编译
- **THEN** 前者通过该检查、后者以 `E0402` 拒绝——差异属影响面清单②（HAL 支持面），非语言层不一致

#### Scenario: cast 位一致跨目标

- **WHEN** 同一含 `tis.cast` 的模块以两个登记目标编译并通过
- **THEN** 两次编译的 cast 静态位一致承诺均成立（`numerics/value-semantics` 既有承诺的总括引用，目标不是其判定输入）
