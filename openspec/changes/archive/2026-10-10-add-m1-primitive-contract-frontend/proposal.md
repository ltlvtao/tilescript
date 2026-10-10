# Proposal: add-m1-primitive-contract-frontend

## Why

1. **M1 第三检查段**（用户指令「启动」）：五段检查管线顺序为 syntax → type-system → primitive-contract → execution-structure → numerics（`toolchain/cli` R3）。前两段已交付（`add-m1-syntax-frontend`、`add-m1-type-system-frontend`，均 2026-10-10 归档）；本 change 实现第三段——原语契约段。
2. **规格就绪而实现为零**：`primitives/memory-ops`（8 Requirement / 33 Scenario，2026-09-28 冻结，段位句经 2026-10-08 一次登记性联动）与 `primitives/compute-ops`（7 Requirement / 34 Scenario，2026-10-08 冻结，reduce 契约经同日一次行为性补洞）合计 15 Requirement / 67 Scenario 定义 `E0402`–`E0408` 全部行为，至今无一行实现。
3. **HAL 能力描述的首个消费段**：`hal/capability-descriptions`（2026-10-08 冻结）定义的 `mma_shapes`/`reduce_scopes` 字段（列表顺序即优先序、两目标登记值入 spec）以「首个消费段 change」为加载时点——`E0402`（显式 MMA 支持面）、`E0403`（整除检查依所选形状）、reduce 显式 scope 支持面（`E0408`）与 `Auto` 确定性选择均以此为数据前提。
4. **两处规格豁口须先闭合**（规格先行）：
   - **缺失必选实参的拒绝路径**（memory-ops R1）：正文只定义「参数集外关键字实参」与「位置实参数量超出参数集总长度」两拒绝路径，`tis.load(src)` 缺失必选 `dst` 的行为 undefined——compute-ops R1 已有对称明文（「缺失必选位置实参时 MUST 以 `E0408` 拒绝」），memory-ops 侧缺口使八原语的参数完整性检查不闭合；
   - **段间执行语义**（memory-ops R8）：定义了「管线顺序语法→类型→原语契约」与跨段同位置 tiebreak，未定义「类型段存在拒绝时原语契约段是否执行」——短路读法（只输出类型段拒绝）与并行收集读法（两段拒绝合并输出）公共可观察输出不同；`language/type-system` R8 已为其前一段（语法→类型）裁决短路语义，本 change 将同一裁决延伸到第三段。
5. **诚实退出时点快照**（`toolchain/cli` R3 Scenario 1）：WHEN「仅语法段与类型系统段已实现」在本 change 后永不可满足，须按仓库先例做登记性 MODIFIED 联动更新。

## What Changes

- **MODIFIED** `primitives/memory-ops`「原语集合与调用结构」（4→5 Scenario）：拒绝路径补全——必选位置实参缺失（位置实参数量少于必选参数个数）时 MUST 以 `E0406` 拒绝（原 undefined→定义，**非 BREAKING**，与 compute-ops R1 对称句同型）；新增 1 个 Scenario 验证 `tis.load(src)` 缺 `dst` 被拒绝。
- **MODIFIED** `primitives/memory-ops`「报告契约与 E04xx 段位」（4→5 Scenario）：补段间短路执行语义——类型系统段存在任一拒绝时原语契约段 MUST NOT 执行；类型系统段零拒绝时原语契约段执行（原「同位置跨段命中只报最早段一条」保持独立规范句地位，短路句为其执行语义，与 `language/type-system` R8 既有短路句同型）。原 undefined 行为被定义，**非 BREAKING**；新增 1 个 Scenario 验证「类型段存在拒绝时原语段不执行」（与 delta Scenario 定稿标题一致）。
- **MODIFIED** `toolchain/cli`「分段实现状态与诚实退出」（2 Scenario 不变，登记性联动）：Scenario 1 快照更新——已实现段为语法段、类型系统段与原语契约段三段（正文泛化句不动）。**非 BREAKING**。
- **实现 `primitives/memory-ops` 全量**（其余语义零 delta，既有契约为准）：
  - `E0404`（转移格承载）：`load` 只承载 Global→Shared / Global→Register / Shared→Register，`store` 只承载 Shared→Global / Register→Global / Register→Shared；原语与格子操作类别不匹配（组合落在另一原语承载格）拒绝并建议正确原语；copy（Shared→Shared）与 move（Register→Register）格无承载原语拒绝；矩阵非法格（Global→Global）仍由类型段 `E0301` 裁决，本段不报。
  - `E0405`（数据维度）：src/dst 须为 Tensor（`Pointer`/标量/状态类归 `E0406`）；dtype 相同；逐维**长度相容**专用判定四支——双常量等值 / 同运行期符号 / 派生表达式结构等价 / 派生维静态折叠为常量与对侧等值（切片 `bm*BR:(bm+1)*BR` 折叠为 `BR`）；报告含两侧类型与不相容维度；不产生类型层等价结论。
  - `E0406`（参数值域与形态）：八原语参数集封闭（未知关键字、位置超量、缺失必选均拒）；`mode` 值域 Sync/Async；`make_tensor` 的 `ptr` 限 `Pointer[d, Global]`、shape 限整数组件，返回 `Tensor[d, shape, Global]`；`alloc_shared` 的 `layout` 值域 RowMajor / `tis.Layout.swizzled(xor=非负 comptime[int])`，返回 `Tensor[dtype, shape, Shared]`（`swizzled` 为分配物理属性不进类型组件）；`zeros`/`full` 的 `scope` 值域 Register/Shared、`full` 的 `value` 限标量表达式形态；`cast` 的 `x` 限 Tensor、`dtype` 限六值封闭集，返回 `Tensor[dtype, shape(x), scope(x)]`；`barrier` 的 `scope` 值域 Block/WarpGroup。
  - `E0407`（语境）：`mode=Async` 仅可出现在 Pipeline produce/consume 嵌套函数体内（kernel 顶层等拒绝）；`load`/`store`/`barrier` 不产生值——赋值右侧、调用实参、`return` 值位置拒绝。
  - 报告契约：四要素、收集全部、位置升序、同位置段内顺序 `E0404`→`E0405`→`E0406`→`E0407` 取首个、重复编译逐条一致。
- **实现 `primitives/compute-ops` 全量**（语义零 delta，既有契约为准）：
  - `E0408`（调用结构与操作数契约）：六原语参数集封闭（`dot` 的 `mma`/`pad` 关键字专属、缺失必选拒、未知关键字拒）；dot 的 A/B 限 `f16` 且 (M,K)/(K,N)、Shared 或 Register、shape 各维编译期常量、K 两侧相等，C 限 `Tensor[f32, (M, N), Register]`，`tis.MMA` 构造限三个 comptime[int]，`pad` 值域封闭四值；reduce 的 `x` 限 Register Tensor、`axis` 限 comptime[int] 且界内、`op` 值域 Sum/Max、`scope` 值域 Auto/Warp/Block 且显式值须在目标 HAL `reduce_scopes` 支持列表内；exp/log 限浮点 dtype Register Tensor；maximum 严格同 dtype/shape/Register；transpose 限二维；**MMA 形状值与 PadPolicy 值为 dot 实参语境专用**——其余任何使用位置（赋值绑定、其他调用实参、算术/比较/条件操作数）拒绝。
  - `E0402`（显式 MMA 支持面）：显式 `mma` 形状不在编译目标 HAL `mma_shapes` 列表时拒绝，报告按登记顺序列出全部支持形状。
  - `E0403`（整除与 pad 策略）：所选 MMA 形状（显式或 `Auto`=列表第一项）不整除 M/N/K 且 `pad=Error`（默认）时拒绝，报告最近合法对齐形状建议；MUST NOT 自动回退非 Tensor Core 路径；`PadZero`/`Mask`/`Split` 三策略编译通过。
  - 结果类型：dot → `Tensor[f32, (M, N), Register]`、reduce → 去 axis 维、maximum/exp/log → 与操作数同型、transpose → 二维互换。
  - 报告契约：段内顺序 `E0408`→`E0402`→`E0403`；与 memory-ops `E0404`–`E0407` 无同位置双命中；`Auto` 选择确定（同输入同目标一致）。
- **HAL 能力描述加载**（`hal/capability-descriptions` 既有契约的实现，零 delta）：两目标登记值（`mma_shapes`/`reduce_scopes` 含顺序）进入实现；`Auto` 选择 = 列表第一项；`E0402` 报告面 = 全列表按登记顺序；小写 `warp`/`block` ↔ 语言值 `Warp`/`Block` 映射；载体形态为工具链实现（Python 冻结常量，满足「静态数据、同目标逐字段一致、无设备在环」契约；YAML 文件载体仍为延期项）。
- **原语调用结果类型承接**（memory-ops Purpose 让渡句的实现）：八存取原语与六计算原语的返回类型在本段推断（`make_tensor`/`zeros`/`full`/`alloc_shared`/`cast`/`dot`/`reduce`/`maximum`/`exp`/`log`/`transpose`；`load`/`store`/`barrier` 不产生值）——类型段对这些表达式的 UNKNOWN 让渡行为**不变**（两段独立运行，段间短路保证类型段零拒绝时本段才执行）；本段结果类型仅供本段实参契约检查与后续合法程序检查使用。
- **管线编排**：`pipeline` 第三段接入（syntax → 段间短路 → type-system → 段间短路 → primitive-contract）；HAL 依赖检查（`E0402`/`E0403`/reduce scope 支持面）以编译目标为输入（CLI `--target` 既有值透传）；CLI 诚实退出实例化 `implemented_stages=["syntax","type-system","primitive-contract"]`；既有 CLI 测试两段断言随 delta 联动更新为三段。
- **无 BREAKING**：三条 delta 均为 undefined→定义或登记性联动；错误码、诊断 JSON Schema、拒绝清单五字段形态均不变（`E04xx` 条目进入清单值域属既有 v1 冻结演进规则内的错误码使用）。

## 非目标

- **`E05xx`/`E06xx` 段实现**：执行结构（`E0501` 计算原语类别清单的 producer 体内拒绝行为、`E0502` Pipeline 契约、`E0505` 内建、`E0506` 入口 HAL 支持面）、数值语义（`E06xx`——原语实参中的算术子表达式先经其定型）——CLI 以 `pending_stages` 如实列出。
- **`buf.<名>` 属性与 Pipeline/buffer 对象类型**：归 execution-structure 段；M1 本段对 `buf.K` 等按未知让渡（依赖它的 `E0405` 维度检查与 dot 操作数检查自然跳过，后续段实现后收紧）。
- **算术/比较子表达式的类型与数值规则**：`numerics/value-semantics`（本段实参检查消费类型段的推断结果，算术表达式仍 UNKNOWN 让渡）。
- **诊断 JSON 通过形态与资源/建议域**：`swizzled` 物理布局记录（memory 段）、`mma_shape` 呈现（compute 段）、资源超限提示——`diagnostics/json-schema` 承载，M2-M3；本段只产拒绝清单。
- **copy/move 格专用显式原语**：memory-ops R2 显式声明延期（同 scope 数据路径由普通赋值承载）。
- **逐元素比较原语、dot A/B 超 f16、批维、跨 Shared 计算**：compute-ops 显式延期项。
- **`full` 的 `value` 到 dtype 数值转换、cast 数值效果**：数值语义段；本段只做标量表达式形态级核对。
- **HAL YAML 文件载体与加载机制**：工具链实现延期项；本 change 以冻结常量承载登记值。
- **新登记目标**：登记表 v1 封闭二成员不动。

## Impact

- 五段管线第三段落地：原语契约检查器成为 execution-structure 段（第四段）的基座（`E0501` 类别清单判定面由本段供给、`E0502` 调用合法性复用本段符号环境）。
- 拒绝清单 JSON 首次出现 `E04xx` 条目（五字段形态不变）；`{"status":"incomplete"}` 的 `implemented_stages` 扩为三段（CLI 测试断言联动）。
- `pipeline.compile_stages` 签名新增编译目标参数（内部 API；CLI `--target` 既有必选值透传，`toolchain/cli` R1/R2 行为零变化）。
- HAL 登记值首次进入实现：`nvidia_h200`（`mma_shapes=[[16,8,16],[16,8,32]]`、`reduce_scopes=["warp","block"]`）与 `ascend_910b`（`mma_shapes=[[16,16,16]]`、`reduce_scopes=["block"]`）——`E0402`/reduce scope 支持面与 `Auto` 选择随目标不同（`hal/capability-descriptions`「跨 HAL 行为不变面」影响面清单①②）。
- 集成样例（veps §7 FlashAttention）在原语段零拒绝：`tis.load` 切片折叠支（(d)）贯通、`tis.dot` 常量维操作数合法、`buf.*` 让渡跳过——该样例升级为三段回归。
- 同一源码的语言层拒绝（`E01xx`/`E03xx`/`E0404`–`E0408` 非 HAL 依赖部分）在两登记目标上逐条一致（`hal` 不变面）；HAL 依赖拒绝（`E0402`/`E0403`/reduce scope）按目标不同。
