# primitives/compute-ops Specification

## Purpose

定义计算原语域的行为契约：六个计算原语（`tis.dot`/`tis.reduce`/`tis.maximum`/`tis.exp`/`tis.log`/`tis.transpose`）的调用结构（参数集、顺序、可省略参数与确定性默认值 `Auto`/`Error`）、操作数与参数值契约（dtype/scope/shape 约束、`tis.MMA` 形状值与 `tis.PadPolicy` 值的 dot 实参语境专用地位）、结果类型与数学语义（dot 累加语义、reduce 去维、maximum 严格同 shape、transpose 二维互换）、计算原语类别清单（闭合 `execution` E0501 的"tis.dot 等"），以及 `E0402`/`E0403`（既有占用正式化）与 `E0408`（新增）报告契约（原语契约段位，段内序 E0408→E0402→E0403）。职责分工：原语调用的语法形式（调用/属性访问/关键字实参）归 `language/syntax-acceptance-set`；原语结果再绑定等价与实参表达式类型推导（`E0303` 豁免）归 `language/type-system`；计算原语实参中的算术子表达式归 `numerics/value-semantics`（先经其定型再进本域契约，子表达式位置与调用位置各自报告）；转移/存取原语（`E0404`–`E0407`）归 `primitives/memory-ops`；类别拒绝行为（E0501）归 `execution/pipeline-structure`（类别判定面由本 capability 供给）；本 capability 承接以上让渡并定义计算原语域规则本身。逐元素比较原语、dot A/B 超 f16 支持面、批维、跨 Shared 计算、数值精度与归约顺序跨硬件位一致（归约顺序依硬件）、诊断 compute 段字段、HAL 能力描述字段、`atomic_add` 为显式延期项；`E0409`–`E0499` 保留给原语域后续扩展。

## Requirements

### Requirement: 原语集合与调用结构

本 capability 定义六个计算原语，其调用结构（参数集、顺序、可省略参数与默认值）MUST 为：`tis.dot(A, B, C, *, mma=Auto, pad=Error)`（`A`/`B`/`C` 为位置实参，`mma`/`pad` 为关键字专属）；`tis.reduce(x, axis, op, scope=Auto)`（`x`/`axis`/`op` 必选，`scope` 可省略）；`tis.maximum(a, b)`；`tis.exp(x)`；`tis.log(x)`；`tis.transpose(x)`。可按位置传递的参数（必选与可省略）MAY 以关键字形式显式传递，MAY 按声明顺序以位置实参传递（如 `tis.reduce(S, 1, Max, Block)` 解析等价于 `axis=1, op=Max, scope=Block`）；`dot` 的 `mma` 与 `pad` 为关键字专属参数，MUST NOT 以位置实参传递。使用参数集之外的任何关键字实参、位置实参数量超出该原语**可按位置传递参数**的总长度（`dot` 为三个：`A`/`B`/`C`；`reduce` 为四个）、或缺失必选位置实参时，编译器 MUST 以 `E0408` 拒绝。默认值（`Auto`、`Error`）固定且写进文档。六原语集合为本 capability 的封闭集；计算原语域的原语名扩展（逐元素比较等）MUST 经显式 change。

#### Scenario: dot 省略关键字参数被接受

- **WHEN** 设备代码调用 `tis.dot(A, B, C)`（省略 `mma` 与 `pad`）
- **THEN** 调用结构检查通过（`mma` 取默认 `Auto`、`pad` 取默认 `Error`）

#### Scenario: dot 关键字专属参数按位置传递被拒绝

- **WHEN** 设备代码调用 `tis.dot(A, B, C, tis.MMA(16, 8, 16))`（`mma` 以第四位置实参传递）
- **THEN** 编译以 `E0408` 拒绝，报告注明 `dot` 可按位置传递的实参只有 `A`/`B`/`C` 三个，`mma`/`pad` 必须以关键字传递

#### Scenario: reduce 未知关键字实参被拒绝

- **WHEN** 设备代码调用 `tis.reduce(S, axis=1, op=Max, keepdims=True)`（`keepdims` 不在参数集内）
- **THEN** 编译以 `E0408` 拒绝，报告注明 `reduce` 的合法参数集为 `x`/`axis`/`op`/`scope`

#### Scenario: reduce 缺失必选实参被拒绝

- **WHEN** 设备代码调用 `tis.reduce(S, axis=1)`（缺失必选 `op`）
- **THEN** 编译以 `E0408` 拒绝，报告注明 `op` 必选且值域为 `Sum`/`Max`

#### Scenario: transpose 位置实参超量被拒绝

- **WHEN** 设备代码调用 `tis.transpose(x, 0, 1)`（两个位置实参）
- **THEN** 编译以 `E0408` 拒绝，报告注明 `transpose` 只接受一个位置实参

### Requirement: dot 的操作数与 MMA 契约

`tis.dot(A, B, C, *, mma, pad)` 的操作数契约：`A` MUST 为 `Tensor[f16, (M, K), Shared 或 Register]`，`B` MUST 为 `Tensor[f16, (K, N), Shared 或 Register]`（A 与 B 的 dtype MUST 相同且限 `f16`——MMA dtype 支持面保守封闭）；`C` MUST 为 `Tensor[f32, (M, N), Register]`（累加初值）；A/B/C 的 shape 各维 MUST 为编译期常量维（运行期派生维以 `E0408` 拒绝——MMA 形状整除检查要求编译期可知的 tile 形状）；`K` 为 A 第二维与 B 第一维的公共长度，两侧不相等以 `E0408` 拒绝。结果 MUST 为 `Tensor[f32, (M, N), Register]`，其值为 `C + A @ B` 的数学语义（数值精度属跨硬件容差，非本 capability 承诺面）。`mma` 实参为 `tis.MMA(m, n, k)` 构造（三个 `comptime[int]`；任一实参类型不符或为运行期 `int` 以 `E0408` 拒绝）或 `Auto`：`Auto` 时编译器 MUST 依 HAL 报告的支持形状列表确定性地选择第一项——同一 HAL 目标的选择 MUST 唯一且写进文档（选择的诊断呈现归 `diagnostics/*`）。显式 `mma` 形状不在该 HAL 支持列表时 MUST 以 `E0402` 拒绝，报告列出该 HAL 支持的全部 MMA 形状。所选 MMA 形状不能整除 M/N/K 且 `pad=Error`（默认）时 MUST 以 `E0403` 拒绝，报告列出最近的合法对齐形状建议；编译器 MUST NOT 自动回退到非 Tensor Core 路径（`Split` 策略的尾部 FMA 是显式选择的结果，不属自动回退）。`pad` 实参值域为封闭四值 `tis.PadPolicy.Error`（默认）/`PadZero`/`Mask`/`Split`（非法值以 `E0408` 拒绝）：`PadZero` 为编译器在 Shared 分配补零对齐、`Mask` 为尾部 predicated MMA、`Split` 为对齐部分 MMA 加尾部 FMA——三策略均编译通过（其物理布局与指令生成细节归实现设计；诊断字段归 `diagnostics/*`）。**MMA 形状值与 PadPolicy 值为 dot 实参语境专用值**（承接 `language/type-system`「表达式结果类型规则」对模块属性访问与原语调用表达式结果类型的让渡；与 `numerics/value-semantics` 布尔值的条件语境专用值同构）：不进 `language/type-system` 封闭类型世界（不新增类型种类），合法使用位置为封闭集——`tis.MMA(...)` 构造值仅作 `dot` 的 `mma=` 关键字实参、`tis.PadPolicy.*` 属性值仅作 `dot` 的 `pad=` 关键字实参；其余任何使用（赋值绑定、其他原语或调用实参、算术或比较操作数、条件操作数等）以 `E0408` 拒绝。

#### Scenario: dot 显式 mma 与 pad 被接受

- **WHEN** `Q_s` 为 `Tensor[f16, (BR, D), Shared]`（BR/D 为 comptime 常量且被所选形状整除）、`K_t` 为 `Tensor[f16, (D, BC), Shared]`、`C0` 为 `Tensor[f32, (BR, BC), Register]`，调用 `tis.dot(Q_s, K_t, C0, mma=tis.MMA(16, 8, 16), pad=tis.PadPolicy.Error)` 且该形状在 HAL 支持列表
- **THEN** 调用合法，结果为 `Tensor[f32, (comptime BR, comptime BC), Register]`（值为 `C0 + Q_s @ K_t` 的数学语义）

#### Scenario: dot 默认参数 Register 操作数被接受

- **WHEN** `A` 为 `Tensor[f16, (BR, BC), Register]`（如 cast 结果）、`B` 为 `Tensor[f16, (BC, D), Shared]`、`C` 为 `Tensor[f32, (BR, D), Register]`，调用 `tis.dot(A, B, C)`（省略 `mma`/`pad`）
- **THEN** 调用合法（A 的 Register scope 在允许集内），`mma` 取 `Auto` 确定性选择，结果为 `Tensor[f32, (comptime BR, comptime D), Register]`

#### Scenario: A 与 B dtype 非 f16 被拒绝

- **WHEN** `A`/`B` 均为 `Tensor[bf16, …]`（dtype 非 f16）
- **THEN** 编译以 `E0408` 拒绝，报告注明 dot 的 A/B 操作数 dtype 限 f16（支持面扩展须显式 change）

#### Scenario: K 维不相容被拒绝

- **WHEN** `A` 为 `Tensor[f16, (BR, D), Shared]`、`B` 为 `Tensor[f16, (D2, BC), Shared]` 且 `D != D2`
- **THEN** 编译以 `E0408` 拒绝，报告两侧 K 维长度

#### Scenario: C 契约不符被拒绝

- **WHEN** `C` 为 `Tensor[f16, (BR, BC), Register]`（dtype 非 f32）
- **THEN** 编译以 `E0408` 拒绝，报告注明 C 必须为 f32 的 (M, N) Register Tensor

#### Scenario: 运行期派生维操作数被拒绝

- **WHEN** `A` 的 shape 含运行期派生维（如切片边界派生）
- **THEN** 编译以 `E0408` 拒绝，报告注明 dot 操作数 shape 必须为编译期常量维

#### Scenario: 显式 mma 形状不支持被拒绝

- **WHEN** `tis.MMA(16, 8, 32)` 不在该 HAL 报告的支持形状列表
- **THEN** 编译以 `E0402` 拒绝，报告列出该 HAL 支持的全部 MMA 形状

#### Scenario: M/N/K 不整除且 Error 策略被拒绝

- **WHEN** 所选 MMA 形状为 (16, 8, 16)，`A` 为 `Tensor[f16, (100, 128), Shared]`（M=100 不被 16 整除），`pad` 取默认 `Error`
- **THEN** 编译以 `E0403` 拒绝，报告列出最近的合法对齐形状建议，MUST NOT 自动回退非 Tensor Core 路径

#### Scenario: 不整除且显式 pad 策略被接受

- **WHEN** 同上一场景的 M=100，`pad=tis.PadPolicy.Mask`
- **THEN** 编译通过（尾部 predicated MMA；策略的物理实现归实现设计，诊断字段归 `diagnostics/*`）

#### Scenario: PadPolicy 值用于实参语境之外被拒绝

- **WHEN** 设备代码出现 `p = tis.PadPolicy.Mask`（赋值绑定）或以 `tis.MMA(16, 8, 16)` 为其他原语实参
- **THEN** 编译以 `E0408` 拒绝，报告注明 MMA 形状值与 PadPolicy 值仅限 `dot` 的 `mma=`/`pad=` 实参位置

### Requirement: reduce 的归约契约

`tis.reduce(x, axis, op, scope=Auto)` 的契约：`x` MUST 为 `Register` scope 的 Tensor（dtype 为封闭 dtype 集合六种之一，任意 shape；跨 scope 输入以 `E0408` 拒绝，恢复建议经显式移动原语对齐）；`axis` MUST 为 `comptime[int]` 且在 `[0, rank(x))` 值域内（类型不符、含运行期 `int` 或越界均以 `E0408` 拒绝）；`op` 值域为封闭二值 `Sum`/`Max`（其他以 `E0408` 拒绝）；`scope` 值域为封闭三值 `Auto`（默认）/`Warp`/`Block`（其他以 `E0408` 拒绝），且显式 `Warp` 或 `Block` MUST 在编译目标 HAL 能力描述的 `reduce_scopes` 支持列表内（字段语义与登记值由 `hal/capability-descriptions` 定义），目标不支持时 MUST 以 `E0408` 拒绝并报告该目标支持的 scope 清单。结果 MUST 为去掉 `axis` 维的 Tensor：dtype 与 `x` 相同、shape 为 `x` 的 shape 删除 `axis` 维、scope 为 `Register`。`scope=Auto` 时编译器 MUST 依 HAL 能力描述确定性地选择 `Warp` 或 `Block` 之一——同一 HAL 目标的选择 MUST 唯一且写进文档。本 capability MUST NOT 承诺跨 HAL 后端的归约结果位一致（归约顺序依实现，数值差异属跨硬件容差——与 `numerics/value-semantics` cast 的静态位一致承诺形成边界对照：类型与去维规则跨硬件一致，数值位模式不承诺）。

#### Scenario: reduce 合法调用去维

- **WHEN** `S` 为 `Tensor[f32, (BR, BC), Register]`，调用 `tis.reduce(S, axis=1, op=Max)`
- **THEN** 结果为 `Tensor[f32, (BR,), Register]`（去掉 axis=1 维，dtype 不变）

#### Scenario: reduce axis 越界被拒绝

- **WHEN** `x` 为二维 Tensor，调用 `tis.reduce(x, axis=2, op=Sum)`
- **THEN** 编译以 `E0408` 拒绝，报告注明 axis 值域为 [0, 2)

#### Scenario: reduce op 非法值被拒绝

- **WHEN** 调用 `tis.reduce(x, axis=0, op=Min)`（`Min` 不在封闭二值内）
- **THEN** 编译以 `E0408` 拒绝，报告注明 op 值域为 `Sum`/`Max`

#### Scenario: reduce scope 非法值被拒绝

- **WHEN** 调用 `tis.reduce(x, axis=0, op=Sum, scope=Cluster)`
- **THEN** 编译以 `E0408` 拒绝，报告注明 scope 值域为 `Auto`/`Warp`/`Block`

#### Scenario: reduce 显式 scope 不在目标支持列表被拒绝

- **WHEN** 编译目标 HAL 能力描述的 `reduce_scopes` 为 `["block"]`（如 `ascend_910b`），调用 `tis.reduce(x, axis=0, op=Sum, scope=Warp)`
- **THEN** 编译以 `E0408` 拒绝，报告注明该目标支持的 scope 清单（列表小写 `block` 对应语言值 `Block`）

#### Scenario: reduce 非 Register 操作数被拒绝

- **WHEN** `x` 为 `Shared` scope Tensor
- **THEN** 编译以 `E0408` 拒绝，恢复建议经显式移动原语对齐后归约

#### Scenario: reduce 整型 dtype 保持

- **WHEN** `x` 为 `Tensor[i32, (BR, BC), Register]`，调用 `tis.reduce(x, axis=1, op=Sum)`
- **THEN** 结果为 `Tensor[i32, (BR,), Register]`（dtype 不变规则对封闭集合六种 dtype 一致）

### Requirement: 元素级与成对计算原语

`tis.exp(x)` 与 `tis.log(x)`：`x` MUST 为 `Register` scope 的 Tensor，dtype 为浮点 dtype（`f16`/`bf16`/`f32`/`f8e4m3`——整型 dtype 以 `E0408` 拒绝）；结果 MUST 与 `x` 同 dtype、同 shape、同 scope（数值精度属跨硬件容差）。`tis.maximum(a, b)`：`a`/`b` MUST 为同 dtype（封闭集合六种之一）、同 shape、同 `Register` scope 的 Tensor（任一不符以 `E0408` 拒绝——严格同 shape，不做逐维广播；shape 不符的恢复建议经 `numerics/value-semantics` 的切片与 `None` 广播或算术组合调整）；结果 MUST 与操作数同 dtype、同 shape、同 scope。三个原语均为元素级/成对计算，不涉及跨线程汇合。

#### Scenario: exp 保持类型与形状

- **WHEN** `x` 为 `Tensor[f32, (BR,), Register]`，调用 `tis.exp(x)`
- **THEN** 结果为 `Tensor[f32, (BR,), Register]`（dtype/shape/scope 不变）

#### Scenario: log 浮点操作数被接受

- **WHEN** `st_l` 为 `Tensor[f32, (BR,), Register]`，调用 `tis.log(st_l)`
- **THEN** 结果为 `Tensor[f32, (BR,), Register]`

#### Scenario: exp 整型操作数被拒绝

- **WHEN** `x` 为 `Tensor[i32, …, Register]`，调用 `tis.exp(x)`
- **THEN** 编译以 `E0408` 拒绝，报告注明 exp/log 的操作数 dtype 限浮点 dtype

#### Scenario: maximum 同 shape 同 dtype 被接受

- **WHEN** `st_m` 与 `m_ij` 均为 `Tensor[f32, (BR,), Register]`，调用 `tis.maximum(st_m, m_ij)`
- **THEN** 结果为 `Tensor[f32, (BR,), Register]`

#### Scenario: maximum shape 不匹配被拒绝

- **WHEN** `a` 为 `Tensor[f32, (BR,), Register]`、`b` 为 `Tensor[f32, (BR, 1), Register]`
- **THEN** 编译以 `E0408` 拒绝（严格同 shape，不做广播），恢复建议经切片与 `None` 广播调整形状后成对取最大

### Requirement: transpose 的数据重排契约

`tis.transpose(x)`：`x` MUST 为恰好二维的 Tensor（其他秩以 `E0408` 拒绝），dtype 为封闭集合六种之一，scope 为 `Shared` 或 `Register`；结果 MUST 为转置 shape（两维互换）的 Tensor，dtype 与 scope 与 `x` 相同。transpose 是纯数据重排（值不变、维序互换）；物理重排时机与布局归实现设计。

#### Scenario: transpose 二维转置被接受

- **WHEN** `buf_K` 为 `Tensor[f16, (BC, D), Shared]`，调用 `tis.transpose(buf_K)`
- **THEN** 结果为 `Tensor[f16, (D, BC), Shared]`（dtype 与 scope 不变）

#### Scenario: transpose 非二维被拒绝

- **WHEN** `x` 为三维 Tensor
- **THEN** 编译以 `E0408` 拒绝，报告注明 transpose 操作数必须为二维 Tensor

### Requirement: 计算原语类别清单

计算原语类别 MUST 为封闭集：`tis.dot`、`tis.reduce`、`tis.maximum`、`tis.exp`、`tis.log`、`tis.transpose`。本清单闭合 `execution/pipeline-structure` E0501 中"计算原语（`tis.dot` 等）"的类别引用：producer warp_group 体内出现上述任一原语的调用 MUST 以 `E0501` 拒绝（拒绝行为与错误码归该 capability，本 capability 只提供类别定义）；consumer 体内不设原语类别限制（该 capability 已裁）。清单扩展（逐元素比较原语等）MUST 经显式 change，并 MUST 同步更新本清单与 E0501 的判定面。

#### Scenario: producer 体内 reduce 被拒绝

- **WHEN** producer warp_group 体内出现 `tis.reduce(S, axis=1, op=Max)`（非 dot 的计算原语）
- **THEN** 以 `E0501` 拒绝（类别清单涵盖全部六原语，不限于 dot；拒绝由 execution 承载）

#### Scenario: consumer 体内计算原语被接受

- **WHEN** consumer warp_group 体内出现 `tis.dot(...)`/`tis.exp(...)` 调用（操作数契约合法）
- **THEN** 不因原语类别被拒绝（consumer 不设类别限制）

### Requirement: 报告契约与 E04xx 段位补充

本 capability 产生的每条拒绝（`E0402`/`E0403`/`E0408`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含违规形态与两侧相关值）、恢复建议（`E0402` MUST 列出该 HAL 支持的全部 MMA 形状；`E0403` MUST 列出最近的合法对齐形状建议）。同一模块存在多条本域拒绝时 MUST 收集输出全部并按源码位置升序排列。同一调用命中多个本 capability 错误码时 MUST 只报告一个，按段内顺序取首个：`E0408`（调用结构与操作数契约）→ `E0402`（MMA 支持面）→ `E0403`（整除与 pad 策略）——先形态契约后硬件支持面后对齐策略。本域错误码与 `primitives/memory-ops` 的 `E0404`–`E0407` 无同位置双命中（原语名唯一决定适用的参数契约集）。检查管线五段顺序（E01xx→E03xx→E04xx→E05xx→E06xx）不变；计算原语实参中的算术子表达式违规按 `numerics/value-semantics` 落 `E06xx`（子表达式位置与调用位置各自报告）。同一输入重复编译 MUST 产生逐条一致的拒绝清单（含 `mma=Auto` 与 `scope=Auto` 的确定性选择——同输入同 HAL 目标产出一致选择）。

错误码段位补充：`E04xx` 原语契约段内，`E0402`（显式 MMA 形状不支持）与 `E0403`（M/N/K 不整除且 Error 策略）为既有占用的语义正式化（承接 `primitives/memory-ops`"既有占用，语义不变"声明，语义来源为既有事实，非重定义）；`E0408`（计算原语调用结构与操作数/参数值域违规）为本 change 新增；`E0409`–`E0499` 继续保留给原语域后续扩展。

#### Scenario: E0408 与 E0402 双命中只报 E0408

- **WHEN** `tis.dot(A, B, C, mma=tis.MMA(16, 8, 32), pad=Bad)` 同时命中 pad 非法值（`E0408`）与该 MMA 形状不支持（`E0402`）
- **THEN** 该调用只报告 `E0408` 一条（段内序 E0408→E0402→E0403）

#### Scenario: 多条拒绝被收集并排序

- **WHEN** 同一模块第 3 行存在 `E0402`（mma 不支持）、第 12 行存在 `E0408`（reduce axis 越界）
- **THEN** 输出恰好两条拒绝，按第 3 行在前、第 12 行在后排序

#### Scenario: Auto 选择确定性重复编译一致

- **WHEN** 同一含 `tis.dot(A, B, C)`（`mma=Auto`）与 `tis.reduce(S, axis=1, op=Sum)`（`scope=Auto`）的模块以同一 HAL 目标连续编译两次
- **THEN** 两次的拒绝清单与 Auto 选择完全一致（确定性承诺）
