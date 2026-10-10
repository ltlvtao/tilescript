# primitives/memory-ops Specification

## Purpose
定义 TileScript 核心存取原语的行为契约：八个原语（`tis.load`/`tis.store`/`tis.barrier`/`tis.make_tensor`/`tis.alloc_shared`/`tis.zeros`/`tis.full`/`tis.cast`）的参数集与封闭性（`E0406`）、作用域转移矩阵格子的承载映射（`E0404`，矩阵非法格仍归 `language/type-system` 的 `E0301`）、load/store 的逐维长度相容判定（`E0405`，专用判定，区别于该 capability 的类型等价）、加载模式与完成语境（`E0407`）、分配与视图构造（`swizzled` 为分配物理属性，不进 Tensor 类型组件）、dtype 显式转换与屏障汇合可见性、`E04xx` 报告契约与段内 tiebreak。与 `language/type-system` 的分工：通用绑定等价（`E0303`）仍由该 capability 裁决（其豁免边界已扩展为 `tis.*` 原语调用实参整体），本 capability 承接其两处让渡（移动原语实参其余维度、`tis.*` 调用结果类型）；Async 完成保证与 WarpGroup 汇合语义让渡执行结构 capability，数值转换规则让渡数值语义 capability。

## Requirements

### Requirement: 原语集合与调用结构

本 capability 定义八个核心存取原语，其调用结构（参数集、顺序、可省略参数与默认值）MUST 为：`tis.load(src, dst, mode=Sync)`；`tis.store(src, dst)`；`tis.barrier(scope=Block)`；`tis.make_tensor(ptr, shape)`；`tis.alloc_shared(shape, dtype, layout=RowMajor)`；`tis.zeros(shape, dtype, scope=Register)`；`tis.full(shape, value, dtype, scope=Register)`；`tis.cast(x, dtype)`。可省略参数 MAY 以关键字形式显式传递（如 `mode=Async`、`layout=tis.Layout.swizzled(xor=0b11100)`），MAY 按声明顺序以位置实参传递（如 `tis.zeros(shape, dtype, Register)` 解析等价于 `scope=Register`）；使用参数集之外的任何关键字实参、位置实参数量超出该原语参数集总长度（必选加可省略）、或位置实参数量少于必选参数个数（如 `tis.load(src)` 缺 `dst`）时，编译器 MUST 以 `E0406` 拒绝。默认值（`Sync`、`Block`、`RowMajor`、`Register`）固定且写进文档。

#### Scenario: 全默认参数的 load 调用被接受

- **WHEN** 设备代码调用 `tis.load(src, dst)`（省略 `mode`）
- **THEN** 调用结构检查通过（`mode` 取默认值 `Sync`）

#### Scenario: 未知关键字实参被拒绝

- **WHEN** 设备代码调用 `tis.load(src, dst, group=g)`（`group` 不在本 capability 定义的参数集内）
- **THEN** 编译以 `E0406` 拒绝，报告注明 `load` 的合法参数集为 `src`/`dst`/`mode`

#### Scenario: 可省略参数按位置传递被接受

- **WHEN** 设备代码调用 `tis.zeros((BR, BC), f32, Register)`（`scope` 以第三位置实参按声明顺序传递）
- **THEN** 调用结构检查通过（解析等价于 `scope=Register` 关键字形式；scope 值域检查由「分配与视图构造原语」承载）

#### Scenario: 位置实参数量超出被拒绝

- **WHEN** 设备代码调用 `tis.cast(x, f16, extra)`（三位置实参）
- **THEN** 编译以 `E0406` 拒绝，报告注明 `cast` 只接受两个位置实参

#### Scenario: 必选位置实参缺失被拒绝

- **WHEN** 设备代码调用 `tis.load(src)`（位置实参数量少于 `load` 的必选参数个数，缺失 `dst`）
- **THEN** 编译以 `E0406` 拒绝，报告注明 `load` 的必选参数为 `src` 与 `dst`（与 `primitives/compute-ops`「缺失必选位置实参」拒绝路径对称）

### Requirement: 转移格承载映射

`tis.load` MUST 只承载 `language/type-system`「作用域转移矩阵」中操作类别为 `load` 的格子（源/目标 scope 组合为 Global→Shared、Global→Register、Shared→Register）；`tis.store` MUST 只承载操作类别为 `store` 的格子（Shared→Global、Register→Global、Register→Shared）。调用实参的 scope 组合落在某格、但所用原语与该格操作类别不匹配时（如以 `tis.store` 承载 Global→Shared），编译器 MUST 以 `E0404` 拒绝，恢复建议指出该组合的操作类别与应使用的原语。scope 组合本身落在矩阵非法格（如 Global→Global）时仍由 `language/type-system` 以 `E0301` 裁决，本 capability MUST NOT 改变该行为。操作类别为 `copy`（Shared→Shared）与 `move`（Register→Register）的格子在本 change 后仍无承载原语（显式声明），以 `tis.load` 或 `tis.store` 承载这两格按 `E0404` 拒绝；同 scope 数据路径由普通赋值承载（`language/type-system` 以类型等价裁决，等价即合法），这两格的专用显式原语延期后续 change。

#### Scenario: load 承载 Global 到 Shared 被接受

- **WHEN** `tis.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)` 的 src 为 Global scope Tensor、dst 为 Shared scope Tensor
- **THEN** 组合落在矩阵 `Global→Shared`（load 格）且原语匹配，检查通过

#### Scenario: store 承载 Register 到 Global 被接受

- **WHEN** `tis.store(x, O[bm*BR:(bm+1)*BR, :])` 的 src 为 Register scope Tensor、dst 为 Global scope Tensor
- **THEN** 组合落在矩阵 `Register→Global`（store 格）且原语匹配，检查通过

#### Scenario: 原语与格子类别不匹配被拒绝

- **WHEN** 以 `tis.store(src, dst)` 承载 Global→Shared 组合（该格操作类别为 load）
- **THEN** 编译以 `E0404` 拒绝，恢复建议注明该组合应使用 `tis.load`

#### Scenario: copy 格无承载原语被拒绝

- **WHEN** 以 `tis.load(src, dst)` 或 `tis.store(src, dst)` 承载 Shared→Shared 组合（copy 格）
- **THEN** 编译以 `E0404` 拒绝，报告注明 copy 类别当前无承载原语

#### Scenario: 矩阵非法格仍由类型层裁决

- **WHEN** 以 `tis.load` 或 `tis.store` 承载 Global→Global 组合（矩阵非法格）
- **THEN** 编译以 `E0301` 拒绝（`language/type-system` 裁决，本 capability 不重复报告 `E04xx`）

### Requirement: load 与 store 的数据维度契约

`tis.load` 与 `tis.store` 的 `src` 与 `dst` MUST 均为 `Tensor`（其类型按 `language/type-system`「表达式结果类型规则」确定，含切片/下标结果），基对象为 `Pointer`、标量或状态类类型时编译器 MUST 以 `E0406` 拒绝。`src` 与 `dst` 的 dtype MUST 相同；shape MUST 逐维**长度相容**。长度相容是本 capability 专用判定，与 `language/type-system`「类型等价规则」的维度等价是两套不同判定：仅用于移动原语实参，不产生类型层等价结论。逐维规则：(a) 两侧同为编译期常量且值相等；(b) 两侧为同一运行期符号；(c) 两侧为派生表达式结构等价的运行期派生维度；(d) 一侧为运行期派生维度、另一侧为编译期常量，且该派生维的长度可静态折叠为编译期常量并与对侧值相等（如切片 `bm*BR:(bm+1)*BR` 的长度折叠为 `BR`）。违反 dtype 或长度相容约束时编译器 MUST 以 `E0405` 拒绝，报告 MUST 包含两侧类型与不相容维度。`tis.load` 与 `tis.store` 不产生值（合法使用形态仅为表达式语句，见「加载模式与完成语义」）。

#### Scenario: dtype 一致的 load 通过维度检查

- **WHEN** `tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)` 两侧均为 `f16`，切片维 `j*BC:(j+1)*BC` 的长度静态折叠为 `BC`、与 buffer 维 `comptime BC` 长度相容（`buf.K` 的类型由执行结构 capability 契约承载，非本 change 原语）
- **THEN** 数据维度契约检查通过（逐维长度相容的折叠支 (d)）

#### Scenario: dtype 不一致被拒绝

- **WHEN** `tis.load` 的 src 为 `Tensor[f16, …, Global]`、dst 为 `Tensor[f32, …, Shared]`
- **THEN** 编译以 `E0405` 拒绝，报告两侧类型并注明 dtype 不同，恢复建议使用 `tis.cast` 显式转换后再移动

#### Scenario: shape 长度不相容被拒绝

- **WHEN** `tis.store` 的 src shape 为 `(BR, D)`（编译期常量维）、dst shape 为 `(运行期 seq_len, D)`
- **THEN** 编译以 `E0405` 拒绝，报告注明不相容维度（编译期常量与运行期符号无相容支）

#### Scenario: Pointer 实参被拒绝

- **WHEN** `tis.load(Q_ptr, Q_s)` 直接以 `Pointer[f16, Global]` 为 src
- **THEN** 编译以 `E0406` 拒绝，恢复建议先以 `tis.make_tensor` 建立 Tensor 视图

### Requirement: 加载模式与完成语义

`tis.load` 的 `mode` 参数值域 MUST 为 `Sync` 或 `Async`（默认 `Sync`），其他值以 `E0406` 拒绝。`Sync` 模式：拷贝在该语句处完成，同一线程的后续语句读取 `dst` 得到已拷贝数据；其他线程的可见性由 `tis.barrier` 承载（见「屏障语义」）。`Async` 模式：调用发起非阻塞拷贝，MUST 只出现在 Pipeline `produce`/`consume` 嵌套函数体内，出现在其他位置（kernel 顶层函数体等）时编译器 MUST 以 `E0407` 拒绝；其完成保证（发起后何时可读 `dst`）由执行结构 capability 的 Pipeline 契约承载，本 capability 不定义。`tis.load` 与 `tis.store` MUST NOT 在产生值的位置使用（赋值右侧、实参、`return` 值），违反时以 `E0407` 拒绝。

#### Scenario: Sync 完成语义

- **WHEN** 设备代码依次执行 `tis.load(Q_s_slice, Q_s, mode=Sync)` 与读取 `Q_s` 的后续语句
- **THEN** 后续语句读取到已拷贝数据（同线程；跨线程可见性需 barrier）

#### Scenario: Async 出现在 kernel 顶层被拒绝

- **WHEN** kernel 顶层函数体（非 Pipeline 嵌套函数）调用 `tis.load(x, y, mode=Async)`
- **THEN** 编译以 `E0407` 拒绝，恢复建议在 Pipeline `produce`/`consume` 内使用 Async 或改用 Sync

#### Scenario: Async 在 produce 内被接受

- **WHEN** Pipeline `produce` 嵌套函数体内调用 `tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)`
- **THEN** 语境检查通过（完成保证由 Pipeline 契约承载）

#### Scenario: 无值原语用于赋值右侧被拒绝

- **WHEN** 设备代码出现 `v = tis.load(src, dst)` 或 `x = tis.barrier()`
- **THEN** 编译以 `E0407` 拒绝，报告注明该原语不产生值、合法形态为表达式语句

### Requirement: 分配与视图构造原语

`tis.make_tensor(ptr, shape)`：`ptr` MUST 为 `Pointer[d, Global]`（scope 非 Global 以 `E0406` 拒绝）；`shape` 为元组，每维 MUST 为 `int` 标量表达式或 `comptime[int]` 常量（非整数组件以 `E0406` 拒绝）；返回 `Tensor[d, shape, Global]`（运行期维成为运行期维度）；语义为零拷贝视图，MUST NOT 分配或移动数据。`tis.alloc_shared(shape, dtype, layout=RowMajor)`：`dtype` MUST 属于 `language/type-system` 的六种 dtype 封闭集合；`shape` 约束同上；`layout` 值域为 `RowMajor`（默认，与省略等价）或 `tis.Layout.swizzled(xor=c)`（`c` 为 `comptime[int]` 且非负，违反以 `E0406` 拒绝；掩码与行宽的可满足性属资源诊断 `memory` 段（建议域），不做编译期拒绝）；返回 `Tensor[dtype, shape, Shared]`。`swizzled` 是分配的物理布局属性而非 Tensor 类型组件：记录于资源诊断 `memory` 段；按 `language/type-system` 的封闭声明，类型层 layout 组件唯一合法值保持 `RowMajor`——`swizzled` 分配的结果 Tensor 的类型组件为 `RowMajor`，该属性 MUST NOT 参与类型等价判定（type-system 无需为 `swizzled` 开辟类型组件值域）。`tis.zeros(shape, dtype, scope=Register)` 与 `tis.full(shape, value, dtype, scope=Register)`：`scope` 值域 MUST 为 `Register` 或 `Shared`（`Global` 以 `E0406` 拒绝——Global 张量必须经 `make_tensor` 从指针建立）；`full` 的 `value` MUST 为标量表达式——三类标量种类的值、数学具名常量或其经一元负得到的表达式（如 `-inf`）——Tensor 等非标量以 `E0406` 拒绝，其到 `dtype` 的数值转换规则由数值语义 capability 承载；分别返回全零与全 `value` 的 `Tensor[dtype, shape, scope]`。

#### Scenario: make_tensor 建立动态 Global 视图

- **WHEN** `Q = tis.make_tensor(Q_ptr, (seq_len, D))`，`Q_ptr: Pointer[f16, Global]`
- **THEN** `Q` 的类型为 `Tensor[f16, (运行期 seq_len, comptime D), Global]`，无数据移动

#### Scenario: make_tensor 用于 Shared Pointer 被拒绝

- **WHEN** 以 `Pointer[f16, Shared]` 为 `ptr` 调用 `tis.make_tensor`
- **THEN** 编译以 `E0406` 拒绝，报告注明 `make_tensor` 只接受 Global scope 的 Pointer

#### Scenario: swizzled 布局分配被接受

- **WHEN** `tis.alloc_shared((BR, D), f16, layout=tis.Layout.swizzled(xor=0b11100))`
- **THEN** 返回 `Tensor[f16, (comptime BR, comptime D), Shared]`（类型 layout 组件为默认 `RowMajor`），分配物理布局 `swizzled(xor=28)` 记录于资源诊断 `memory` 段

#### Scenario: 负数 xor 掩码被拒绝

- **WHEN** `tis.alloc_shared((BR, D), f16, layout=tis.Layout.swizzled(xor=-1))`
- **THEN** 编译以 `E0406` 拒绝，报告注明 `xor` 必须为非负 `comptime[int]`

#### Scenario: zeros 到 Global 被拒绝

- **WHEN** `tis.zeros((BR, BC), f32, Global)`
- **THEN** 编译以 `E0406` 拒绝，恢复建议 Global 张量经 `make_tensor` 从指针建立

#### Scenario: full 的具名常量值被接受

- **WHEN** `tis.full((BR,), -inf, f32, Register)`（`value` 为数学具名常量 `inf` 的一元负）
- **THEN** 返回 `Tensor[f32, (comptime BR,), Register]`（数值转换规则由数值语义 capability 承载）

### Requirement: dtype 显式转换

`tis.cast(x, dtype)`：`x` MUST 为 `Tensor`（标量、`Pointer` 或状态类类型以 `E0406` 拒绝）；`dtype` MUST 属于六种 dtype 封闭集合（其他值以 `E0406` 拒绝）；返回 `Tensor[dtype, shape(x), scope(x)]`——dtype 替换为目标值，shape 与 scope 不变。dtype 转换 MUST NOT 隐式发生：普通绑定两侧 dtype 不同的拒绝由 `language/type-system`「类型不匹配拒绝」（`E0303`）裁决，本 capability 只提供显式转换路径。转换的数值效果（舍入、饱和、精度损失及其诊断）由数值语义 capability 承载，本 capability 不定义。算术结果表达式作为 `x` 传入时，其结果类型同样由数值语义 capability 承载（该 capability 定义前，此路径仅可做形态级核对）。

#### Scenario: cast 返回同 shape 新 dtype

- **WHEN** `tis.cast(P, f16)`，`P` 为 `Tensor[f32, (BR, BC), Register]`
- **THEN** 结果类型为 `Tensor[f16, (BR, BC), Register]`（dtype 换、shape 与 scope 不变）

#### Scenario: cast 标量被拒绝

- **WHEN** 以 `int` 标量为 `x` 调用 `tis.cast(x, f32)`
- **THEN** 编译以 `E0406` 拒绝，报告注明 `cast` 只接受 Tensor

#### Scenario: 隐式转换不发生

- **WHEN** 将 `Tensor[f16, …]` 值赋给 `Tensor[f32, …]` 目标（未经 `tis.cast`）
- **THEN** 按 `language/type-system` 以 `E0303` 拒绝（本 capability 不引入隐式转换路径）

### Requirement: 屏障语义

`tis.barrier(scope=Block)`：`scope` 值域 MUST 为 `Block`（默认）或 `WarpGroup`（其他值以 `E0406` 拒绝）。`Block` 模式：同 Block 的全部线程在该语句处汇合，汇合前任一线程对 Shared 内存 的写在汇合后对同 Block 全部线程可见。`WarpGroup` 模式的上下文约束（合法出现位置、与 `warp_group` 的关系）与汇合/可见性语义均由执行结构 capability 承载，本 capability 只定义值域与 `Block` 模式语义。`tis.barrier` 不产生值（值使用以 `E0407` 拒绝，见「加载模式与完成语义」）。

#### Scenario: barrier 后跨线程可见

- **WHEN** 设备代码依次执行线程 A 的 `tis.load(Q[切片], Q_s, mode=Sync)`、`tis.barrier()`、线程 B 读取 `Q_s`
- **THEN** 线程 B 在汇合后读取到线程 A 拷贝的数据（同 Block）

#### Scenario: 未知 scope 值被拒绝

- **WHEN** `tis.barrier(scope=Grid)`（值域外）
- **THEN** 编译以 `E0406` 拒绝，报告注明值域为 Block/WarpGroup

#### Scenario: barrier 用于赋值右侧被拒绝

- **WHEN** 设备代码出现 `x = tis.barrier()`
- **THEN** 编译以 `E0407` 拒绝（不产生值的原语）

### Requirement: 报告契约与 E04xx 段位

本 capability 产生的每条拒绝（`E04xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含两侧类型、格子组合或违规参数）、恢复建议。同一模块存在多条 `E04xx` 拒绝时 MUST 收集输出全部并按源码位置升序排列。检查管线顺序为语法（`E01xx`）→ 类型（`E03xx`）→ 原语契约（`E04xx`）；类型系统段存在任一拒绝时原语契约段 MUST NOT 执行（段间短路——与 `language/type-system`「语法段存在任一拒绝时类型系统段 MUST NOT 执行」同型；类型系统段零拒绝时原语契约段执行）；同一源码位置跨段命中时 MUST 只报告最早段一条。原语调用实参不适用 `language/type-system` 的绑定等价规则（`E0303`，经本 change 对该 capability 的 MODIFIED 裁决），原语实参的违规一律落 `E04xx`。同一源码位置命中多个本 capability规则时 MUST 只报告一个错误码，按段内顺序取首个：`E0404`（转移格类别）→ `E0405`（数据维度）→ `E0406`（参数值域与形态）→ `E0407`（语境）。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E04xx` 为原语参数契约段——`E0402`/`E0403` 为计算原语（MMA/pad）既有占用，语义不变；`E0404` 转移格类别不匹配、`E0405` 数据维度不一致、`E0406` 参数值域与形态违规、`E0407` 语境违规为本 change 新增；`E0408` 为计算原语调用结构与操作数/参数值域违规（由 `primitives/compute-ops` 定义——经该 change 对本条的 MODIFIED 起用）；`E0409`–`E0499` 保留给原语域后续扩展。

#### Scenario: 多条 E04xx 拒绝被收集并排序

- **WHEN** 同一模块第 12 行存在 `E0404`、第 30 行存在 `E0405`
- **THEN** 输出恰好两条拒绝，按第 12 行在前、第 30 行在后排序

#### Scenario: 同位置类型与原语双命中只报类型错误

- **WHEN** 同一 `tis.load` 调用同时命中 `E0303`（类型层）与 `E0405`（维度层）
- **THEN** 该位置只报告管线更早的 `E0303` 一条（类型层裁决优先）

#### Scenario: 同位置多个原语规则命中只报段内首个

- **WHEN** 同一 `tis.load` 调用同时命中 `E0404`（格类别）与 `E0405`（dtype 不一致）
- **THEN** 该位置只报告段内顺序更早的 `E0404` 一条

#### Scenario: 重复编译原语拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E04xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）

#### Scenario: 类型段存在拒绝时原语段不执行

- **WHEN** 同一模块既存在类型系统段拒绝（如第 5 行 `E0303`）、又存在原语契约段命中（如第 12 行本应触发的 `E0405`）
- **THEN** 输出只含类型系统段的拒绝（原语契约段未执行，第 12 行的 `E04xx` 不出现——段间短路；与 `language/type-system` Scenario「语法段存在拒绝时类型段不执行」同型）
