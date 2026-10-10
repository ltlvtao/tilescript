# primitives/memory-ops Specification（delta）

## MODIFIED Requirements

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
