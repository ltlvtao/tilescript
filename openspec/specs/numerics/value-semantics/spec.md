# numerics/value-semantics Specification

## Purpose

定义数值语义的行为契约：算术/比较/逻辑运算的操作数与结果规则（家族区分、无隐式 dtype 提升、scope 一致、逐维 shape 广播）、一元负与数学具名常量（`inf`）、布尔值的条件语境专用地位、数值常量与标量值到 dtype 位置的绑定（封闭枚举）、`comptime[int]` 折叠与 `tis.cdiv` 编译期求值、`tis.cast`/常量绑定的数值转换效果与跨硬件位一致承诺，以及 `E0601`–`E0606` 报告契约（数值语义段位，跨段管线五段的末段）。职责分工：dtype 标量种类与 shape 组件、再绑定等价（`E0303`）归 `language/type-system`；运算符与字面量/具名常量的语法接受集归 `language/syntax-acceptance-set`；`tis.full`/`tis.cast` 的参数契约（`E0406`）与视图派生长度折叠归 `primitives/memory-ops`（计算原语域为后续 change）；`tis.cdiv` 形态契约（`E0505`）与语句结构归 `execution/pipeline-structure`；本 capability 承接以上四处让渡并定义数值域规则本身。跨硬件运行期数值容差（rtol 类数字）属里程碑验收（`veps`），不属本 capability；诊断 JSON 字段归 `diagnostics/*`。

## Requirements

### Requirement: 算术运算的操作数与结果规则

二元算术运算（`+` `-` `*` `/`）与增强算术赋值（`+=` `-=` `*=` `/=`）的结果规则按操作数家族裁决。**int 家族**（两侧操作数均为 `comptime[int]` 或 `int` 标量）：`+` `-` `*` 的结果为 int 家族标量——两侧均为 `comptime[int]` 时结果为 `comptime[int]`（值按「comptime 折叠」求值），否则为 `int`；`/` 对 int 家族操作数 MUST NOT 使用（以 `E0604` 拒绝，恢复建议使用 `tis.cdiv` 承载整除）。**dtype 家族**（操作数为 dtype 标量或 Tensor）：两侧操作数的 dtype MUST 相同（家族内跨 dtype 以 `E0602` 拒绝，恢复建议显式 `tis.cast`；本语言 MUST NOT 发生隐式 dtype 提升）；`/` 仅允许浮点 dtype（`f16`/`bf16`/`f32`/`f8e4m3`），整型 dtype（`i8`/`i32`）的 `/` 以 `E0604` 拒绝（Tensor 整除零承载，建议标量路径 `tis.cdiv`）；其余运算符（`+` `-` `*`）对同 dtype 的合法组合结果 dtype 不变。**混家族**（一侧 int 家族、一侧 dtype 家族）的任何算术运算以 `E0601` 拒绝——int 家族值（含 int 字面量与 `comptime[int]` 编译期常量）MUST NOT 作 dtype 家族算术的操作数：int 编译期常量到 dtype 位置的绑定经「数值常量与标量值到 dtype 位置的绑定」裁决且限于原语 dtype 标量参数位，不在算术操作数位豁免（恢复建议说明当前版本不支持混家族算术，数值转换须由宿主侧完成或经 Tensor 显式 `tis.cast`）。Tensor 操作数的 scope MUST 相同（跨 scope 组合以 `E0603` 拒绝，报告两侧 scope，恢复建议核对 scope 或经显式移动原语对齐）；dtype 标量无 scope 组件，标量与 Tensor 运算的结果取该 Tensor 的 scope；同 scope Tensor 组合的结果 scope 为该 scope。结果 shape 按逐维广播派生：两侧均为标量时结果为标量；标量与 Tensor 运算结果为该 Tensor 的 shape；Tensor 与 Tensor 运算右对齐逐维比较——两维相等则保留、一侧为 1 则扩展为另一侧长度、不相等且均非 1 以 `E0603` 拒绝（恢复建议核对操作数维度，或经切片与 `None` 广播调整形状至兼容）；结果维的编译期常量性：参与合流的维均为编译期常量维时结果为编译期常量维，否则为运行期派生维（派生表达式为该运算表达式）。增强算术赋值按等价展开裁决（`S *= scale` 同 `S = S * scale`）：运算违规按本 Requirement 报 `E06xx`，展开后再绑定与目标已绑定类型不等价由 `language/type-system`「类型不匹配拒绝」（`E0303`）裁决。算术运算的操作数与结果规则承接 `language/type-system`「表达式结果类型规则」的数值语义让渡；下标与切片的 shape 规则仍由该 capability 承载。

#### Scenario: Tensor 与同 dtype 标量运算广播

- **WHEN** `S` 为 `Tensor[f32, (BR, BC), Register]`（BR/BC 为 comptime 常量维）、`scale` 为 `f32` 标量，表达式为 `S * scale`
- **THEN** 运算合法，结果为 `Tensor[f32, (comptime BR, comptime BC), Register]`（dtype 不变，标量广播到全 shape）

#### Scenario: 逐维广播派生结果 shape

- **WHEN** `alpha` 为 `Tensor[f32, (BR,), Register]`，表达式为 `alpha[:, None] * st_O_acc`（后者为 `Tensor[f32, (BR, D), Register]`）
- **THEN** 运算合法，结果为 `Tensor[f32, (comptime BR, comptime D), Register]`（dim0 相等保留，dim1 的 1 扩展为 D）

#### Scenario: int 家族运算的种类推导

- **WHEN** `j` 为运行期 `int` 标量、`BC` 为 `comptime[int]`，表达式为 `j * BC` 与 `j + 1`
- **THEN** 两表达式结果均为运行期 `int` 标量（非全 comptime 不折叠）

#### Scenario: 家族内跨 dtype 运算被拒绝

- **WHEN** `Tensor[f16, …]` 与 `Tensor[f32, …]` 直接相加（未显式 cast）
- **THEN** 编译以 `E0602` 拒绝，报告两侧 dtype 与位置，恢复建议显式 `tis.cast`

#### Scenario: 混家族运算被拒绝

- **WHEN** `int` 标量与 `f32` 标量直接相乘
- **THEN** 编译以 `E0601` 拒绝，恢复建议说明混家族算术须在宿主侧转换或经 Tensor 显式 cast

#### Scenario: int 编译期常量作算术操作数被拒绝

- **WHEN** 表达式为 `S * 2`（`S` 为 `Tensor[f32, …]`，`2` 为 int 字面量）
- **THEN** 编译以 `E0601` 拒绝（int 常量绑定限原语 dtype 标量参数位，不在算术操作数位豁免），恢复建议使用 float 字面量（`S * 2.0`，按 `S` 的 dtype 定型）

#### Scenario: 跨 scope Tensor 运算被拒绝

- **WHEN** `Register` scope Tensor 与 `Shared` scope Tensor（同 dtype）直接相加
- **THEN** 编译以 `E0603` 拒绝，报告两侧 scope，恢复建议核对 scope 或经显式移动原语对齐

#### Scenario: int 家族除法被拒绝

- **WHEN** 表达式为 `seq_len / BC`（两侧 int 家族）
- **THEN** 编译以 `E0604` 拒绝，恢复建议使用 `tis.cdiv`

#### Scenario: 广播不兼容被拒绝

- **WHEN** `Tensor[f32, (64, 32), Register]` 与 `Tensor[f32, (128, 32), Register]` 相加（dim0 不等且均非 1）
- **THEN** 编译以 `E0603` 拒绝，报告两侧 shape，恢复建议核对维度

#### Scenario: 增强算术赋值合法

- **WHEN** `S` 已绑定为 `Tensor[f32, (BR, BC), Register]`，语句为 `S *= scale`（`scale: f32` 标量）
- **THEN** 等价展开 `S = S * scale` 运算合法（E0602/E0603 不命中）且再绑定等价，接受

### Requirement: 一元负与数学具名常量

一元负（`-`）的操作数与结果：int 家族标量结果留在 int 家族（`comptime[int]` 的值按「comptime 折叠」求值，`int` 结果为 `int`）；dtype 家族（标量或 Tensor）结果为同 dtype、同 shape。布尔值 MUST NOT 作一元负操作数（以 `E0605` 拒绝）。**数学具名常量 `inf`** 是语言预置名称（不经 `import` 引入；语法层以名称引用类别接受，其语言层地位由本 capability 定义——承接 `language/type-system`「表达式结果类型规则」对具名常量的让渡）：`inf` 的值语义为 IEEE 754 标准精度浮点 dtype（`f16`/`bf16`/`f32`）的正无穷；`-inf`（具名常量经一元负）为负无穷。`inf`/`-inf` 绑定到浮点 dtype 位置的规则见「数值常量与标量值到 dtype 位置的绑定」；出现在整型 dtype 位置或 int 家族位置以 `E0606` 拒绝（无穷无整数表示）；出现在 `f8e4m3` 位置以 `E0606` 拒绝（OCP FP8 E4M3 无无穷表示，与「dtype 转换数值效果」的饱和行为分工：cast 的运行期溢出饱和到 ±448 是行为承诺，常量绑定语境的 `inf` 是表示能力缺失，编译期即可判定）。`inf`/`-inf` 为待定型常量：无 dtype 语境的裸绑定（如 `x = inf` 的首次绑定）以 `E0606` 拒绝（与 float 字面量裸绑定同构，恢复建议经 `tis.full` 构造或与已定型值运算）。`nan` 不在本语言具名常量集内（保守封闭，扩展须显式 change）。

#### Scenario: -inf 绑定浮点 dtype 位置被接受

- **WHEN** `tis.full((BR,), -inf, f32, Register)`（value 为具名常量一元负，目标 dtype 为 `f32`）
- **THEN** 常量绑定合法，结果 Tensor 各元素为 `f32` 负无穷（IEEE 754 binary32 的 `-inf` 位模式）

#### Scenario: inf 用于整型 dtype 位置被拒绝

- **WHEN** `tis.full((4,), inf, i32, Register)`
- **THEN** 编译以 `E0606` 拒绝，报告注明无穷常量只能绑定浮点 dtype 位置

#### Scenario: inf 用于 f8e4m3 位置被拒绝

- **WHEN** `tis.full((2,), inf, f8e4m3, Register)`
- **THEN** 编译以 `E0606` 拒绝，报告注明 E4M3 无无穷表示（max finite ±448）

#### Scenario: inf 裸绑定被拒绝

- **WHEN** 设备代码出现 `x = inf`（首次绑定，无 dtype 语境）
- **THEN** 编译以 `E0606` 拒绝，报告注明具名常量需要明确的浮点 dtype 语境（与 float 字面量裸绑定同构）

#### Scenario: 一元负保持 dtype 与 shape

- **WHEN** `P` 为 `Tensor[f32, (BR, BC), Register]`，表达式为 `-P`
- **THEN** 结果为 `Tensor[f32, (comptime BR, comptime BC), Register]`（dtype 与 shape 不变）

### Requirement: 比较逻辑与条件语境的布尔封闭

比较运算（`<` `<=` `>` `>=` `==` `!=`，仅两操作数单比较）的操作数 MUST 为**同家族标量**：int 家族两标量（`comptime[int]` 与 `int` 可按单向兼容混用），或 dtype 家族同 dtype 两标量；违反（混家族、家族内跨 dtype、Tensor 参与比较、以待定型常量为操作数）均以 `E0605` 拒绝（Tensor 参与比较的恢复建议说明逐元素比较归计算原语域后续 change）。逻辑运算（`and` `or` `not`）的操作数 MUST 为布尔值，违反以 `E0605` 拒绝。比较与逻辑运算的结果均为**布尔值**。`if`/`elif` 条件的操作数 MUST 为布尔值（标量语境）；以非布尔值（int 家族、dtype 标量、Tensor 等）为条件以 `E0605` 拒绝——本语言不采用真值语义（truthiness）。布尔值的类型层地位为**条件语境专用值**：不进 `language/type-system` 封闭类型世界（不新增类型种类），合法使用位置为封闭集——`if`/`elif` 条件操作数、逻辑运算（含 `not`）操作数、`comptime` 参数默认值（`language/syntax-acceptance-set` `E0104` 既有，经 `True`/`False` 字面形式）；其余任何使用（赋值绑定、原语或调用实参、返回值、下标、算术或比较操作数、一元负操作数等）以 `E0605` 拒绝。`comptime` 参数默认值语境的布尔常量取整数值：`True` 为 1、`False` 为 0（编译期常量值绑定，承接 `E0104` 已接受的默认值形式）。`None` 不是布尔值（切片新增维专用，语义由 `language/type-system` 承载）。

#### Scenario: 合法比较条件被接受

- **WHEN** 设备代码出现 `if seq_len > 0:`（`seq_len: int` 与 comptime 字面量比较，结果布尔值作条件）
- **THEN** 比较与条件契约检查通过（int 家族两标量可混用按单向兼容）

#### Scenario: 非布尔条件被拒绝

- **WHEN** 设备代码出现 `if seq_len:`（int 标量直接作条件）
- **THEN** 编译以 `E0605` 拒绝，报告注明条件必须为布尔值（不支持真值语义），恢复建议显式比较（如 `seq_len > 0`）

#### Scenario: 布尔值赋值被拒绝

- **WHEN** 设备代码出现 `flag = a > b` 或以 `a > b` 为原语实参
- **THEN** 编译以 `E0605` 拒绝，报告注明布尔值仅限条件语境封闭集

#### Scenario: Tensor 参与比较被拒绝

- **WHEN** 设备代码出现 `if S == T:`（两侧为 Tensor）
- **THEN** 编译以 `E0605` 拒绝，恢复建议说明逐元素比较归计算原语域后续 change

### Requirement: 数值常量与标量值到 dtype 位置的绑定

dtype 标量位置的常量与标量值绑定规则（承接 `language/type-system`「标量类型与 shape 组件规则」的字面量绑定让渡与 `primitives/memory-ops` 的 `full` value 转换让渡）。本 Requirement 适用的位置集合为封闭枚举：① `tis.*` 原语的 dtype 标量参数位（当前为 `tis.full` 的 `value`；后续原语经各自契约登记）；② 算术运算操作数位——仅 float 字面量与具名常量 `inf`/`-inf`（int 家族常量不在其列，见「算术运算的操作数与结果规则」混家族条款）；③ 已绑定浮点 dtype 变量的再绑定赋值源（按已绑定 dtype 定型）。比较运算操作数位不接受待定型常量（比较限已定型标量，见「比较逻辑与条件语境的布尔封闭」）。各值类的绑定规则：**float 字面量**只在浮点 dtype 语境合法——绑定值为字面量十进制值经 roundTiesToEven 到目标 dtype 的最近表示（正确舍入）；无 dtype 语境的裸绑定（如 `x = 1.5` 的首次绑定）以 `E0606` 拒绝（无法确定 dtype，恢复建议经 `tis.full` 构造或与已定型值运算）。**int 家族编译期常量**（int 字面量与 `comptime[int]` 常量）MAY 绑定到位置集合①的 dtype 标量参数位（原语参数位）：绑定到整型 dtype（`i8`/`i32`）时值 MUST 在目标 dtype 补码值域内（越界以 `E0606` 拒绝，恢复建议核对值域），绑定到浮点 dtype 时按「dtype 转换数值效果」的整到浮规则舍入。**运行期 `int` 标量值 MUST NOT 绑定 dtype 标量位置**（运行期整型到 dtype 转换无原语路径，以 `E0606` 拒绝，恢复建议由宿主侧传入目标 dtype 标量或经 Tensor 显式 cast 路径）。**dtype 标量值**绑定 dtype 标量位置时 MUST 与目标 dtype 相同（家族内不同 dtype 以 `E0606` 拒绝——与算术的 `E0602` 同理，无隐式转换；恢复建议显式 `tis.cast` 路径）。**具名常量 `inf`/`-inf`** 绑定 IEEE 754 标准精度浮点 dtype（`f16`/`bf16`/`f32`）位置为该精度无穷；`f8e4m3`、整型与 int 家族位置以 `E0606` 拒绝（见「一元负与数学具名常量」）。`tis.full` 的 `value` 按本 Requirement 裁决（其标量形态合法性由 `primitives/memory-ops` `E0406` 先行裁决）。

#### Scenario: int 字面量绑定整型 dtype 位置被接受

- **WHEN** `tis.full((4,), 0, i32, Register)`
- **THEN** 常量绑定合法（0 在 i32 值域内），结果各元素为 i32 的 0

#### Scenario: int 字面量越界被拒绝

- **WHEN** `tis.full((4,), 300, i8, Register)`（300 超 i8 补码值域 −128–127）
- **THEN** 编译以 `E0606` 拒绝，报告目标 dtype 值域与实际值

#### Scenario: float 字面量正确舍入绑定

- **WHEN** `tis.full((2,), 3.14159, f16, Register)`（目标 f16）
- **THEN** 绑定值为 3.14159 经 roundTiesToEven 到 f16 的最近表示（正确舍入，非双精度先舍入再二次舍入）

#### Scenario: float 字面量裸绑定被拒绝

- **WHEN** 设备代码出现 `x = 1.5`（首次绑定，无 dtype 语境）
- **THEN** 编译以 `E0606` 拒绝，报告注明 float 字面量需要明确的浮点 dtype 语境

#### Scenario: float 字面量再绑定按已绑定 dtype 定型

- **WHEN** `x` 已绑定为 `f32` 标量，语句为 `x = 1.5`
- **THEN** float 字面量按 `f32` 定型（roundTiesToEven 到 f32 的最近表示），再绑定等价成立，接受

#### Scenario: 运行期 int 绑定 dtype 位置被拒绝

- **WHEN** `tis.full((4,), n, f32, Register)`（`n: int` 运行期标量）
- **THEN** 编译以 `E0606` 拒绝，恢复建议由宿主侧传入 `f32` 标量

#### Scenario: 不同 dtype 标量绑定被拒绝

- **WHEN** `tis.full((2,), v, f16, Register)`（`v` 为 `f32` 标量）
- **THEN** 编译以 `E0606` 拒绝，恢复建议显式转换后取值

### Requirement: comptime 折叠与 cdiv 编译期求值

`comptime[int]` 常量参与的一元负与二元算术（`+` `-` `*`）在编译期求值（求值语义为整数精确算术）；该折叠的适用域为**全 `comptime[int]` 操作数**的常量表达式。折叠服务于：int 家族种类推导（全 `comptime[int]` 操作数的运算结果为 `comptime[int]`）、常量子表达式求值（如 `BC * 2 - 1` 化简为单一常量值——`primitives/memory-ops` `E0405` 静态折叠支中常量部分的求值依据；**含运行期符号的派生长度折叠（如 `(bm+1)*BR − bm*BR` 化简为 `BR`）的判定机制仍由该 capability 的 `E0405` 支 (d) 承载，本 capability 不定义符号折叠**）与 `tis.cdiv` 折叠（承接 `execution/pipeline-structure` 的让渡）。`tis.cdiv(a, b)` 两实参均为 `comptime[int]` 时结果为 `comptime[int]` 且值 MUST 在编译期求值为 `ceil(a/b)`（数学向上取整；`b` 为 0 已由 `E0505` 拒绝，本 capability 不重复报告）。`tis.block_idx` 结果为运行期 `int`（`execution/pipeline-structure` 已裁），不参与折叠。折叠结果 MUST 确定：同一表达式重复折叠产生相同值。`comptime` 参数默认值的常量表达式形式（如 `D: comptime[int] = 64 * 1024`）仍由 `language/syntax-acceptance-set` `E0104` 拒绝（该层仅接受单个字面量），本 capability 的折叠不改变该语法层约束。

#### Scenario: cdiv 全 comptime 折叠

- **WHEN** `tis.cdiv(128, 64)`（两实参均为 comptime 字面量）
- **THEN** 结果为 `comptime[int]` 且编译期值为 2（`ceil(128/64)`），可用于 comptime 位置

#### Scenario: cdiv 非全 comptime 不折叠

- **WHEN** `tis.cdiv(seq_len, BC)`（`seq_len: int`）
- **THEN** 结果为运行期 `int`（不折叠；种类推导由 `execution/pipeline-structure` 承载）

#### Scenario: comptime 算术折叠值确定

- **WHEN** 表达式含 `BC * 2 - 1`（`BC: comptime[int] = 64`）
- **THEN** 折叠值为 127，结果种类为 `comptime[int]`

### Requirement: dtype 转换数值效果与跨硬件一致

`tis.cast`（承接 `primitives/memory-ops` 的数值效果让渡；其参数契约与 `E0406` 不变）与常量绑定（「数值常量与标量值到 dtype 位置的绑定」）的数值转换效果按目标 dtype 家族规则确定：**浮到浮**（`f32`/`f16`/`bf16`/`f8e4m3` 之间）：roundTiesToEven 最近舍入；结果超出目标 dtype 值域时，IEEE 754 标准精度（`f32`/`f16`/`bf16`）饱和到 ±inf，`f8e4m3` 饱和到 ±448（最大有限值，OCP FP8 Specification E4M3 无 inf）；NaN 输入产生 NaN 输出（`f8e4m3` 按 E4M3 NaN 编码）。**浮到整**（到 `i8`/`i32`）：roundTowardZero 向零舍入；越界饱和到目标整型补码边界（MIN/MAX）；NaN 输入产生 0。**整到浮**：roundTiesToEven（整数值可精确表示时结果精确）。**整到整**：窄化（`i32` 到 `i8`）为补码低 8 位截断（环绕语义，跨硬件唯一确定）；宽化（`i8` 到 `i32`）为符号扩展。同 dtype 转换结果位模式不变。上述效果为语言级行为承诺：同一输入值（含编译期常量绑定值）在任何 HAL 后端 MUST 产生相同的结果位模式（确定性；实现机制归后端 design）。运行期值的转换失败（越界、NaN）MUST NOT 产生编译期拒绝——按上述饱和/NaN 规则确定结果（行为承诺而非拒绝路径）；编译期可判定的位置违规（具名常量到整型位置、字面量越界）由 `E0606` 承载。精度损失的诊断呈现（建议域字段）归 `diagnostics/*`，本 capability 不定义。

#### Scenario: f32 到 f16 最近舍入含 tie 取偶

- **WHEN** `tis.cast(x, f16)` 的输入元素值为 f32 的 1.00048828125（恰为 f16 相邻两表示 1.0 与 1.0009765625 的中点）
- **THEN** 输出为 f16 的 1.0（roundTiesToEven 等距取偶；跨后端位一致）

#### Scenario: f32 到 f16 溢出饱和到无穷

- **WHEN** 输入元素值为 f32 的 1e5（超出 f16 max finite 65504）
- **THEN** `tis.cast(x, f16)` 输出为 f16 的 +inf

#### Scenario: f32 到 f8e4m3 溢出饱和到最大有限值

- **WHEN** 输入元素值为 f32 的 1e4（超出 E4M3 max finite 448）
- **THEN** `tis.cast(x, f8e4m3)` 输出饱和到 +448（E4M3 无 inf）

#### Scenario: 浮到整向零截断与饱和

- **WHEN** `tis.cast(x, i32)` 的输入元素值分别为 f32 的 −2.7、1e10、NaN
- **THEN** 输出分别为 −2（向零截断）、2147483647（i32 MAX 饱和）、0

#### Scenario: i32 到 i8 环绕

- **WHEN** `tis.cast(x, i8)` 的输入元素值为 i32 的 300
- **THEN** 输出为 i8 的 44（300 的补码低 8 位，环绕语义）

#### Scenario: 跨后端转换位一致

- **WHEN** 同一源码的同一 cast 调用分别以 NVIDIA 与 Ascend HAL 目标编译并运行于相同输入值
- **THEN** 两后端的结果位模式逐元素相同（语言级承诺）

### Requirement: 报告契约与 E06xx 段位

本 capability 产生的每条拒绝（`E06xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含违规形态、两侧家族/dtype/shape 或位置类别）、恢复建议。同一模块存在多条 `E06xx` 拒绝时 MUST 收集输出全部并按源码位置升序排列。检查管线顺序为语法（`E01xx`）→ 类型（`E03xx`）→ 原语契约（`E04xx`）→ 执行结构（`E05xx`）→ 数值语义（`E06xx`）；同一源码位置跨段命中时 MUST 只报告最早段一条（算术表达式运算违规落 `E06xx`，其赋值再绑定不等价落 `E0303`——同位置双命中时只报 `E0303`）。同一源码位置命中多个本 capability 规则时 MUST 只报告一个错误码，按段内顺序取首个：`E0601`（混家族算术）→ `E0602`（家族内跨 dtype）→ `E0603`（shape 广播不兼容与 Tensor scope 不一致）→ `E0604`（int 家族或整型 dtype 除法）→ `E0605`（布尔封闭与比较条件规则）→ `E0606`（常量与标量值绑定位置）。同一表达式命中多个操作数的同类违规时 MUST 合并为一条报告并列出全部违规操作数。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E06xx` 为数值语义段——`E0601`–`E0606` 为本 change 新增（`E06xx` 段此前空白，无既有事实冲突）；`E0607`–`E0699` 保留给数值语义域后续扩展。

#### Scenario: 多条 E06xx 拒绝被收集并排序

- **WHEN** 同一模块第 5 行存在 `E0602`（跨 dtype 相加）、第 18 行存在 `E0606`（float 字面量裸绑定）
- **THEN** 输出恰好两条拒绝，按第 5 行在前、第 18 行在后排序

#### Scenario: 同位置类型与数值双命中只报类型错误

- **WHEN** `x` 已绑定为 `int` 标量，语句 `x = f32_tensor * f16_tensor` 同时命中 `E0602`（跨 dtype）与再绑定不等价
- **THEN** 该位置只报告管线更早的 `E0303` 一条

#### Scenario: 同表达式多个操作数违规合并报告

- **WHEN** 三元连加 `a + b + c` 中 `a` 与 `c` 均为混家族违规操作数
- **THEN** 只报告一条 `E0601`，列出全部违规操作数位置

#### Scenario: 重复编译数值语义拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E06xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）
