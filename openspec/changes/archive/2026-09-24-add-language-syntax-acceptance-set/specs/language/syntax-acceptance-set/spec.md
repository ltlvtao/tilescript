# language/syntax-acceptance-set 规格增量

## ADDED Requirements

### Requirement: 源码载体为 Python 3.10 语法基线

TileScript 模块源文件的规范扩展名为 `.tis`（命名约定，语法接受集不因扩展名拒绝源文件）。模块源文本 MUST 是以 Python 3.10 语法可解析的合法源文本（等价于 `ast.parse(..., feature_version=(3, 10))` 的接受范围）。源文本不可解析、或使用了高于 3.10 的语法（如 3.12 的类型参数语句）时，编译器 MUST 拒绝该模块并以错误码 `E0101` 报告；报告 MUST 包含行列位置与原始解析错误描述，并附恢复建议：改写为 Python 3.10 兼容语法。

TileScript MUST NOT 提供模块包装语法：一个模块恰好是一个 Python 源文件；`module <name>:` 形式（见 `veps/design.md` §7 示例）是设计文档的文档组织伪代码，不是语言构造，按 `E0101` 拒绝。

#### Scenario: 非 Python 语法被拒绝并定位

- **WHEN** 模块源文本包含 Python 3.10 无法解析的语法（如 `def f( {`）
- **THEN** 编译以 `E0101` 拒绝该模块，报告包含行列位置、原始解析错误描述与恢复建议

#### Scenario: module 伪代码语法被拒绝

- **WHEN** 模块源文本首行是 `module flash_attention:`
- **THEN** 编译以 `E0101` 拒绝（该行不是合法 Python 语句），恢复建议说明模块即源文件本身

#### Scenario: 高于基线的 Python 语法被拒绝

- **WHEN** 模块使用了 Python 3.11 及以上版本引入的语法
- **THEN** 编译以 `E0101` 拒绝，恢复建议指出本语言的语法基线为 Python 3.10

#### Scenario: 合法 Python 3.10 源文本通过载体检查

- **WHEN** 模块源文本可按 Python 3.10 语法解析
- **THEN** 不产生 `E0101`，模块进入后续结构检查

### Requirement: 模块顶层结构白名单

模块顶层 MUST 只接受以下结构：`import` 语句、`@tis.state` 装饰的类定义、被设备入口装饰器（见「设备入口装饰器识别」）装饰的函数定义。顶层出现白名单之外的语句时，编译器 MUST 以 `E0102` 拒绝；报告包含被拒绝结构的类别与行列位置，恢复建议说明顶层仅接受的三类结构。

#### Scenario: 顶层裸语句被拒绝

- **WHEN** 模块顶层出现赋值语句或表达式调用语句
- **THEN** 编译以 `E0102` 拒绝，报告定位该语句、注明类别并附恢复建议

#### Scenario: 顶层 kernel 定义被接受

- **WHEN** 模块顶层存在 `@tis.kernel` 装饰的函数定义
- **THEN** 顶层结构检查通过（函数体内部由「设备代码语句接受集」检查）

### Requirement: 设备入口装饰器识别

模块中出现的装饰器 MUST 属于已识别集合，且目标类别匹配：`tis.state`（仅模块顶层类）、`tis.kernel`、`tis.persistent_kernel`、`tis.fused_kernel`（仅模块顶层函数）、Pipeline 实例的 `produce`/`consume` 属性装饰器（形如 `@<pipeline 实例名>.produce`、`@<pipeline 实例名>.consume`，仅设备代码内嵌套函数；按属性名 `produce`/`consume` 识别，实例名任意）。出现集合外装饰器、或装饰器与目标类别不匹配时，编译器 MUST 以 `E0103` 拒绝；报告包含装饰器名称、目标类别与行列位置，恢复建议列出已识别装饰器集合及其适用目标。

#### Scenario: 未知装饰器被拒绝

- **WHEN** 函数使用 `@tis.autotune` 等集合外装饰器
- **THEN** 编译以 `E0103` 拒绝，报告包含装饰器名称、位置与已识别装饰器清单

#### Scenario: pipe.produce 嵌套函数被接受

- **WHEN** 设备代码内定义 `@pipe.produce` 装饰的嵌套函数（`pipe` 为任意 Pipeline 实例名）
- **THEN** 装饰器识别通过（函数体由「设备代码语句接受集」检查）

#### Scenario: 入口装饰器用于嵌套函数被拒绝

- **WHEN** 设备代码内出现 `@tis.kernel` 装饰的嵌套函数
- **THEN** 编译以 `E0103` 拒绝（目标类别不匹配：入口装饰器仅用于模块顶层函数）

### Requirement: kernel 函数签名规则

设备入口函数（`tis.kernel`、`tis.persistent_kernel`、`tis.fused_kernel` 装饰的函数）的每个参数 MUST 带类型注解，注解形式 MUST 属于：`Pointer[...]`、`Tensor[...]`、dtype 名（`f16`、`bf16`、`f32`、`f8e4m3`、`i8`、`i32`）、宿主标量注解 `int`、编译期常量注解 `comptime[int]`。参数缺少类型注解或注解形式不在上述集合时，编译器 MUST 以 `E0104` 拒绝；报告定位该参数，恢复建议给出合法注解形式清单。`comptime` 参数 MAY 携带默认值，默认值 MUST 是单个 int 字面量（十进制/二进制/八进制/十六进制形式）或布尔常量（`True`/`False`）；默认值形式不满足该句法定义时 MUST 以 `E0104` 拒绝，恢复建议说明默认值仅接受字面量形式（常量表达式求值由后续 capability 承载）。非 `comptime` 参数携带默认值时 MUST 以 `E0104` 拒绝，恢复建议说明默认值仅允许用于 `comptime` 参数。

#### Scenario: 参数缺类型注解被拒绝

- **WHEN** 设备入口函数存在无注解参数（如 `def k(src, n: int)` 中的 `src`）
- **THEN** 编译以 `E0104` 拒绝，报告定位该参数并附合法注解形式清单

#### Scenario: comptime 默认值被接受

- **WHEN** 签名包含 `D: comptime[int] = 64`
- **THEN** 签名规则检查通过

#### Scenario: 运行期参数带默认值被拒绝

- **WHEN** 签名包含 `scale: f32 = 1.0`
- **THEN** 编译以 `E0104` 拒绝，报告定位该参数并说明默认值仅限 `comptime`

#### Scenario: comptime 默认值非常量字面量被拒绝

- **WHEN** 签名包含 `D: comptime[int] = 64 * 1024`（常量表达式而非单个字面量）
- **THEN** 编译以 `E0104` 拒绝，报告定位该参数并说明默认值仅接受 int 字面量或布尔常量

### Requirement: 设备代码语句接受集

设备代码（设备入口函数体与 Pipeline `produce`/`consume` 嵌套函数体）MUST 只接受以下语句类别：单目标赋值（无类型注解）、增强算术赋值（运算符限于「设备代码表达式接受集」接受的运算符集）、表达式语句、`for` 语句（可迭代表达式仅为 `range(...)` 或 `tis.tile_iter()` 调用）、`if`/`elif`/`else`、`with` 语句（上下文表达式仅为 `tis.warp_group(...)` 调用）、`return`、`pass`、Pipeline `produce`/`consume` 装饰的嵌套函数定义。设备代码出现类别之外的语句时，编译器 MUST 以 `E0105` 拒绝；报告包含被拒绝语句类别与行列位置，恢复建议给出该类别最接近的合法替代。被拒绝类别包括：`while`、`break`、`continue`、`try`、`raise`、`assert`、`del`、`global`、`nonlocal`、`match`、注解式局部赋值（形如 `x: f32 = 0.0`，含带右值与无右值两种形态）、设备代码内 `import`、设备代码内类定义。

语句类别本身在白名单内、但其子表达式被「设备代码表达式接受集」拒绝时，编译器 MUST 按该 Requirement 以 `E0106` 报告被拒子表达式，MUST NOT 因子表达式对该语句另报 `E0105`。

#### Scenario: 白名单语句组合被接受

- **WHEN** 设备代码只使用白名单语句（如 `veps/design.md` §7 的 FlashAttention 案例函数体）
- **THEN** 语句接受集检查通过

#### Scenario: while 循环被拒绝并给出替代

- **WHEN** 设备代码包含 `while` 语句
- **THEN** 编译以 `E0105` 拒绝，报告注明类别为 `while`、定位并建议改写为 `for ... in range(...)` 或 `tis.tile_iter()`

#### Scenario: match 语句被拒绝

- **WHEN** 设备代码包含 `match` 语句（Python 3.10 可解析）
- **THEN** 编译以 `E0105` 拒绝，报告注明类别为 `match` 并定位

#### Scenario: 注解式局部赋值被拒绝

- **WHEN** 设备代码包含 `x: f32 = 0.0` 或无右值的 `x: f32`
- **THEN** 编译以 `E0105` 拒绝，报告注明类别为注解式局部赋值，恢复建议为使用无注解赋值（局部类型由推断承载）

#### Scenario: 白名单语句含被拒表达式只报表达式错误

- **WHEN** 赋值语句右值为 lambda（如 `f = lambda: 1`）
- **THEN** 编译以 `E0106` 拒绝该 lambda 表达式，不另报 `E0105`

### Requirement: 设备代码表达式接受集

设备代码 MUST 只接受以下表达式类别：名称引用、属性访问、下标与切片、调用、以接受运算符集构成的二元运算（算术 `+` `-` `*` `/`；比较 `<` `<=` `>` `>=` `==` `!=`，仅两操作数单比较；逻辑 `and` `or`）、一元负 `-` 与 `not`、数值常量（int 字面量——十进制/二进制/八进制/十六进制形式均可——与 float 字面量）、布尔常量 `True`/`False`、`None`、元组字面量（仅切片下标与调用实参位置）、字典字面量（仅调用关键字实参值位置，且键 MUST 为字符串常量）、字符串常量（仅调用关键字实参值位置）。

设备代码出现类别之外的表达式时，编译器 MUST 以 `E0106` 拒绝；报告包含被拒绝表达式类别与行列位置，恢复建议给出最接近的合法表达方式。被拒绝类别包括：lambda、列表/集合/字典推导与生成器表达式、f-string 与任意字符串格式化表达式、`yield`、`await`、星号解包、海象运算符、条件（三元）表达式、链式比较（如 `a < b < c`）、白名单之外的运算符（`//`、`%`、`**`、`<<`、`>>`、`&`、`|`、`^`、`~`）、字符串常量出现在调用关键字实参值之外的位置。

#### Scenario: 切片与广播索引被接受

- **WHEN** 表达式包含 `Q[bm*BR:(bm+1)*BR, :]` 与 `m_new[:, None]`
- **THEN** 表达式接受集检查通过

#### Scenario: 调用实参中的元组与字典字面量被接受

- **WHEN** 表达式包含 `tis.zeros((BR, BC), f32, Register)`、`tis.make_tensor(L_ptr, (seq_len,))` 与 `tis.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})`
- **THEN** 表达式接受集检查通过（元组在调用实参位置、字典作为关键字实参值且键为字符串常量）

#### Scenario: 列表推导被拒绝

- **WHEN** 设备代码包含 `[x*2 for x in row]`
- **THEN** 编译以 `E0106` 拒绝，报告注明类别为列表推导并定位

#### Scenario: 白名单外运算符被拒绝

- **WHEN** 设备代码包含 `a // b` 或 `a ** b`
- **THEN** 编译以 `E0106` 拒绝，报告注明被拒运算符并定位

### Requirement: 状态类类体结构

`@tis.state` 类的类体 MUST 只包含带类型注解且无右值的字段声明（形如 `O_acc: Tensor[f32, (BR, D), Register]`）。类体出现其他结构——方法定义、带右值的字段赋值、无注解赋值、表达式语句等——时，编译器 MUST 以 `E0107` 拒绝；报告包含被拒绝结构与行列位置，恢复建议说明字段初始值由 `pipe.run(init=...)` 提供、不得在类体书写。

#### Scenario: 纯字段状态类被接受

- **WHEN** 类体只包含带注解且无右值的字段声明
- **THEN** 状态类结构检查通过（字段类型规则由 `language/type-system` 承载）

#### Scenario: 状态类包含方法被拒绝

- **WHEN** `@tis.state` 类体内定义方法
- **THEN** 编译以 `E0107` 拒绝，报告定位该方法

#### Scenario: 字段带右值被拒绝

- **WHEN** 类体包含 `m: Tensor[f32, (BR,), Register] = tis.zeros(...)`
- **THEN** 编译以 `E0107` 拒绝，恢复建议指向 `pipe.run(init=...)`

### Requirement: 语法拒绝报告契约

语法接受集产生的每条拒绝（E01xx）MUST 报告四项要素：错误码、源码行列位置、被拒绝的语法类别、恢复建议（内容由所命中错误码的定义给出）。同一模块存在多条语法拒绝时，除 `E0101`（无法建立 AST 时仅能报告单条）外，编译器 MUST 收集并输出全部拒绝，MUST NOT 在第一条拒绝处停止；多条拒绝 MUST 按源码位置升序排列，同一源码位置命中多个检查阶段时 MUST 只报告检查管线最早的一个错误码。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

#### Scenario: 多条拒绝被收集并按位置排序

- **WHEN** 同一模块的设备代码分别在第 12 行包含 `while`、第 30 行包含列表推导
- **THEN** 输出恰好两条拒绝（`E0105` 与 `E0106`），按 12 行在前、30 行在后排序

#### Scenario: 同位置多命中只报最早检查阶段

- **WHEN** 同一源码位置的构造同时满足 `E0105` 与 `E0106` 的触发条件
- **THEN** 该位置只报告检查管线更早的 `E0105` 一条拒绝

#### Scenario: 重复编译拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的拒绝清单逐条一致（错误码、位置、类别、恢复建议完全相同）
