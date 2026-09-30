# execution/pipeline-structure 规格增量

## ADDED Requirements

### Requirement: Pipeline 构造契约

`tis.Pipeline(stages=…, buffers=…)`：`stages` 为必选关键字实参，MUST 为 `comptime[int]` 且值不小于 1（其他形态或值域以 `E0502` 拒绝）；`buffers` 为必选关键字实参，MUST 为非空字典字面量——键为字符串常量且内容形如标识符（字典键句法由 `language/syntax-acceptance-set` 承载）、互不相同、值为 `Tensor[…, Shared]`（Register/Global scope 张量或其他形态以 `E0502` 拒绝）。参数集之外的任何关键字实参、或任何位置实参以 `E0502` 拒绝。构造调用 MUST 只出现在 kernel 顶层函数体（`produce`/`consume` 嵌套函数体内构造以 `E0502` 拒绝）；每个 kernel 函数体内至多构造一个 Pipeline 实例（M1 封闭，第二个以 `E0502` 拒绝）。构造调用的结果是 **Pipeline 值**：其类型不在 `language/type-system` 的封闭类型世界内，合法使用位置为封闭集——作为 `produce`/`consume` 属性装饰器的接收者、作为 `run` 方法调用的接收者；除赋值语句的首次绑定（`pipe = tis.Pipeline(…)`）外，其余任何使用（作为原语实参、参与运算、条件判断等）以 `E0502` 拒绝。`run`/`produce`/`consume` 属性的合法接收者仅为 Pipeline 值；在其他任何对象上访问这些成员以 `E0502` 拒绝。`stages` 语义为缓冲深度：同一逻辑 buffer 的 `stages` 份物理副本循环复用（复用安全见「Async 完成保证」）。

#### Scenario: 合法构造被接受

- **WHEN** kernel 顶层函数体出现 `pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})`，`STAGES: comptime[int] = 2`，`K_s`/`V_s` 均为 `Tensor[f16, (comptime BC, comptime D), Shared]`
- **THEN** 构造契约检查通过，`pipe` 绑定为 Pipeline 值（合法使用位置封闭集内）

#### Scenario: stages 值域违规被拒绝

- **WHEN** `tis.Pipeline(stages=0, buffers={"K": K_s})`
- **THEN** 编译以 `E0502` 拒绝，报告注明 `stages` 必须为不小于 1 的 `comptime[int]`

#### Scenario: 非 Shared 缓冲被拒绝

- **WHEN** `buffers` 值为 `Tensor[f32, (BR,), Register]`
- **THEN** 编译以 `E0502` 拒绝，恢复建议缓冲区必须是 Shared scope Tensor（`tis.alloc_shared` 产物）

#### Scenario: Pipeline 值误用被拒绝

- **WHEN** 设备代码出现 `tis.load(x, pipe)` 或 `y = pipe + 1`
- **THEN** 编译以 `E0502` 拒绝，报告注明 Pipeline 值的合法使用位置封闭集（produce/consume 装饰器与 run 调用）

### Requirement: produce 与 consume 嵌套函数契约

每个 Pipeline 实例 MUST 至多装饰一个 `produce` 嵌套函数与一个 `consume` 嵌套函数（重复装饰以 `E0502` 拒绝；装饰器识别由 `language/syntax-acceptance-set` 承载）。`produce` 嵌套函数签名 MUST 恰为两形参：第一形参带 `int` 注解、第二形参（buffer 形参）无注解（违反以 `E0502` 拒绝）。`consume` 嵌套函数签名 MUST 恰为三形参：第一形参带 `int` 注解、第二形参无注解、第三形参带状态类类型注解，且返回注解 MUST 为与第三形参注解**同一**状态类类型（同一类声明；违反以 `E0502` 拒绝）。形参按位置裁决，名称任意。produce/consume 嵌套函数 MUST NOT 被显式调用（形如 `fetch(0, K_s)`），其名称 MUST NOT 作为值使用（赋值源、实参、返回值等）——嵌套函数执行的唯一入口是 `pipe.run` 的迭代展开，违反以 `E0502` 拒绝。buffer 形参带类型注解时的合法性由 `language/type-system` 的类型表达式规则裁决（`tis.BufferSlot` 不在该 capability 定义的类型表达式封闭集内，本 change 不引入该类型；具体拒绝路径以该 capability 的权威文本为准），本 capability 不重复报告。两嵌套函数体内的语句接受集由 `language/syntax-acceptance-set`（`E0105`）承载；`mode=Async` 的调用语境合法性由 `primitives/memory-ops`（`E0407`）承载。

#### Scenario: 合法 produce 与 consume 签名被接受

- **WHEN** `@pipe.produce def fetch(j: int, buf): …` 与 `@pipe.consume def attend(j: int, buf, st: AttnState) -> AttnState: …`
- **THEN** 签名契约检查通过（produce 两形参、consume 三形参与同类返回注解）

#### Scenario: consume 缺返回注解被拒绝

- **WHEN** `@pipe.consume def f(j: int, buf, st: AttnState): …`（无返回注解）
- **THEN** 编译以 `E0502` 拒绝，报告注明 consume 必须带与状态形参同一状态类的返回注解

#### Scenario: 返回注解与状态形参不同类被拒绝

- **WHEN** `@pipe.consume def f(j: int, buf, st: AttnState) -> OtherState: …`（两类声明不同名）
- **THEN** 编译以 `E0502` 拒绝（nominal 不同类，状态链断裂）

#### Scenario: produce 重复装饰被拒绝

- **WHEN** 同一 Pipeline 实例装饰两个 `@pipe.produce` 嵌套函数
- **THEN** 编译以 `E0502` 拒绝，报告注明每实例至多一个 produce 与一个 consume

#### Scenario: 嵌套函数被显式调用或作值使用被拒绝

- **WHEN** 设备代码出现 `fetch(0, K_s)`（显式调用 produce 嵌套函数）、`f = fetch` 或以 `fetch` 为实参
- **THEN** 编译以 `E0502` 拒绝，报告注明 produce/consume 嵌套函数的唯一执行入口是 `pipe.run` 的迭代展开

### Requirement: run 调用契约

`pipe.run(...)` MUST 恰接受两实参：位置 0 为 `range(...)` 内建调用（其他形态以 `E0502` 拒绝）；`init=` 必选关键字实参，其实参 MUST 为状态类类型的值（其他形态以 `E0502` 拒绝），且其状态类 MUST 与 `consume` 第三形参注解为同一类声明（不同类以 `E0502` 拒绝）。`run` 调用 MUST 只出现在 kernel 顶层函数体，且每个 Pipeline 实例恰调用一次（缺失或重复以 `E0502` 拒绝；调用前必须已装饰 `produce` 与 `consume`，缺失以 `E0502` 拒绝）。`run` 的结果类型为 `init` 实参的状态类类型（承接 `language/type-system`「表达式结果类型规则」的执行结构方法调用让渡）。迭代语义 MUST 为：设迭代序列长度为 `n`（`range` 实参的运行期值），状态初值 `st₀ = init`；对每个 `j = 0…n−1`，先执行 `produce(j, buf)`、后执行 `consume(j, buf, stⱼ)`，`stⱼ₊₁` 为该次 `consume` 的返回值；`run` 的结果为 `stₙ`。`n ≤ 0` 时为空迭代：不执行任何 `produce`/`consume`，`run` 的结果为 `init`。该语义为语言级承诺，编译器 MAY 经多缓冲与展开实现（见「Async 完成保证」）。

#### Scenario: 合法 run 调用被接受并推导结果类型

- **WHEN** `st = pipe.run(range(tis.cdiv(seq_len, BC)), init=AttnState(O_acc=…, m=…, l=…))` 且 consume 状态形参注解为 `AttnState`
- **THEN** 调用契约检查通过，`st` 的类型为 `AttnState`（= init 的状态类类型）

#### Scenario: init 状态类与 consume 注解不同类被拒绝

- **WHEN** `init=` 实参为 `OtherState` 构造而 consume 状态形参注解为 `AttnState`
- **THEN** 编译以 `E0502` 拒绝（状态链两端不同类）

#### Scenario: 第一实参非 range 调用被拒绝

- **WHEN** `pipe.run(K_s, init=st)`（以 Tensor 为第一实参）
- **THEN** 编译以 `E0502` 拒绝，报告注明第一实参必须为 `range(...)` 调用

#### Scenario: 嵌套体内调用 run 被拒绝

- **WHEN** `produce` 或 `consume` 嵌套函数体内出现 `pipe.run(...)`
- **THEN** 编译以 `E0502` 拒绝（run 仅 kernel 顶层，M1 单实例单调用封闭）

#### Scenario: 空迭代结果为 init

- **WHEN** `range` 实参的运行期值为 0（如 `seq_len` 为 0 时 `tis.cdiv(seq_len, BC)` 求值为 0）
- **THEN** 空迭代：不执行任何 `produce`/`consume`，`run` 的结果为 `init`（类型仍为该状态类）

### Requirement: buffer 命名空间

`produce`/`consume` 的 buffer 形参绑定该 Pipeline `buffers` 字面量引入的命名空间。`buf.<名>` 属性访问的结果类型 MUST 为 `buffers` 中同名键值的 Tensor 类型（类型推导唯一，承接 `language/type-system`「表达式结果类型规则」的 Pipeline buffer 属性让渡）；访问未注册名以 `E0503` 拒绝，报告列出已注册名。buffer 形参值的使用位置为封闭集：仅在其所在嵌套函数体内作属性访问的接收者（作为原语实参、赋值源、返回值等传出以 `E0503` 拒绝）。`buf.<名>` 的属性值是该迭代的逻辑 buffer（Shared Tensor），`produce` 体内 `mode=Async` 拷贝的 `dst` 写入目标。

#### Scenario: buffer 属性类型推导

- **WHEN** `buffers={"K": K_s}` 且 `K_s: Tensor[f16, (comptime BC, comptime D), Shared]`，consume 体内访问 `buf.K`
- **THEN** `buf.K` 的类型为 `Tensor[f16, (comptime BC, comptime D), Shared]`

#### Scenario: 未注册名被拒绝

- **WHEN** `produce` 体内访问 `buf.Q`（`buffers` 只注册了 `K`/`V`）
- **THEN** 编译以 `E0503` 拒绝，报告列出已注册名 `K`/`V`

#### Scenario: buffer 形参逃逸被拒绝

- **WHEN** `consume` 体内出现 `x = buf`（整对象赋值）、以 `buf` 为实参的调用、或 `return buf`（属性访问 `buf.K` 之外的任何整对象传出形态）
- **THEN** 编译以 `E0503` 拒绝，报告注明 buffer 形参仅限体内属性访问接收者

### Requirement: Async 完成保证与组管理不可见

`produce` 嵌套函数体内以 `mode=Async` 发起的拷贝，其完成保证为语言级承诺：对每个迭代 `j`，`consume(j, buf, st)` 体执行开始前，`produce(j, buf)` 发起的全部 Async 拷贝 MUST 已完成且对执行 `consume` 的线程可见。物理缓冲槽循环复用 MUST 安全：第 `j + stages` 迭代对某逻辑 buffer 物理槽的写入 MUST NOT 早于第 `j` 迭代 `consume` 对该槽读取的完成（无 use-before-ready、无 overwrite-before-consumed）。上述保证由编译器插入的组管理操作（commit/wait）与同步实现——组管理对用户不可见：用户 MUST NOT 手写异步组管理原语（`tis.commit_group`/`tis.wait_group` 不在本语言已定义原语集内，`tis.load` 的 `group=` 参数已由 `primitives/memory-ops` 以 `E0406` 拒绝）。Pipeline 展开后的每条底层操作（异步拷贝、等待）MUST 可定位到源码行列位置（诊断字段形式由 `diagnostics/*` 承载）。

#### Scenario: consume 读到已完成拷贝

- **WHEN** `produce(j, buf)` 内 `tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)`，`consume(j, buf, st)` 内读取 `buf.K`
- **THEN** `consume` 读到的内容为该迭代 `produce` 拷贝完成后的数据（完成与可见性由语言承诺）

#### Scenario: 手写组管理不可达

- **WHEN** 设备代码出现 `tis.wait_group(0)` 或 `tis.load(x, y, mode=Async, group=g)`
- **THEN** 组管理不存在用户可见路径：`tis.wait_group` 不是本语言已定义的原语，不存在可用的调用形态（未定义 `tis.*` 符号的拒绝错误码归属待原语总集 capability 正式化，本 change 不指定）；`group=` 参数由 `primitives/memory-ops` 以 `E0406` 拒绝（已裁，本 Requirement 不重复报告）

#### Scenario: 展开产物可定位源码

- **WHEN** Pipeline 展开 pass 将 produce 体的一条 Async 拷贝展开为多条底层操作
- **THEN** 每条底层操作均携带源码行列定位（可追溯到 produce 体内的发起语句）

### Requirement: warp_group 上下文契约

`with tis.warp_group(role=r, warps=w) as g:`：`role` 值域 MUST 为 `"producer"` 或 `"consumer"`（其他值以 `E0504` 拒绝）；`warps` MUST 为 `comptime[int]` 且不小于 1（以 `E0504` 拒绝）。`tis.warp_group(...)` 调用 MUST 只作为 `with` 语句的上下文表达式（语句位置约束由 `language/syntax-acceptance-set` 承载；其他表达式位置以 `E0504` 拒绝）；`with` 目标绑定名（`g`）的合法使用位置仅为 `tis.warp_group_sync` 的实参（其他值使用以 `E0504` 拒绝）。`producer` 角色体内 MUST 只允许内存原语（`tis.load`/`tis.store`/`tis.barrier`）、普通赋值与语句控制流（语句接受集由 `language/syntax-acceptance-set` 承载）；出现计算原语（`tis.dot` 等）以 `E0501` 拒绝（既有语义，正式化为 warp_group 语境违规）。`consumer` 角色体内不设原语类别限制。`tis.warp_group_sync(g1, g2, barrier_id=c)`：两位置实参 MUST 为两个**不同** warp_group 语句的绑定名（同一绑定以 `E0504` 拒绝），`barrier_id` 为 `comptime[int]` 且非负（违反以 `E0504` 拒绝）；调用 MUST 只以表达式语句形态出现在 kernel 顶层函数体且位于两个 warp_group with 语句之后（其他位置以 `E0504` 拒绝），不产生值（值使用以 `E0504` 拒绝，与 `primitives/memory-ops` 无值原语条款同构）；语义为两组全部线程在该语句处汇合，汇合前任一组线程的写汇合后对另一组可见。`tis.barrier(scope=WarpGroup)` MUST 只出现在 warp_group with 体内（其他位置以 `E0501` 拒绝——语境违规），其汇合范围为同 warp_group 的全部线程，汇合与可见性语义同 `primitives/memory-ops` 的 Block 模式（承接其 WarpGroup 让渡）。

#### Scenario: producer 体内 dot 被拒绝

- **WHEN** `with tis.warp_group(role="producer", warps=2) as pg:` 体内出现 `tis.dot(...)`
- **THEN** 编译以 `E0501` 拒绝，报告注明 producer 角色只允许内存原语

#### Scenario: 合法 producer 与 consumer 配对

- **WHEN** producer（warps=2，体内 Async load）与 consumer（warps=6，体内 dot）两个 warp_group，后随 `tis.warp_group_sync(pg, cg, barrier_id=0)`
- **THEN** 契约检查通过，两组在 sync 语句汇合且写入互可见

#### Scenario: role 值域违规被拒绝

- **WHEN** `tis.warp_group(role="dma", warps=2)`
- **THEN** 编译以 `E0504` 拒绝，报告注明 role 值域为 producer/consumer

#### Scenario: WarpGroup barrier 出现在顶层被拒绝

- **WHEN** kernel 顶层函数体（无 warp_group 语境）出现 `tis.barrier(scope=WarpGroup)`
- **THEN** 编译以 `E0501` 拒绝，恢复建议在 warp_group 体内使用或改用 Block

#### Scenario: WarpGroup barrier 在 warp_group 体内汇合同组线程

- **WHEN** warp_group 体内执行 `tis.barrier(scope=WarpGroup)`，此前同组线程 A 写入 Shared
- **THEN** 汇合后同组全部线程可见该写入

### Requirement: 索引与整数内建

`tis.block_idx(dim)`：`dim` MUST 为 `comptime[int]` 且非负（违反以 `E0505` 拒绝）；结果为运行期 `int` 标量。`tis.cdiv(a, b)`：`a` 与 `b` MUST 均为 `int` 或 `comptime[int]` 种类的标量（其他形态以 `E0505` 拒绝）；`b` 为 `comptime[int]` 且值为 0 时以 `E0505` 拒绝；结果种类按实参推导——两实参均为 `comptime[int]` 时结果为 `comptime[int]`（值的编译期折叠归数值语义 capability），否则为 `int`。`range(n)` 内建调用 MUST 恰接受一个实参，实参为 `int` 或 `comptime[int]` 种类（违反以 `E0505` 拒绝）；结果为**可迭代值**——类型不在 `language/type-system` 封闭类型世界内，合法使用位置为封闭集：`for` 语句可迭代表达式（`language/syntax-acceptance-set` 已裁）与 `pipe.run` 第一实参（本 capability）；其余使用以 `E0502` 拒绝。三者的结果类型与种类推导承接 `language/type-system`「表达式结果类型规则」的内建调用让渡。

#### Scenario: block_idx 返回运行期 int

- **WHEN** `bm = tis.block_idx(0)`
- **THEN** `bm` 的类型为运行期 `int` 标量

#### Scenario: cdiv 结果种类推导

- **WHEN** `tis.cdiv(seq_len, BC)`（`seq_len: int` 运行期、`BC: comptime[int]`）
- **THEN** 结果为运行期 `int`（`tis.cdiv(64, BC)` 两 comptime 则结果为 `comptime[int]`）

#### Scenario: cdiv 常量零除数被拒绝

- **WHEN** `tis.cdiv(seq_len, 0)`（除数为 `comptime[int]` 值 0）
- **THEN** 编译以 `E0505` 拒绝

#### Scenario: range 值误用被拒绝

- **WHEN** 设备代码出现 `x = range(10)`（绑定后参与运算）或 `tis.load(range(10), y)`
- **THEN** 编译以 `E0502` 拒绝，报告注明可迭代值仅限 for 可迭代表达式与 run 第一实参

### Requirement: 报告契约与 E05xx 段位

本 capability 产生的每条拒绝（`E05xx`）MUST 报告四要素：错误码、源码行列位置、被违反的规则（含违规形态、名称或位置类别）、恢复建议。同一模块存在多条 `E05xx` 拒绝时 MUST 收集输出全部并按源码位置升序排列。检查管线顺序为语法（`E01xx`）→ 类型（`E03xx`）→ 原语契约（`E04xx`）→ 执行结构（`E05xx`）；同一源码位置跨段命中时 MUST 只报告最早段一条。执行结构调用（`pipe.run`、`range` 等无源码层形参注解的调用）的实参不适用 `language/type-system` 绑定等价规则（`E0303`，经本 change 对该 capability 的 MODIFIED 裁决），实参违规一律落 `E05xx`。同一源码位置命中多个本 capability 规则时 MUST 只报告一个错误码，按段内顺序取首个：`E0501`（warp_group 语境）→ `E0502`（Pipeline 构造、签名与调用契约，及 Pipeline 值与 range 可迭代值的封闭使用位置）→ `E0503`（buffer 命名空间，含 buffer 对象的封闭使用违规）→ `E0504`（warp_group 参数契约，含 warp_group 调用位置与 with 绑定名的封闭使用）→ `E0505`（索引与整数内建参数）。同一调用命中多个实参的同类契约违规时 MUST 合并为一条报告并列出全部违规实参。同一输入重复编译 MUST 产生逐条一致的拒绝清单。

错误码段位：`E05xx` 为执行结构段——`E0501` 为 warp_group 语境违规（`veps` 既有事实"producer 体内出现 dot"的正式化，语义涵盖 producer 体内计算原语与 `barrier(scope=WarpGroup)` 语境违规，原场景语义不变）；`E0502`/`E0503`/`E0504`/`E0505` 为本 change 新增；`E0506`–`E0599` 保留给执行结构域后续扩展。

#### Scenario: 多条 E05xx 拒绝被收集并排序

- **WHEN** 同一模块第 8 行存在 `E0502`（buffers 形态违规）、第 21 行存在 `E0503`（未注册 buffer 名）
- **THEN** 输出恰好两条拒绝，按第 8 行在前、第 21 行在后排序

#### Scenario: 同位置原语与执行结构双命中只报原语错误

- **WHEN** 同一 `tis.load(x, y, mode=Async)` 调用位置同时命中 `E0407`（Async 语境）与某 `E05xx`
- **THEN** 该位置只报告管线更早的 `E0407` 一条（原语契约段先于执行结构段）

#### Scenario: 同位置多个执行结构规则命中只报段内首个

- **WHEN** 同一 `pipe.run(Tensor_obj, init=t)` 调用同时命中 `E0502`（第一实参形态与 init 形态）
- **THEN** 该位置只报告 `E0502` 一条（同码合并，报告全部违规实参）

#### Scenario: 重复编译执行结构拒绝清单一致

- **WHEN** 同一非法模块连续编译两次
- **THEN** 两次输出的 `E05xx` 拒绝清单逐条一致（错误码、位置、规则描述、恢复建议完全相同）
