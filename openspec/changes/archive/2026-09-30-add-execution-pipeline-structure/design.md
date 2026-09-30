# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| execution/pipeline-structure | 新增执行结构行为契约（纯规格，无代码交付） | specs/execution/pipeline-structure/spec.md | 第 1 节 |
| language/type-system | MODIFIED「类型不匹配拒绝」：等价豁免边界第二次扩展（执行结构调用实参） | specs/language/type-system/spec.md | 第 1 节 |

## 1. execution/pipeline-structure

### 目标与规范依据

目标 Requirements：`specs/execution/pipeline-structure/spec.md` 的全部 ADDED Requirements——Pipeline 构造契约、produce/consume 嵌套函数契约、run 调用契约、buffer 命名空间、Async 完成保证与组管理不可见、warp_group 上下文契约、索引与整数内建、报告契约与段位。

规范依据：`veps/design.md` §2.1 原则 1（显式优先：stages/produce-consume 分工用户显式指定）与原则 4（类型系统承担安全）、§2.3 warp 特化（warp_group 签名、E0501 事实、warp_group_sync）、§2.5 Pipeline 抽象（构造签名、produce/consume 示例、半自动原则、诊断映射、形式化验证承诺）、§7（八类执行结构用法）。承接已归档 specs 的三处显式让渡：type-system「表达式结果类型规则」（执行结构方法/内建调用结果类型、Pipeline buffer 属性）、memory-ops「加载模式与完成语义」（Async 完成保证）与「屏障语义」（WarpGroup 上下文与汇合）。

### 当前实现

仓库尚无编译器代码。执行结构事实只存在于 `veps/design.md`：§2.5 一段示例（`tis.Pipeline(stages=3, buffers={…})`/`@pipe.produce`/`@pipe.consume`/`pipe.run(range(…), init=…)`）加"半自动"原则与 Z3 承诺；§2.3 warp_group 签名与 E0501 一句话事实；§7 八类用法（Pipeline×1、produce×1、consume×1、run×1、buf 属性×4、range×1、cdiv×1、block_idx×1）零参数级契约。语法层已接受相关形态（`with tis.warp_group(...)`、`@pipe.produce/consume`、`for … in range(...)`）。既有错误码事实：`E0501`（veps，未进 stable）。type-system MODIFIED 后的豁免只覆盖 `tis.*` 原语调用，`pipe.run`/`range` 实参按现行 stable 属 E0303 适用集（"执行结构方法与内建调用的实参绑定"）——与这些调用无源码层形参注解的事实矛盾，形成双解源。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| Pipeline 构造 | §2.5 示例签名（stages=3 字面量、buffers 字典） | 参数种类/值域/位置约束、Pipeline 值的类型层地位均未定义 |
| produce/consume | §2.5 示例两签名（§2.5 的 `buf: tis.BufferSlot` 与 §7 的无注解 buf 不一致） | 签名契约（形参个数/注解/返回）、配对唯一性、BufferSlot 地位未裁决 |
| run 调用 | §2.5 `pipe.run(range(num_blocks), init=AttnState(...))` | 实参契约、结果类型、迭代语义（状态链）未定义 |
| buffer 属性 | type-system 让渡句挂起；memory-ops Scenario 以 buf.K 已是 Shared Tensor 为前提 | 属性类型推导、未注册名、buf 使用位置封闭未定义 |
| Async 完成保证 | memory-ops 显式让渡本域；§2.5 半自动原则 + Z3 承诺 | 完成语义、槽复用安全、组管理不可见的语言级承诺未定义 |
| warp_group | §2.3 签名与 E0501 一句话；memory-ops 让渡 WarpGroup 汇合 | role/warps 契约、producer 体内限制、sync 契约、WarpGroup barrier 语境与汇合语义未定义 |
| 索引内建 | §7 三用法（block_idx(0)/cdiv(seq_len, BC)/range(...)） | 参数契约与结果种类推导未定义 |
| E05xx 段位 | 仅 E0501 veps 事实 | 段位语义、tiebreak、与 E0303 的边界未定义 |

### 修改方案

本 change 为纯规格 change，无代码交付。关键裁决：

- **三个特殊值的类型层地位（合法使用位置封闭集）**：Pipeline 值、buffer 命名空间对象（buf）、range 可迭代值均不进 type-system 封闭类型世界（不引入新类型种类、不改 R1 封闭声明），各自定义为封闭的使用位置集合（Pipeline：装饰器接收者+run 接收者；buf：体内属性访问接收者；range：for 可迭代表达式+run 第一实参），违反按 `E0502`/`E0503` 裁决。理由：三者是无源码层注解形态的语境绑定对象，等价规则对其无意义；封闭集模式与 type-system"封闭集合"风格同构。封闭集的**反向出口**一并封闭（设计审查 round 1 major）：produce/consume 嵌套函数名 MUST NOT 被显式调用或作值使用（唯一入口是 run 迭代展开，`E0502`）；`run`/`produce`/`consume` 成员的合法接收者仅为 Pipeline 值（其他对象上访问 `E0502`）；`tis.warp_group(...)` 仅可作 with 上下文表达式、with 绑定名仅可作 warp_group_sync 实参（`E0504`）。
- **E0303 豁免第二次扩展（MODIFIED，BREAKING）**：执行结构调用（`pipe.run`/`range`）与 `tis.*` 原语同构——无源码层形参注解，实参约束是调用契约而非类型化绑定。MODIFIED 将豁免从"tis.* 原语调用"扩为"tis.* 原语调用与执行结构调用"，适用集表述改为"带源码层形参注解的调用（状态类构造调用等）"。状态类构造（有字段注解）仍走等价。影响面：E0303 适用集再次收敛；无存量受众，迁移路径为无。
- **buf 形参无注解（裁决 §2.5/§7 不一致）**：`tis.BufferSlot` 不引入类型世界（引入需 type-system MODIFIED 且无必要性——buf 的类型由 buffers 推导，§7 事实即无注解）。buffer 形参带类型注解时的合法性以判据式让渡 type-system 裁决（`tis.BufferSlot` 不在其类型表达式封闭集内，具体拒绝路径以该 capability 权威文本为准——该断言对 E0302 触发条件含注解位置的解读属解释性依赖，spec 已注明；如后续 type-system change 澄清注解位置覆盖与本裁决不一致，以该 change 为准），本 change 不重复报告。
- **consume 状态链契约**：consume 第三形参注解状态类 S、返回注解必须同一 S（nominal 同类），run 的 init 必须同一 S、run 结果类型 = S。状态链四个环节（init→st₀→consume 返回→run 结果）同 S，断裂即 `E0502`。理由：迭代间状态携带的类型安全是 Pipeline 的核心承诺；多状态类交替无事实。
- **M1 封闭**：每 kernel 至多一个 Pipeline 实例、恰一次 run、均在 kernel 顶层函数体；produce/consume 各至多一个。多实例/嵌套 run/复用无 §7 事实，保守封闭（扩展须显式 change）。嵌套体内构造 Pipeline 或调用 run 以 `E0502` 拒绝。
- **构造参数契约**：stages 为 `comptime[int]` ≥1（§7 STAGES=2、§2.5 字面量 3——编译期确定是分配的前提）；buffers 非空字典字面量、键为字符串常量且内容形如标识符（字典键句法由 `language/syntax-acceptance-set` 承载）、互不相同、值必须 Shared Tensor（多缓冲在 Shared 上展开，Global/Register buffer 无语义事实）。参数集外关键字或任何位置实参 `E0502`。
- **迭代语义为语言级承诺**：st₀=init；每 j 先 produce(j) 后 consume(j, stⱼ)，stⱼ₊₁=返回值；结果 stₙ；n≤0 空迭代结果为 init（设计审查 round 1 minor：公式对 n≤0 失效的补全）。编译器 MAY 多缓冲展开实现——展开的正确性承诺见完成保证条款。
- **Async 完成保证为行为承诺、机制延期**：consume 执行前 produce 的 Async 拷贝完成且可见；槽复用无 use-before-ready/overwrite-before-consumed（§2.5 Z3 验证的**对象**写为 MUST 行为，Z3 工具链归实现 design——行为 spec 承诺可观察结果，不规定验证手段）；组管理（commit/wait）由编译器插入、用户不可手写（memory-ops 已封闭手写路径，本 change 只声明不可见性承诺，不重复报告）；展开产物可定位源码行列（origin 字段形式归 diagnostics/*）。
- **E0501 正式化与精化**：从 veps 一句话升格为"warp_group 语境违规"段内首码——涵盖 producer 体内计算原语（原语义不变）与 `barrier(scope=WarpGroup)` 语境违规（承接 memory-ops 让渡的上下文约束）。WarpGroup barrier 汇合范围 = 同 warp_group 全部线程，可见性同 Block 模式。
- **producer 体内限制**：只允许内存原语（load/store/barrier）、普通赋值与语句控制流（§2.3"只允许 load/commit/barrier"——commit 对用户不可见，故封闭为三内存原语；语句接受集由语法层承载）；consumer 不设限（§2.3 只约束 producer，producer 拒 dot 的事实即角色分工）。warp_group_sync：两位置实参为两个**不同** warp_group 语句的绑定名、`barrier_id` 为 `comptime[int]` 非负，调用仅以表达式语句形态出现在 kernel 顶层函数体且位于两个 with 语句之后，不产生值（值使用按 `E0504` 拒绝）。
- **索引内建（E0505）**：block_idx(dim)——dim comptime 非负、结果运行期 int（§7 bm 用途即运行期标量）；cdiv(a, b)——两实参 int/comptime 标量、comptime 零除数拒绝、结果种类按实参推导（全 comptime → comptime[int]，值折叠归数值语义——shape 组件用 cdiv 的路径在数值语义前按种类推导可用）；range(n) 恰一实参（§7 事实；Python 三态 range 不引入）、可迭代值封闭使用位置。
- **E05xx 段位与 tiebreak**：E0501（warp_group 语境）→E0502（Pipeline 构造/签名/调用契约，及 Pipeline 值与 range 可迭代值的封闭使用位置）→E0503（buffer 命名空间，含 buffer 对象封闭使用违规）→E0504（warp_group 参数契约，含 warp_group 调用位置与 with 绑定名的封闭使用）→E0505（内建参数）——先语境后契约、先结构后成员、先整体后参数；同码多实参违规合并一条报告并列出全部违规实参；跨段管线 E01xx→E03xx→E04xx→E05xx（执行结构检查最后：其输入依赖前段已定型的类型与原语信息）。E0506–E0599 保留。
- **充分性标准**：覆盖 §7 全部八类执行结构用法（Pipeline、produce、consume、run、buf.K/buf.V 属性、range、cdiv、block_idx）及 warp_group 的 E0501 语义核对（§2.3 事实），核对任务见 tasks。
- **确定性影响**：本 change 无实现，不改变任何 `deterministic_hash`；"重复编译 E05xx 清单一致"为将来实现固化确定性要求。全部检查为编译期静态契约（参数形态、签名、语境、封闭位置），完成保证条款为运行前静态可验证的语言级承诺（编译器负责证明），不涉及运行时反馈。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/execution/pipeline-structure/spec.md`；按本 change 的 MODIFIED delta 更新 `openspec/specs/language/type-system/spec.md`——「类型不匹配拒绝」Requirement 以豁免边界第二次扩展后的完整重述替换既有块（Scenario 标题保持 canonical 不变）。
- designs：无（Pipeline 展开 pass 的实现设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `execution/pipeline-structure` 索引并同步 `language/type-system` 行描述（豁免边界第二次扩展）。
