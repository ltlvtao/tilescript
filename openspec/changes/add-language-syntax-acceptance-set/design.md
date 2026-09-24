# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| language/syntax-acceptance-set | 新增 M1 前端语法接受集行为契约（纯规格，无代码交付） | specs/language/syntax-acceptance-set/spec.md | 第 1 节 |

## 1. language/syntax-acceptance-set

### 目标与规范依据

目标 Requirements：`specs/language/syntax-acceptance-set/spec.md` 的全部 ADDED Requirements——源码载体、模块顶层白名单、装饰器识别、kernel 签名、语句接受集、表达式接受集、状态类类体、拒绝报告契约。

### 当前实现

仓库尚无编译器代码。语法形态只由 `veps/design.md` §2、§7 的示例隐式表达：`module` 结构、`@ts.state`/`@ts.kernel`/`@pipe.produce`/`@pipe.consume` 装饰器（veps 示例现仍用旧前缀 `ts.*`，见下方命名裁决的映射）、带类型注解的签名（含 `comptime[int] = 64` 默认值）、切片与 `[:, None]` 广播、`for`/`with`/`return` 语句。既有错误码事实：`E0301`（作用域转移）、`E0402`/`E0403`（MMA）、`E0501`（warp 角色），提示段位按检查阶段分配（E03xx 类型/作用域、E04xx 计算原语、E05xx 执行结构）。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| 权威语法接受集（白名单 + 显式拒绝清单） | 仅示例隐式表达 | 无定义；实现会隐式决定语言边界 |
| Python 语法基线版本 | 未定义 | 未钉基线，接受范围随宿主解释器漂移 |
| `module` 结构语义 | §7 示例出现 `module flash_attention:` | 未裁决是真语法还是文档伪代码 |
| 字面量（元组/字典/字符串）接受位置 | §7 案例只在调用实参与关键字实参值位置使用 | 未按位置裁决；一刀切拒绝或全放都会破坏旗舰案例或扩大接受面 |
| 语法错误码段位 | E03/E04/E05 段已被占用 | E01xx 未分配 |
| 拒绝报告要素与格式 | 未定义 | 无"错误码 + 位置 + 类别 + 恢复建议"契约 |
| 拒绝收集与排序 | 未定义 | fail-fast 与收集式未裁决；AI 优化循环（`veps/design.md` §3.3 编译错误回传 LLM）需要收集式 |

### 修改方案

本 change 为纯规格 change，无代码交付：

- **命名前缀与扩展名裁决**：TileScript 缩写为 `tis`（`ts` 与 TypeScript 缩写冲突，2026-09-24 决定），语言 API 前缀一律 `tis.`，源文件规范扩展名 `.tis`。本 change 全部示例与白名单采用 `tis.*`；`veps/design.md` 示例仍用旧前缀 `ts.*`，映射为机械替换（`ts.` → `tis.`），示例其余内容不受影响，规格与示例冲突处以规格为准。扩展名取舍：`.tis` 以独立命名约定句进入 E0101 Requirement（事实陈述，不与 MUST 义务混句），不挂错误码、语法接受集不因扩展名拒绝源文件——理由：M1 Parser 只检查源文本语法，按扩展名识别/拒收文件属工具链行为，留待后续 change 细化；`E0101` 触发条件仅由源文本可解析性闭合。
- 白名单与拒绝清单按"精确清单"表达（不使用"等"），后续扩展只能通过显式 change 的 MODIFIED Requirement 进行，防实现漂移。
- 白名单充分性标准：覆盖 `veps/design.md` §7 FlashAttention 完整案例与 M1 五个 kernel（MatMul/FlashAttention/LayerNorm/Softmax/Transpose）所需的全部语法类别；拒绝清单不得命中上述源码。核对任务见 tasks。
- **载体基线裁决**：语法基线钉为 Python 3.10（等价 `ast.parse(..., feature_version=(3, 10))` 接受范围），M1 复用 Python 标准库 `ast` 解析器，接受范围不随宿主解释器版本漂移；高于 3.10 的语法按 `E0101` 拒绝。
- **module 裁决**：模块 = 源文件，语言不提供 `module <name>:` 包装语法。§7 的 `module flash_attention:` 是设计文档的文档组织伪代码，不是语言构造——该行本就不是合法 Python 语句，自动落入 `E0101`，无需专门错误码。裁决理由：引入模块包装语法违反"一个概念一个原语"（文件即模块已由 Python 载体承载），且为 M1 增加无收益的解析负担。
- **字面量按位置接受裁决**：元组字面量接受于切片下标与调用实参位置（覆盖 `tis.zeros((BR, BC), ...)`、`tis.make_tensor(L_ptr, (seq_len,))`）；字典字面量仅接受为调用关键字实参值且键为字符串常量（覆盖 `tis.Pipeline(stages=..., buffers={"K": K_s, "V": V_s})`）；字符串常量仅接受为调用关键字实参值。其他位置一律 `E0106` 拒绝。裁决理由：旗舰案例所需的最小接受面，不引入通用元组/字典/字符串一等值语义（那属于 `language/type-system` 的决策域）。
- **运算符白名单穷举**：算术 `+ - * /`、比较 `< <= > >= == !=`（仅两操作数单比较，链式比较拒绝）、逻辑 `and or`、一元 `-` 与 `not`；`// % ** << >> & | ^ ~` 全部在拒绝清单（整数除法/取模/幂/位运算由 `primitives/*` 原语承载，不由运算符语法承载——"一个概念一个原语"）。
- **E0105/E0106 报告边界**：语句类别在白名单内而子表达式被拒时，只报 `E0106`、不另报 `E0105`；同一源码位置命中多个检查阶段时，只报检查管线最早的错误码（tiebreak 规则）。保证"每个语法构造恰好一条拒绝"，杜绝双报。
- **局部注解赋值裁决**：设备代码拒绝注解式局部赋值（AnnAssign，含 `x: f32 = 0.0` 与无右值两种形态）。理由：局部变量类型由推断与原语返回类型承载，签名注解已是唯一强制类型表达点（"一个概念一个原语"）；`veps/design.md` §7 案例未使用该形态；未来若类型锚定需要，通过 MODIFIED Requirement 显式扩展。
- **comptime 默认值句法化**：`comptime` 参数默认值仅接受单个 int 字面量（十进制/二进制/八进制/十六进制形式）或布尔常量；常量表达式（如 `64 * 1024`）的编译期求值由后续 capability（常量折叠）承载，当前按 `E0104` 拒绝——精确清单哲学，避免默认值槽位成为隐式表达式语言。
- **E0107 形态裁决**：状态类字段唯一接受形态是"带类型注解且无右值"（`O_acc: Tensor[...]`）；带右值一律拒绝，恢复建议指向 `pipe.run(init=...)`——状态初值语义归属 pipeline 运行期，不归属类体语法。
- **装饰器识别裁决**：Pipeline 钩子装饰器按属性名 `produce`/`consume` 识别（形如 `@<任意实例名>.produce`），不限定实例必须叫 `pipe`。
- 错误码段位分配：`E0100`–`E0199` 归语法接受集，按检查管线顺序编号——`E0101` 载体、`E0102` 顶层、`E0103` 装饰器、`E0104` 签名、`E0105` 语句、`E0106` 表达式、`E0107` 状态类体；`E0108`–`E0199` 保留给语法接受集后续扩展。
- 拒绝裁决为收集式：除 `E0101` 无法建 AST 外，检查器收集全部语法拒绝按位置升序输出。理由：AI agent 每轮迭代获得全部语法错误，减少迭代轮数（服务 H1 假设的验证效率）。
- 拒绝报告要素（错误码、行列位置、类别、恢复建议）是与 `diagnostics/*` 未来 schema 的最小交接面——恢复建议按 config.yaml 错误码契约三要件（触发条件 + 消息要素 + 恢复建议）落入每个 E01xx 定义；本 change 不定义诊断 JSON 的整体结构。
- 确定性影响：本 change 无实现，不改变任何 `deterministic_hash`；报告契约的"重复编译拒绝清单一致"Requirement 为将来实现固化了确定性要求。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/language/syntax-acceptance-set/spec.md`。
- designs：无（M1 前端实现设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `language/syntax-acceptance-set` 索引。
