# Design: add-m1-execution-structure-frontend

## 导航

- 行为契约：`openspec/specs/execution/pipeline-structure/spec.md`（8 Requirement / 33 Scenario——E0501–E0505 行为面 7R + 报告契约 1R）+ `openspec/specs/hal/capability-descriptions/spec.md`「入口 HAL 支持面」（E0506，3 Scenario）+ 本 change 两条 delta（execution 段间短路句、cli 快照）。
- 实现先例：`tilescript/primitives/`（第三段——路由 checker、`check_module(tree, target)` 编排、`_reject(node, category, suggestion)` 收集、`frontend.report.finalize` 收口）；`tilescript/hal.py`（能力加载——`mma_shapes`/`reduce_scopes` 消费面，其余字段随消费段加载）。
- 本 change 不改 stable specs 行为（除两条登记性 delta）。

## GAP 分析（现状 → 目标）

| 面 | 现状 | 目标 |
|---|---|---|
| E0501–E0505 | spec 已定义、零实现（四段管线 pending 含 execution-structure） | 全量实现（构造/签名/run/buffer/warp_group/内建） |
| E0506 | hal spec 已定义（含两目标登记值）、`hal.py` 未加载 `persistent_kernel` | 字段加载 + 入口扫描拒绝 |
| 管线 | 三段（syntax/type-system/primitive-contract），原语段终段 | 四段，原语段非空短路执行段（delta R 新 Scenario） |
| CLI 快照 | 三段 / pending 两段 | 四段 / pending 仅 numerics |
| compute-ops「类别清单」R | E0501 拒绝行为让渡（归 execution） | 本段承载：producer 体内六计算原语 → E0501（该 R 的 2 Scenario 转为可验收） |

## D1 架构与模块边界

新增 `tilescript/execution/` 包，公共入口唯一：`check_module(tree, target) -> list[Rejection]`（与 `primitives.check_module` 同签名同收口：`frontend.report.finalize`）。

- `__init__.py`：编排——comptime/registry 复跑（与原语段同输入）、形态环境构建入口、结构扫描、finalize。
- `shape_env.py`：轻量形态环境（D3）。
- `pipeline_ops.py`：E0502（Pipeline 构造、produce/consume 签名与调用封闭、run 契约、Pipeline 值与 range 值封闭使用）+ E0503（buffer 命名空间）。
- `warp_group.py`：E0501（语境：producer 体内计算原语、WarpGroup barrier 语境）+ E0504（参数与位置、绑定名封闭、sync 契约）。
- `builtins.py`：E0505（block_idx/cdiv/range）。
- `entry_hal.py`：E0506（模块级装饰器扫描，消费 `hal.capability(target).persistent_kernel`）。

依赖边界：execution 只依赖 `frontend`（report/top_level/装饰器辅助）、`typecheck` 公共面（`symbols.kernel_params`/`kernel_comptime_names`/`state_fields.check`/`annotations`）、`hal`、`primitives.compute_ops` 的计算原语类别清单（E0501 判定面——compute-ops R6 供给的封闭六原语，复用而非复抄，清单扩展 MUST 经 change 的同步义务由引用承载；实施时在 `compute_ops` 暴露模块级公共只读名（如 `COMPUTE_PRIMS`），execution 引用该公共面而非 `_PRIM_SPECS` 私有名——实现内部公共接口，零 spec 影响）。**不依赖** `typecheck.infer`/`primitives.traverse`（不重放类型推断，理由见 D2）。

## D2 纯结构扫描（不重放类型推断）

原语段以 `PrimitiveInferencer` 遍历取实参类型；本段的类型需求面窄得多——仅 scope（buffers 值是否 Shared）、标量种类（cdiv/range/block_idx 实参 int/comptime）、状态类名（consume 注解与 init 值）。重放整条类型推断成本高且耦合 `primitives` 内部接口；故自建**形态环境**（D3）按需跟踪，遍历为单遍语句流结构扫描（与 `bindings`/`traverse` 同构的 kernel→produce/consume 结构、快照恢复），差异：不做表达式级类型推断，只在判定点查形态环境。

`buf.<名>` 的 Tensor 类型推导承诺（buffer 命名空间 R）以**键集判定**承载：注册名接受（其 Tensor 类型与键值一致由 Pipeline 构造时 buffers 值形态判定保证——Shared Tensor 形态已核）、未注册名 E0503；独立类型推导无可观察静态输出差异（原语段先跑、`buf.*` 让渡维持——上 change residual 1 不因本段解除）。

## D3 形态环境（shape env）

名 → 形态枚举（粗类型，**不含 dtype/dims**——本段无此判定面）：

- `SHARED_TENSOR` / `REGISTER_TENSOR` / `GLOBAL_TENSOR` / `UNKNOWN_TENSOR`（tensor 但 scope 不可知）
- `INT`（运行期 int）/ `COMPTIME_INT` / `DTYPE_SCALAR`
- `STATE_CLASS(类名)` / `PIPELINE` / `RANGE_VALUE` / `UNKNOWN`

来源（单遍遍历同步维护）：
1. kernel 形参注解（`symbols.kernel_params` + 形态归约：Tensor→按 scope、`int`→INT、`comptime[int]`→COMPTIME_INT、dtype 名→DTYPE_SCALAR、状态类→STATE_CLASS）；
2. 赋值绑定按源形态：`tis.alloc_shared(...)`→SHARED_TENSOR、`tis.make_tensor(...)`→GLOBAL_TENSOR、`tis.zeros`/`tis.full(...)`→REGISTER_TENSOR、`tis.dot(...)`→REGISTER_TENSOR、`tis.cast(x, ...)`→转发 x 形态、`tis.reduce/maximum/exp/log/transpose(x)`→转发 x 形态（reduce/maximum/exp/log 的 x 恒 Register→实际即 REGISTER_TENSOR；transpose 同 x）、`tis.block_idx(...)`→INT、`tis.cdiv(a,b)`→两实参 COMPTIME_INT 则 COMPTIME_INT 否则 INT、`tis.Pipeline(...)`→PIPELINE（合法构造后）、`range(...)`→RANGE_VALUE、`AttnState(...)` 状态类构造→STATE_CLASS(注解查 registry)、纯名赋值→转发；其余（算术、`buf.*`、UNKNOWN 源）→UNKNOWN；`pipe.run` 合法调用的结果绑定形态特判归 D4.4（STATE_CLASS(init 类名)）——非法 run（拒绝已记录）的绑定为 UNKNOWN；
3. produce/consume 形参（第一形参 `int`→INT、buffer 形参→专档 BUFFER_PARAM（逃逸判定用，D6）、状态形参→STATE_CLASS(注解)）；嵌套函数退出快照恢复（与 bindings 同构）。

**buffers 值判定三分**（E0502）：SHARED_TENSOR→接受；REGISTER_TENSOR/GLOBAL_TENSOR/INT/COMPTIME_INT/DTYPE_SCALAR/PIPELINE 等确定非 Shared 形态→E0502（报文建议 `tis.alloc_shared` 产物）；UNKNOWN/UNKNOWN_TENSOR→让渡（键仍注册进命名空间，`buf.<键>` 接受）。让渡面 residual：经 UNKNOWN 链（算术切片派生等）传入的 Shared 值不判（可后续收紧）。

## D4 E0502 判定链与 Pipeline 实例表

段内 order：`E0501=1 → E0502=2 → E0503=3 → E0504=4 → E0505=5 → E0506=6`（与 spec 段内 tiebreak 一致；stage=`"execution-structure"`）。

Pipeline 实例表（per-kernel，模块内多 kernel 各自独立）：`{绑定名, buffers 键集(有序), produce 函数名, consume 函数名, run 已调用标志}`。判定链（单遍扫描 + 函数级收尾）：

1. **构造面**：`tis.Pipeline` 调用——位置仅 kernel 顶层语句（produce/consume/warp_group 体内→E0502）；参数集恰 `stages=`/`buffers=` 两必选关键字（位置实参/未知关键字/缺失→E0502）；`stages` 值 comptime 判定（int 字面量≥1、comptime[int] 名让渡——值不可知；int 字面量<1、运行期 int 名/其他形态→E0502，口径同 compute 段 `_comptime_int_node`）；`buffers` 非空 dict 字面量、键互不相同（重复键→E0502）、值按 D3 三分；同 kernel 第二个构造→E0502（M1 单实例封闭）。
2. **值使用封闭**（PIPELINE 形态名的引用位置分类）：合法位恰三——`@<名>.produce`/`@<名>.consume` 装饰器接收者、`<名>.run` 方法调用接收者、首次赋值绑定 target；其余 Name 引用（原语实参、算术操作数、条件、return、二次赋值源等）→E0502。`run`/`produce`/`consume` 属性访问在非 Pipeline 对象上→E0502。
3. **produce/consume 面**：装饰器识别复用 `frontend` 装饰器辅助；同实例重复装饰→E0502；签名裁决（produce 恰两形参：第一 `int` 注解、第二无注解；consume 恰三形参：第一 `int` 注解、第二无注解、第三状态类注解、返回注解与第三形参同类——nominal 比类名，registry 供类名集）；嵌套函数显式调用（Name 作 Call.func）或名作值（赋值源/实参/return）→E0502。
4. **run 面**：接收者须 PIPELINE 形态名；恰两实参（位置 0 为 `range(...)` Call 形态——其他→E0502；`init=` 必选、值形态 STATE_CLASS 且与 consume 第三形参同类→不同类 E0502；缺失/多余/未知关键字→E0502）；仅 kernel 顶层（嵌套体内→E0502）；per-kernel 收尾判定：每实例恰一次 run（零或多次→E0502，多次报首个之后每处/零次报装饰缺失处——报文按 spec「缺失或重复」），run 前须已装饰 produce 与 consume（按源码行位序判「前」）。`run` 结果绑定形态=STATE_CLASS(init 类名)（`st` 后续字段访问让渡维持——原语段先跑）。
5. **同码合并**：run 同调用多实参违规（第一实参非 range + init 异类）合并一条 E0502 列全部违规实参（spec 报告契约「同码合并」）。

## D5 E0503 buffer 命名空间

produce/consume 体内：buffer 形参名的 Name 引用位置封闭——仅属性访问接收者（`buf.K` 的 `buf`）；整对象形态（赋值源、调用实参、return 值、算术操作数）→E0503。`buf.<名>` 属性名不在实例表键集→E0503（报文列已注册名，登记序）。键集来自构造面（含让渡值的键）。

## D6 warp_group（E0504 参数面 + E0501 语境面）

**E0504**：`tis.warp_group(role=…, warps=…)` 调用——`role` 值域 `"producer"`/`"consumer"`（字符串常量；其他→E0504）、`warps` comptime[int] 且≥1（口径同 D4 stages；违规→E0504）、恰此两关键字（位置实参/未知/缺失→E0504）；调用位置仅 `with` 语句上下文表达式（其余表达式位置→E0504——扫描全部 `tis.warp_group` Call 按父节点判定）；with 绑定名（`as g`）使用封闭——仅 `tis.warp_group_sync` 实参位（其余值用→E0504）。`tis.warp_group_sync(g1, g2, barrier_id=c)`：恰两位置实参为**不同** with 绑定名（同名/非绑定名→E0504）；`barrier_id` 必选关键字且 comptime 非负（缺失或违规→E0504；必选性裁决 D8.6）；仅表达式语句、仅 kernel 顶层、位于两个 warp_group with 语句之后（源码行位序）；值使用→E0504。

**E0501 语境**：遍历跟踪 warp_group 体内/外与 role；producer 体内 `tis.*` 调用三分（D8 裁决表）；kernel 顶层（warp_group 体外）`tis.barrier(scope=WarpGroup)`→E0501。produce/consume 体内同面（体内 WarpGroup barrier 语境=undefined，D8）。

## D7 E0505 索引与整数内建

- `tis.block_idx(dim)`：恰一实参；comptime 非负判定（int 字面量≥0 / comptime[int] 名让渡 / 负字面量、运行期名→E0505）；结果 INT。
- `tis.cdiv(a, b)`：恰两实参；a/b 形态 INT/COMPTIME_INT（其他→E0505）；`b` 为 int 字面量 0→E0505（comptime 名值不可知→让渡）；结果两 COMPTIME_INT→COMPTIME_INT 否则 INT。
- `range(n)`（裸名内建，非 `tis.` 前缀）：恰一实参、INT/COMPTIME_INT（违规→E0505）；结果 RANGE_VALUE。**range 值封闭**（E0502 承载）：RANGE_VALUE 名/调用位置封闭——合法位恰二：`for` 可迭代表达式（syntax 已裁）、`run` 第一实参；其余（绑定后运算、原语实参等）→E0502。

## D8 裁决表（spec 空白的显式裁决，不静默扩张）

1. **producer 体内 `tis.*` 三分**：`tis.load`/`tis.store`/`tis.barrier`（内存原语）接受；六计算原语（`compute_ops` 封闭清单，公共只读名见 D1 实施注记）→E0501（类别面消费 compute-ops R6）；**其余 `tis.*`**（`zeros`/`full`/`make_tensor`/`alloc_shared`/`cast`/`block_idx`/`cdiv`/`Pipeline` 构造等）spec 只写「MUST 只允许内存原语…出现计算原语以 E0501 拒」——拒绝路径仅明文计算原语面，其余 undefined、M1 不裁（residual 登记，后续补洞 change 裁决；方向与 memory-ops D6 值位置收窄同型）。
2. **persistent/fused 入口体深检查**：前三段遍历面均仅 `@tis.kernel`（现状），本段同构——`@tis.persistent_kernel` 在 `persistent_kernel=true` 目标通过 E0506 后，其函数体四段均不深查（语法段签名/语句白名单面已覆盖三类入口）；E0506 拒绝时体同样不查（入口被拒）。residual 登记（hal spec「正常参与后续编译」的深检查读法留待后续 change 统一裁决）。
3. **warp_group 与 Pipeline/produce-consume 嵌套**：with 语句出现在 produce/consume 体内、Pipeline 构造/装饰出现在 warp_group 体内——spec 未定义合法性，不裁（residual）。
4. **`st`（run 结果）的字段访问**：STATE_CLASS 形态绑定后 `st.l` 访问形态 UNKNOWN——本段无该面检查（原语段先跑让渡），仅 D3 环境记录名存在。
5. **重复键/非字符串键**：dict 字面量重复键→E0502（spec「互不相同」）；非字符串常量键形态→syntax 段字典键句法已裁（不重复报）。
6. **`barrier_id` 缺省**：spec 形态含 `barrier_id=c`，按必选关键字处理（缺失→E0504）。

## D9 段间交互与确定性

- **短路**：`pipeline.compile_stages` 原语段非空即返回（执行段不执行）——delta R 新 Scenario 直测（公共管线面）。
- **确定性**：纯结构扫描（输入=AST+冻结 HAL 常量）+ 实例表/形态环境按语句序构建 + `finalize` 稳定排序——同输入同目标逐条一致（重复编译测试）。
- **跨 HAL 不变面**：语言层 E0501–E0505 与目标无关；E0506 按目标分化（ascend 拒/h200 接受）——两目标一致性测试（同源语言层逐条一致）。
- **E0506 与语法的同位置**：装饰器行违规（如 E0103）由语法段先拒（短路），E0506 不出现——hal spec Scenario 3 由短路结构性承载。

## D10 测试映射（全口径 39S）

execution spec 33S + compute-ops R6 类别清单 2S（本段承载后转可验收）+ 本 change delta 新增短路 Scenario 1S（报告契约 4S→5S）+ hal spec E0506 3S，合计 39S：

| Requirement | Scenario | 承载 |
|---|---|---|
| Pipeline 构造 4S | 全静态直测 | 3.x |
| produce/consume 5S | 全静态直测 | 4.x |
| run 5S | 4 静态直测 + 空迭代 1 运行时间接（静态接受面间接承载——合法 run 调用零拒绝） | 5.x |
| buffer 3S | 全静态直测 | 6.x |
| Async 3S | consume 可见 1 运行时间接 + 手写组管理 1 静态（`tis.wait_group` 让渡零拒绝负例 + `group=` E0406 先例）+ 展开定位 1 后端延期 | 6.x |
| warp_group 5S | 4 静态直测 + 汇合可见 1 运行时间接 | 7.x |
| 内建 4S | 全静态直测 | 8.x |
| 报告契约 5S（delta 后） | 4 静态直测 + 1 静态（delta 新增短路 Scenario） | 9.x |
| E0506 3S（hal spec） | 全静态直测（ascend/h200 分化、语法先报短路承载） | 9.x |
| compute-ops 类别 2S | 全静态直测（producer reduce 拒、consumer 计算原语接受） | 7.x |

**验收面 34S**（静态直测 30 + 运行时间接 3 + 让渡负例 1——execution stable + compute-ops 口径内）；**口径对账：39S = 34S + delta 新增 1S（验收，静态直测）+ E0506 3S（验收，静态直测）+ 延期 1S（展开产物定位，后端 pass）**，与上表行合计逐项一致。

## D11 residual risk（登记，非本 change 缺陷）

1. producer 体内非计算原语 `tis.*`（分配/视图/内建）undefined 不裁（D8.1）。
2. `@tis.persistent_kernel`/`@tis.fused_kernel` 入口体四段不深查（D8.2；E0506 通过后与被拒均不查）。
3. warp_group 与 Pipeline 嵌套合法性 undefined 不裁（D8.3）。
4. buffers 值 UNKNOWN 链让渡（D3 三分之第三支——`buf.<键>` 仍接受）。
5. `buf.<名>` 独立 Tensor 类型推导无静态可观察输出（键集判定承载；上 change residual 1 维持——原语段先跑）。
6. `tis.wait_group` 等未定义 `tis.*` 符号拒绝码待原语总集 capability（spec 既有让渡）。
7. 展开产物源码定位（diagnostics 字段形式）归后端与 `diagnostics/*`。
8. `st` 字段访问/`pipe.run` 表达式内算术——UNKNOWN 链让渡（numerics 段后收紧）。
9. import 别名/from-import 空白（上 change residual 9 延续——本段同按字面 `tis`/`range` 识别）。

## 长期基线刷新计划（归档时执行）

1. `openspec archive`：`execution/pipeline-structure`「报告契约与 E05xx 段位」MODIFIED（4→5 Scenario，补段间短路）与 `toolchain/cli` R3 MODIFIED（快照四段）并入 stable。
2. `overview.md`：当前状态改四段已实现、pending 仅 numerics；`execution/pipeline-structure` 基线行追加首个实现注记（D10 全口径 39S 中类别清单 2S 由本段承载转验收）；`hal/capability-descriptions` 基线行追加 `persistent_kernel` 消费落地注记；`toolchain/cli` 基线行追加快照联动注记（三次登记性 MODIFIED 并述）。
3. CLAUDE.md 无需更新。
4. `veps/` 不回改。
