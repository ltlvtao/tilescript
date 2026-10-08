# M1 诊断 JSON Schema 用法映射核对记录

> 本文是 `openspec/changes/add-diagnostics-json-schema` tasks.md task 2/3 的验证证据（非正式过程记录）。
> 核对对象：`specs/diagnostics/json-schema/spec.md` 的 8 个 Requirement（ADDED）。
> 依据：`veps/design.md` §3.1（诊断 JSON Schema v1 示例，L237-291）、§3.2（精度表与诚实性声明，L293-303）、§2.4（PadPolicy 三诊断字段，L182-189）、§2.1 原则 3（L92-93）、L153/L211/L362、§7（FlashAttention 完整案例）。
> 引用约定：DR1–DR8 依次指 spec 的「输出形态分离与产出条件」「顶层字段与确定性承诺」「字段精度等级与标注义务」「resources 段字段集」「memory 段字段集」「async 段字段集与展开溯源」「compute 段字段集」「portability 段与 suggestions 建议域」。

核对方法：以 veps §3.1 示例的**全部字段**为对照全集逐字段映射（零缺漏标准）；§3.2 精度表逐行、§2.4 三字段逐项定级核对；三处让渡承接点（六处文本）逐处落地核对；形态分离与五段报告契约逐条零冲突核对；§7 案例作诊断域视角推演（该案例的诊断 JSON 应含哪些条目）。

## 1. veps §3.1 字段集零缺漏映射

| veps §3.1 字段 | spec 承载 | 核对 |
|---|---|---|
| 顶层 `schema_version` | DR1（值恒 `"1.0"`；v1 冻结演进规则） | ✅ |
| 顶层 `target` | DR2（稳定标识符；单目标承诺；登记表延期 HAL 域） | ✅ |
| 顶层 `kernel` | DR2（本份文档对应入口 kernel 函数名；per-kernel 产出） | ✅ |
| 顶层 `compile_time_ms` | DR2（非确定字段封闭清单唯一成员） | ✅ |
| 顶层 `deterministic_hash` | DR2（确定性承诺载体：同源码+同目标+同入口 kernel 一致） | ✅ |
| `resources.registers_per_thread` | DR4（整数，exact） | ✅ |
| `resources.register_spill_bytes` | DR4（整数，exact） | ✅ |
| `resources.shared_memory_bytes` | DR4（整数，exact；无分配时 0 不省略） | ✅ |
| `resources.occupancy.active_warps_per_sm` / `.limiter` | DR4（对象，exact 理论值） | ✅ |
| `resources.estimated_bound` | DR4（字符串枚举 compute/memory；§3.2 L303 roofline 降级裁决的落实；不参与数值定级、前缀为语义标注——round 1 M2 修复后表述） | ✅ |
| `memory[].tensor` / `.src_line` / `.layout` | DR5（绑定名/分配行/确定性布局描述） | ✅ |
| `memory[].bank_conflict.read_way` / `.write_way` / `.worst_access_line` | DR5（对象或 `"unknown"`；条目限 alloc_shared 产物——round 1 m1 修复） | ✅ |
| `async[].pipeline` / `.stages` | DR6 汇总条目 | ✅ |
| `async[].estimated_gap_cycles` | DR6（整数，estimated，前缀） | ✅ |
| `async[].wait_coverage` | **规范化为 `estimated_wait_coverage`**（DR6；同为静态估算，与 gap 一致补前缀——D3 裁决，§3.1 示例的唯一字段名修正，见 §6 勘误） | ✅ |
| `async[].note` | DR6（确定性文字；DR3 旁路封堵：不得呈现 estimated_ 字段未承载的量化估算）。值级判定（代码审查 round 1 m2 补）：§3.1 示例值 "wait_group at line 71 blocks 9% of steady-state iterations" 中 9% 与 `estimated_wait_coverage: 0.91` 互补——属「MAY 对已承载值作确定性文字说明」的合法路径（非未承载新估算）；若示例 note 出现任何 estimated_ 字段未承载的百分比/周期数则违规 | ✅ |
| `async[]` 展开条目（L211 `origin`） | DR6 展开操作条目：`op`（async_copy/wait 封闭二值）+ `origin` 四字段（pipeline/stage/iteration_class 三值/src_line） | ✅ |
| `compute[].op` / `.src_line` / `.instruction_count` | DR7 通用三字段（六原语名封闭集） | ✅ |
| `compute[].mma_shape` / `.tensor_core` / `.pad_policy` | DR7 dot 专属必带（mma_shape 承接 L153「Auto 选择出现在诊断 JSON」） | ✅ |
| `compute[].tail_masked` | DR7 条件必带（pad_policy=Mask） | ✅ |
| `compute[].pad_waste_ratio` / `.tail_fma_ratio` | DR7 条件必带（PadZero/Split；§2.4 表 L186/L188 落实；exact——静态可计算） | ✅ |
| `portability[].issue` / `.src_line` / `.severity` | DR8（severity 封闭 info/warning；触发=静态判定其他 HAL 目标行为差异——round 1 m2 修复，不引用 `--target=all`） | ✅ |
| `suggestions[].code` / `.src_line` / `.msg` | DR8（S+四位唯一；非强制） | ✅ |

**结论：§3.1 示例全部字段 + §3.2/§2.4/L153/L211/L362 补充字段（estimated_bound、pad_waste_ratio、tail_fma_ratio、origin）零缺漏承载；唯一字段名修正为 wait_coverage→estimated_wait_coverage（D3）。**

## 2. §3.2 精度表逐行定级核对

| §3.2 行 | veps 表述 | spec 定级 | 核对 |
|---|---|---|---|
| registers / spill / shared | `ptxas -v` / CANN 日志，精确 | DR4 exact（来源工具属实现机制，非目标；行为侧只定级） | ✅ |
| bank_conflict | 静态可判定场景精确；动态索引标注 unknown | DR5：可判定 exact；不可判定 MUST 取字面 `"unknown"`、MUST NOT 省略/猜测 | ✅ |
| occupancy | 由 registers/shared 查表计算，精确（理论值） | DR4 exact（理论值） | ✅ |
| async gap | 静态估算，粗估，明确标注 estimated | DR6 estimated + `estimated_` 前缀（结构性标注） | ✅ |
| 运行时 stall reason | 可选 `--profile` 模式，精确但慢 | 非目标（工具层；DR3 末句「静态诊断不替代运行期 profiling」承接诚实性声明） | ✅ |
| roofline 降级字段 | `estimated_bound: compute|memory` 单字段，不承担预测性能 | DR4 字符串枚举二值 + 「不承担性能预测、不构成运行期承诺」 | ✅ |
| 关键诚实性声明 | 静态诊断不能替代 profiling | DR3 unknown/estimated 义务 + 末句 | ✅ |

## 3. 让渡承接落地（六处文本、四个承接点）

| # | 既有 spec 文本（grep 核实位置） | 承接点 | 本 change 落地 | 核对 |
|---|---|---|---|---|
| 1 | numerics L5（Purpose）「诊断 JSON 字段归 `diagnostics/*`」 | 域级让渡 | 本 capability 即该域首个实体（全域承载） | ✅ |
| 2 | numerics L176「精度损失的诊断呈现（建议域字段）归 `diagnostics/*`」 | cast 精度损失建议 | DR8 suggestions 建议域结构承载（具体建议内容随实现登记——建议码完整清单为非目标） | ✅ |
| 3 | compute-ops L40「（Auto 选择）写进文档（选择的诊断呈现归 `diagnostics/*`）」 | mma 选择呈现 | DR7 `mma_shape`（呈现显式值或 Auto 确定性选择结果；与「写进文档」义务互证） | ✅ |
| 4 | compute-ops L40「三策略均编译通过（……诊断字段归 `diagnostics/*`）」 | pad 三策略字段 | DR7 `pad_policy` + 条件必带三字段（Mask⇒tail_masked、PadZero⇒pad_waste_ratio、Split⇒tail_fma_ratio） | ✅ |
| 5 | compute-ops L85（Scenario）「诊断字段归 `diagnostics/*`」 | 同 #4（Scenario 侧） | DR7 Mask 条件字段（与该 Scenario 的编译通过路径互补） | ✅ |
| 6 | execution L113「（展开后每条底层操作）诊断字段形式由 `diagnostics/*` 承载」 | origin 标签形式 | DR6 展开操作条目：origin 四字段（pipeline/stage/iteration_class 封闭三值/src_line） | ✅ |

**纯 ADDED 论证**：六处文本均为「归/由 `diagnostics/*`」域级指向，本 change 落地后指向有实体、原句语义不变——无需 MODIFIED（对照：compute-ops 对 memory-ops 的 MODIFIED 因既有文本「E0408–E0499 保留」字面为假，本域无同类字面冲突）。

## 4. 形态分离与确定性边界核对（task 3）

| # | 核对面 | 核对结果 |
|---|---|---|
| ① | 与五段拒绝报告契约零冲突 | 五段各「报告契约」Requirement（syntax 四要素/收集/排序/tiebreak、type-system E03xx、memory-ops E04xx、execution E05xx、numerics E06xx）义务不变；DR1 只新增「拒绝时 MUST NOT 产出任何诊断 JSON」的形态分离义务与「通过时 per-kernel 产出」义务，不改任何既有句；诊断 JSON 不携带拒绝条目（无 errors 段）——信息无双形态混装 ✅ |
| ② | 跨段管线不变 | E01xx→E03xx→E04xx→E05xx→E06xx 检查管线语义与各段定义零改动；诊断域在管线全部通过后介入（产出侧）✅ |
| ③ | 非确定字段封闭清单完备性 | v1 顶层+六段全部字段逐一审查：schema_version（恒值）、target/kernel（编译输入决定）、deterministic_hash（承诺载体）、六段字段（静态分析/lowering 产物、含自由文本 note/issue/msg 均受逐字段一致约束）；v1 冻结字段集内无路径、版本串、时间戳类字段——唯一天然非确定为 compile_time_ms（墙钟）✅ |
| ④ | 不新增错误码 | spec 全文 E0xxx 出现处均为引用既有段位（E01xx–E06xx 管线描述），无新码；违反诊断契约（如缺 src_line）属实现缺陷而非源码可触发拒绝——不与任何既有段位重叠 ✅ |
| ⑤ | per-kernel 产出与 syntax 一致 | DR1 引用「识别与数量由 language/syntax-acceptance-set 定义」——三个入口装饰器（tis.kernel/persistent_kernel/fused_kernel）无数量上限的事实与「各产出一份」表述一致；确定性承诺单位（同源码+同目标+同入口 kernel）无歧义 ✅ |
| ⑥ | 多目标行为 | DR2 target 单目标承诺 + 跨目标批量每目标各自产出（批量组织形态属工具链）；portability 段为单目标文档内对其他目标差异的提示（静态判定触发）——`--target=all` CLI 语义零引用（grep 核实 spec/proposal）✅ |

## 5. §7 FlashAttention 案例诊断域推演

单入口 `@tis.kernel def flash_attn_fwd` → **一份诊断 JSON**（DR1；kernel 字段 `flash_attn_fwd`）。推演该案例的诊断 JSON 条目面（合法路径零冲突标准）：

| 段 | §7 事实 | 推演条目 | 核对 |
|---|---|---|---|
| resources | 三次 alloc_shared（(64,64) f16 ×2 + (64,64) f16 ×1，各 8192B）+ 状态类 Register | `shared_memory_bytes` = 3×8192 = 24576；occupancy/registers/estimated_bound 为实现期值（字段恒在） | ✅ 与 memory-ops 分配语义一致（swizzled 为分配物理属性） |
| memory | `alloc_shared` ×3（Q_s/K_s/V_s，均 `layout=tis.Layout.swizzled(xor=0b11100)`） | 3 条目：tensor=Q_s/K_s/V_s、src_line=各自分配行、layout 呈现 swizzle 掩码、bank_conflict 按访问模式静态可判（已知 layout；若含动态索引则该条 `"unknown"`） | ✅ 限 alloc_shared 产物——make_tensor ×5（Q/K/V/O/L，Global 视图）与 zeros 不入段（m1 修复后无歧义） |
| async | `pipe = tis.Pipeline(stages=STAGES(=2), buffers={"K","V"})` + fetch 内 Async load ×2/迭代 | 1 汇总条目（pipeline="pipe"、src_line=pipe 构造行、stages=2、两个 estimated 字段、note）+ 展开操作条目若干（fetch 每迭代 async_copy ×2 + 编译器插入 wait；各带 origin{pipeline="pipe", stage="fetch", iteration_class ∈ prologue/steady/epilogue, src_line=fetch 体内对应行}） | ✅ 溯源义务与 execution「展开后可定位」一致 |
| compute | 九次计算原语调用（compute 映射 §1 #1–#9） | 9 条目：dot ×2（#1 显式 mma → mma_shape 呈现 "m16n8k16"、pad_policy="Error" 无条件字段；#8 Auto → mma_shape 呈现该 HAL 确定性选择、pad_policy="Error"）、transpose ×1、reduce ×2、maximum ×1、exp ×2、log ×1（均通用三字段） | ✅ 与 compute-ops 契约零冲突；mma 呈现两路径（显式/Auto）均覆盖 |
| portability | D/BR/BC=64 对 (16,8,16) 与两目标候选形状全整除（compute 映射已核） | 无对齐差异条目（空数组合法；swizzle/Placement 差异是否存在由静态判定定） | ✅ |
| suggestions | v1 无登记建议码 | 空数组（段恒在） | ✅ |

**结论：§7 全部诊断相关事实经推演零冲突——产出单位（单入口一份）、memory 条目集（3 条 alloc_shared）、async 两类条目（汇总 + 展开溯源）、compute 九条目（含 mma 两路径）、空段恒在均有承载。**

## 6. 勘误登记

- `veps/design.md` §3.1 示例 `wait_coverage` 字段正式化为 `estimated_wait_coverage`（D3 裁决：同为静态估算来源，补齐 `estimated_` 前缀一致性；§3.1 为非正式示例不追溯改写，本节为对照说明）。
