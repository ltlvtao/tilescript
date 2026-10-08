# Design

## 设计范围

| 受影响 capability | 目标变化 | delta specs | 设计章节 |
|---|---|---|---|
| primitives/compute-ops | 新增计算原语行为契约（纯规格，无代码交付） | specs/primitives/compute-ops/spec.md | 第 1 节 |
| primitives/memory-ops | 「报告契约与 E04xx 段位」登记性 MODIFIED（保留区间联动） | specs/primitives/memory-ops/spec.md | 第 1 节 |

本 change 为 ADDED 为主、一条登记性 MODIFIED：`primitives/memory-ops` 的原语集合是其 capability 级封闭集（"本 capability 定义八个核心存取原语"），语言级不存在原语全集封闭规则（语法层调用与属性访问为通用表达式类别），新增 capability 不触碰其原语集合与行为；`E0402`/`E0403` 为既有占用的语义正式化（memory-ops 明文"既有占用，语义不变"）。唯一 MODIFIED 为段位保留区间的登记性联动（E0408 起用后"E0408–E0499 保留"字面为假，保留区间改 E0409–E0499 并指明 E0408 归属；无行为变化），无 BREAKING（设计审查 round 1 M2 裁决）。

## 1. primitives/compute-ops

### 目标与规范依据

目标 Requirements：`specs/primitives/compute-ops/spec.md` 的全部 ADDED Requirements——原语集合与调用结构、dot 的操作数与 MMA 契约、reduce 的归约契约、元素级与成对计算原语、transpose 的数据重排契约、计算原语类别清单、报告契约与 E04xx 段位补充。

规范依据：`veps/design.md` §2.1 原则 1（显式优先——不自动回退 SIMD、Auto 选择确定且文档化）与原则 2（一个概念一个原语）；§2.3 计算原语签名（dot 签名与 mma=Auto 确定性、reduce 签名与 scope=Auto HAL 依赖）；§2.4 形状不匹配策略（PadPolicy 四策略表与跨硬件对齐差异）；§7 FlashAttention 案例六类计算原语用法。承接四个方向的既有指向：memory-ops「报告契约与 E04xx 段位」的 E0402/E0403 既有占用声明与 E0408–E0499 保留、`numerics/value-semantics` Purpose（计算原语域为后续 change）与 R3 两处（逐元素比较归计算原语域后续 change）、`execution/pipeline-structure` E0501 的"计算原语（`tis.dot` 等）"类别引用。

### 当前实现

仓库尚无编译器代码。计算原语事实只存在于 `veps/design.md`：§2.3 dot 签名草案（A `Tensor[..., (M, K), Shared|Register]`、B `Tensor[..., (K, N), Shared|Register]`、C `Tensor[f32, (M, N), Register]`、`mma: MMAShape | Auto = Auto`、`pad: PadPolicy = Error`；mma=Auto 取 HAL 列表第一项；E0402 列出全部支持形状；E0403 列出最近合法形状、不自动回退 SIMD）与 reduce 签名草案（`tis.reduce(x, axis, op=Sum|Max, scope=Auto|Warp|Block)`；scope=Auto 依 HAL 选择）；§2.4 PadPolicy 四策略表（Error/PadZero/Mask/Split 及诊断字段名）；§7 六类用法（dot ×2——`mma=tis.MMA(16, 8, 16)`/`pad=tis.PadPolicy.Error` 显式与全默认、`transpose` ×1（Shared 上 (BC, D)→(D, BC)）、reduce ×2（`op=Max`/`op=Sum` 均 `axis=1`、scope 默认）、maximum ×1（f32 (BR,) 同 shape）、exp ×2 与 log ×1（f32 元素级））。`maximum`/`exp`/`log`/`transpose` 连签名草案都没有；E0402/E0403 语义无权威规格；E0501 的"等"未闭合；numerics 算术规则以计算原语结果类型为前提类型（三个用法映射文档多处标注，`m ← maximum` 一处形态级残留）。

### GAP 分析

| 规范目标 | 当前事实 | 差距 |
|---|---|---|
| 原语集合与调用结构 | 六原语中两个有签名草案、四个零签名 | 参数集、必选/可省略、默认值、调用结构违规码未定义 |
| dot 操作数与 MMA 契约 | 签名草案 + §7 两处用法；E0402/E0403 仅 veps 事实 | A/B/C dtype/scope/shape 规则、K 相容、累加语义、运行期维边界、MMA 构造、Auto 确定性、四策略接受面未规格化 |
| reduce 契约 | 签名草案 + §7 两处用法 | axis/op/scope 值域、dtype 集合、结果去维、Auto 确定性、跨硬件数值一致边界未定义 |
| 元素级与成对原语 | 仅 §7 用法（零签名） | exp/log 浮点限制、maximum 同 shape 规则与广播边界未定义 |
| transpose | 仅 §7 用法 | 秩、dtype、scope、结果规则未定义 |
| 类别清单 | E0501"tis.dot 等"未闭合 | 六原语封闭集与 E0501 判定面的接口未定义 |
| E04xx 段位 | E0402/E0403 占用悬空、E0408 起未用 | 段内 tiebreak、与 E0404–E0407 关系、确定性未定义 |

### 修改方案

本 change 为纯规格 change，无代码交付。关键裁决：

- **六原语封闭集**（对称 memory-ops「原语集合与调用结构」模式）：`dot`/`reduce`/`maximum`/`exp`/`log`/`transpose`。调用结构违规（未知关键字、位置超量、必选缺失）统一 `E0408`（与 memory-ops 的 `E0406` 对称——各 capability 的参数契约码互不越界）。**位置实参计数基准为"可按位置传递参数"**（设计审查 round 1 M1 修正）：dot 的 `mma`/`pad` 为关键字专属（签名 `*` 之后）MUST NOT 位置传递，dot 可位置实参上限为 A/B/C 三个——消除"参数集总长度（必选加可省略）"读法下 5 位置实参不拒的双解；可省略参数位置传递示例覆盖 `reduce` 第 4 位（`scope`）。`transpose` 归计算原语域（数据重排、计算图节点），不扩 memory-ops 视图构造集合（其五原语集合不变）。`tis.atomic_add` 为内存原语域扩展候选，不属本域。
- **dot A/B 限 f16（保守封闭）**：MMA dtype 直接关联硬件路径与精度语义（f32 MMA 常经 TF32 降精度、bf16/f8e4m3 支持面依硬件差异大），非纯类型规则可泛化——保守封闭为 §7 唯一事实 f16，扩展须显式 change（与 numerics 把 `/` 开放全浮点的"规则可泛化"形成对照：那边是纯类型规则、这边是硬件支持面）。C 与结果恒 f32（签名草案明确）。**累加语义 C + A@B** 只承诺类型契约与数学语义（数值精度归容差）。
- **dot 操作数 shape 限编译期常量维**：M/N/K 整除检查（E0403）与 pad 策略以编译期可知的 tile 形状为前提；运行期派生维（切片边界派生）以 E0408 拒绝。§7 全部 dot 操作数为 alloc_shared/cast/zeros 产物（comptime 维），零误命中。
- **E0402/E0403 语义正式化**：按 veps §2.3 既有事实逐条承接——E0402 报告列出该 HAL 全部支持形状；E0403 列出最近合法对齐形状、MUST NOT 自动回退非 Tensor Core 路径（原则 1：编译器不替用户降级性能；Split 的尾部 FMA 是显式选择不属自动回退）。memory-ops"语义不变"声明兑现：本 change 是从既有事实正式化，非重定义。**memory-ops 段位句登记性 MODIFIED**（设计审查 round 1 M2 裁决）：E0408 起用后"E0408–E0499 保留"字面为假（与 execution 管线前缀列举"列举截止各自段位仍为真"先例不同类——保留区间部分起用属字面冲突），联动更新保留区间为 E0409–E0499 并指明 E0408 归属；纯登记性、无行为变化、非 BREAKING。
- **MMA 形状值与 PadPolicy 值的类别地位**（设计审查 round 1 m2 补全）：dot 实参语境专用值（承接 type-system L146 对模块属性访问与原语调用表达式结果类型的让渡；与 numerics 布尔值条件语境专用值同构）——不进类型世界，合法位置封闭集（仅 dot 的 `mma=`/`pad=` 实参），他处使用 E0408；`tis.MMA(m, n, k)` 三个 comptime[int] 类型不符以 E0408 拒绝（m1 补全，`axis` 同）。
- **mma=Auto 与 scope=Auto 的确定性承诺**：依 HAL 报告确定性选择（Auto 取支持列表第一项；reduce 依 HAL 能力描述选 Warp/Block 之一）、同一 HAL 目标唯一、写进文档；诊断呈现归 `diagnostics/*`。硬件示例（H200 m16n8k16、910B m16n16k16；NVIDIA Warp/Ascend Block）是 veps 层事实举例，不写入行为 spec（HAL 能力描述字段为独立域）。
- **PadPolicy 四策略只承诺接受面**：Error（默认拒绝路径）+ PadZero/Mask/Split 三策略编译通过；物理布局与指令生成细节归实现设计，诊断字段（dot.pad_waste_ratio 等）归 `diagnostics/*`。不承诺策略间数值位一致（FMA 与 MMA 尾部的数值差属跨硬件容差）。
- **reduce dtype 全六开放、结果 dtype 不变**：Sum/Max 对封闭集合六种 dtype 良定义且为纯类型规则（与 dot A/B 的保守封闭形成对照）；axis 限 comptime[int] 值域 [0, rank)；x 限 Register scope（跨线程归约结果落寄存器，跨 scope 输入经显式移动原语对齐——与 numerics 算术 scope 一致原则同构）。**跨硬件归约位一致不承诺**（归约顺序依实现）——与 numerics cast 的静态位一致承诺形成边界对照：cast 是输入值到输出值的确定函数（位级可定义），reduce 的求和顺序依硬件（数值容差归里程碑）。
- **maximum 严格同 shape（不做广播）**：§7 唯一用法为同 shape；广播已由 numerics 算术承载（切片 + `None` 机制），成对原语不重复承载广播语义（一个概念一个原语）；shape 不符恢复建议指向 numerics 调整手段。dtype 全六开放（int max 良定义）。
- **exp/log 限浮点 dtype**：数学函数对整型无定义；结果 dtype/shape/scope 不变（f8e4m3 的 exp 精度归容差域）。
- **类别清单闭合 E0501**：六原语封闭集供 execution E0501 producer 语境判定（拒绝行为与错误码归 execution——本 capability 只提供类别定义，不改动其 Requirement）；扩展须显式 change 并同步两处。numerics R3 指向的"逐元素比较归计算原语域后续 change"经本 change 落位裁决：仍不引入（零用法零签名，E0605 恢复建议文字保持有效——"后续 change"不限定为本 change）。
- **段内 tiebreak E0408→E0402→E0403**（先形态契约后硬件支持面后对齐策略）：与 memory-ops E0404–E0407 无同位置双命中（原语名唯一决定适用集）；五段管线不变。dot 实参中的算术子表达式违规落 E06xx（子表达式位置与调用位置各自报告——§7 事实：`tis.dot(tis.cast(P, f16), ...)` 的 cast 实参先经 memory-ops/numerics 裁决定型，再进 dot 操作数契约检查）。
- **充分性标准**：覆盖 §7 全部六类计算原语用法（dot ×2——显式 mma/pad 与全默认、transpose、reduce ×2——op=Max/op=Sum、maximum、exp ×2/log ×1），合法路径全部覆盖、E0402/E0403 零误命中（§7 的 BR/BC/D 取 comptime 常量且被 MMA 形状整除的语境事实）；核对任务见 tasks。
- **确定性影响**：本 change 无实现，不改变任何 `deterministic_hash`；"重复编译清单一致"与 Auto 选择确定性为将来实现固化确定性要求。

质量属性影响：无新增黑盒质量目标（可验证性由各 Requirement 的 Scenario 承载；数值精度/容差属 veps 里程碑验收不写入）。

## 长期基线刷新计划

- stable specs：归档时新增 `openspec/specs/primitives/compute-ops/spec.md`（补写 Purpose——计算原语域职责与四个方向让渡分工）；memory-ops 的 MODIFIED delta 归档时自动应用（「报告契约与 E04xx 段位」段位句保留区间更新），应用后核对 stable 文本与 compute-ops 的 E0408 归属表述一致。其余既有 specs 文本不变。
- designs：无（dot/reduce 的 lowering 与 pad 策略代码生成设计由实现 change 承载）。
- overview：在「稳定基线」节登记 `primitives/compute-ops` 索引（E0402/E0403 语义正式化、E0408 起用），并在 memory-ops 条目补注其段位句经本 change 联动更新。
