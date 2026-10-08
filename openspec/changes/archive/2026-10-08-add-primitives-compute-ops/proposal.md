# Proposal

## Why

计算原语域是当前规格基线的最大悬空面，四个方向同时指向它：

1. **既有错误码占用悬空**：`primitives/memory-ops`「报告契约与 E04xx 段位」已登记"`E0402`/`E0403` 为计算原语（MMA/pad）既有占用，语义不变"——但这两码的语义只存在于 `veps/design.md`（§2.3：E0402 = 显式 MMA 形状硬件不支持、列出全部支持形状；E0403 = M/N/K 不整除且 `PadPolicy.Error`、列出最近合法形状、不自动回退 SIMD），无权威规格；同条还保留 `E0408`–`E0499` 给"原语域后续扩展"，起用无主。
2. **数值语义的显式让渡**：`numerics/value-semantics` Purpose 写明"计算原语域为后续 change"，其比较规则（R3）两处把"Tensor 逐元素比较"指向计算原语域后续 change；`numerics` 的算术规则以计算原语结果类型为前提类型（dot/reduce/maximum/exp/log 的返回类型），此前三个用法映射文档中多处以此为"计算原语域前提类型"、`m ← maximum` 一处仍为"形态级核对"残留。
3. **执行结构的类别让渡**：`execution/pipeline-structure` E0501 规定 producer warp_group 体内"出现计算原语（`tis.dot` 等）"以 `E0501` 拒绝——"等"的类别清单未闭合。
4. **案例零规格**：`veps/design.md` §7 计算原语用法共六类（`dot` ×2——一处显式 `mma`/`pad`、一处默认参数；`transpose` ×1；`reduce` ×2——`op=Max`/`op=Sum`；`maximum` ×1；`exp` ×2 与 `log` ×1），其中只有 `dot`/`reduce` 有 §2.3 签名草案，`maximum`/`exp`/`log`/`transpose` 连签名都没有；`dot` 的 A/B/C 操作数契约、`mma=Auto` 确定性、PadPolicy 四策略行为在 `veps/design.md` §2.3–§2.4 有描述但全部无权威规格。

## What Changes

- 新增 capability `primitives/compute-ops`，定义计算原语行为契约：
  - 原语集合与调用结构：六原语（`tis.dot(A, B, C, *, mma=Auto, pad=Error)`、`tis.reduce(x, axis, op, scope=Auto)`、`tis.maximum(a, b)`、`tis.exp(x)`、`tis.log(x)`、`tis.transpose(x)`）的参数集、可省略参数与默认值、调用结构违规（`E0408`）
  - `tis.dot` 操作数与 MMA 契约：A/B/C 的 dtype/shape/scope 规则（A/B 限 `f16`、C 与结果恒 `f32`、shape 为编译期常量维、K 相容、累加语义 C + A@B）、`tis.MMA(m, n, k)` 与 `mma=Auto` 的确定性选择、`tis.PadPolicy` 四策略接受面、`E0402`（形状不支持）与 `E0403`（不整除且 Error 策略）既有占用语义正式化
  - `tis.reduce` 契约：axis 值域、`op` 封闭二值（`Sum`/`Max`）、`scope` 值域与 `Auto` 确定性、结果去维规则与跨硬件数值一致边界（归约顺序依硬件，位一致不承诺——与 `numerics` cast 位一致承诺的边界对照）
  - 元素级与成对原语：`exp`/`log`（浮点 dtype、dtype/shape/scope 不变）、`maximum`（同 dtype 同 shape、dtype 全六开放）
  - `tis.transpose`：二维转置、dtype/scope 不变
  - 计算原语类别清单：六原语封闭集，闭合 `execution` E0501 的"`tis.dot` 等"类别引用
  - `E04xx` 报告契约补充：`E0408` 新增、段内 tiebreak、与 memory-ops `E0404`–`E0407` 的关系（不同原语调用各自适用）
- 对 `primitives/memory-ops`「报告契约与 E04xx 段位」做一条**纯登记性 MODIFIED**（非 BREAKING，无行为变化）：其段位句明文"`E0408`–`E0499` 保留给原语域后续扩展"，本 change 起用 `E0408` 后该句字面为假（保留区间部分起用——与 execution 管线"前缀列举仍为真"先例不同类），故联动更新为 `E0408` 由 `primitives/compute-ops` 定义、保留区间改 `E0409`–`E0499`；该 Requirement 其余文本与全部 Scenario 原样保留。
- **无 BREAKING**：`primitives/memory-ops` 的原语集合是其 capability 级封闭集（"本 capability 定义八个核心存取原语"），语言级无原语全集封闭规则，本 change 新增 capability 不触碰其原语集合与行为；E0402/E0403 语义正式化承接其"既有占用，语义不变"声明（语义来源为 `veps` 既有事实，非新定义）；唯一 MODIFIED 为上述登记性联动。

## Capabilities

### New Capabilities

- `primitives/compute-ops`：计算原语行为契约——六原语参数与结果规则、MMA 形状与 pad 策略、归约实现域确定性、元素级函数类型规则、transpose 数据重排、计算原语类别清单、`E0402`/`E0403` 正式化与 `E0408` 报告行为。

### Modified Capabilities

- `primitives/memory-ops`：「报告契约与 E04xx 段位」——段位保留区间联动更新（`E0408` 起用归属 `primitives/compute-ops`，保留区间改 `E0409`–`E0499`）；纯登记性，无行为变化，非 BREAKING。

## 非目标

- 逐元素比较原语（`numerics/value-semantics` R3 恢复建议所指"计算原语域后续 change"）：零用法零签名，本 change 仍不引入（该表述指向本域的后续扩展，不限定为本 change）；`E0605` 相关恢复建议文字保持有效。
- `dot` 的 A/B dtype 超出 `f16` 的支持面（`bf16`/`f32`/`f8e4m3` 的 MMA 路径）：MMA dtype 直接关联硬件路径与精度语义，跨硬件支持面不一致，保守封闭为 §7 唯一事实 `f16`，扩展须显式 change。
- `dot` 批维（签名草案 `Tensor[..., (M, K), ...]` 的前导维）：零用法。
- `reduce` 的 `x` 与元素级原语操作数超出 `Register` scope（跨 `Shared` 直接计算）：零用法；跨 scope 组合经显式移动原语对齐（与 `numerics` 算术 scope 一致原则同构）。
- 计算原语的数值精度（`exp`/`log` 的 ULP、`reduce` 跨硬件归约顺序差异、MMA 与 FMA 尾部的数值差）：属跨硬件数值容差（里程碑验收，`veps`），不写入行为 spec；`reduce` 的跨硬件位一致**不承诺**（与 `numerics` cast 静态位一致承诺形成明确边界）。
- 诊断 JSON `compute` 段字段（`dot.pad_waste_ratio`、`dot.tail_masked`、`dot.tail_fma_ratio`、MMA `Auto` 选择的诊断记录）：`diagnostics/*` 延期项；本 change 只承诺行为级（接受面、拒绝、确定性选择存在）。
- HAL 能力描述字段（MMA 支持形状列表的内容与格式、归约实现域报告字段）：HAL 能力域独立 change；本 change 以"HAL 报告"为输入引用其存在（`E0402` 报告列出全部支持形状的行为承诺依赖该输入）。
- PadZero 补零的物理布局细节、Mask/Split 的指令生成策略：实现/HAL 层设计，非行为契约。
- `tis.atomic_add`（`veps` §2.3 内存操作清单）：内存原语域扩展候选，非计算原语，零用法不引入。
- `E0409`–`E0499` 保留给原语域后续扩展。

## Impact

- 承接四向悬空：E0402/E0403 语义正式化（memory-ops"既有占用"声明闭合）、E0408 起用（"保留给原语域后续扩展"闭合——起用后 memory-ops 段位句经本 change 的登记性 MODIFIED 同步更新）、`numerics` R3 两处指向的计算原语域显式落位（本 change 裁决逐元素比较仍不引入）、execution E0501"计算原语（`tis.dot` 等）"的类别清单闭合。
- `numerics` 算术规则的前提类型（dot/reduce/maximum/exp/log 返回类型）自本 change 落实；既有三个用法映射文档中"计算原语域前提类型"与"`m ← maximum` 形态级"残留标注在后续核对记录中升级闭合（不追溯改写已归档文档）。
- 与 `primitives/memory-ops` 的边界：八存取原语集合与 `E0404`–`E0407` 不变；`transpose` 为计算原语域数据重排原语（非 memory-ops 视图构造——`make_tensor`/`alloc_shared`/`zeros`/`full`/`cast` 集合不变）；同位置无跨集错误码双命中（原语名决定适用集）。其「报告契约与 E04xx 段位」经本 change 登记性 MODIFIED（保留区间联动，见 What Changes）。
- 与 `numerics/value-semantics` 的边界：计算原语实参中的算术子表达式按 `numerics` 裁决（R4 位置①已预留"后续原语经各自契约登记"——本 change 六原语均为 Tensor 操作数位，不新增 dtype 标量参数位，该枚举不变）；计算原语结果的数值效果（舍入/精度）归容差域，类型/shape/scope 结果规则归本 change。
- 与 `execution/pipeline-structure` 的边界：E0501 拒绝行为与 `E05xx` 不变，本 change 只提供类别清单输入。
- 五段错误码管线不变（`E04xx` 段内部扩容，跨段顺序 E01xx→E03xx→E04xx→E05xx→E06xx 不变）。
- `veps/design.md` §7 六类计算原语用法自此获得权威依据；示例与规格冲突时以规格为准。
- 无生产代码影响（纯规格先行 change）。
