# Design: add-diagnostics-json-schema

## 设计范围

| 对象 | 说明 |
|---|---|
| `openspec/changes/add-diagnostics-json-schema/specs/diagnostics/json-schema/spec.md` | 新 capability（ADDED）：8 Requirement——输出形态分离与产出条件、顶层字段与确定性承诺、字段精度等级与标注义务、resources/memory/async/compute 段字段集、portability 与 suggestions 建议域 |
| 既有五段拒绝报告契约（syntax/type-system/memory-ops/execution/numerics 各「报告契约」Requirement） | **零改动**：形态分离为本 capability 新增义务（拒绝时 MUST NOT 产出诊断 JSON），不触碰各段错误码、四要素与排序定义 |
| 既有 specs 六处让渡文本（设计审查 round 1 m3 修正计数） | **承接不修改**：均为域级 `diagnostics/*` 指向，本 change 落地后语义依然成立——numerics Purpose（L5）与 cast Requirement（L176）、compute-ops dot Requirement 正文两处（L40）与 Scenario（L85）、execution Async Requirement（L113） |

本 change 无代码交付（项目尚无编译器实现；诊断 JSON 行为契约为实现先行规格）。

## 当前实现（基线事实）

- `veps/design.md` §3.1（L237-291）：诊断 JSON Schema v1 示例——顶层 `schema_version`/`target`/`kernel`/`compile_time_ms`/`deterministic_hash` + 六段（resources/memory/async/compute/portability/suggestions）；标注「v1，冻结」。
- `veps/design.md` §3.2（L293-303）：字段精度表——registers/spill/shared「精确」、bank_conflict「静态可判定场景精确；动态索引标注 unknown」、occupancy「精确（理论值）」、async gap「粗估，明确标注 estimated」；roofline 模型降级为 `estimated_bound: compute|memory` 单字段；关键诚实性声明（静态诊断不替代 profiling）。
- `veps/design.md` §2.1 原则 3（L92-93）：每个原语的 lowering 结果必须能在诊断 JSON 中以源码行号定位。
- `veps/design.md` §2.4（L182-189）：PadPolicy 三策略的诊断体现——`pad_waste_ratio`/`tail_masked`/`tail_fma_ratio` 三字段。
- `veps/design.md` L153：mma=Auto 的确定性选择「出现在诊断 JSON 里」。
- `veps/design.md` L211：展开后每条 `cp.async`/`wait_group` 带 `origin: {pipeline, stage, iteration_class: prologue|steady|epilogue, src_line}`。
- `veps/design.md` L362：portability 段建议（如 `Placement.Fast`）——「建议由诊断给出，决策由用户/AI 做」。
- 既有 specs 让渡句四处（numerics Purpose「诊断 JSON 字段归 diagnostics/*」、numerics cast Requirement「精度损失的诊断呈现（建议域字段）归 diagnostics/*」、compute-ops CR2「选择的诊断呈现归 diagnostics/*」「三策略……诊断字段归 diagnostics/*」、execution Async Requirement「诊断字段形式由 diagnostics/* 承载」）。
- CLAUDE.md 价值主张 2（JSON Schema v1 冻结向后兼容、六段、行号映射）与 3（同源码+同目标产出一致 `deterministic_hash`）。

## GAP 分析

1. **产出条件未定义**：诊断 JSON 何时产出、拒绝（E01xx–E06xx）时是否产出/部分产出，任何 spec 均未规定（§3.1 示例无 errors 段，暗示分离，但未成文）。
2. **确定性边界未定义**：`deterministic_hash` 无 spec 承载；`compile_time_ms` 天然非确定，哪些字段豁免确定性承诺未划界。
3. **精度等级未成文**：§3.2 表是 veps 层事实；estimated 的标注义务（如何标注）、unknown 的禁止猜测义务无规格。
4. **字段集未冻结**：六段字段名与类型只有示例；「v1 冻结」的演进语义（允许何种演进）未定义。
5. **四处承接点（六处让渡文本）未落地**：compute-ops 的 mma 选择呈现与 pad 三策略字段（正文两处 + Scenario 一处文本）、execution 的 origin 四字段、numerics 的精度损失建议域字段与 Purpose 域级让渡。
6. **建议域结构未定义**：S0201 仅一例；code 格式、唯一性、非强制性未规格。

## 修改方案（含裁决记录）

**D1 形态分离（DR1）**：裁决拒绝时 MUST NOT 产出诊断 JSON（而非携带 errors 段或部分产出）。理由：(a) §3.1 schema 无 errors 段——诊断 JSON 的字段（registers、occupancy、instruction_count 等）只在完整 lowering 后可得，拒绝路径上半产出无法保证字段集闭合与确定性；(b) 拒绝清单已有五段规格各自的四要素契约，混装会造成同一信息双形态。副作用：使用方在失败轮次拿不到诊断——与诚实性声明一致（失败轮次的信息就是拒绝清单本身）。

**D2 非确定字段封闭清单 = {compile_time_ms}（DR2）**：唯一豁免为墙钟时间；其余字段（含自由文本 note/issue/msg、layout 描述）全部参与「同输入逐字段一致」承诺——这把 CLAUDE.md 价值主张 3 的 `deterministic_hash` 承诺具体化：hash 一致 + 除豁免字段外逐字段一致为可验证的验收面。「异输入 hash 不同」明确不做承诺（hash 碰撞域非行为承诺）。

**D3 精度三值 {exact, estimated, unknown} + estimated_ 前缀（DR3）**：§3.2 的「精确/静态可判定/粗估」梳理为字段级固定定级的三值枚举。estimated 的标注义务落为**名称前缀 `estimated_`**（结构性、机器可检查，§3.1 示例 `estimated_gap_cycles` 先例）。裁决：示例中的 `wait_coverage` 规范化为 `estimated_wait_coverage`（同为静态估算来源，与 gap 一致；veps 为非正式文档，正式化时补齐前缀一致性——此为对 §3.1 示例的唯一字段名修正，记于长期基线刷新时同步 veps 勘误）。unknown 为值级状态（仅 bank_conflict 登记）：不可静态判定时取字面 `"unknown"`，禁止省略、禁止猜测——诚实性声明的规格化。

**D4 estimated_bound 纳入 resources（DR4）**：§3.2 L303 的降级裁决（roofline 降级为单字段）落实为 resources 段第五字段，取值封闭二值 compute|memory，定级 estimated，并明文「不承担性能预测」。

**D5 severity 封闭 {info, warning}（DR8）**：无 `"error"`——错误属拒绝域，而拒绝时不产出诊断 JSON（D1），故 severity 无 error 值不产生缺口。issue 类目与 severity 赋值规则延期至 portability/HAL 域（本 change 只冻结条目结构、确定性覆盖与建议非强制性）。

**D6 compute 段通用三字段 + dot 专属扩展（DR7）**：veps 仅有 dot 条目示例；推广为：全部六原语条目带 `op`/`src_line`/`instruction_count` 通用三字段（原则 3「每个原语都有诊断挂点」的落实——非 dot 原语不因示例缺失而失去挂点），dot 条目追加 mma_shape/tensor_core/pad_policy 与策略关联字段。`mma_shape` 承接 compute-ops「Auto 选择呈现」让渡：呈现实际生效形状（显式值或 Auto 确定性选择结果），与该 capability「选择写进文档」义务互证。

**D7 展开操作条目（DR6）**：execution 让渡的落实形式——async 段两类条目：pipeline 汇总条目（§3.1 示例五字段）+ 展开操作条目（`op` 封闭二值 `"async_copy"`/`"wait"` + `origin` 四字段）。`op` 二值对应 veps L211 的 `cp.async`/`wait_group` 与 execution「异步拷贝、等待」表述（抽象为 HAL 中立命名）。

**D8 pad 策略关联字段条件必带（DR7）**：§2.4 表三字段（`pad_waste_ratio`/`tail_masked`/`tail_fma_ratio`）并入 compute 段 v1 字段集，按 `pad_policy` 取值条件必带（Mask⇒tail_masked、PadZero⇒pad_waste_ratio、Split⇒tail_fma_ratio；Error⇒均不带）。pad_waste_ratio/tail_fma_ratio 定级 exact（由 tile 形状与 MMA 形状静态可计算，非估算）。

**D9 建议码格式（DR8）**：`S` + 四位十进制（S0201 先例），同文档内唯一。完整建议码清单（逐条语义）延期——v1 只约束格式、唯一性、确定性与非强制性（「建议由诊断给出，决策由用户/AI 做」的规格化：MUST NOT 因未采纳而拒绝或改变行为）。

**D10 target 字段最小语义（DR2）**：仅承诺「稳定标识符字符串」；取值登记表与 HAL 能力报告归 HAL 域后续 change（与 compute-ops 对 HAL 支持列表的引用边界一致）。设计审查 round 1 m2 后补充多目标裁决：本份文档描述单一目标编译，跨目标批量编译时每目标各自产出其诊断 JSON（批量组织形态属工具链）——portability 段是**单目标文档内**对其他目标差异的提示，不依赖 `--target=all` 语义（该 CLI 语义无 spec 承载，正文与 Scenario 均不引用）。

**D11 产出单位为 kernel（DR1，设计审查 round 1 M3 修复）**：`language/syntax-acceptance-set`「设备入口装饰器识别」允许模块多个入口装饰函数（`tis.kernel`/`tis.persistent_kernel`/`tis.fused_kernel`，无数量上限），故「产出一个诊断 JSON」+ 单数 `kernel` 字段在多入口模块下未定义。裁决：**每个入口装饰函数各产出一份诊断 JSON**（单入口模块即一份，与 veps §3.1 示例的单 kernel 视角一致；顶层 `kernel` 字段为该份文档对应的入口函数名）；确定性承诺单位相应为「同一源码 + 同一目标 + 同一入口 kernel」。不采用 kernel 字段改数组（破坏 veps 顶层形态）或限单入口（收紧 syntax 接受集，属另一 capability 的 BREAKING 变更）。

**D12 estimated_bound 移出数值定级（DR3/DR4，设计审查 round 1 M2 修复）**：原稿把字符串字段 `resources.estimated_bound` 登记进 estimated 数值定级与前缀义务清单，与「字符串与布尔字段不参与定级」的范围句自相矛盾。裁决：精度三值定级仅适用数值字段（v1 前缀义务清单只登记 `estimated_gap_cycles`/`estimated_wait_coverage`）；`estimated_bound` 为字符串枚举字段、不参与数值定级，其名称中的 `estimated_` 前缀保留为**语义标注**（DR4 自洽描述：值为静态方向估算，不构成运行期承诺）。

**D13 src_line 定位义务统一（DR1/DR6，设计审查 round 1 M1 修复）**：原稿 DR1 要求条目级段每条条目 MUST 含 `src_line` 字段，但 DR6 pipeline 汇总条目五字段冻结集无 `src_line`、展开操作条目的定位在 `origin.src_line`——两种读法互相违反。裁决：(a) 汇总条目补 `src_line`（六字段，定位 Pipeline 构造行）；(b) DR1 义务句改为「每条条目 MUST 含恰好一个源码行定位：直接 `src_line`，或（仅展开操作条目）`origin.src_line`」——保持普适义务的简洁，展开操作条目不冗余加顶层字段。

**D14 memory 段条目限定 Shared 分配（DR5，设计审查 round 1 m1 修复）**：原稿「静态分配的张量」未限定分配类别，`zeros`（Register）与 `make_tensor`（视图）是否入段双解。裁决：条目对应 `tis.alloc_shared` 产物——`bank_conflict` 分析以 Shared 存储为对象，Register 分配与视图构造不入段。

**D15 自由文本量化估算旁路封堵（DR3，设计审查 round 1 m4 修复）**：`note`（字符串）可携带估算性数字描述（veps 示例 "blocks 9%"），绕过 estimated 定级义务。裁决：量化估算的权威载体为 `estimated_` 前缀数值字段；`note`/`issue`/`msg` MUST NOT 呈现任何 `estimated_` 字段未承载的量化估算（MAY 对已承载值作确定性文字说明）。

**错误码**：本 change 不新增错误码（诊断 JSON 为编译通过输出；拒绝路径归既有五段，形态分离义务自身不产生新拒绝码——违反诊断契约属实现缺陷，非源码可触发行为）。

**Scenario 设计说明**：验收语义可推导——产出条件（通过/拒绝两路径、多入口逐 kernel 产出）、确定性（同输入一致/豁免清单）、标注义务（前缀/unknown/猜测违规/自由文本旁路）、字段集（各段齐全/条件必带/空段恒在）、承接（展开溯源/Auto 呈现/建议非强制）。计数：8 Requirement / 26 Scenario（DR1=5、DR2=3、DR3=3、DR4=2、DR5=3、DR6=3、DR7=3、DR8=4；设计审查 round 1 M3 修复新增「多入口模块逐 kernel 产出」）。

## 长期基线刷新计划

归档时：
1. `openspec/specs/diagnostics/json-schema/spec.md` 由 archive 自动生成（8 Requirement）；补写 Purpose（诊断域职责：诊断 JSON v1 行为契约与四处让渡承接；确定性编译承诺载体；与五段拒绝报告的形态分工；延期项——HAL 能力字段、--profile、建议码清单、portability 类目登记）。
2. `openspec/overview.md` 稳定基线登记第 7 个 capability `diagnostics/json-schema`（2026-10-08）。
3. 四处让渡句无需回改（域级 `diagnostics/*` 指向语义不变）；numerics/compute-ops/execution 各条目在 overview 的描述不涉及本 change 修改。
4. veps 勘误（非规范层，随手可做）：`veps/design.md` §3.1 示例 `wait_coverage` 注明正式化为 `estimated_wait_coverage`（D3 裁决的对照说明；veps 不追溯改写示例，勘误说明落 `veps/m1-diagnostics-usage-mapping.md`）。

## tasks 验证方式

见 `tasks.md`：strict validation、§3.1/§2.4/§3.2 与 §7 案例对照映射（字段集零缺漏、精度定级逐字段核对、让渡四处落地核对）落 `veps/m1-diagnostics-usage-mapping.md`。
