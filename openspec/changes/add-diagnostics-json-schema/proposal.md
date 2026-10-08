# Proposal: add-diagnostics-json-schema

## Why

结构化诊断是 TileScript 三大价值主张之一（源码级结构化诊断：诊断 JSON Schema v1 冻结、向后兼容、覆盖六段、每条诊断映射回源码行号），但目前四处悬空：

1. **让渡句悬空**：`numerics/value-semantics`（诊断 JSON 字段归 `diagnostics/*`；精度损失的诊断呈现归建议域字段）、`primitives/compute-ops`（mma=Auto 选择的诊断呈现、PadPolicy 三策略的诊断字段）、`execution/pipeline-structure`（Pipeline 展开后每条底层操作的源码定位，诊断字段形式由 `diagnostics/*` 承载）均指向尚无实体的域。
2. **确定性编译承诺悬空**：CLAUDE.md 价值主张 3（同一源码 + 同一目标硬件产出一致的 `deterministic_hash`）尚无任何 spec 承载；诊断 JSON 中天然非确定字段（墙钟时间）与确定性承诺的边界未定义。
3. **精度承诺悬空**：`veps/design.md` §3.2 的字段精度表（精确 / 静态可判定 / 动态索引 `unknown` / 粗估 `estimated`）与「关键诚实性声明」（静态诊断不替代 profiling）无行为契约；estimated 字段的标注义务、unknown 的禁止猜测义务未规格化。
4. **schema 演进规则悬空**：「v1 冻结、向后兼容」的具体语义（何种演进允许、何种禁止）未定义。

## What Changes

- **ADDED** capability `diagnostics/json-schema`：诊断 JSON 的产出条件与形态分离（拒绝清单 vs 诊断 JSON）、顶层字段与确定性承诺（`deterministic_hash` 语义、非确定字段封闭清单）、字段精度等级封闭枚举与标注义务、六段（resources/memory/async/compute/portability/suggestions）v1 字段集冻结与逐字段精度定级、挂点映射义务（每条条目可定位源码行、Pipeline 展开后的 origin 标签）、建议域非强制性、schema v1 冻结与向后兼容演进规则。
- 无 MODIFIED / REMOVED：四处让渡句均为域级 `diagnostics/*` 指向，本 change 落地后语义依然成立；五段错误码报告契约（E01xx–E06xx 拒绝清单）不受影响。
- 不新增错误码：诊断 JSON 是编译**通过**时的输出形态，拒绝路径仍由既有五段承载（本 change 仅定义形态分离义务）。
- 无 BREAKING。

## 非目标

- **HAL 能力描述字段语义**：`target` 字段仅承诺为稳定标识符，其取值登记表与 HAL 能力报告（支持 MMA 形状列表等）归 HAL 域后续 change。
- **`--profile` 运行期模式**：运行期 profiling（Nsight Compute / Ascend Insight 调取）属工具层，非编译诊断输出；其行为契约不在本 change。
- **建议码完整清单**：本 change 只规格化建议条目结构（`code` 格式约束、`src_line`、`msg`）与非强制性；`S0xxx` 逐条语义随实现逐步登记（后续 change）。
- **诊断准确率验收数字**（如 bank conflict 静态判定与 Nsight 一致率 ≥ 90%）：属里程碑验收（`veps` 层），不写入行为 spec。
- **agent 优化循环**（`veps/design.md` §3.3）：随 1.0 发布的工具，非语言行为契约。
- **后端实现机制**：字段值的具体来源（`ptxas -v` / CANN 编译日志解析）是实现设计；本 change 只规格化行为侧（字段集、精度等级、确定性、标注义务）。
- **诊断 JSON Schema 的机器可读形态**（JSON Schema 定义文件本身的发布形态与版本化仓库位置）：属工具链交付，本 change 规格化 schema 的行为语义。

## Impact

- 承接让渡（既有 specs 共六处 `diagnostics/*` 让渡文本、四个承接点，均为承接而非修改）：`numerics/value-semantics` Purpose（诊断 JSON 字段归 `diagnostics/*`）与 cast Requirement（精度损失的诊断呈现归建议域字段）两处；`primitives/compute-ops` dot Requirement 正文两处（mma 选择呈现、pad 三策略诊断字段）与一个 Scenario（tail_masked 诊断字段）；`execution/pipeline-structure` Async Requirement 一处（展开操作定位的诊断字段形式）。
- 确定性编译承诺（`deterministic_hash`）自此获得首个 spec 承载（顶层字段语义 + 非确定字段封闭清单）。
- 既有五段拒绝报告契约零改动（形态分离为新增义务，不触碰各段错误码与四要素定义）。
