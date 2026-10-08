## MODIFIED Requirements

### Requirement: reduce 的归约契约

`tis.reduce(x, axis, op, scope=Auto)` 的契约：`x` MUST 为 `Register` scope 的 Tensor（dtype 为封闭 dtype 集合六种之一，任意 shape；跨 scope 输入以 `E0408` 拒绝，恢复建议经显式移动原语对齐）；`axis` MUST 为 `comptime[int]` 且在 `[0, rank(x))` 值域内（类型不符、含运行期 `int` 或越界均以 `E0408` 拒绝）；`op` 值域为封闭二值 `Sum`/`Max`（其他以 `E0408` 拒绝）；`scope` 值域为封闭三值 `Auto`（默认）/`Warp`/`Block`（其他以 `E0408` 拒绝），且显式 `Warp` 或 `Block` MUST 在编译目标 HAL 能力描述的 `reduce_scopes` 支持列表内（字段语义与登记值由 `hal/capability-descriptions` 定义），目标不支持时 MUST 以 `E0408` 拒绝并报告该目标支持的 scope 清单。结果 MUST 为去掉 `axis` 维的 Tensor：dtype 与 `x` 相同、shape 为 `x` 的 shape 删除 `axis` 维、scope 为 `Register`。`scope=Auto` 时编译器 MUST 依 HAL 能力描述确定性地选择 `Warp` 或 `Block` 之一——同一 HAL 目标的选择 MUST 唯一且写进文档。本 capability MUST NOT 承诺跨 HAL 后端的归约结果位一致（归约顺序依实现，数值差异属跨硬件容差——与 `numerics/value-semantics` cast 的静态位一致承诺形成边界对照：类型与去维规则跨硬件一致，数值位模式不承诺）。

#### Scenario: reduce 合法调用去维

- **WHEN** `S` 为 `Tensor[f32, (BR, BC), Register]`，调用 `tis.reduce(S, axis=1, op=Max)`
- **THEN** 结果为 `Tensor[f32, (BR,), Register]`（去掉 axis=1 维，dtype 不变）

#### Scenario: reduce axis 越界被拒绝

- **WHEN** `x` 为二维 Tensor，调用 `tis.reduce(x, axis=2, op=Sum)`
- **THEN** 编译以 `E0408` 拒绝，报告注明 axis 值域为 [0, 2)

#### Scenario: reduce op 非法值被拒绝

- **WHEN** 调用 `tis.reduce(x, axis=0, op=Min)`（`Min` 不在封闭二值内）
- **THEN** 编译以 `E0408` 拒绝，报告注明 op 值域为 `Sum`/`Max`

#### Scenario: reduce scope 非法值被拒绝

- **WHEN** 调用 `tis.reduce(x, axis=0, op=Sum, scope=Cluster)`
- **THEN** 编译以 `E0408` 拒绝，报告注明 scope 值域为 `Auto`/`Warp`/`Block`

#### Scenario: reduce 显式 scope 不在目标支持列表被拒绝

- **WHEN** 编译目标 HAL 能力描述的 `reduce_scopes` 为 `["block"]`（如 `ascend_910b`），调用 `tis.reduce(x, axis=0, op=Sum, scope=Warp)`
- **THEN** 编译以 `E0408` 拒绝，报告注明该目标支持的 scope 清单（列表小写 `block` 对应语言值 `Block`）

#### Scenario: reduce 非 Register 操作数被拒绝

- **WHEN** `x` 为 `Shared` scope Tensor
- **THEN** 编译以 `E0408` 拒绝，恢复建议经显式移动原语对齐后归约

#### Scenario: reduce 整型 dtype 保持

- **WHEN** `x` 为 `Tensor[i32, (BR, BC), Register]`，调用 `tis.reduce(x, axis=1, op=Sum)`
- **THEN** 结果为 `Tensor[i32, (BR,), Register]`（dtype 不变规则对封闭集合六种 dtype 一致）

