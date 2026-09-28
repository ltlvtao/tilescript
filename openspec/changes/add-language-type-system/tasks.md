# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-language-type-system --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出
- [x] 核对类型规则充分性：以 `veps/design.md` §7 FlashAttention 案例的全部类型用法（AttnState 三字段、五个 Pointer 签名参数、int/f32/comptime[int] 标量、make_tensor 动态 shape、zeros/full 的 Register Tensor、状态构造与 pipe.run(init=...) 实参绑定）逐项映射到本 change 的 Requirement，白名单必须全部覆盖、拒绝规则不得命中
  - 来源：design「修改方案」充分性标准；验证：无法自动化的 code review 检查点——列出案例源码每个类型用法并映射到 Requirement 条目，形成逐项映射记录（与 `veps/m1-kernel-syntax-mapping.md` 同类，落 `veps/`）
- [x] 核对错误码段位与既有事实不冲突：本 change 的 `E0302`–`E0304` 不与 `E0301`（本段保留）/`E0101`–`E0107`（语法段）/`E0402`/`E0403`/`E0501` 重叠，且段位分配规则与 design「修改方案」一致
  - 来源：design「GAP 分析」与「修改方案」段位分配；验证：错误码清单比对记录（并入充分性映射记录）
