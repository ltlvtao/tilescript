# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-language-syntax-acceptance-set --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出
  - 验证结果（2026-09-24）：通过，exit 0（主智能体每轮修订后执行 + 设计审查 subagent round 2/3、cycle 2 round 1/2 独立复跑，累计 9 次全通过）
- [x] 核对语句/表达式白名单充分性：以 `veps/design.md` §7 FlashAttention 完整案例与 M1 五个 kernel（MatMul/FlashAttention/LayerNorm/Softmax/Transpose）所需语法逐类别比对，白名单必须全部覆盖、拒绝清单不得命中
  - 来源：design「修改方案」白名单充分性标准；验证：无法自动化的 code review 检查点——列出案例源码使用的每个语法类别并映射到接受集白名单条目，形成逐项映射记录
  - 验证结果（2026-09-24）：逐项映射记录见 `veps/m1-kernel-syntax-mapping.md`——§7 逐构造映射通过（拒绝清单零命中）；其余四 kernel 按算法语义推导，语法需求全部被白名单条目覆盖（`for range`、`if`、比较 `<`、float 常量、变量整数下标五类超出 §7 案例实际使用、由 R5/R6 白名单条目直接承载），无新增类别
- [x] 核对错误码段位与既有事实不冲突：本 change 的 `E0101`–`E0107` 不与 `veps/design.md` 已使用的 `E0301`/`E0402`/`E0403`/`E0501` 重叠，且段位分配规则与 design「修改方案」一致
  - 来源：design「GAP 分析」与「修改方案」段位分配；验证：错误码清单比对记录
  - 验证结果（2026-09-24）：比对记录见 `veps/m1-kernel-syntax-mapping.md` §3——无重叠，"段位=检查阶段"分配原则与既有事实一致
