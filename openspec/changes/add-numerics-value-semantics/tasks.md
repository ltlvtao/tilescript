# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-numerics-value-semantics --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出——2026-10-01 实跑 `npx openspec validate add-numerics-value-semantics --strict` 输出 `Change 'add-numerics-value-semantics' is valid`；`npx openspec validate --all --strict` 输出 `Totals: 5 passed, 0 failed (5 items)`（4 stable specs + 本 change，仅 INFO 级长度提示无 ERROR）
- [x] 核对数值语义契约充分性：以 `veps/design.md` §7 FlashAttention 案例的全部数值用法（int 索引算术 `bm*BR`/`(bm+1)*BR`/`j*BC`/`(j+1)*BC`、Tensor-标量 `S * scale`、Tensor-Tensor 同 shape 算术、逐维广播算术含除法 `st.O_acc / st.l[:, None]`、`-inf` 常量绑定、`tis.cdiv(seq_len, BC)` 折叠边界用法——六类）逐项映射到本 change 的 Requirement，合法路径必须全部覆盖、拒绝规则不得命中
  - 来源：design「修改方案」充分性标准；验证：无法自动化的 code review 检查点——列出案例源码每个数值用法并映射到 Requirement 条目，形成逐项映射记录（与 `veps/m1-type-usage-mapping.md`、`veps/m1-primitive-usage-mapping.md`、`veps/m1-execution-usage-mapping.md` 同类，落 `veps/`）。注意：此前两个映射文档中六处"形态级核对——依赖数值语义"标注的路径（cast 派生表达式实参、`init` 内构造实参、store 一维 src 等）在本 change 定义下升级为完整核对并在记录中逐项闭合
  - 验证记录：`veps/m1-numerics-usage-mapping.md` §1——十三项逐用法映射（六类全覆盖：int 索引算术乘法 10 表达式 + 内层加法子表达式 5 个、Tensor-标量、同 shape、广播含除法、-inf 绑定、cdiv 折叠边界；代码审查 round 1 F2 补入原漏列的 L515 加法、F3 补计数口径），零 E0601–E0606 命中；§2——六处"形态级核对"标注升级闭合（1 完整闭合、4 规则闭合、1 如实保留形态级——maximum 路径不含运算符表达式不属本 change 承接面）
- [x] 核对五处让渡闭合与错误码段位：type-system/memory-ops/execution 的让渡句逐处对照本 change 承接条款；`E0601`–`E0606` 不与 `E0101`–`E0107`/`E0301`–`E0304`/`E0402`–`E0407`/`E0501`–`E0505` 重叠，跨段管线（五段）与段内 tiebreak（E0601→…→E0606）与 design 一致
  - 来源：design「GAP 分析」与「修改方案」；验证：让渡闭合与错误码比对记录（并入充分性映射记录）
  - 验证记录：`veps/m1-numerics-usage-mapping.md` §3——五处让渡逐处闭合核对（①type-system L36 字面量绑定→R4；②L150 运算规则→R1/R3/R4；③memory-ops L111 full value→R4；④L145 cast 数值效果与算术实参→R6+R1；⑤execution L161 cdiv 折叠→R5）；§4——错误码段位核对（四既有段与 E06xx 零重叠、五段管线与段内 tiebreak 与 design 一致、E0303 跨段 tiebreak 单列）
