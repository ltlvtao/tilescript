# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-execution-pipeline-structure --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出 `Change 'add-execution-pipeline-structure' is valid`（2026-09-30 复跑；同轮 `npx openspec validate --all --strict` 4 passed 0 failed，设计审查者 round 3 亦独立复跑一致）
- [x] 核对执行结构契约充分性：以 `veps/design.md` §7 FlashAttention 案例的全部执行结构用法（Pipeline 构造、produce 签名、consume 签名、run 调用、buf.K/buf.V 属性访问、range、cdiv、block_idx——八类）逐项映射到本 change 的 Requirement，合法路径必须全部覆盖、拒绝规则不得命中；并核对 warp_group 的 E0501 语义与 §2.3 事实一致
  - 来源：design「修改方案」充分性标准；验证：无法自动化的 code review 检查点——列出案例源码每个执行结构用法并映射到 Requirement 条目，形成逐项映射记录（与 `veps/m1-type-usage-mapping.md`、`veps/m1-primitive-usage-mapping.md` 同类，落 `veps/`）。注意：`init=AttnState(…)` 的外层绑定经 MODIFIED 豁免 E0303，但构造调用**内部**的关键字实参绑定仍走 type-system 等价规则——两层分开核对
  - 验证结果：`veps/m1-execution-usage-mapping.md` §1 九项逐层映射（八类用法全覆盖、零 E05xx 命中）、§2 init 两层分开核对（外层豁免/内层 E0303 路径归属正确，内层类型核对受数值语义延期约束为形态级）、§3 §2.3/§2.5 事实核对（E0501 原语义不变、warp_group 示例合法、BufferSlot 不一致已裁决、半自动原则承接）
- [x] 核对 E0303 豁免二次扩展的影响面与错误码段位：`pipe.run`/`range` 实参移出 E0303 适用集后 §7 用法无拒绝命中；`E0501`–`E0505` 不与 `E0101`–`E0107`/`E0301`–`E0304`/`E0402`–`E0407` 重叠，跨段管线（E01xx→E03xx→E04xx→E05xx）与段内 tiebreak（E0501→…→E0505）与 design 一致
  - 来源：design「GAP 分析」与「修改方案」；验证：错误码与豁免影响比对记录（并入充分性映射记录）
  - 验证结果：`veps/m1-execution-usage-mapping.md` §4（run/range 实参豁免后零命中、构造内部与赋值保留面不受影响、无存量受众）与 §5（四段错误码无重叠、语法层形态先决与 E05xx 语义契约互补无双解、跨段与段内 tiebreak 与 design 一致）
