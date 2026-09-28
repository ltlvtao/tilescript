# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-primitives-memory-ops --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出——实施阶段终跑 `Change 'add-primitives-memory-ops' is valid`；`--all --strict` 3 passed 0 failed（2026-09-28，设计审查者 round 1–3 亦每轮独立复跑通过）
- [x] 核对原语契约充分性：以 `veps/design.md` §7 FlashAttention 案例的全部原语用法（load 的 Sync 与 Async 两模式、store×2、barrier、make_tensor×5、alloc_shared×3 含 swizzled、zeros×3、full 的 -inf 值、cast×2 含派生表达式）逐项映射到本 change 的 Requirement，合法路径必须全部覆盖、拒绝规则不得命中
  - 来源：design「修改方案」充分性标准；验证：无法自动化的 code review 检查点——逐项映射记录已落 `veps/m1-primitive-usage-mapping.md`（20 条原语用法逐条过 tiebreak 四层检查，十类充分性标准全覆盖、零拒绝规则命中；cast 派生表达式实参按依赖顺序限制做形态级核对）
  - 注意：cast 派生表达式实参（`st.O_acc / st.l[:, None]` 类）的结果类型依赖数值语义 capability，该 capability 定义前此项按形态级核对（设计审查 round 1 minor 6 的依赖顺序裁决）——已执行
- [x] 核对错误码段位与既有事实不冲突：本 change 的 `E0404`–`E0407` 不与 `E0402`/`E0403`（本段既有 MMA 占用）/`E0101`–`E0107`（语法段）/`E0301`–`E0304`（类型段）/`E0501`（执行结构段）重叠，且跨段管线顺序（E01xx→E03xx→E04xx）与段内 tiebreak（E0404→E0405→E0406→E0407）与 design「修改方案」一致
  - 来源：design「GAP 分析」与「修改方案」段位分配；验证：错误码清单比对记录已并入 `veps/m1-primitive-usage-mapping.md` §4（五段全表核对无重叠，管线与 tiebreak 逐句相符）
