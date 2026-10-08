# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-primitives-compute-ops --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出——2026-10-08 实跑输出 `Change 'add-primitives-compute-ops' is valid`；`npx openspec validate --all --strict` 输出 `Totals: 6 passed, 0 failed (6 items)`（5 stable specs + 本 change）
- [x] 核对计算原语契约充分性：以 `veps/design.md` §7 FlashAttention 案例的全部计算原语用法（dot ×2——显式 `mma=tis.MMA(16, 8, 16)`/`pad=tis.PadPolicy.Error` 与全默认、`transpose` ×1、reduce ×2——`op=Max`/`op=Sum`、maximum ×1、exp ×2 与 log ×1——六类）逐项映射到本 change 的 Requirement，合法路径必须全部覆盖、拒绝规则（`E0402`/`E0403`/`E0408`）不得命中；同时核对 `numerics` 既有映射文档中的"计算原语域前提类型"标注逐处落实（dot/reduce/maximum/exp/log 返回类型）与 "`m ← maximum` 形态级"残留的最终闭合
  - 来源：design「修改方案」充分性标准；验证：逐项映射记录落 `veps/`（与 `veps/m1-type-usage-mapping.md`、`veps/m1-primitive-usage-mapping.md`、`veps/m1-execution-usage-mapping.md`、`veps/m1-numerics-usage-mapping.md` 同类）
  - 验证记录：`veps/m1-compute-usage-mapping.md` §1——九项逐调用映射（六类全覆盖，dot 两处覆盖 A 为 Shared 与 Register 两 scope 路径、D/BR/BC=64 对 (16,8,16) 与 Auto 候选形状整除零 E0403），零 E0402/E0403/E0408 命中、E0501 零命中（全部调用在 consumer 与 kernel 顶层）；§2——numerics"计算原语域前提类型"六处落实 + §2 #17/#20"仅 exp/log 契约"残留兑现 + m 行形态级残留最终闭合（maximum 返回类型与字段声明等价可判 E0303）
- [x] 核对承接面闭合与错误码段位：E0402/E0403 语义正式化对照 memory-ops 段位声明（"既有占用，语义不变"）与 veps §2.3 事实逐条一致；计算原语类别清单闭合 E0501"tis.dot 等"；numerics R3 两处指向落位（本 change 裁决逐元素比较仍不引入）；`E0408` 不与既有各段错误码重叠、段内 tiebreak（E0408→E0402→E0403）与跨段五段管线（E01xx→E03xx→E04xx→E05xx→E06xx）与 design 一致；memory-ops MODIFIED delta 完整重述与既有 stable 文本逐字对照（仅段位句变动，其余与全部 Scenario 原样）
  - 来源：design「GAP 分析」与「修改方案」；验证：承接闭合与错误码比对记录（并入充分性映射记录）
  - 验证记录：`veps/m1-compute-usage-mapping.md` §3——六向承接面逐处闭合（E0402/E0403 对照 veps L153-155/L184/L187 逐条、MODIFIED diff 逐字对照唯一差异为段位句、E0501 类别清单、numerics R3 落位、type-system L146 让渡闭合）；§4——五段错误码核对（E0408 零重叠 grep 核实、段内 tiebreak 与跨段管线与 design 一致、E0303 豁免覆盖计算原语实参）
