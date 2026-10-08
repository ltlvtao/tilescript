# Tasks

## 1. 规格自身验证（本 change 无代码交付）

- [x] 运行 `npx openspec validate add-diagnostics-json-schema --strict` 与 `npx openspec validate --all --strict`，预期通过
  - 来源：design「修改方案」；验证：实际命令输出——2026-10-08 实跑输出 `Change 'add-diagnostics-json-schema' is valid`；`npx openspec validate --all --strict` 输出 `Totals: 7 passed, 0 failed (7 items)`（6 stable specs + 本 change；设计审查两轮各自独立复跑一致）
- [x] 核对字段集与精度定级充分性：以 `veps/design.md` §3.1（v1 示例全部字段）、§3.2（精度表逐行）、§2.4（PadPolicy 三诊断字段）、L153/L211/L362（mma 呈现/origin 标签/portability 建议）为对照全集，逐字段映射到本 change Requirement——示例字段零缺漏、精度定级逐字段一致（含 `wait_coverage`→`estimated_wait_coverage` 规范化裁决的登记）、三处让渡字段（compute-ops mma 呈现与 pad 策略、execution origin、numerics 建议域）逐处落地
  - 来源：design「GAP 分析」与「修改方案」D3–D8；验证：映射记录落 `veps/`（与 `veps/m1-*-usage-mapping.md` 同类）
  - 验证记录：`veps/m1-diagnostics-usage-mapping.md` §1——§3.1 全字段 + §3.2/§2.4/L153/L211/L362 补充字段零缺漏映射（唯一字段名修正 wait_coverage→estimated_wait_coverage，§6 勘误登记）；§2——精度表七行逐行定级一致；§3——六处让渡文本（四个承接点）逐处落地 + 纯 ADDED 论证；§5——§7 案例诊断域推演（单入口一份、memory 3 条 alloc_shared、async 汇总+展开两类、compute 九条目含 mma 两路径、空段恒在）零冲突
- [x] 核对形态分离与确定性承诺边界：拒绝时 MUST NOT 产出诊断 JSON 与五段拒绝报告契约（E01xx–E06xx）零冲突；非确定字段封闭清单完备性（除 `compile_time_ms` 无其他天然非确定字段）；四处让渡句经本 change 落地后语义依然成立（纯 ADDED 论证）；不新增错误码与既有 E0xxx 段位零重叠
  - 来源：design「修改方案」D1/D2/D10 与「设计范围」；验证：承接核对记录（并入映射记录文档）
  - 验证记录：`veps/m1-diagnostics-usage-mapping.md` §4——六向核对（五段契约逐条零冲突、跨段管线不变、封闭清单完备——v1 字段集内无路径/版本/时间戳类字段、E0xxx 零新增、per-kernel 与 syntax 一致、多目标行为定义且 `--target=all` 零引用）；§3 纯 ADDED 论证（六处文本均域级指向、无字面冲突——对照 compute-ops MODIFIED 先例说明差异）。注：task 描述原文「四处让渡句」系起草时口径，设计审查 round 1 m3 修正为六处文本/四个承接点（design/proposal 已同步），验证按修正后口径覆盖
