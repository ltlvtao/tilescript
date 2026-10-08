# Tasks: add-hal-capability-descriptions

## 1. 规格自洽验证

- [x] 1.1 运行 `npx openspec validate add-hal-capability-descriptions --strict` 与 `npx openspec validate --all --strict`，零 ERROR / 零 WARNING
- [x] 1.2 计数核对：grep 实数 Requirement/Scenario 与 design「Scenario 计数」、proposal What Changes 各处口径一致

## 2. 字段集与登记值对照（veps §4.1 零缺漏映射）

- [x] 2.1 产出 `veps/m1-hal-capability-mapping.md`：§4.1 两目标 YAML 全字段逐项映射到 HR3（含缺席裁决：`ascend_910b` 的 `registers_per_sm` 可选缺席、`ub_bytes` 随 Placement/UB 延期）
- [x] 2.2 既有 specs HAL 消费面逐处落地核对：design「当前实现」清单 7 处（compute-ops×3、diagnostics×2、memory-ops×1、execution 段位句×1）逐处标注落点 Requirement

## 3. MODIFIED 逐字对照（diff 证据）

- [x] 3.1 execution delta 与 stable 原文（L183–207）diff：唯一差异为段位句一处
- [x] 3.2 compute-ops delta 与 stable 原文（L92–125）diff：唯一差异为 scope 值域句扩展 + 新增 1 个 Scenario

## 4. 管线与错误码无冲突核对

- [x] 4.1 `E0506`：段位归属（execution E05xx）、段内顺位（最后）、与最早段规则交互（HR5 Scenario 3）、五段管线顺序不变
- [x] 4.2 `E0408` 值域扩展：与既有 E0408 命中面（scope 非法值/axis/op/跨 scope）无重叠歧义；compute-ops E04xx 段内 tiebreak（`E0408`→`E0402`→`E0403`）不受影响

## 5. 流程闭环

- [x] 5.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）
- [x] 5.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design 末节）
