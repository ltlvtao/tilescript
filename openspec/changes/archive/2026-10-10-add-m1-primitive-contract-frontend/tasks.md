# Tasks: add-m1-primitive-contract-frontend

约定：每个 task 先写表达验收语义的失败测试（TDD），实现到通过后以实际验证证据勾选；只改 active change 与生产代码。依赖序：1.x（HAL 数据）→ 2.x（折叠与结果类型基座）→ 3.x/4.x（存取/计算原语，可并行）→ 5.x（管线接入）→ 6.x（集成回归）。

## 1. HAL 能力描述加载（hal/capability-descriptions 实现面）

- [x] 1.1 `tilescript/hal.py` 扩展：两目标冻结常量（`mma_shapes`/`reduce_scopes` 含登记顺序）、`Auto` 选择（列表第一项）、小写 `warp`/`block` ↔ 语言值映射、`E0402` 报告面文本——测试断言登记值与 hal spec 逐字段一致（hal R「字段集」两目标 Scenario + 「列表顺序即优先序」3S）——验证：`tests/test_hal.py` 8 passed（登记值/Auto/映射/报告面文本）
- [x] 1.2 `pipeline.compile_stages` 签名扩展（`*, target` 必选关键字）与 `cli` 透传——既有测试联动（`--target` 行为零变化，R1/R2 不动）——验证：`.venv/bin/python -m pytest -q` → 197 passed（189 既有 + 8 HAL；15 处测试调用点补 `target="nvidia_h200"`，`--target` 行为零变化）

## 2. 基座：折叠与结果类型

- [x] 2.1 `primitives/fold.py`：Dim → 整数系数符号多项式规范化（`+`/`-`/`*`/一元负；`/` → None；`//`/`%` 双 ConstDim 整数求值，含符号操作数或除数 0 → None；`slice` 维差分）——测试含 `bm*BR:(bm+1)*BR → BR`、`(bm+1)*BC - bm*BC → BC`、`64-0 → 64`、`(128//8,) → 16`、`x//0 → None`、含 `/` 不可折叠——验证：`tests/test_fold.py` 7 passed（六例 + 代数基元/零系数清理）
- [x] 2.2 `primitives/results.py` + shape 实参组件推断：`make_tensor`/`alloc_shared`/`zeros`/`full`/`cast`/`dot`/`reduce`/`maximum`/`exp`/`log`/`transpose` 合法结果类型表（design D4）；违规调用返回 UNKNOWN（拒绝已记录）——验证：`tests/test_results.py` 12 passed（组件推断三态 4 + D4 表 8 行）
- [x] 2.3 `primitives/traverse.py` 设备函数遍历骨架：复用 `typecheck.symbols` env 构建 + `Inferencer`（`tis.*` Call 改走 results）；赋值绑定、循环变量 UNKNOWN、嵌套 produce/consume 进入/退出跟踪；收尾断言 `Inferencer.rejections` 为空（design D3 防御）——测试断言公共行为：合法片段遍历零拒绝、原语结果类型经后续 E0405/E0408 检查间接可观察——验证：`tests/test_traverse.py` 5 passed（改道/env 流动/语境进出/作用域快照恢复/漂移防御断言）

## 3. 存取原语（memory-ops E0404/E0405/E0406/E0407）

- [x] 3.1 E0406 参数集与值域（R1×5 + R5×6 + R6×3 + R7 值域）：八原语未知关键字/位置超量/缺失必选；`mode` Sync/Async；`make_tensor` ptr Global Pointer + shape 整数组件；`alloc_shared` layout RowMajor/swizzled(xor 非负 comptime)；`zeros`/`full` scope Register/Shared、value 标量形态；`cast` x Tensor + dtype 六值；`barrier` scope Block/WarpGroup；load/store 实参非 Tensor（Pointer/标量/状态类）——验证：`tests/test_memory_ops.py` 22 passed（R1 结构含缺失必选/mode 值域、R5 六面、R6 三面、R7 值域、实参类别）
- [x] 3.2 E0404 转移格承载（R2×5）：load/store 承载格集合；类别不匹配建议正确原语；copy/move 无承载；Global→Global 不报（E0301 归类型段——类型段已拒场景经管线短路，直调验证不重复）——验证：`tests/test_memory_ops.py` TestTransferCell 7 passed（load/store 格接受、双向类别不匹配、copy/move 无承载、非法格零 E04xx）
- [x] 3.3 E0405 数据维度（R3×4）：dtype 相同 + 逐维四支（双常量/同符号/派生结构等价/折叠支 (d)）；dtype 差建议 cast；不相容维报告两侧类型与维度——验证：`tests/test_memory_ops.py` TestDataDims 5 passed（Scenario 1 等价构造折叠支贯通、dtype/维度数/不相容维拒、(b)(c) 支接受）
- [x] 3.4 E0407 语境（R4 静态 3S + R7 值位置 1S；运行时 2S 间接承载）：Async 限嵌套 produce/consume 体内（kernel 顶层拒、produce 内接受）；load/store/barrier 值位置（赋值右侧/实参/return）拒绝——验证：`tests/test_memory_ops.py` TestContext 7 passed（Async 顶层拒/produce 内接受/三值位置拒/封闭集外不裁/E0406 先于 E0407）

## 4. 计算原语（compute-ops E0408/E0402/E0403）

- [x] 4.1 E0408 调用结构（R1×5）：六原语参数集；dot mma/pad 关键字专属（位置第 4+ 拒）；缺失必选拒；未知关键字拒；transpose 超量拒——验证：`tests/test_compute_ops.py` TestCallStructure 8 passed（R1 五 Scenario + 位置全形态/关键字形式等价面）
- [x] 4.2 E0408 dot 操作数契约（R2 操作数各 Scenario）：A/B f16 (M,K)/(K,N) Shared|Register、K 相容、C f32 (M,N) Register、shape 常量维（运行期派生维拒；折叠纯常数的运行期切片维接受——`_dim_comptime_known` 固定裁决）、MMA 构造 comptime 三参（运行期 int/float/两参拒）、pad 封闭四值——验证：`tests/test_compute_ops.py` TestDotOperands 13 passed（含 UNKNOWN 让渡/C 形状 (M,N) 相容/rank 面）
- [x] 4.3 E0402/E0403（R2 硬件面各 Scenario）：显式 mma 不在目标列表拒 + 报告全列表按登记顺序（两目标差异）；所选形状（Auto=第一项/显式）不整除 M/N/K 且 pad=Error 拒 + 最近对齐建议（ceil(d/s)*s，D5 窄域仅 ConstDim 参与）；PadZero/Mask/Split 通过；comptime 符号维/显式 comptime 名 MMA 无数值 → E0402/E0403 让渡（负例固定）——验证：`tests/test_compute_ops.py` TestMmaSupportAndAlignment 8 passed
- [x] 4.4 E0408 MMA/PadPolicy 实参语境专用（R2 末 Scenario，design D7 独立扫描）：赋值绑定/其他原语实参/算术操作数/dot 位置实参位拒；合法 mma=/pad= 关键字位零叠加；hit 协调单条——验证：`tests/test_compute_ops.py` TestSpecialValueContext 8 passed
- [x] 4.5 E0408 reduce/maximum/exp/log/transpose（R3×7 + R4×5 + R5×2）：reduce Register/axis comptime 界内（comptime 名让渡）/op 二值/scope 三值 + ascend_910b 支持面拒（报目标支持清单）；maximum 严格同 dtype/shape（相容语义）/Register；exp/log 浮点四 dtype；transpose 恰二维 Shared|Register——验证：`tests/test_compute_ops.py` TestReduce 8 + TestMaximum 5 + TestUnaryMath 3 + TestTranspose 3 passed
- [x] 4.6 段内序与确定性（R7×3）：E0408（pad 形态/操作数面）+ E0402/E0403 同调用只报 E0408；跨 checker 多拒绝位置升序；Auto/显式选择同输入同目标重复编译逐条一致——验证：`tests/test_compute_ops.py` TestStageOrderAndDeterminism 4 passed

## 5. 管线接入与 CLI 联动

- [x] 5.1 `primitives/__init__.py` check_module(tree, target) + `pipeline` 第三段：类型段非空短路（delta R8 新 Scenario 公共管线直测）；三段完整序；IMPLEMENTED_STAGES 三段——验证：`tests/test_pipeline.py` TestPrimitiveStageWiring 4 passed（短路/E0404 上报/三段序/段清单）
- [x] 5.2 `cli` 快照三段（R3 Scenario 1 delta）——test_cli 断言联动更新——验证：`tests/test_cli.py` + `tests/test_integration.py` implemented_stages 三段/pending 两段断言更新，全量 326 passed

## 6. 集成回归与收口

- [x] 6.1 FLASH_ATTENTION 样例升级三段回归：`tis.load` 切片折叠支贯通、`buf.*` 让渡跳过（三段零拒绝）；`tis.dot` 已知面（buf.K → K_s，comptime 符号维 + E0402 命中 + E0403 全符号不触发）非让渡路径零拒绝——验证：`tests/test_integration.py` TestFlashAttentionThreeStages 3 passed
- [x] 6.2 两目标语言层一致性（hal「跨 HAL 行为不变面」Scenario 3）：同源语言层拒绝两目标逐条一致（to_dict 含建议文本）；HAL 依赖拒绝按目标分化（h200 仅 E0406 / ascend 追加 E0408 scope-unsupported 报支持清单）——验证：`tests/test_integration.py` TestCrossTargetConsistency 1 passed
- [x] 6.3 `.venv/bin/python -m pytest -q` 全量通过（328 passed：既有 197 + 本 change 131）；`npx openspec validate --all --strict` 通过（10 items）；主智能体读全部 diff（8 个修改文件 + primitives 五文件 + 六个新测试文件）→ commit 候选代码

## 7. 流程闭环

- [x] 7.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）——验证：cycle 1 三轮闭环（round1 CHANGES_REQUESTED 6 minor + 3 info → round2 CHANGES_REQUESTED 1 minor + 1 info（修订引入一行级）→ round3 PASS 11/11 resolved）；H1 APPROVED 2026-10-10
- [x] 7.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design「长期基线刷新计划」）——验证：cycle 1 两轮闭环（round1 CHANGES_REQUESTED 3 minor + 1 info → 修复 commit a219175 → round2 PASS 全 resolved + 1 info → commit 5d95334 措辞补全）；H2 APPROVED 2026-10-10；push 9e409d0..00ef8da（pre-push strict 10 items + 331 passed）；归档与基线刷新随本 commit
