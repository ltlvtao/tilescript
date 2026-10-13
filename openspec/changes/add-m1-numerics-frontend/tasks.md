# Tasks: add-m1-numerics-frontend

约定：每个 task 先写表达验收语义的失败测试（TDD），实现到通过后以实际验证证据勾选；只改 active change 与生产代码。依赖序：1.x（数值环境基座）→ 2.x–5.x（各错误码面，按 spec R 序）→ 6.x（折叠与 cast）→ 7.x（报告契约 + 管线接入）→ 8.x（CLI + 集成回归）→ 9.x（流程闭环）。

## 1. 数值环境基座（D2/D3）

- [x] 1.1 `numerics/value_env.py` 数值类型环境与表达式推断入口：env 起点 `symbols.kernel_params` + comptime 值表（形参默认值）；`_SilentInferencer` 委托（类型域拒绝零重放）；`tis.*` 返回类型表（复用 `primitives/results` 构造器 + `infer_shape_component`）；kernel/produce/consume 体遍历（快照恢复同构）；赋值绑定/For（range→int、tile_iter→UNKNOWN）/With（UNKNOWN）/AugAssign 等价展开骨架——测试断言公共行为：合法模块（含原语调用与算术赋值）零拒绝、扫描不触既有四段（同输入四段结果不变）——验证：`tests/test_numerics_env.py`（env 基座行为 + 四段零扰动）——已验证：`.venv/bin/python -m pytest tests/test_numerics_env.py -q` 8 passed
- [x] 1.2 段编排 `numerics/__init__.py` `check_module(tree)`（段名 `numerics`、order 1–6、`report.finalize` 收口）+ `binary_ops`/`constants` 域骨架——测试：合法 FLASH 全链片段零拒绝、重复编译一致——验证：`tests/test_numerics_env.py` 增补——已验证：同上（TestStageWiring 4 例全过：段名/无 target 签名/位置排序/四要素）

## 2. R1 算术操作数与结果规则（E0601–E0604，10S）

- [x] 2.1 家族与 dtype/scope 裁决（Scenario 3/4/5/6/7/8）：int 家族种类推导（全 comptime→COMPTIME_INT 带 folding、否则 int）；家族内跨 dtype E0602（同表达式多操作数合并一条列全部）；混家族 E0601（int 字面量/comptime 常量不豁免）；跨 scope E0603（报两侧 scope）；`/` 的 int 家族与整型 dtype E0604（建议 tis.cdiv）——验证：`tests/test_numerics_arith.py` TestFamily（≥10 例）——已验证：31 passed（TestFamily 12 例含 R7 同码合并与不同码各报）
- [x] 2.2 广播派生与 shape 判定（Scenario 1/2/9/10）：标量⊗Tensor 广播；Tensor⊗Tensor 右对齐逐维（相等保留取左、一侧 1 扩展、fold 等价判定——`ConstDim(64)` vs `32*2` 不误报）；不相等且均非 1 E0603（报两侧 shape）；UnknownDim/fold None 让渡；增强赋值等价展开（`S *= 2` → E0601 int 常量操作数；`S *= scale` 合法）——验证：`tests/test_numerics_arith.py` TestBroadcast + TestAugAssign（≥8 例）——已验证：同上（TestBroadcast 6 + TestAugAssign 7）
- [x] 2.3 算术赋值再绑定等价（R7 明文授权面，design D2）：Assign/AugAssign 源为算术/一元负表达式且目标已绑定非 UNKNOWN——等价判定（违规时类别近似类型，复用 type-system equivalent/compatible 语义）；不等价 E0303（stage/type-system 口径、择一机制：同位置双命中只构造 E0303 一条）；单命中各照报（`x = alpha * scale` 后 x:int → 恰一条 E0303；双命中 `x = f32_t * f16_t` → 只报 E0303 不报 E0602——R7 Scenario 2 直测）；非算术源再绑定让渡负例——验证：`tests/test_numerics_arith.py` TestRebinding（≥5 例）——已验证：同上（TestRebinding 6 例）

## 3. R2/R4 待定型常量与 dtype 位置绑定（E0606，12S）

- [x] 3.1 待定型与算术操作数位/裸绑定（R2 Scenario 4/5 + R4 Scenario 4/5）：float 字面量（含 `-1.5` 一元负形态）与 `inf`/`-inf` 待定型；与 dtype 家族运算按其 dtype 定型；与 int 家族 E0601；裸绑定（`x = 1.5`/`x = inf`）E0606；位置③再绑定（已绑 f32 ← 1.5 定型合法；已绑 int ← 1.5 E0606；已绑 f32 ← int 字面量 E0606）；`inf` 整型/f8e4m3 位置 E0606；一元负家族面（`-P` 保持 dtype/shape——R2 Scenario 5；int 家族折叠值取负）——验证：`tests/test_numerics_const.py` TestPending（≥9 例）——已验证：74 passed 全量数值段（TestPending 11 例）
- [x] 3.2 `tis.full` value 位绑定（R4 Scenario 1/2/3/6/7 + R2 Scenario 1/2/3）：int 字面量整型值域（i8 −128–127 越界 E0606 报值域与实际值；浮点 dtype 整到浮合法）；float 字面量浮点 dtype 合法；运行期 int 名 E0606；dtype 标量跨 dtype E0606；`inf`→f32 合法/i8 与 f8e4m3 E0606；`-inf`→f32 value 位合法（R2 Scenario 1——f32 负无穷位模式承诺注记）——验证：`tests/test_numerics_const.py` TestFullValue（≥8 例）——已验证：同上（TestFullValue 14 例）

## 4. R3 比较逻辑与布尔封闭（E0605，4S）

- [x] 4.1 比较与条件（Scenario 1/2/4）：同家族标量比较（int 家族混用合法）→ 布尔；非布尔条件 E0605（`if seq_len:` 建议显式比较）；Tensor 参与比较 E0605（建议计算原语后续 change）；混家族/待定型比较操作数 E0605——验证：`tests/test_numerics_bool.py` TestCompare（≥6 例）——已验证：22 passed（TestCompare 10 例——含 while 同构/UNKNOWN 让渡/待定型比较位）
- [x] 4.2 布尔使用封闭集（Scenario 3 + Requirement 正文）：`and`/`or`/`not` 操作数布尔、结果布尔；封闭集外使用 E0605（赋值绑定/实参/返回/下标/算术操作数/比较操作数/一元负）；UNKNOWN 让渡——验证：`tests/test_numerics_bool.py` TestClosure（≥6 例）——已验证：同上（TestClosure 12 例——赋值/原语实参/状态构造实参/返回/下标/一元负/逻辑运算/not/合法条件负例）

## 5. R5 comptime 折叠与 R6 cast（3S + 6S 间接）

- [x] 5.1 折叠与 cdiv 值（Scenario 1/2/3）：comptime 值表（形参默认值）；全 comptime `+ - *`/一元负折叠值确定；cdiv 两 comptime → COMPTIME_INT 值 `ceil(a/b)`、非全 comptime → int；R5×R4 连接（`BC * 4 + 68` 折叠 324 绑 i8 value 位 → E0606；`BC * 2 - 1` 绑 i32 合法）；`b` 值 0 让渡不报（E0505 已拒）——验证：`tests/test_numerics_const.py` TestFold（≥5 例）——已验证：同上（TestFold 6 例；修复 fold_value 查 const_values 侧表——cdiv/已折叠子表达式经节点取值）
- [x] 5.2 cast 间接承载（R6 六 Scenario）：`cast` 返回类型派生（dtype 换目标、shape/scope 承接——参与后续算术断言）；合法 cast 零拒绝负例；运行期承诺不上卡（spec 引用注记）——验证：`tests/test_numerics_const.py` TestCast（≥3 例）——已验证：同上（TestCast 4 例）

## 6. R7 报告契约 + 管线第五段接入（delta 5S）

- [ ] 6.1 数值段报告契约：多条 E06xx 位置升序；段内 order（同位置 E0601 先于 E0606）；同码合并；重复编译逐条一致；同位置跨段双命中只报最早段（E0303×E06xx 双命中由 2.3 择一机制承载——`x = f32_t * f16_t`（x:int）恰一条 E0303）——验证：`tests/test_numerics_env.py` TestReportContract + `tests/test_pipeline.py` 增补（≥5 例）——已验证：`.venv/bin/python -m pytest tests/test_numerics_env.py tests/test_pipeline.py -q` 37 passed（TestReportContract 5 例——位置主导排序/同位 order tiebreak 直测 finalize/同码合并/重复一致/择一 E0303 stage 口径；pipeline 增补 TestNumericsReportContract 1 例）
- [ ] 6.2 `pipeline.py` 第五段接入 + 段间短路（delta 新 Scenario）：`IMPLEMENTED_STAGES` 五段；原语/执行段非空短路数值段（E04xx/E05xx 存在则 E06xx 不出现）；`check_module(tree)` 无 target——验证：`tests/test_pipeline.py` TestNumericsStageIntegration + TestShortCircuit（≥4 例）——已验证：`.venv/bin/python -m pytest tests/test_pipeline.py -q` 23 passed（TestNumericsStageIntegration 3 + TestNumericsShortCircuit 2；短路构造修正——类型段 E0303 须注解形参目标，tis.zeros 局部在类型段 UNKNOWN 不报）

## 7. CLI passed 联动 + 集成回归

- [x] 7.1 CLI passed（cli delta 3S）：五段零命中 → `{"status": "passed"}` 单字段退出码 0；重复运行逐字节一致；拒绝优先（E06xx → rejected 退出码 1）；既有 incomplete 断言联动翻转——验证：`tests/test_cli.py` 增补与联动更新（≥4 例）——已验证：`.venv/bin/python -m pytest tests/test_cli.py -q` 16 passed（TestR3HonestExit 5 例：passed 最小形态/逐字节一致/E0606 拒绝优先/monkeypatch 过渡期 incomplete/status 封闭值；联动翻转 3 处——扩展名/子进程/status 断言）
- [x] 7.2 FLASH_ATTENTION 五段零拒绝 → passed 回归（全合法面含算术/比较/常量）；E06xx 注入面；两目标语言层一致性（E06xx 无目标分化——`check_module` 无 target 的行为注记）——验证：`tests/test_integration.py` 增补与联动更新（≥3 例）——已验证：`.venv/bin/python -m pytest tests/test_integration.py -q` 16 passed（TestFlashAttentionFiveStages 更名+CLI passed 翻转、TestNumericsInjection 2 例——注入恰一条 E0602 + 两目标逐条一致）；修复两处实现缺陷——compute 六原语 `_bind_args` 分派（memory/compute 两套 _PRIM_SPECS）与 maximum 槽名 a/b；再绑定 env 语义（让渡路径改写为源类型，拒绝路径保旧——`x = 5` 后 `x + 16` 不按陈旧 f32 误报）
- [x] 7.3 `.venv/bin/python -m pytest -q` 全量通过；`npx openspec validate --all --strict` 通过；主智能体读全部 diff → commit 候选代码——已验证：579 passed + strict 10/10；全部 diff 已读（cli/pipeline 生产面最小改动；numerics 包四文件编译检查 + 未用 import 清理；测试五文件）；commit d17b041（23 文件，+2794/−53）

## 8. 流程闭环

- [x] 8.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）
- [ ] 8.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design「长期基线刷新计划」）——cycle 1 round 1 CHANGES_REQUESTED（4 findings：blocker AugAssign 值双重推断 / major 待定型常量算术操作数位语境缺失 / minor comptime[int] 布尔默认值不入值表 / minor 私有 _COMPUTE_PRIMS 复制）已全部修复：`_aug_assign` 分支互斥 + 值表达式开帧（`framed` 公开化）、binary_ops 判定序 8 步 + pending 语境 E0606 分支、constants 收 bool 默认值 + full_value_verdict 走 report 进帧合并（R7 同表达式同码合并不排除 E0606——实现口径修正）、value_env 消费公共 COMPUTE_PRIMS；回归测试 +11 例（TestPendingOperandContext 8 例、TestAugAssign 2 例、TestFold 1 例）+ 联动翻转 1 处——已验证：`.venv/bin/python -m pytest -q` 590 passed + strict 10/10
