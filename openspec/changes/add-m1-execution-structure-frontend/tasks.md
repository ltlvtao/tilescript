# Tasks: add-m1-execution-structure-frontend

约定：每个 task 先写表达验收语义的失败测试（TDD），实现到通过后以实际验证证据勾选；只改 active change 与生产代码。依赖序：1.x（HAL 扩展）→ 2.x（形态环境与遍历基座）→ 3.x–8.x（各错误码面，按序）→ 9.x（E0506 + 管线接入 + CLI）→ 10.x（集成回归）→ 11.x（流程闭环）。

## 1. HAL 能力加载扩展

- [x] 1.1 `tilescript/hal.py` 加载 `persistent_kernel` 字段（Capability 冻结 dataclass 加必选字段；两目标登记值 h200=true / ascend=false，与 hal spec「能力描述字段集」登记值逐字段一致）——测试断言两目标登记值与默认行为——验证：`tests/test_hal.py` 增补（登记值断言；既有 8 测试零回归）——已执行：`.venv/bin/python -m pytest tests/test_hal.py -q` → 9 passed（新增 1 + 既有 8 零回归；先红 AttributeError 后绿）

## 2. execution 基座：形态环境与遍历骨架

- [x] 2.1 `execution/shape_env.py` 形态枚举与归约（design D3）：注解→形态（Tensor 按 scope/int/comptime/dtype 名/状态类）、赋值源形态规则（alloc_shared/make_tensor/zeros/full/dot/cast·reduce·maximum·exp·log·transpose 转发/block_idx/cdiv 两 comptime 则 comptime/Pipeline/range/状态类构造/纯名转发；其余 UNKNOWN）、buffers 值三分（确定 Shared 接受/确定非 Shared 拒/UNKNOWN 让渡）——验证：`tests/test_execution_shape_env.py`（形态归约各支 + 三分面）——已执行：`.venv/bin/python -m pytest tests/test_execution_shape_env.py -q` → 11 passed（先红 ModuleNotFoundError 后绿；修复一次形态枚举名遮蔽 types 常量的缺陷）
- [x] 2.2 `execution/__init__.py` check_module 骨架 + `pipeline_ops` 遍历骨架（design D2：单遍语句流、kernel→produce/consume 结构、快照恢复、warp_group 语境跟踪位；Rejection stage="execution-structure"、段内 order 1–6；finalize 收口）——测试断言公共行为：合法模块零拒绝、扫描不触既有段（同输入三段结果不变）——验证：`tests/test_execution_ops.py` 骨架用例（合法 FLASH 片段零拒绝）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py -q` → 2 passed（先红 collection error 后绿；创建 pipeline_ops/warp_group/builtins/entry_hal 四域骨架 + shape_env.params_of；骨架检查判定留空随 3.x–9.x 填充；全量 345 passed 零回归；修正 _FLASH_LIKE 测试源两处维度名缺陷 BR/BC）

## 3. E0502 Pipeline 构造与值封闭（构造 4S）

- [x] 3.1 构造参数集与值域（R Scenario 1/2/3）：合法构造接受（stages comptime、buffers Shared Tensor 字典）；stages 形态/值域违规（0、运行期名、字面量缺失）拒；buffers 值 Register/Global/非 tensor 形态拒（建议 tis.alloc_shared 产物）；位置实参/未知关键字/缺失必选拒；重复键/空字典拒；同 kernel 第二实例拒（M1 封闭）——验证：`tests/test_execution_ops.py` TestPipelineCtor（≥8 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestPipelineCtor -q` → 14 passed（先红 13/14 后绿；含报告契约同码合并例——多实参违规合并一条列全部；全量 359 passed 零回归）
- [x] 3.2 构造位置与 Pipeline 值使用封闭（R Scenario 4）：produce/consume 体内构造拒；PIPELINE 名引用位置封闭（装饰器接收者/run 接收者/首次绑定外任一使用拒——load 实参、算术、条件、return、二次绑定）；非 Pipeline 对象访问 run/produce/consume 属性拒——验证：TestPipelineValue（≥6 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestPipelineValue -q` → 9 passed（先红 9/9 后绿；run 调用接收者/装饰器接收者/成员值取出三路 member-receiver + 五种值使用位 escape；全量 368 passed 零回归）

## 4. E0502 produce/consume 签名与调用封闭（5S）

- [x] 4.1 签名契约（R Scenario 1/2/3）：合法两/三形参接受；consume 缺返回注解拒；返回注解与状态形参不同类拒（nominal）；produce/consume 形参数/注解形态违规拒——验证：TestProduceConsumeSig（≥6 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestProduceConsumeSig -q` → 12 passed（先红 11/12 后绿；produce 两形参/consume 三形参与 nominal 同类返回逐位裁决；全量 380 passed 零回归）
- [x] 4.2 调用封闭（R Scenario 4/5）：同实例重复装饰拒；嵌套函数显式调用（`fetch(0, K_s)`）拒；名作值（赋值源/实参/return）拒——验证：TestNestedFnClosure（≥4 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestNestedFnClosure -q` → 6 passed（先红 6/6 后绿；dup-decorate 于装饰登记时判定，nested-fn-value 于 Name 值位判定；全量 386 passed 零回归）

## 5. E0502 run 调用契约（4S 直测 + 空迭代间接）

- [x] 5.1 run 参数与位置（R Scenario 1/2/3/4）：合法 run 接受（range 第一实参 + init 同类状态）；第一实参非 range 拒；init 缺失/非状态类/与 consume 注解不同类拒；run 在嵌套体内拒；同码合并（第一实参与 init 同时违规→一条列全部）——验证：TestRunContract（≥7 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestRunContract -q` → 8 passed（先红 7/8 后绿；修正 enter_nested 嵌套环境语义缺陷——外层名可见/体内绑定不外泄，与 primitives/traverse 的 kernel_env 拷贝同构；非 Pipeline 接收者时调用契约让渡只报 member-receiver；全量 394 passed 零回归）
- [x] 5.2 run 单次性与前置装饰（R「每实例恰一次/调用前已装饰」+ 空迭代 Scenario 间接承载）：零 run/多 run 拒（收尾判定）；run 先于装饰拒；合法调用（含空迭代形态源码——`range` 实参为运行期值）零拒绝（空迭代结果=init 为运行时承诺，静态面间接承载）——验证：TestRunLifecycle（≥4 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestRunLifecycle -q` → 5 passed（先红 4/5 后绿；零 run 收尾报构造行/多 run 即时报第二处/前置装饰缺项列名；收尾判定生效联动修正测试模板为合法全链——构造违规例因实例不登记而保持面隔离；装饰登记以接收者为 Pipeline 值为前提；全量 399 passed 零回归）

## 6. E0503 buffer 命名空间与 Async 静态面（3S + 让渡负例）

- [x] 6.1 键集与逃逸（R Scenario 1/2/3）：`buf.<注册名>` 接受（类型面无静态输出差异——键集承载）；未注册名拒（报文列已注册名登记序）；buffer 形参整对象逃逸（赋值源/实参/return）拒——验证：TestBufferNamespace（≥5 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestBufferNamespace -q` → 6 passed（先红 5/6 后绿；未注册名报文按构造键登记序列出；buf.<名> → SHARED_TENSOR）
- [x] 6.2 Async 静态面（R「手写组管理不可达」+「consume 可见」间接）：`tis.wait_group(0)` 让渡零拒绝（未定义符号——spec 让渡，负例固定）；`group=` 已由 E0406 拒（先例不重复报——原语段短路后执行段不跑的联动负例）；合法 produce/consume Async 配对零拒绝（完成保证运行时承诺间接承载）——验证：TestAsyncStaticFace（≥3 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestAsyncStaticFace -q` → 3 passed（让渡面骨架即绿 + 原语段 E0406 联动断言；全量 408 passed 零回归）

## 7. E0501/E0504 warp_group（4S + 汇合间接 + compute-ops 类别 2S）

- [x] 7.1 E0504 参数与位置（R Scenario「role 值域」等）：role 值域外拒；warps 形态/值域拒；调用非 with 上下文表达式位置拒；with 绑定名非 sync 实参位使用拒；warp_group_sync 契约（同名/非绑定名/barrier_id 违规/非表达式语句/嵌套体内/先于 with 语句拒；合法配对接受）——验证：TestWarpGroupContract（≥9 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_warp.py::TestWarpGroupContract -q` → 15 passed（先红 10/15 后绿；E0504 同码合并；绑定名 sync 实参位特判；修复 warp_bindings 登记遗漏与无装饰嵌套函数 in_nested 置位两处缺陷）
- [x] 7.2 E0501 语境（R Scenario「producer dot」「顶层 WarpGroup barrier」+ compute-ops R6 2S）：producer 体内六计算原语任一拒（dot + reduce 各一——类别清单涵盖性）；consumer 体内计算原语接受；顶层 `barrier(scope=WarpGroup)` 拒（建议 warp_group 体内或 Block）；warp_group 体内接受；producer 体内三内存原语接受；汇合可见为运行时承诺间接承载（合法配对零拒绝）——验证：TestWarpGroupContext（≥7 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_warp.py::TestWarpGroupContext -q` → 7 passed（COMPUTE_PRIMS 公共名判定面；全量 430 passed 零回归）

## 8. E0505 索引与整数内建（4S）

- [x] 8.1 block_idx/cdiv/range 参数（R Scenario 1/2/3）：block_idx comptime 非负（负字面量/运行期名拒、comptime 名让渡）；cdiv 恰两实参 int/comptime 种类（Tensor/浮点拒）、`b` 字面量 0 拒（comptime 名让渡）、结果种类推导（两 comptime→comptime，供 range/再绑定消费）；range 恰一实参 int/comptime——验证：TestIndexBuiltins（≥8 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestIndexBuiltins -q` → 12 passed（先红 9/12 后绿；_int_kind 种类裁决 comptime/int/bad/让渡四分；结果种类经 block_idx 消费链两例验证——两 comptime 零拒绝 vs 运行期链 E0505；修复 -1 为 UnaryOp 而非 Constant 的字面量形态遗漏；全量 447 passed 零回归）
- [x] 8.2 range 值封闭（R Scenario 4，E0502 承载）：`x = range(10)` 绑定后运算/原语实参拒；直接调用形态 `tis.load(range(10), y)` 位拒；合法位（for 可迭代/run 第一实参）接受——验证：TestRangeValueClosure（≥4 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_ops.py::TestRangeValueClosure -q` → 5 passed（先红 3/5 后绿；_name_use_check 加 RANGE_VALUE 分支（绑定名逃逸）+ check_range_use 值位调用拒；for 可迭代合法位由语法段 L94 裁决「仅为 range(...)/tis.tile_iter() 调用」承载——绑定名作可迭代属其余使用）

## 9. E0506 入口 HAL 支持面 + 管线接入 + CLI 联动

- [x] 9.1 E0506（hal spec「入口 HAL 支持面」3S）：ascend 拒 `@tis.persistent_kernel`（四要素——装饰器行位置、建议 tis.kernel 替代或换目标）；h200 接受同一入口（E0506 零命中）；装饰器行同位置语法段拒绝时只报一条（短路承载）——验证：TestEntryHal（≥3 例）——已执行：`.venv/bin/python -m pytest tests/test_execution_entry.py -q` → 4 passed（先红 2/4 后绿；ascend 四要素（装饰器行 line=3）/h200 零命中/短路承载（语法段非空执行段不跑，装饰器行恰一条 E0103）/重复编译一致；实现 entry_hal.scan 消费 hal.capability(target).persistent_kernel，扫描面模块顶层函数，__init__ 统一 _reject 构造；全量 451 passed 零回归）
- [x] 9.2 `pipeline.py` 第四段接入 + 报告契约（delta R 5S）：`IMPLEMENTED_STAGES` 四段；原语段非空短路执行段（delta 新 Scenario 公共管线直测——E04xx 存在则 E05xx 不出现）；多条 E05xx 位置升序；段内序（同位置 E0501 先于 E0502 等）；重复编译逐条一致——验证：`tests/test_pipeline.py` 增补 + TestReportContract（≥5 例）——已执行：`.venv/bin/python -m pytest tests/test_pipeline.py -q` → 18 passed（先红 3/18 后绿；接入面 2 + 短路 3（E0406/E0303 短路 + 同调用 E0407>E0502）+ 报告契约 3（两码升序/同码合并/重复一致）+ 段清单四段；联动修正类型段载体测试 6 例——显式调用与 Pipeline 缺参载体形态现归 E0502 执行段承载，`_nested`/`_pipeline` helper 改类型段直测保持焦点，test_explicit_call_legality 翻转为段间分工双断言（类型段不报 + 四段管线 E0502 nested-fn-value））
- [x] 9.3 `cli` 快照四段（cli delta Scenario 1）——`tests/test_cli.py`/`tests/test_integration.py` 断言联动更新（四段/pending 仅 numerics）——验证：联动测试更新后全量通过——已执行：CLI 段清单消费 pipeline.IMPLEMENTED_STAGES（唯一来源，代码零改动）；两处断言更新四段 + pending 仅 numerics；`.venv/bin/python -m pytest -q` → 459 passed 零回归（FLASH_ATTENTION 四段零拒绝由既有 test_three_stages_zero_rejections 顺带承载）

## 10. 集成回归与收口

- [x] 10.1 FLASH_ATTENTION 四段零拒绝回归（Pipeline 构造/produce/consume/run/cdiv/range/block_idx 全套合法面、buf.*/st.*/pipe.run 让渡链维持）；E0506 注入面（FLASH + persistent_kernel 装饰两目标分化）；两目标语言层一致性（同源 E05xx 语言层逐条一致 + E0506 按目标分化）——验证：`tests/test_integration.py` TestFlashAttentionFourStages + TestCrossTargetE0506（≥3 例）——已执行：`.venv/bin/python -m pytest tests/test_integration.py -q` → 14 passed（TestFlashAttentionThreeStages 更名 FourStages 承载四段零拒绝回归；TestCrossTargetE0506 三例——入口换装饰 h200 零拒绝/ascend E0506@装饰行、E0502 注入两目标逐条一致、分化合成 ascend=语言层+E0506 升序；修正装饰行号断言 10→7）；全量 462 passed 零回归
- [x] 10.2 `.venv/bin/python -m pytest -q` 全量通过；`npx openspec validate --all --strict` 通过；主智能体读全部 diff → commit 候选代码——已执行：全量 462 passed（22 文件）；strict 10 passed/0 failed；主智能体通读全部 diff（hal.py persistent_kernel 字段/compute_ops COMPUTE_PRIMS 公共名/pipeline.py 四段编排三处既有面小改 + execution/ 五文件新域 + 测试七文件 + change 目录）后 commit

## 11. 流程闭环

- [ ] 11.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）
- [ ] 11.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design「长期基线刷新计划」）
