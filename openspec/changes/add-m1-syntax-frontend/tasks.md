# Tasks: add-m1-syntax-frontend

## 1. 规格自洽验证

- [x] 1.1 运行 `npx openspec validate add-m1-syntax-frontend --strict` 与 `npx openspec validate --all --strict`，零 ERROR / 零 WARNING（INFO 长度提示与既往同级可接受）
- [x] 1.2 计数核对：grep 实数 `toolchain/cli` 3R/10S、`syntax-acceptance-set` 8R/28S 与 design「Scenario 计数」、proposal What Changes 各处口径一致

## 2. 工程骨架（来源：design D1/D2/D9；验证：pytest 收集发现全部测试文件、pre-push 干跑）

- [x] 2.1 `pyproject.toml`：`requires-python >= 3.10`、零运行时依赖、pytest 开发依赖
- [x] 2.2 `tilescript` 包骨架：`__init__.py`/`__main__.py`/`cli.py`/`hal.py`（登记表 `frozenset`）+ `frontend/` 八模块空接口（`carrier`/`top_level`/`decorators`/`signature`/`statements`/`expressions`/`state_class`/`report`）
- [x] 2.3 `.husky/pre-push` 追加 `python3 -m pytest -q`（pytest 不可导入即阻断并提示安装）；CLAUDE.md 工程约束节登记 test 命令与挂接

## 3. 报告基座 report.py（来源：syntax R8 + toolchain/cli R2；验证：`tests/test_report_contract.py`）

- [x] 3.1 `Rejection` 数据类 + 稳定排序键 `(line, col, 检查器序)` + 同位置最早检查器 tiebreak + 确定性收集序（承载 syntax R8 全部 3 个 Scenario 的机制面）
- [x] 3.2 序列化辅助：`{"status": "rejected", "rejections": [...]}`、条目五字段固定序、`json.dumps` 固定参数（承载 toolchain R2 Scenario 1/2 的输出面）

## 4. 语法段七检查器（TDD：每条 Scenario 先写失败测试再实现；来源：`language/syntax-acceptance-set` 对应 Requirement）

- [x] 4.1 `carrier.py` E0101——R1×4S：`ast.parse(feature_version=(3, 10))` 等价语义、SyntaxError 映射（行列/原始描述/恢复建议）、`module x:` 伪代码、高于基线语法的显式后检查（D4 residual risk 集）、单条特例
- [x] 4.2 `top_level.py` E0102——R2×2S：顶层白名单三类、类别报告与恢复建议
- [x] 4.3 `decorators.py` E0103——R3×3S：已识别集合（state/kernel/persistent_kernel/fused_kernel/produce/consume 属性装饰器按属性名识别）、目标类别匹配
- [ ] 4.4 `signature.py` E0104——R4×4S：注解形式封闭集、comptime 默认值限单 int 字面量或布尔、非 comptime 默认值拒绝
- [x] 4.5 `statements.py` E0105——R5×5S：白名单类别（赋值/增强赋值/表达式/for-range|tile_iter/if/with-warp_group/return/pass/produce|consume 嵌套函数）、拒绝清单 13 类别、子表达式短路规则（L96：白名单语句因子表达式只报 E0106）
- [x] 4.6 `expressions.py` E0106——R6×4S：白名单类别（含元组/字典/字符串的位置约束）、拒绝清单（lambda/推导/f-string/yield/await/星号/海象/三元/链式比较/白名单外运算符）、位置限制类别
- [x] 4.7 `state_class.py` E0107——R7×3S：类体仅带注解无右值字段、方法/右值/无注解/表达式语句拒绝

## 5. CLI（TDD；来源：`toolchain/cli` R1–R3；验证：`tests/test_cli.py` 子进程级断言退出码/stdout/stderr）

- [x] 5.1 R1×5S：`python -m tilescript compile <源文件> --target <标识符>`；缺 `--target`/未登记标识符/缺源文件位置参数/文件不可读 → stderr 一行 + exit 2 + 不读源文件（登记表清单说明仅辖 `--target` 相关拒绝）；非 `.tis` 扩展名不拒绝
- [x] 5.2 R2×3S：存在拒绝 → stdout 单 JSON 对象 + exit 1；同输入重复运行逐字节一致；v1 字段冻结（测试断言字段集恰为五字段）
- [x] 5.3 R3×2S：已实现段零命中 → `status="incomplete"` + `implemented_stages=["syntax"]` + `pending_stages` 其余四段 + exit 0；`passed` 无产出路径（枚举全部输出路径断言）

## 6. 集成与收口验证

- [x] 6.1 集成样例：veps §7 FlashAttention 源码（Scenario L100 实例化；须去除 `module flash_attention:` 文档伪代码行——该行按 `E0101` 拒绝，veps §7 的源码以源文件本身为模块）语法段零拒绝 + 拒绝注入样例（while/列表推导双拒绝排序）
- [x] 6.2 `python3 -m pytest -q` 全量通过；`npx openspec validate --all --strict` 通过；`.husky/pre-push` 实际触发验证
- [x] 6.3 主智能体读全部 diff（实施结果第一读者）→ commit 候选代码

## 7. 流程闭环

- [x] 7.1 设计语义审查（`reviews/design-review.jsonl`，三轮上限）→ H1 人工确认（`reviews/h1.jsonl`）
- [ ] 7.2 代码语义审查（`reviews/code-review.jsonl`，三轮上限）→ H2 人工确认（`reviews/h2.jsonl`）→ push → 归档 + 长期基线刷新（design「长期基线刷新计划」）
