# Design: add-m1-syntax-frontend

## 设计范围导航

| 文件/目录 | 操作 | 内容 |
|---|---|---|
| `specs/toolchain/cli/spec.md` | ADDED | 3 Requirement / 10 Scenario：编译入口与目标参数、拒绝清单 JSON 序列化、分段实现状态与诚实退出（工具链域首个 capability） |
| `pyproject.toml` | 新增 | `requires-python >= 3.10`、零运行时依赖、pytest 开发依赖 |
| `tilescript/`（包） | 新增 | `__main__.py`/`cli.py`（入口与序列化）、`hal.py`（登记表标识符集）、`frontend/`（carrier/top_level/decorators/signature/statements/expressions/state_class/report 八模块） |
| `tests/` | 新增 | 语法段 8 Requirement 的 Scenario 逐条行为测试 + toolchain/cli 10 Scenario + 确定性/集成测试 |
| `.husky/pre-push` | 修改 | 追加 `python3 -m pytest -q`（openspec strict 之后；pytest 缺失即阻断并提示安装） |
| `CLAUDE.md` | 修改 | 工程约束节登记 test 命令与门禁挂接 |
| `proposal.md` / `tasks.md` / `reviews/` | — | 五件套其余 |

实现对象 `language/syntax-acceptance-set`（8 Requirement / 28 Scenario）零 spec delta——本 change 是既有契约的实现者，唯一新规格为 `toolchain/cli`。

## 当前实现（事实基线）

- 零生产代码、无 Python 基建；开发环境 Python 3.14.4（`requires-python >= 3.10` 与语言语法基线一致）。
- 行为契约来源：`language/syntax-acceptance-set` 全文（L11 直接引用 `ast.parse(..., feature_version=(3, 10))` 等价语义）；`hal/capability-descriptions` HR2 登记表封闭二成员 `{nvidia_h200, ascend_910b}`（`--target` 校验的唯一数据）。
- `veps` §5.2 M1 定义（Parser/类型检查/TSR/作用域检查——本 change 为其语法片）；§7 FlashAttention 案例（28 个 Scenario 之一 L100 引用，兼作集成样例）。
- `.husky/pre-push` 现仅 openspec strict。

## GAP 分析

| 缺口 | 落点 |
|---|---|
| 语法段规格冻结 vs 零实现 | `frontend/` 七检查器 + `report.py`（tasks 2–9） |
| 工具链输出面（入口/序列化/退出码）无规格 | `toolchain/cli` ADDED（3R/10S）→ `cli.py` |
| 无 build/test 基建 | pyproject + pytest + pre-push 挂接 + CLAUDE.md 更新 |
| `--target` 校验无载体 | `hal.py` 内置登记表标识符集（YAML 加载延期，见 D2） |

## 修改方案（裁决记录）

**D1 宿主语言 Python 3.10+、零三方运行时依赖**。veps M1 明示「Parser(Python AST)」，syntax spec 以 `ast.parse(feature_version=(3, 10))` 为接受范围定义——标准库 `ast` 是唯一自然载体；零依赖缩小确定性输入面与供应链面。运行时依赖仅标准库（`ast`/`argparse`/`json`/`sys`）；pytest 为开发依赖。

**D2 包结构**。`tilescript/cli.py`（入口 + JSON 序列化 + 工具链入口拒绝）、`tilescript/hal.py`（登记表 `frozenset`——仅标识符集，YAML 载体与字段加载随首个消费段 change）、`tilescript/frontend/` 每 Requirement 一模块：`carrier.py`（E0101）、`top_level.py`（E0102）、`decorators.py`（E0103）、`signature.py`（E0104）、`statements.py`（E0105）、`expressions.py`（E0106）、`state_class.py`（E0107）、`report.py`（Rejection 数据类 + 收集/排序/tiebreak/序列化辅助）。模块边界=错误码边界，与 spec Requirement 一一对应。

**D3 检查器执行序与短路规则**。检查器按错误码数字序执行（E0101→E0107）；「同位置命中多个检查阶段只报最早」的段内实例化为：同 `(line, col)` 多命中取检查器序最早一条。E0105/E0106 的两面：spec L96 **明文**规定白名单语句的子表达式被拒时按 E0106 报、不另报 E0105；而「语句类别被 E0105 拒绝时短路——不再深入其子表达式报 E0106」是 spec 沉默处的 **design 裁决**（不短路则如 `while a // b:` 会产出 E0105+E0106 两条噪声，tiebreak 规则只辖同位置、不辖此情形），裁决理由：被拒语句整体的子表达式检查无恢复价值，且方向与 L177-180「最早检查阶段优先」一致。E0101 无 AST，单条立即返回（spec L170 特例）。跨段 tiebreak（`E01xx` vs 更晚段同位置）本 change 只有语法段，`report.py` 的稳定排序键预留段序字段，机制随后续段实现自然生效。

**D4 解析与「高于基线语法」**。`ast.parse(source, filename=..., feature_version=(3, 10))`，`SyntaxError` 映射 `E0101`（`lineno`/`offset`/`msg` 直通报告——「原始解析错误描述」）。`module x:` 伪代码天然落 SyntaxError 路径。已知限制：`feature_version` 对新语法的拒绝不完整（官方语义为「尽量一致」），补充已知 3.11+ 构造的显式后检查——后检查以实现内**冻结节点表**承载（如 `ast.TryStar`、3.12 类型参数语句节点，与 D6 slug 冻结表同纪律），表内每个节点配一条回归测试锁定；未尽构造以 Scenario 驱动补齐——登记为 residual risk。运行时对该参数的 `DeprecationWarning` 在 CLI 层过滤（不改变行为）。

**D5 位置语义**。`ast` 的 `col_offset` 0 起，报告 `col = col_offset + 1`、`line = lineno`（`toolchain/cli` R2 冻结 1 起始）；仅用起始位置。

**D6 `category` slug 与 `suggestion` 文本**。`category` 为小写 ASCII+连字符 slug（如 `while-loop`/`list-comprehension`/`annotated-assignment`/`unknown-decorator`），每错误码的 slug 值域在实现中登记为冻结表（同一拒绝类别恒定）；`suggestion` 为中文恢复建议，文本对应 syntax spec 各条款的恢复建议句（自由文本确定性：同输入逐字符一致即可，语言不约束——与 diagnostics 的 note/issue/msg 同纪律）。

**D7 CLI 形态**。`argparse` 子命令 `compile`；`--target` 不用 `argparse` 的 `choices`（避免其 exit 2 通用用法错误与「含登记表清单的拒绝说明」混杂），统一自校验后走工具链入口拒绝路径（stderr 一行 + exit 2，不触源文件）。stdout JSON：`ensure_ascii=False`、字段序固定（`status` 在前；条目按 `code/line/col/category/suggestion`）、无文件名字段（四要素不含文件名——单文件编译，兼得确定性）。退出码封闭 `0/1/2`（R3：0=已实现段全过；R2：1=存在拒绝；R1：2=工具链入口）。

**D8 测试结构**。`tests/` 按 Requirement 分文件（`test_carrier.py` … `test_state_class.py`、`test_report_contract.py`、`test_cli.py`），28 个语法 Scenario + 10 个 toolchain Scenario 逐条至少一个断言公共可观察结果的测试（错误码/位置/类别/建议/退出码/stdout JSON/确定性）；集成样例：veps §7 FlashAttention 源码语法段全过（Scenario L100 的实例化）+ M1 验收语法片的注入样例。

**D9 门禁与 CLAUDE.md**。pre-push 在 openspec strict 后追加 `python3 -m pytest -q`；pytest 不可导入时明确报错阻断并提示安装（诚实优先于便利）。CLAUDE.md 工程约束节更新：test 命令 `python3 -m pytest`、pre-push 挂接说明、「当前尚无代码与质量脚本」句改为已引入 pytest（lint/build 仍缺）。

**D10 确定性**。输出无时间戳/路径/环境字段；`json.dumps` 固定参数；排序稳定键 `(line, col, 检查器序)`，并列时按确定性 AST 遍历的收集序（结构化递归遍历，子节点序固定）——同输入两次运行逐字节一致（R2 Scenario 承载）。

**D11 顶层 `import` 与名字不管控**。语法段只查结构类别（`ast.Import`/`ast.ImportFrom` 均接受），导入名与导入目标的合法性不属语法段（后续段或工具链 change 裁决）——与 spec 顶层白名单的措辞一致（「`import` 语句」为类别级）。

## 长期基线刷新计划（归档时执行）

1. `openspec archive`：新建 stable `toolchain/cli/spec.md`，补写 Purpose（工具链域定位：入口/序列化/退出码/诚实退出；`passed` 形态与 `--profile`、批量编译为显式延期项）。
2. `overview.md`：登记 `toolchain/cli` 行 + 仓库转入规格+实现双轨注记（M1 前端启动，首个实现 change）。
3. `CLAUDE.md` 工程约束更新已随 change 实施（非归档时补）——归档时复核一致性。
4. `veps/` 不回改。

## Scenario 计数

- `toolchain/cli` ADDED：**3 Requirement / 10 Scenario**（R1×5、R2×3、R3×2——grep 实数核对；R1 第 5 个 Scenario 为设计审查 cycle 1 finding 2 修复时补入的缺位置参数边界）
- 语法段实现映射：**8 Requirement / 28 Scenario**（R1×4、R2×2、R3×3、R4×4、R5×5、R6×4、R7×3、R8×3——grep 实数核对；无 delta，测试映射表见 tasks）
- 语言行为规格零新增零修改（无 BREAKING）
