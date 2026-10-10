# Proposal: add-m1-syntax-frontend

## Why

1. **M1 前端启动**（用户指令）：`veps` §5.2 M1 = Parser（Python AST）/类型检查/TSR/作用域检查。本 change 是其第一片——Parser 载体检查与语法段（`E01xx`）全量实现，五段检查管线的第一段获得载体。
2. **规格就绪而实现为零**：`language/syntax-acceptance-set`（8 Requirement / 28 Scenario）自 2026-09-24 冻结至今无一行实现；本仓库尚无生产代码、无 build/test 基建。
3. **工具链输出面豁口**：拒绝报告四要素、收集/排序/tiebreak/确定性均已规格化，但其公共可观察的序列化形态——编译入口（命令行形态与参数）、拒绝清单的 stdout 格式、退出码语义——未在任何 spec 定义。这是调用方（CI、AI agent、人类）依赖的公共契约，按「规格先行」必须在实现前闭合。
4. **工程基建挂账到期**：CLAUDE.md「当前尚无代码与质量脚本（build/test/lint）；引入后把对应检查挂入 `.husky/` 门禁并更新本文件」——本 change 首次兑现（pytest 挂 pre-push）。

## What Changes

- **ADDED** capability `toolchain/cli`（3 Requirement / 10 Scenario，工具链域首个实体）：
  - 编译入口与目标参数——`python -m tilescript compile <源文件> --target <登记表成员>`；未提供 `--target`/未登记标识符/缺少源文件路径位置参数/源文件不可读 → 工具链入口拒绝（stderr + 退出码 2，非 `E0xxx`——承接 `hal/capability-descriptions` HR2）；扩展名不拒绝。
  - 拒绝清单 JSON 序列化——`{"status": "rejected", "rejections": [...]}` 输出至 stdout，每条拒绝四要素序列化为五字段 `code`/`line`/`col`/`category`/`suggestion`（位置要素 = `line`+`col` 两字段，行列 1 起始、`category` 为稳定 slug），字段集 v1 冻结演进规则；存在拒绝退出码 1；同输入重复运行逐字节一致。
  - 分段实现状态与诚实退出——已实现段全过时 `{"status": "incomplete", "implemented_stages": [...], "pending_stages": [...]}`、退出码 0；MUST NOT 静默跳过未实现段；`passed` 状态（含诊断 JSON 输出挂接）值域登记但形态留待 M2-M3 诊断 v0 change。
- **实现 `language/syntax-acceptance-set` 全量**（零 spec delta，既有契约为准）：`E0101`（载体：`ast.parse(..., feature_version=(3, 10))` 等价语义、`module` 伪代码拒绝）→ `E0102`（顶层白名单）→ `E0103`（装饰器识别与目标类别）→ `E0104`（kernel 签名注解与 comptime 默认值）→ `E0105`（设备代码语句接受集）→ `E0106`（表达式接受集）→ `E0107`（状态类类体）七检查器 + 报告契约（收集全部、位置升序、同位置最早检查器 tiebreak、重复编译一致；`E0101` 无 AST 单条特例）。
- **工程骨架**：`pyproject.toml`（`requires-python >= 3.10`、零运行时三方依赖——仅标准库 `ast`/`argparse`/`json`）；`tilescript` 包（`cli` + `frontend` 分模块 + HAL 登记表校验）；pytest 测试树（28 个语法 Scenario 逐条转行为测试 + toolchain 10 个新 Scenario）；`.husky/pre-push` 追加 pytest；CLAUDE.md 工程约束节更新（test 命令与门禁挂接）。
- **无 BREAKING**：纯新增实现 + 小 ADDED 规格，不改任何既有 spec 文本与错误码语义。

## 非目标

- **`E03xx`–`E06xx` 各段实现**：类型检查、原语契约、执行结构、数值语义的检查器属后续 change 逐段实现；本 change CLI 以 `pending_stages` 如实列出（`toolchain/cli` 诚实退出条款）。
- **TSR 构造与诊断 JSON 输出**：TSR（Tile 级 IR）随类型检查 change 起步；per-kernel 诊断 JSON 属 M2-M3 诊断 v0（`passed` 状态的输出形态届时挂接）。
- **HAL 能力描述加载**：本 change 仅校验 `--target` 为登记表成员（标识符层面）；YAML 载体与字段加载随首个消费段（类型/lowering）实现。
- **批量编译、目录输入、`--profile`、配置文件、插件机制**：工具链后续 change。
- **性能**：编译时间验收数字属 `veps` 里程碑层（§5.3），不入本 change。
- **lint/format 工具**：可随后续 change 引入（CLAUDE.md 门禁挂接届时同步）；本 change 仅挂 pytest。

## Impact

- 仓库从纯规格转入规格 + 实现双轨；TDD 自本 change 生效（既有 Scenario 即验收面，行为测试断言公共可观察结果：拒绝 JSON、退出码、确定性）。
- `toolchain/cli` 成为工具链域基座 capability（未来 `--profile`、批量组织、诊断 JSON 输出等扩展挂靠）。
- 五段管线不变：本 change 实现第一段（语法段），同位置跨段 tiebreak 语义（最早段优先）在后续段实现时自然生效。
- pre-push 门禁追加 pytest（openspec strict 保留）；CLAUDE.md 工程约束节同步登记 test 命令。
