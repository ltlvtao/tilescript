# Proposal

## Why

TileScript 的语言边界目前只存在于 `veps/design.md` 的示例代码中：哪些 Python 语法被设备代码接受、哪些被拒绝、以什么错误码拒绝，没有任何权威定义。M1 前端（Parser + 类型检查）即将开发，若没有先行规格，解析器实现将隐式决定语言边界——后续每个前端改动都可能成为无人察觉的事实语言变更。TileScript 语言用户和 AI agent（优化循环依赖结构化编译错误，见 `veps/design.md` §3.3）都需要一份可依赖的语法契约来编写与改写 kernel。

## What Changes

- 新增 capability `language/syntax-acceptance-set`，定义 M1 前端的语法接受集行为契约：
  - 源码载体：模块必须是 Python 3.10 语法基线内可解析的源文本（规范扩展名 `.tis`），模块即源文件、无 `module` 包装语法（`E0101`）；语言 API 前缀为 `tis.`（如 `tis.kernel`、`tis.state`）
  - 模块顶层结构白名单（`E0102`）
  - 装饰器识别集合（`E0103`）
  - kernel 函数签名规则（`E0104`）
  - 设备代码语句接受集：白名单与显式拒绝清单（`E0105`）
  - 设备代码表达式接受集：白名单与显式拒绝清单（`E0106`），元组/字典/字符串字面量按位置接受
  - `@tis.state` 状态类类体结构：仅带类型注解且无右值的字段声明（`E0107`）
  - 语法拒绝（E01xx）的报告契约：错误码 + 行列定位 + 被拒类别 + 恢复建议，收集式输出与确定性排序
- 分配错误码段位 `E0100`–`E0199` 给语法接受集，按检查管线顺序编号；与既有事实段位（`E03xx` 类型/作用域、`E04xx` 计算原语、`E05xx` 执行结构，见 `veps/design.md`）不重叠。

## Capabilities

### New Capabilities

- `language/syntax-acceptance-set`：M1 前端的语法接受集——接受哪些 Python 语法、以什么错误码拒绝哪些，以及拒绝报告的格式、收集与确定性。

## 非目标

- 类型检查与作用域转移语义（`language/type-system`，含 `E0301` 体系与"状态字段必须 Register scope"的类型规则）——本 change 只约束语法类别，不约束类型合法性。
- 原语语义（`primitives/*`）：`tis.load`/`tis.dot` 等的参数与行为契约。
- 诊断 JSON Schema（`diagnostics/*`）：编译诊断的结构化输出格式。本 change 的报告契约只定义语法拒绝（E01xx）的报告要素，跨错误码段的通用错误报告契约由后续 change 定义。
- 名称解析与预定义名（如 `inf`）：`-inf` 在语法层是一元负 + 名称引用，被本接受集接受；名字的解析与语义由后续 capability 定义。
- M1 前端（Parser、AST 检查器）的实现：本 change 只建立行为契约与验收依据，实现由后续 change 承载。

## Impact

- 无生产代码影响（纯规格先行 change）。
- `veps/design.md` 的示例代码从本 change 起获得权威语法依据；示例与规格冲突时以规格为准。
- 为 M1 验收（"5 个 kernel 源码全部通过类型检查；20 个注入错误全部被捕获"）提供语法层验收依据。
