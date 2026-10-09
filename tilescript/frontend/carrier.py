"""E0101 载体检查（syntax spec R1「源码载体为 Python 3.10 语法基线」）。

- 解析：`ast.parse(..., feature_version=(3, 10))` 等价语义（spec L11）。
- `module <name>:` 伪代码：设计文档的文档组织形式，不是语言构造（spec L13）——
  SyntaxError 路径上的正则预检给出专属类别与建议。
- 高于基线语法：`feature_version` 对新语法的拒绝不完整（CPython 语义为「尽量一致」，
  design D4 residual risk）——冻结节点表后检查 + 逐节点回归测试锁定。
"""

import ast
import re
import warnings

from .report import Rejection

# 检查管线序（design D3：E0101→E0107 数字序即执行序）。
_ORDER = 1

# module 伪代码形态：`module <name>:`（spec L13 引用 veps §7 的文档组织伪代码）。
_MODULE_PSEUDOCODE = re.compile(r"^module\s+\w+\s*:")

# 3.11+ 已知构造冻结节点表（design D4：与 D6 slug 冻结表同纪律；表内节点以
# tests/test_carrier.py 回归测试锁定，未尽构造以 Scenario 驱动补齐）。
_FROZEN_NEWER_NODES = (
    "TryStar",    # 3.11 except*
    "TypeAlias",  # 3.12 type X = ...
)


def parse(source: str, filename: str = "<tilescript>"):
    """解析模块源文本。

    返回 `(tree, None)`：载体检查通过（后续结构检查接管）。
    返回 `(None, rejection)`：E0101 单条（无 AST 特例——报告契约 L170）。
    """
    try:
        with warnings.catch_warnings():
            # feature_version 在部分运行时发 DeprecationWarning（design D4：
            # 过滤不改变行为，仅不泄漏给用户）。在解析点过滤比 CLI 层更完整。
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(source, filename=filename, feature_version=(3, 10))
    except SyntaxError as e:
        return None, _from_syntax_error(source, e, filename)
    except (ValueError, RecursionError) as e:  # 空字节/极端嵌套等解析器层面失败
        return None, Rejection(
            code="E0101", line=1, col=1, category="syntax-error",
            suggestion=f"源文本不是合法的 Python 3.10 语法（解析失败：{e}）；"
                       "TileScript 以 Python 3.10 语法为载体，请改写为 Python 3.10 兼容语法。",
            order=_ORDER,
        )
    return tree, _check_newer_nodes(tree)


def _from_syntax_error(source: str, e: SyntaxError, filename: str) -> Rejection:
    first_line = source.splitlines()[0] if source else ""
    if e.lineno == 1 and _MODULE_PSEUDOCODE.match(first_line):
        return Rejection(
            code="E0101", line=e.lineno or 1, col=e.offset or 1,
            category="module-pseudocode",
            suggestion="module <name>: 是设计文档的文档组织伪代码，不是语言构造；"
                       "一个模块恰好是一个 Python 源文件（模块即源文件本身），请删除该行。",
            order=_ORDER,
        )
    # 区分「高于基线语法」与「一般语法错误」：无版本限制重解析成功 ⇒ 语法本身
    # 合法但为 3.10 之后引入（feature_version 拒绝）——精确且不依赖错误消息文本。
    if _parses_without_version_limit(source, filename):
        return _newer_syntax_rejection(source, e)
    return Rejection(
        code="E0101", line=e.lineno or 1, col=e.offset or 1, category="syntax-error",
        suggestion=f"源文本不是合法的 Python 3.10 语法（解析错误：{e.msg}）；"
                   "TileScript 以 Python 3.10 语法为载体，请改写为 Python 3.10 兼容语法。",
        order=_ORDER,
    )


def _parses_without_version_limit(source: str, filename: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ast.parse(source, filename=filename)
        return True
    except (SyntaxError, ValueError, RecursionError):
        return False


def _newer_syntax_rejection(source: str, e: SyntaxError) -> Rejection:
    line, col = e.lineno or 1, e.offset or 1
    # 优先取新语法节点的精确位置（重解析成功的树中首个冻结表节点）。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            newer_tree = ast.parse(source, filename="<newer-check>")
    except (SyntaxError, ValueError, RecursionError):
        newer_tree = None
    if newer_tree is not None:
        for node in ast.walk(newer_tree):
            if type(node).__name__ in _FROZEN_NEWER_NODES or getattr(node, "type_params", None):
                line = getattr(node, "lineno", line)
                col = getattr(node, "col_offset", col - 1) + 1
                break
    return Rejection(
        code="E0101", line=line, col=col, category="newer-python-syntax",
        suggestion="该构造是 Python 3.10 之后版本引入的语法；"
                   "TileScript 的语法基线为 Python 3.10，请改写为 3.10 兼容形式。",
        order=_ORDER,
    )


def _check_newer_nodes(tree: ast.Module):
    """冻结节点表后检查：解析成功但含 3.11+ 构造（feature_version 拒绝不完整）。"""
    for node in ast.walk(tree):
        if type(node).__name__ in _FROZEN_NEWER_NODES or getattr(node, "type_params", None):
            return Rejection(
                code="E0101", line=getattr(node, "lineno", 1),
                col=getattr(node, "col_offset", 0) + 1,
                category="newer-python-syntax",
                suggestion="该构造是 Python 3.10 之后版本引入的语法；"
                           "TileScript 的语法基线为 Python 3.10，请改写为 3.10 兼容形式。",
                order=_ORDER,
            )
    return None
