"""E0102 顶层结构白名单（syntax spec R2「模块顶层结构白名单」）。

顶层仅接受三类：`import` 语句、`@tis.state` 装饰的类定义、被设备入口装饰器装饰的
函数定义。类别级判定——import 的名字与目标不管控（design D11，属后续段）。
装饰器本身的识别细节（集合外/类别不匹配）由 decorators 承载（E0103）。
"""

import ast

from .report import Rejection

_ORDER = 2

_SUGGESTION = (
    "模块顶层仅接受三类结构：import 语句、@tis.state 装饰的类定义、"
    "被设备入口装饰器（@tis.kernel/@tis.persistent_kernel/@tis.fused_kernel）"
    "装饰的函数定义。"
)

# 类别 slug 冻结表（design D6：同一拒绝类别恒定）。
_CATEGORY = {
    "Assign": "top-level-assignment",
    "AugAssign": "top-level-assignment",
    "AnnAssign": "top-level-annotated-assignment",
    "Expr": "top-level-expression",
    "ClassDef": "top-level-class",
    "FunctionDef": "top-level-function",
    "AsyncFunctionDef": "top-level-function",
}


def check(tree: ast.Module) -> "list[Rejection]":
    rejections: list[Rejection] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef) and _has_tis_decorator(node, "state"):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_entry_function(node):
            continue
        category = _CATEGORY.get(type(node).__name__, "top-level-statement")
        rejections.append(Rejection(
            code="E0102", line=node.lineno, col=node.col_offset + 1,
            category=category, suggestion=_SUGGESTION, order=_ORDER,
        ))
    return rejections


def iter_entry_functions(tree: ast.Module):
    """设备代码检查根：仅模块顶层的入口函数。

    嵌套的入口装饰函数已被 E0103（目标类别不匹配）与 E0105（function-definition
    短路）拒绝，其子树不再独立检查（design D3——被拒构造子树不深入）；
    statements/signature/expressions 的检查根统一取本迭代器。
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _is_entry_function(node):
            yield node


def _has_tis_decorator(node: ast.AST, name: str) -> bool:
    """`@tis.<name>` 形态装饰器（Attribute：value 为 Name 'tis'、attr 为 name）。"""
    return any(
        isinstance(d, ast.Attribute) and d.attr == name
        and isinstance(d.value, ast.Name) and d.value.id == "tis"
        for d in getattr(node, "decorator_list", [])
    )


def _is_entry_function(node: ast.AST) -> bool:
    """被任一设备入口装饰器装饰（kernel/persistent_kernel/fused_kernel）。"""
    return any(
        _has_tis_decorator(node, entry)
        for entry in ("kernel", "persistent_kernel", "fused_kernel")
    )
