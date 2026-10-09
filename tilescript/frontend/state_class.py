"""E0107 状态类类体结构（syntax spec R7「状态类类体结构」）。

`@tis.state` 类体仅接受带类型注解且无右值的字段声明；字段类型规则
（nominal 等价、dtype/scope 约束）由 language/type-system 承载。
"""

import ast

from .report import Rejection
from .top_level import _has_tis_decorator

_ORDER = 7


def check(tree: ast.Module) -> "list[Rejection]":
    rejections: list[Rejection] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _has_tis_decorator(node, "state"):
            rejections.extend(_check_body(node))
    return rejections


def _check_body(cls: ast.ClassDef) -> "list[Rejection]":
    out: list[Rejection] = []
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.value is None:
            if isinstance(stmt.target, ast.Name):
                continue  # 合法字段声明
            out.append(_rej(stmt, "other-class-body",
                            "状态类类体仅接受带类型注解且无右值的字段声明。"))
        elif isinstance(stmt, ast.AnnAssign):
            out.append(_rej(stmt, "field-with-value",
                            "状态类字段不得书写初始值；字段初始值由 pipe.run(init=...) 提供。"))
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_rej(stmt, "class-method",
                            "状态类类体不包含方法；行为请写在 kernel 设备代码中"
                            "（字段初始值由 pipe.run(init=...) 提供）。"))
        elif isinstance(stmt, ast.Assign):
            out.append(_rej(stmt, "unannotated-field",
                            "状态类字段必须带类型注解且无右值"
                            "（形如 O_acc: Tensor[f32, (BR, D), Register]）。"))
        else:
            out.append(_rej(stmt, "other-class-body",
                            "状态类类体仅接受带类型注解且无右值的字段声明。"))
    return out


def _rej(stmt: ast.stmt, category: str, suggestion: str) -> Rejection:
    return Rejection(
        code="E0107", line=stmt.lineno, col=stmt.col_offset + 1,
        category=category, suggestion=suggestion, order=_ORDER,
    )
