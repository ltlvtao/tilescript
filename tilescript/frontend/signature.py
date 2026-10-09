"""E0104 kernel 函数签名规则（syntax spec R4「kernel 函数签名规则」）。

辖域：设备入口函数（tis.kernel/tis.persistent_kernel/tis.fused_kernel 装饰）的
参数；produce/consume 嵌套函数签名契约由 execution/pipeline-structure 承载。
注解形式封闭集为语法层形态（Pointer[...]/Tensor[...]/dtype 名/int/comptime[int]），
参数化内部结构由 type-system 段承载。
"""

import ast

from .report import Rejection
from .top_level import _is_entry_function

_ORDER = 4

_DTYPES = ("f16", "bf16", "f32", "f8e4m3", "i8", "i32")
_SUBSCRIPT_FORMS = ("Pointer", "Tensor")

_FORMS = "Pointer[...]、Tensor[...]、dtype 名（f16/bf16/f32/f8e4m3/i8/i32）、int、comptime[int]"


def check(tree: ast.Module) -> "list[Rejection]":
    rejections: list[Rejection] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and _is_entry_function(node)):
            continue
        for arg, default in _all_args(node.args):
            rejections.extend(_check_arg(arg, default))
    return rejections


def _all_args(args: ast.arguments):
    """产出 (arg, default)；default=None 表示无默认值。

    位置参数的 defaults 从右向左对齐（无默认值的前缀参数对应 None）；
    kwonlyargs 与 kw_defaults 一一对应；vararg/kwarg 无默认值语法位。
    """
    positional = args.posonlyargs + args.args
    pad = [None] * (len(positional) - len(args.defaults))
    for arg, default in zip(positional, pad + list(args.defaults)):
        yield arg, default
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        yield arg, default
    for a in (args.vararg, args.kwarg):
        if a is not None:
            yield a, None


def _check_arg(arg: ast.arg, default: "ast.expr | None") -> "list[Rejection]":
    out: list[Rejection] = []
    if arg.annotation is None:
        out.append(Rejection(
            code="E0104", line=arg.lineno, col=arg.col_offset + 1,
            category="missing-annotation",
            suggestion=f"入口函数参数必须带类型注解；合法注解形式：{_FORMS}。",
            order=_ORDER,
        ))
        return out  # 无注解时无 comptime 判定依据，默认值检查跳过

    is_comptime = _is_comptime(arg.annotation)
    if not _valid_annotation(arg.annotation):
        out.append(Rejection(
            code="E0104", line=arg.lineno, col=arg.col_offset + 1,
            category="invalid-annotation",
            suggestion=f"参数 {arg.arg} 的注解形式不在合法集合内；合法注解形式：{_FORMS}。",
            order=_ORDER,
        ))
    if default is None:
        return out

    if not is_comptime:
        out.append(Rejection(
            code="E0104", line=arg.lineno, col=arg.col_offset + 1,
            category="default-on-non-comptime",
            suggestion=f"参数 {arg.arg} 的默认值仅允许用于 comptime 参数"
                       "（comptime[int]）。",
            order=_ORDER,
        ))
    elif not _literal_default(default):
        out.append(Rejection(
            code="E0104", line=arg.lineno, col=arg.col_offset + 1,
            category="invalid-comptime-default",
            suggestion=f"参数 {arg.arg} 的 comptime 默认值仅接受单个 int 字面量"
                       "（十进制/二进制/八进制/十六进制）或布尔常量（True/False）；"
                       "常量表达式求值由后续能力承载。",
            order=_ORDER,
        ))
    return out


def _is_comptime(annotation: ast.expr) -> bool:
    """`comptime[int]` 精确形态。"""
    return (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "comptime"
        and isinstance(_slice_element(annotation), ast.Name)
        and _slice_element(annotation).id == "int"
    )


def _slice_element(subscript: ast.Subscript):
    """单元素下标（3.9+ ast.slice 是表达式本身或 Tuple）。"""
    return subscript.slice


def _valid_annotation(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id in _DTYPES or annotation.id == "int"
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id in _SUBSCRIPT_FORMS:
            return True  # 参数化内部结构由 type-system 段承载
        return _is_comptime(annotation)
    return False


def _literal_default(default: ast.expr) -> bool:
    """单个 int 字面量（任意进制——AST 层同 Constant(int)）或布尔常量。

    注意 bool 是 int 的子类，先排除 bool 再判 int（True/False 单独接受）。
    """
    if isinstance(default, ast.Constant):
        return type(default.value) is int or type(default.value) is bool
    return False
