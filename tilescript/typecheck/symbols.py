"""符号表构建（type-system 段）：kernel 形参表、设备函数注解表、注册表。"""

import ast

from tilescript.frontend.top_level import _has_tis_decorator

from .annotations import parse_annotation
from .types import UNKNOWN


def kernel_comptime_names(tree: ast.Module) -> frozenset:
    """kernel 形参中注解为 comptime[int] 的名字集（shape 符号 comptime 性来源）。

    M1 单 kernel 语义：取首个 @tis.kernel 函数；无 kernel 返回空集。
    """
    kernel = find_kernel(tree)
    if kernel is None:
        return frozenset()
    names = set()
    for arg in _params(kernel):
        if _is_comptime_int_annotation(arg.annotation):
            names.add(arg.arg)
    return frozenset(names)


def find_kernel(tree: ast.Module):
    """首个 @tis.kernel 函数节点；无则 None。"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and _has_tis_decorator(node, "kernel"):
            return node
    return None


def kernel_params(tree: ast.Module, comptime_syms=frozenset()) -> dict:
    """kernel 形参名 → 类型（首次注解解析失败的形参按 UNKNOWN 让渡）。"""
    kernel = find_kernel(tree)
    if kernel is None:
        return {}
    env = {}
    for arg in _params(kernel):
        env[arg.arg] = _param_type(arg.annotation, comptime_syms, registry=None)
    return env


def annotated_functions(tree: ast.Module, registry=None, comptime_syms=frozenset()) -> dict:
    """模块级 produce/consume 函数：名 → {"params": [(形参名, 类型)],
    "returns": 返回注解类型 | None}。

    无注解或注解结构解析失败（E0302 已另行报告）的形参按 UNKNOWN 让渡
    （绑定检查遇 UNKNOWN 跳过）；returns 为 None 表示无返回注解（不查）。
    契约违规（形参个数/位置形态）归 E0502，本表只消费可解析部分（design D3）。
    """
    out = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not (_has_tis_decorator(node, "produce") or _has_tis_decorator(node, "consume")):
            continue
        params = []
        for arg in _params(node):
            params.append((arg.arg, _param_type(arg.annotation, comptime_syms,
                                                registry)))
        returns = None
        if node.returns is not None and not _is_none_node(node.returns):
            annotation, rejection = parse_annotation(node.returns,
                                                      comptime_syms=comptime_syms,
                                                      registry=registry)
            returns = None if rejection is not None else annotation
        out[node.name] = {"params": params, "returns": returns}
    return out


def _params(fn):
    return [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]


def _param_type(annotation, comptime_syms, registry):
    if annotation is None:
        return UNKNOWN  # 无注解形参：让渡（绑定检查跳过）
    parsed, rejection = parse_annotation(annotation, comptime_syms=comptime_syms,
                                         registry=registry)
    if rejection is not None:
        return UNKNOWN
    return parsed


def _is_none_node(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_comptime_int_annotation(annotation) -> bool:
    return (isinstance(annotation, ast.Subscript)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "comptime")
