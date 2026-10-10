"""执行结构段形态环境（design D3）——名 → 粗形态枚举。

粗类型，不含 dtype/dims（本段无此判定面）；STATE_CLASS 带类名
（consume 返回注解与 init 的 nominal 比较）。注解归约复用
`typecheck.annotations.parse_annotation`（解析失败 → UNKNOWN，
E0302 已由类型段在短路前置报，此处防御性让渡）。
"""

import ast
from dataclasses import dataclass

from ..typecheck import annotations as _annotations
from ..typecheck import types as _types
from ..typecheck.types import ScalarType, StateType, TensorType

# 形态枚举（D3；string simple-enums + StateShape 参数化形态）。
SHARED_TENSOR = "SHARED_TENSOR"
REGISTER_TENSOR = "REGISTER_TENSOR"
GLOBAL_TENSOR = "GLOBAL_TENSOR"
UNKNOWN_TENSOR = "UNKNOWN_TENSOR"   # tensor 但 scope 不可知（预留：UNKNOWN 链派生）
INT = "INT"                          # 运行期 int
COMPTIME_INT = "COMPTIME_INT"
DTYPE_SCALAR = "DTYPE_SCALAR"
PIPELINE = "PIPELINE"
RANGE_VALUE = "RANGE_VALUE"
BUFFER_PARAM = "BUFFER_PARAM"        # D3 专档：produce/consume buffer 形参（逃逸判定）
WARP_BINDING = "WARP_BINDING"        # with 绑定名（E0504 使用封闭判定）
UNKNOWN = "UNKNOWN"

_SCOPE_SHAPES = {"Shared": SHARED_TENSOR, "Register": REGISTER_TENSOR,
                 "Global": GLOBAL_TENSOR}


@dataclass(frozen=True)
class StateShape:
    """状态类形态（nominal：仅比类名）。"""
    class_name: str


def from_annotation(annotation, comptime_syms=frozenset(), registry=None):
    """注解 AST 节点 → 形态（无注解/解析失败 → UNKNOWN）。"""
    if annotation is None:
        return UNKNOWN
    parsed, rejection = _annotations.parse_annotation(
        annotation, comptime_syms=comptime_syms, registry=registry)
    if rejection is not None:
        return UNKNOWN
    return from_type(parsed)


def from_type(t):
    """类型对象 → 形态（kernel 形参与 produce/consume 形参共用归约）。"""
    if isinstance(t, TensorType):
        return _SCOPE_SHAPES[t.scope]
    if isinstance(t, ScalarType):
        if t.kind == _types.INT:
            return INT
        if t.kind == _types.COMPTIME_INT:
            return COMPTIME_INT
        return DTYPE_SCALAR
    if isinstance(t, StateType):
        return StateShape(t.class_name)
    return UNKNOWN


def comptime_int_form(node, comptime):
    """comptime[int] 形态口径（stages/warps/block_idx/barrier_id 共用，D4）。

    → ("const", 值)（int 字面量，值可判）| ("name", None)（comptime 名，值让渡）
    | None（运行期名/其他形态——违规）。
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return ("const", node.value)
    if isinstance(node, ast.Name) and node.id in comptime:
        return ("name", None)
    return None


def buffers_verdict(shape):
    """buffers 值三分（E0502 构造面消费）：accept / reject / defer。"""
    if shape == SHARED_TENSOR:
        return "accept"
    if shape in (UNKNOWN, UNKNOWN_TENSOR):
        return "defer"
    return "reject"


def params_of(fn) -> "list":
    """函数形参 AST 节点展开（posonly + positional + kwonly；不含 varargs）。"""
    a = fn.args
    return [*a.posonlyargs, *a.args, *a.kwonlyargs]
