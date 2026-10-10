"""原语调用结果类型规则（design D4）。

两类职责：
1. `infer_shape_component`：shape 实参组件 AST → Dim 三态——
   Dim（可判定整数类别）/ UnknownDim（未绑定名、非常规表达式：让渡传播，
   D11 residual 2）/ None（浮点/布尔字面量、已知非标量绑定：非整数类别，
   调用方记 E0406）。
2. 结果类型表：八存取/六计算原语合法调用下的结果 TensorType 构造器。
   违规调用的结果为 UNKNOWN（拒绝已由 memory_ops/compute_ops 记录，
   在 traverse 层衔接，design D3）。

类型段的「无数值折叠」不受本模块影响——构造器只重排既有 Dim。
"""

import ast

from ..typecheck.types import (
    COMPTIME_INT,
    INT,
    ConstDim,
    DerivedDim,
    PointerType,
    ScalarType,
    SymbolDim,
    TensorType,
    UnknownDim,
)

_BINOPS = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//",
           "Mod": "%"}


def infer_shape_component(node, env):
    """shape 实参单组件 → Dim | UnknownDim | None（None = 非整数类别）。

    env：变量名 → 类型（traverse 维护的同构 env）。整数类别封闭集：
    int 字面量 / comptime[int] 与 int 标量名 / 其算术组合（与类型段
    shape 注解组件规则同构，D4）。
    """
    if isinstance(node, ast.Constant):
        if type(node.value) is int:  # bool 是 int 子类，显式排除
            return ConstDim(node.value)
        if node.value is None or isinstance(node.value, str):
            return UnknownDim()  # 非数值常量：不可判定，让渡
        return None  # float 字面量：非整数类别 → E0406 证据
    if isinstance(node, ast.Name):
        bound = env.get(node.id)
        if bound is None:
            return UnknownDim()  # 未绑定名：让渡（类型段同样不报）
        if bound == ScalarType(COMPTIME_INT):
            return SymbolDim(node.id, True)
        if bound == ScalarType(INT):
            return SymbolDim(node.id, False)
        return None  # 已知非标量绑定（Tensor/Pointer/状态类）：E0406 证据
    if isinstance(node, ast.BinOp) and type(node.op).__name__ in _BINOPS:
        left = infer_shape_component(node.left, env)
        right = infer_shape_component(node.right, env)
        if left is None or right is None:
            return None
        if isinstance(left, UnknownDim) or isinstance(right, UnknownDim):
            return UnknownDim()
        return DerivedDim(_BINOPS[type(node.op).__name__], (left, right),
                          _comptime(left) and _comptime(right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = infer_shape_component(node.operand, env)
        if operand is None:
            return None
        if isinstance(operand, UnknownDim):
            return UnknownDim()
        return DerivedDim("-", (operand,), _comptime(operand))
    return UnknownDim()  # Attribute/Call/Subscript 等非常规组件：让渡


def _comptime(dim) -> bool:
    if isinstance(dim, ConstDim):
        return True
    return dim.comptime


# ---- 结果类型表（D4 逐行） -----------------------------------------------

def make_tensor_result(ptr: PointerType, dims) -> TensorType:
    """make_tensor(ptr, shape)：dtype 承接 ptr，scope 恒 Global（R5）。"""
    return TensorType(ptr.dtype, dims, "Global")


def alloc_shared_result(dtype, dims) -> TensorType:
    """alloc_shared(shape, dtype, layout)：scope 恒 Shared。"""
    return TensorType(dtype, dims, "Shared")


def alloc_result(dtype, dims, scope) -> TensorType:
    """zeros(shape, dtype, scope) / full(shape, value, dtype, scope) 共用。"""
    return TensorType(dtype, dims, scope)


def cast_result(x: TensorType, dtype) -> TensorType:
    """cast(x, dtype)：shape/scope 承接操作数（R6）。"""
    return TensorType(dtype, x.dims, x.scope)


def dot_result(m, n) -> TensorType:
    """dot(A, B, C, ...)：f32 (M, N) Register（compute-ops R2）。"""
    return TensorType("f32", (m, n), "Register")


def reduce_result(x: TensorType, axis: int) -> TensorType:
    """reduce(x, axis, op, scope)：去 axis 维，scope 恒 Register（R3）。"""
    return TensorType(x.dtype, x.dims[:axis] + x.dims[axis + 1:], "Register")


def elementwise_result(x: TensorType) -> TensorType:
    """maximum/exp/log(x)：与操作数同 dtype/shape/scope（R4/R5）。"""
    return TensorType(x.dtype, x.dims, x.scope)


def transpose_result(x: TensorType) -> TensorType:
    """transpose(x)：二维互换（R5）。"""
    return TensorType(x.dtype, x.dims[::-1], x.scope)
