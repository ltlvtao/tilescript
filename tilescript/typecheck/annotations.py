"""E0302 注解解析器（type-system spec R1「类型表达式结构规则」/ R2 shape 组件）。

AST 注解表达式 → 类型对象，或 E0302 Rejection（order=1，段内最优先）。
一注解一结果：结构失败不产类型（R8「同位置一注解一结果」）。

位置约定：报告定位注解表达式根节点起点（line 1 起始 / col 1 起始）。
"""

import ast

from tilescript.frontend.report import Rejection
from tilescript.frontend.top_level import _has_tis_decorator

from .types import (
    COMPTIME_INT,
    DTYPES,
    INT,
    LAYOUT,
    SCOPES,
    ConstDim,
    DerivedDim,
    PointerType,
    ScalarType,
    SymbolDim,
    TensorType,
)

# E0302 段内管线序（design D9：E0302=1/E0304=2/E0301=3/E0303=4）。
_ORDER = 1


def parse_annotation(node, *, comptime_syms=frozenset(), runtime_syms=frozenset(),
                     registry=None):
    """注解 AST 节点 → (类型 | None, Rejection | None)。

    comptime_syms：shape 组件符号中属 comptime[int] 的名字集；其余名字（含
    runtime_syms 与未知名）按运行期符号处理（同名判等仍有效，design D5）。
    registry：状态类注册表（类名 → StateType）；Name 命中注册名返回 StateType。
    runtime_syms 仅为显式性保留，不参与判定（非 comptime 即运行期）。
    """
    if isinstance(node, ast.Name):
        return _parse_name(node, registry)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return _parse_subscript(node, comptime_syms, registry)
    return None, _reject(node, "invalid-annotation-form")


def _parse_name(node, registry):
    if node.id in DTYPES:
        return ScalarType(node.id), None
    if node.id == INT:
        return ScalarType(INT), None
    if registry is not None and node.id in registry:
        return registry[node.id], None
    return None, _reject(
        node, "invalid-annotation-form",
        "注解形式不在类型表达式合法集合内；合法形式："
        "Tensor[dtype, shape, scope[, RowMajor]]、Pointer[dtype, scope]、"
        "comptime[int]、int、dtype 名（f16/bf16/f32/f8e4m3/i8/i32）或已注册状态类名。",
    )


def _parse_subscript(node, comptime_syms, registry):
    name = node.value.id
    params = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
    if name == "Tensor":
        if len(params) not in (3, 4):
            return None, _reject(
                node, "invalid-param-count",
                "Tensor 需要 3 或 4 个参数：(dtype, shape, scope[, RowMajor])。")
        return _parse_tensor(node, params, comptime_syms)
    if name == "Pointer":
        if len(params) != 2:
            return None, _reject(
                node, "invalid-param-count",
                "Pointer 需要 2 个参数：(dtype, scope)。")
        pointer_type, rejection = _parse_tensor(
            node, (params[0], None, params[1]), comptime_syms, pointer=True)
        if rejection is not None:
            return None, rejection
        return PointerType(pointer_type.dtype, pointer_type.scope), None
    if name == "comptime":
        if (len(params) == 1 and isinstance(params[0], ast.Name)
                and params[0].id == INT):
            return ScalarType(COMPTIME_INT), None
        return None, _reject(
            node, "invalid-annotation-form",
            "comptime 注解唯一合法形式为 comptime[int]。")
    return None, _reject(
        node, "invalid-annotation-form",
        f"注解形式「{name}[...]」不在类型表达式合法集合内；"
        "合法下标形式：Tensor[dtype, shape, scope[, RowMajor]]、"
        "Pointer[dtype, scope]、comptime[int]。")


def _parse_tensor(node, params, comptime_syms, pointer=False):
    # 参数 1：dtype 位（R1 封闭集；scope 名占位判参数顺序错）。
    dtype_node = params[0]
    if not (isinstance(dtype_node, ast.Name) and dtype_node.id in DTYPES):
        if isinstance(dtype_node, ast.Name) and dtype_node.id in SCOPES:
            return None, _reject(
                node, "invalid-param-order",
                "参数顺序错误：第一参数必须是 dtype 名"
                f"（{'/'.join(DTYPES)}），其后依次为 shape 与 scope。")
        shown = dtype_node.id if isinstance(dtype_node, ast.Name) else "（非名字）"
        return None, _reject(
            node, "unknown-dtype",
            f"未知 dtype「{shown}」不合法；合法 dtype 封闭集：{'/'.join(DTYPES)}。")

    # 参数 2：shape（元组逐维；单组件按单维张量处理——形式无损泛化，design D5）。
    # Pointer 无 shape 参数（占位 None），dims 为空。
    shape_node = params[1]
    if shape_node is None:
        dims = ()
    else:
        components = (shape_node.elts if isinstance(shape_node, ast.Tuple)
                      else (shape_node,))
        dims = tuple(_dim_expr(c, comptime_syms) for c in components)
        if any(d is None for d in dims):
            return None, _reject(
                node, "invalid-shape-component",
                "shape 组件必须是整数类别表达式（int 字面量、comptime[int]/int 符号"
                "或其算术组合）；浮点/布尔/非整数组件不合法。")

    # 参数 3：scope 位（封闭集三值）。
    scope_node = params[2]
    if not (isinstance(scope_node, ast.Name) and scope_node.id in SCOPES):
        shown = scope_node.id if isinstance(scope_node, ast.Name) else "（非名字）"
        return None, _reject(
            node, "unknown-scope",
            f"未知 scope「{shown}」不合法；合法 scope 封闭集：{'/'.join(SCOPES)}。")

    if pointer:
        return TensorType(dtype_node.id, dims, scope_node.id), None

    # 参数 4（可选）：layout 唯一合法值。
    if len(params) == 4:
        layout_node = params[3]
        if not (isinstance(layout_node, ast.Name) and layout_node.id == LAYOUT):
            return None, _reject(
                node, "invalid-layout",
                f"layout 第四参数唯一合法值为 {LAYOUT}（或省略）。")

    return TensorType(dtype_node.id, dims, scope_node.id), None


def _dim_expr(node, comptime_syms):
    """shape 组件 → Dim 三形态之一；非整数类别返回 None（调用方报 E0302）。"""
    if isinstance(node, ast.Constant):
        if type(node.value) is int:  # bool 是 int 子类，显式排除
            return ConstDim(node.value)
        return None
    if isinstance(node, ast.Name):
        # comptime 性仅由 comptime_syms 决定；运行期符号与未知名同为运行期。
        return SymbolDim(node.id, node.id in comptime_syms)
    if isinstance(node, ast.BinOp) and type(node.op).__name__ in _BINOPS:
        left = _dim_expr(node.left, comptime_syms)
        right = _dim_expr(node.right, comptime_syms)
        if left is None or right is None:
            return None
        return DerivedDim(_BINOPS[type(node.op).__name__], (left, right),
                          _is_comptime(left) and _is_comptime(right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _dim_expr(node.operand, comptime_syms)
        if operand is None:
            return None
        return DerivedDim("-", (operand,), _is_comptime(operand))
    return None


_BINOPS = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//", "Mod": "%"}


def _is_comptime(dim):
    """子组件 comptime 性：常量恒 True，符号/派生看标志。"""
    if isinstance(dim, ConstDim):
        return True
    return dim.comptime


def _reject(node, category, suggestion):
    return Rejection(code="E0302", line=node.lineno, col=node.col_offset + 1,
                     category=category, suggestion=suggestion, order=_ORDER,
                     stage="type-system")


def check(tree: ast.Module, comptime_syms=frozenset(), registry=None):
    """E0302 三类检查位置（design D5）：

    1. kernel 签名形参与返回注解（不查注册表——状态类名在此非法）；
    2. @tis.state 字段（由 state_fields.check 承载，此处不重复扫描）；
    3. produce/consume 形参与返回注解（查注册表——状态类名 → StateType）。

    无注解形参与 None 返回注解跳过（D7：无注解形参位不查）。
    三类位置之外的普通函数注解不查（M1 边界）。
    """
    rejections: "list[Rejection]" = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_kernel = _has_tis_decorator(node, "kernel")
        is_produce_consume = (_has_tis_decorator(node, "produce")
                              or _has_tis_decorator(node, "consume"))
        if not (is_kernel or is_produce_consume):
            continue
        reg = None if is_kernel else registry
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if arg.annotation is not None:
                _, rejection = parse_annotation(arg.annotation,
                                                comptime_syms=comptime_syms,
                                                registry=reg)
                if rejection is not None:
                    rejections.append(rejection)
        if node.returns is not None and not _is_none(node.returns):
            _, rejection = parse_annotation(node.returns,
                                            comptime_syms=comptime_syms,
                                            registry=reg)
            if rejection is not None:
                rejections.append(rejection)
    return rejections


def _is_none(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None
