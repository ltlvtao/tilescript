"""类型表示与等价判定（type-system spec R3「类型等价规则」全量实现面）。

Dim 三形态（design D2）：ConstDim（int 字面量）/ SymbolDim（参数符号，comptime 标志
承载常量/运行期）/ DerivedDim（算术派生，操作数 comptime 性递归判定）。跨形态不等
（无数值折叠——折叠归 numerics 段，design D11 residual 1）。

UNKNOWN 是让渡面（R6 让渡句）的类型占位：equivalent 对其恒 False——「不产生等价
结论」；绑定检查器在调用前跳过（design D4）。
"""

from dataclasses import dataclass, field

# dtype/scope 封闭集（spec R1：六 dtype / 三 scope；冻结，扩展只能经显式 change）。
DTYPES = ("f16", "bf16", "f32", "f8e4m3", "i8", "i32")
SCOPES = ("Global", "Shared", "Register")
LAYOUT = "RowMajor"  # 唯一合法值（显式与省略等价）


@dataclass(frozen=True)
class ConstDim:
    """编译期常量维（int 字面量，四进制在 AST 层同 Constant(int)）。"""
    value: int


@dataclass(frozen=True)
class SymbolDim:
    """符号维：comptime=True（comptime[int] 参数）/ False（int 运行期参数）。"""
    name: str
    comptime: bool


@dataclass(frozen=True)
class DerivedDim:
    """算术派生维：op 为运算符符号，operands 为子 Dim；comptime 递归判定。"""
    op: str
    operands: tuple
    comptime: bool


@dataclass(frozen=True)
class TensorType:
    dtype: str
    dims: tuple
    scope: str
    layout: str = LAYOUT  # 省略即默认，等价判定恒真


@dataclass(frozen=True)
class PointerType:
    """指针类型（独立类别：与 Tensor 跨类别不等，R3）。"""
    dtype: str
    scope: str


@dataclass(frozen=True)
class ScalarType:
    """标量种类：int / comptime_int / dtype 名（六值——运行期 dtype 标量）。"""
    kind: str


@dataclass(frozen=True)
class StateType:
    """状态类类型：名义等价（仅比类名）；fields 有序（类体声明序）。"""
    class_name: str
    fields: tuple  # tuple[(名, 类型), ...]


@dataclass(frozen=True)
class _Unknown:
    """让渡面占位（单例 UNKNOWN）——R6 让渡句的 M1 实现语义。"""


UNKNOWN = _Unknown()


@dataclass(frozen=True)
class UnknownDim:
    """UNKNOWN 组件传播的维度占位（design D4）：含此维的 Tensor 整体按
    UNKNOWN 参与绑定（等价全有或全无——不做部分等价判定）。"""


def treats_as_unknown(t) -> bool:
    """类型是否按 UNKNOWN 参与绑定（D4）：UNKNOWN 本体，或含 UNKNOWN 维的 Tensor。"""
    if isinstance(t, _Unknown):
        return True
    if isinstance(t, TensorType):
        return any(isinstance(d, UnknownDim) for d in t.dims)
    return False

# 标量种类封闭值（ScalarType.kind 合法域）。
INT = "int"
COMPTIME_INT = "comptime_int"


def equivalent(a, b) -> bool:
    """两类型等价判定（R3 逐组件；跨类别恒 False；UNKNOWN 恒 False）。"""
    if isinstance(a, _Unknown) or isinstance(b, _Unknown):
        return False
    if isinstance(a, TensorType) and isinstance(b, TensorType):
        if any(isinstance(d, UnknownDim) for d in (*a.dims, *b.dims)):
            return False  # UNKNOWN 维不参与部分等价（D4 全有或全无）
        return (a.dtype == b.dtype and a.scope == b.scope
                and len(a.dims) == len(b.dims)
                and all(x == y for x, y in zip(a.dims, b.dims)))  # layout 恒等价
    if isinstance(a, PointerType) and isinstance(b, PointerType):
        return a.dtype == b.dtype and a.scope == b.scope
    if isinstance(a, ScalarType) and isinstance(b, ScalarType):
        return a.kind == b.kind
    if isinstance(a, StateType) and isinstance(b, StateType):
        return a.class_name == b.class_name  # 名义等价
    return False


def compatible(source, target) -> bool:
    """绑定单向兼容（R2）：comptime[int] 值可绑定 int 位置；其余须 equivalent。"""
    if isinstance(source, ScalarType) and isinstance(target, ScalarType):
        if source.kind == COMPTIME_INT and target.kind == INT:
            return True
    return equivalent(source, target)


def _dim_desc(dim) -> str:
    if isinstance(dim, ConstDim):
        return str(dim.value)
    if isinstance(dim, SymbolDim):
        return f"{'comptime' if dim.comptime else '运行期'} {dim.name}"
    if isinstance(dim, DerivedDim):
        return f"{'comptime' if dim.comptime else '运行期派生'} {_dim_expr(dim)}"
    if isinstance(dim, UnknownDim):
        return "未知维度"
    return repr(dim)


def _dim_expr(dim) -> str:
    if isinstance(dim, ConstDim):
        return str(dim.value)
    if isinstance(dim, SymbolDim):
        return dim.name
    if isinstance(dim, DerivedDim):
        if len(dim.operands) == 2:
            left, right = dim.operands
            return f"{_dim_expr(left)} {dim.op} {_dim_expr(right)}"
        return f"{dim.op}({', '.join(_dim_expr(o) for o in dim.operands)})"
    return repr(dim)


def desc(t) -> str:
    """类型→描述文本（拒绝报告「两侧类型」与建议用；冻结格式）。"""
    if isinstance(t, _Unknown):
        return "未知类型（结果类型由后续检查段承载）"
    if isinstance(t, TensorType):
        inner = ", ".join(_dim_desc(d) for d in t.dims)
        return f"Tensor[{t.dtype}, ({inner}), {t.scope}]"
    if isinstance(t, PointerType):
        return f"Pointer[{t.dtype}, {t.scope}]"
    if isinstance(t, ScalarType):
        if t.kind == COMPTIME_INT:
            return "comptime[int]"
        return t.kind  # int 与 dtype 名原样
    if isinstance(t, StateType):
        return t.class_name
    return repr(t)
