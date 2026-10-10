"""原语结果类型表测试（design D4——合法结果构造 + shape 实参组件推断）。

结果类型表：八存取/六计算原语在合法调用下的结果 TensorType 构造器；
shape 实参组件推断：AST 组件 → Dim 三态（Dim=可判定 / UnknownDim=让渡 /
None=非整数类别，调用方记 E0406）。违规调用返回 UNKNOWN 的段内语义由
traverse/检查器承载（2.3/3.x/4.x 集成面）。
"""

import ast

from tilescript.primitives import results
from tilescript.typecheck.types import (
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

# make_tensor ptr 形参 / kernel comptime·运行期标量的典型 env。
_ENV = {
    "ptr": PointerType("f16", "Global"),
    "BR": ScalarType(COMPTIME_INT),
    "bm": ScalarType(INT),
}


def _comp(expr: str):
    return results.infer_shape_component(ast.parse(expr, mode="eval").body,
                                         _ENV)


class TestShapeComponentInference:
    def test_literal_and_symbols(self):
        """int 字面量 → ConstDim；comptime/int 符号 → SymbolDim 双态。"""
        assert _comp("128") == ConstDim(128)
        assert _comp("BR") == SymbolDim("BR", True)
        assert _comp("bm") == SymbolDim("bm", False)

    def test_arithmetic_derived(self):
        """算术组合 → DerivedDim（记录 op；子组件 comptime 性不在此裁决）。"""
        d = _comp("bm*BR + 8")
        assert isinstance(d, DerivedDim) and d.op == "+"

    def test_non_integer_category_none(self):
        """浮点/布尔字面量与已知非标量绑定 → None（E0406 证据）。"""
        assert _comp("2.5") is None
        assert _comp("True") is None
        assert _comp("ptr") is None  # Pointer 绑定名占组件位

    def test_unknown_and_exotic_deferred(self):
        """未绑定名与非常规表达式（属性/调用）→ UnknownDim 让渡。"""
        assert isinstance(_comp("q"), UnknownDim)
        assert isinstance(_comp("buf.K"), UnknownDim)
        assert isinstance(_comp("(bm + BR) * q"), UnknownDim)  # UNKNOWN 传播


class TestResultTypeTable:
    """D4 表逐行：合法实参组合 → 结果 TensorType。"""

    def test_make_tensor(self):
        t = results.make_tensor_result(_ENV["ptr"], (SymbolDim("bm", False),
                                                     ConstDim(64)))
        assert t == TensorType("f16", (SymbolDim("bm", False), ConstDim(64)),
                               "Global")

    def test_alloc_shared(self):
        dims = (ConstDim(64), SymbolDim("BR", True))
        assert results.alloc_shared_result("f32", dims) == \
            TensorType("f32", dims, "Shared")

    def test_zeros_full(self):
        dims = (ConstDim(64),)
        assert results.alloc_result("f16", dims, "Register") == \
            TensorType("f16", dims, "Register")  # zeros
        assert results.alloc_result("f32", dims, "Shared") == \
            TensorType("f32", dims, "Shared")    # full scope=Shared

    def test_cast(self):
        x = TensorType("f16", (ConstDim(64), SymbolDim("bm", False)), "Register")
        assert results.cast_result(x, "f32") == \
            TensorType("f32", x.dims, "Register")

    def test_dot(self):
        assert results.dot_result(ConstDim(64), ConstDim(128)) == \
            TensorType("f32", (ConstDim(64), ConstDim(128)), "Register")

    def test_reduce(self):
        x = TensorType("f32", (ConstDim(64), ConstDim(8), ConstDim(2)),
                       "Register")
        assert results.reduce_result(x, 1) == \
            TensorType("f32", (ConstDim(64), ConstDim(2)), "Register")

    def test_elementwise(self):
        x = TensorType("f16", (ConstDim(64),), "Shared")
        assert results.elementwise_result(x) == x  # maximum/exp/log 同构

    def test_transpose(self):
        x = TensorType("f32", (ConstDim(64), ConstDim(8)), "Register")
        assert results.transpose_result(x) == \
            TensorType("f32", (ConstDim(8), ConstDim(64)), "Register")
