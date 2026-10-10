"""类型表示与等价判定测试（type-system spec R3「类型等价规则」×4 Scenario + 表示面）。

Scenario 映射：
- 逐组件相等判定等价（L66——layout 省略与显式 RowMajor 等价）
- 同符号动态维度判定等价（L71）
- 常量与运行期符号判定不等价（L76）
- 不同 dtype 标量判定不等价（L81）
"""

from tilescript.typecheck.types import (
    UNKNOWN,
    ConstDim,
    DerivedDim,
    PointerType,
    ScalarType,
    StateType,
    SymbolDim,
    TensorType,
    desc,
    equivalent,
)


def _t(dtype="f32", dims=None, scope="Register", layout=None):
    if layout is None:
        return TensorType(dtype=dtype, dims=tuple(dims or ()), scope=scope)
    return TensorType(dtype=dtype, dims=tuple(dims or ()), scope=scope, layout=layout)


class TestDimEquivalence:
    def test_const_dim_value_equality(self):
        assert ConstDim(64) == ConstDim(64)
        assert ConstDim(64) != ConstDim(65)

    def test_symbol_dim_name_and_phase(self):
        assert SymbolDim("seq_len", comptime=False) == SymbolDim("seq_len", comptime=False)
        # 同名不同相（comptime vs 运行期）不等——类别不同
        assert SymbolDim("D", comptime=True) != SymbolDim("D", comptime=False)
        assert SymbolDim("a", comptime=False) != SymbolDim("b", comptime=False)

    def test_derived_dim_structural_equality(self):
        # (bm+1)*BR 结构等价：同运算符、逐操作数等价
        a = DerivedDim("*", (DerivedDim("+", (SymbolDim("bm", False), ConstDim(1)), False),
                             SymbolDim("BR", True)), False)
        b = DerivedDim("*", (DerivedDim("+", (SymbolDim("bm", False), ConstDim(1)), False),
                             SymbolDim("BR", True)), False)
        assert a == b
        c = DerivedDim("*", (DerivedDim("+", (SymbolDim("bm", False), ConstDim(2)), False),
                             SymbolDim("BR", True)), False)
        assert a != c

    def test_cross_shape_inequality(self):
        """跨形态不等（无数值折叠）。"""
        assert ConstDim(64) != SymbolDim("D", comptime=True)  # 64 vs 符号 D
        assert ConstDim(128) != DerivedDim("*", (SymbolDim("BR", True), ConstDim(2)), True)
        assert SymbolDim("seq_len", False) != DerivedDim("-", (SymbolDim("seq_len", False),
                                                               ConstDim(1)), False)


class TestTensorEquivalence:
    def test_layout_omission_equivalent(self):
        """Scenario L66：Tensor[f32,(64,64),Register] ≡ Tensor[f32,(64,64),Register,RowMajor]。"""
        a = TensorType("f32", (ConstDim(64), ConstDim(64)), "Register", "RowMajor")
        b = TensorType("f32", (ConstDim(64), ConstDim(64)), "Register")  # 省略 layout
        assert equivalent(a, b) and equivalent(b, a)

    def test_same_runtime_symbol_dim_equivalent(self):
        """Scenario L71：两侧某维均为同一运行期符号 seq_len。"""
        a = _t("f16", (SymbolDim("seq_len", False), SymbolDim("D", True)), "Global")
        b = _t("f16", (SymbolDim("seq_len", False), SymbolDim("D", True)), "Global")
        assert equivalent(a, b)

    def test_const_vs_symbol_dim_not_equivalent(self):
        """Scenario L76：一侧常量 64、另一侧运行期符号 seq_len——不等价。"""
        a = _t("f32", (ConstDim(64),))
        b = _t("f32", (SymbolDim("seq_len", False),))
        assert not equivalent(a, b)

    def test_component_mismatch_not_equivalent(self):
        assert not equivalent(_t("f32", (ConstDim(64),)), _t("f16", (ConstDim(64),)))
        assert not equivalent(_t("f32", (ConstDim(64),), "Global"), _t("f32", (ConstDim(64),), "Shared"))
        assert not equivalent(_t("f32", (ConstDim(64),)), _t("f32", (ConstDim(64), ConstDim(64))))


class TestScalarAndStateEquivalence:
    def test_different_dtype_scalars_not_equivalent(self):
        """Scenario L81：f32 标量 × f16 标量不等价。"""
        assert not equivalent(ScalarType("f32"), ScalarType("f16"))

    def test_scalar_kind_equivalence(self):
        assert equivalent(ScalarType("int"), ScalarType("int"))
        assert equivalent(ScalarType("comptime_int"), ScalarType("comptime_int"))
        # 不同种类之间不等价（comptime→int 单向兼容由 bindings 裁，equivalent 恒 False）
        assert not equivalent(ScalarType("int"), ScalarType("comptime_int"))

    def test_state_type_nominal_equivalence(self):
        """状态类名义等价：同类名等价、字段结构相同但类名不同不等价。"""
        fields = (("O_acc", _t("f32", (ConstDim(64),))),)
        assert equivalent(StateType("AttnState", fields), StateType("AttnState", fields))
        assert equivalent(StateType("AttnState", fields),
                          StateType("AttnState", (("O_acc", _t("f16", (ConstDim(64),))),)))
        assert not equivalent(StateType("AttnState", fields), StateType("OtherState", fields))

    def test_cross_category_not_equivalent(self):
        assert not equivalent(_t("f32", (ConstDim(64),)), ScalarType("f32"))
        assert not equivalent(StateType("S", ()), _t("f32", (ConstDim(64),)))

    def test_unknown_never_equivalent(self):
        """UNKNOWN 恒不产生等价结论（bindings 层先跳过，防御层返回 False）。"""
        assert not equivalent(UNKNOWN, _t("f32", (ConstDim(64),)))
        assert not equivalent(UNKNOWN, UNKNOWN)


class TestPointer:
    def test_pointer_equivalence(self):
        """Pointer 逐组件等价：同 dtype+scope 等；任一组件异不等。"""
        assert equivalent(PointerType("f16", "Global"), PointerType("f16", "Global"))
        assert not equivalent(PointerType("f16", "Global"), PointerType("f32", "Global"))
        assert not equivalent(PointerType("f16", "Global"), PointerType("f16", "Shared"))

    def test_pointer_cross_category(self):
        """Pointer 与 Tensor/标量跨类别不等（空 dims Tensor 不等于 Pointer）。"""
        assert not equivalent(PointerType("f16", "Global"),
                              TensorType("f16", (), "Global"))
        assert not equivalent(PointerType("f16", "Global"), ScalarType("f16"))

    def test_pointer_desc(self):
        assert desc(PointerType("f16", "Global")) == "Pointer[f16, Global]"


class TestDescription:
    def test_desc_frozen_text(self):
        assert desc(_t("f16", (SymbolDim("seq_len", False), SymbolDim("D", True)), "Global")) == \
            "Tensor[f16, (运行期 seq_len, comptime D), Global]"
        assert desc(_t("f32", (ConstDim(64), ConstDim(64)), "Register", "RowMajor")) == \
            "Tensor[f32, (64, 64), Register]"
        assert desc(ScalarType("int")) == "int"
        assert desc(ScalarType("comptime_int")) == "comptime[int]"
        assert desc(ScalarType("f32")) == "f32"
        assert desc(StateType("AttnState", (("m", ScalarType("int")),))) == "AttnState"
        assert desc(UNKNOWN) == "未知类型（结果类型由后续检查段承载）"
