"""表达式类型推断测试（type-system spec R6「表达式结果类型规则」）。

Scenario 映射：
- 切片结果类型派生（L159——Q[bm*BR:(bm+1)*BR, :]）
- 广播维度派生（L165——m_new[:, None]）
- 标量下标被拒绝（L169——n[0] → E0303）
让渡面（design D4）与 UNKNOWN 组件传播逐项直测。
"""

from tilescript.frontend import carrier
from tilescript.typecheck import infer as infer_mod
from tilescript.typecheck import state_fields, symbols
from tilescript.typecheck.types import (
    UNKNOWN,
    ConstDim,
    DerivedDim,
    PointerType,
    ScalarType,
    SymbolDim,
    TensorType,
    UnknownDim,
    treats_as_unknown,
)


def _infer_expr(expr: str, env=None, prelude="", comptime=()):
    """在给定环境下推断单个表达式的类型（直测推断器本体）。"""
    source = f"import tis\n\n{prelude}\ndef __f__():\n    return {expr}\n"
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    comptime_syms = frozenset(comptime) | symbols.kernel_comptime_names(tree)
    registry, _ = state_fields.check(tree, comptime_syms=comptime_syms)
    env = dict(env or {})
    inc = infer_mod.Inferencer(env, registry)
    result = inc.infer(tree.body[-1].body[0].value)
    return result, inc.rejections


def _q():
    """Scenario L159 的 Q：Tensor[f16, (运行期 seq_len, comptime D), Global]。"""
    return TensorType("f16", (SymbolDim("seq_len", False), SymbolDim("D", True)),
                      "Global")


class TestConstantsAndNames:
    def test_int_literal_is_comptime(self):
        t, rs = _infer_expr("64")
        assert rs == [] and t == ScalarType("comptime_int")
        t, _ = _infer_expr("0x10")
        assert t == ScalarType("comptime_int")

    def test_relinquished_literals_unknown(self):
        for expr in ("2.5", "True", "None", "inf"):
            t, rs = _infer_expr(expr)
            assert rs == [] and t is UNKNOWN, expr

    def test_name_lookup_and_unbound_unknown(self):
        known = TensorType("f32", (ConstDim(64),), "Register")
        t, _ = _infer_expr("x", env={"x": known})
        assert t is known
        t, rs = _infer_expr("not_defined")
        assert rs == [] and t is UNKNOWN

    def test_arithmetic_and_calls_unknown(self):
        """让渡清单：算术/比较/逻辑/一元、tis.* 调用、pipe.run/range、属性访问。"""
        for expr in ("a + b", "a * 2", "a > 1", "a and b", "not a", "a if b else c",
                     "tis.zeros(64)", "tis.dot(a, b)", "pipe.run(1)",
                     "range(8)", "pipe.n_stages", "buf.K", "g.a"):
            t, rs = _infer_expr(expr, env={"a": ScalarType("int"),
                                           "b": ScalarType("int"),
                                           "c": ScalarType("int")})
            assert rs == [] and t is UNKNOWN, expr

    def test_unknown_base_derivation_unknown(self):
        base = TensorType("f32", (ConstDim(4),), "Register")
        t, rs = _infer_expr("u[1:3]", env={"u": UNKNOWN})
        assert rs == [] and t is UNKNOWN
        t, rs = _infer_expr("u.f", env={"u": UNKNOWN})
        assert rs == [] and t is UNKNOWN


class TestFieldAccess:
    def _env(self):
        st = state_fields.check(carrier.parse(
            "import tis\n\n@tis.state\nclass S:\n"
            "    m: Tensor[f32, (64,), Register]\n")[0])[0]["S"]
        return {"st": st}

    def test_state_field_access_declared_type(self):
        t, rs = _infer_expr("st.m", env=self._env())
        assert rs == []
        assert t == TensorType("f32", (ConstDim(64),), "Register")

    def test_unknown_field_and_non_state_base_unknown(self):
        t, rs = _infer_expr("st.nope", env=self._env())
        assert rs == [] and t is UNKNOWN
        x = TensorType("f32", (ConstDim(64),), "Register")
        t, rs = _infer_expr("x.m", env={"x": x})
        assert rs == [] and t is UNKNOWN


class TestSubscriptDerivation:
    def test_integer_index_drops_dim(self):
        base = TensorType("f32", (ConstDim(64), ConstDim(64)), "Register")
        t, rs = _infer_expr("x[2]", env={"x": base})
        assert rs == [] and t.dims == (ConstDim(64),)

    def test_slice_derivation_scenario(self):
        """Scenario L159：Q[bm*BR:(bm+1)*BR, :] → (运行期派生, comptime D)。"""
        bm = ScalarType("int"); BR = ScalarType("comptime_int")
        t, rs = _infer_expr("Q[bm*BR:(bm+1)*BR, :]",
                            env={"Q": _q(), "bm": bm, "BR": BR},
                            comptime=("BR",))
        assert rs == []
        assert t.dtype == "f16" and t.scope == "Global"
        d0 = t.dims[0]
        assert isinstance(d0, DerivedDim) and d0.comptime is False
        assert t.dims[1] == SymbolDim("D", True)  # `:` 保留原维

    def test_const_slice_const_dim(self):
        base = TensorType("f32", (SymbolDim("n", False),), "Global")
        t, rs = _infer_expr("x[2:8]", env={"x": base})
        assert rs == [] and t.dims == (ConstDim(6),)

    def test_none_broadcast_scenario(self):
        """Scenario L165：m_new[:, None] → (运行期 BR, comptime 1)。"""
        m_new = TensorType("f32", (SymbolDim("BR", False),), "Register")
        t, rs = _infer_expr("m_new[:, None]", env={"m_new": m_new})
        assert rs == []
        assert t.dims == (SymbolDim("BR", False), ConstDim(1))

    def test_none_leading_broadcast(self):
        base = TensorType("f32", (ConstDim(8),), "Register")
        t, rs = _infer_expr("x[None, :]", env={"x": base})
        assert rs == [] and t.dims == (ConstDim(1), ConstDim(8))

    def test_open_ended_slice_unknown_dim(self):
        """缺省边界（x[2:]）→ 未知维（上界不可静态派生）。"""
        base = TensorType("f32", (SymbolDim("n", False),), "Global")
        t, rs = _infer_expr("x[2:]", env={"x": base})
        assert rs == [] and isinstance(t.dims[0], UnknownDim)


class TestUnknownPropagation:
    def test_unknown_boundary_propagates_to_whole_tensor(self):
        """D4：切片边界含 UNKNOWN 操作数 → 维 UNKNOWN → 整体按 UNKNOWN。"""
        base = _q()
        t, rs = _infer_expr("Q[u:(u+1), :]", env={"Q": base, "u": UNKNOWN})
        assert rs == []
        assert isinstance(t.dims[0], UnknownDim)
        assert treats_as_unknown(t) and not treats_as_unknown(base)

    def test_equivalent_all_or_nothing(self):
        a = TensorType("f16", (UnknownDim(), SymbolDim("D", True)), "Global")
        b = TensorType("f16", (UnknownDim(), SymbolDim("D", True)), "Global")
        from tilescript.typecheck.types import equivalent
        assert not equivalent(a, b)  # 含未知维不判等（绑定层跳过）


class TestSubscriptBaseViolations:
    def test_scalar_subscript_rejected(self):
        """Scenario L169：n[0]（int 标量基对象）→ E0303。"""
        t, rs = _infer_expr("n[0]", env={"n": ScalarType("int")})
        assert t is UNKNOWN and len(rs) == 1
        r = rs[0]
        assert r.code == "E0303" and r.category == "subscript-base"
        assert "Tensor" in r.suggestion

    def test_state_and_pointer_base_rejected(self):
        st = state_fields.check(carrier.parse(
            "import tis\n\n@tis.state\nclass S:\n"
            "    m: Tensor[f32, (64,), Register]\n")[0])[0]["S"]
        t, rs = _infer_expr("st[0]", env={"st": st})
        assert len(rs) == 1 and rs[0].category == "subscript-base"
        t, rs = _infer_expr("p[0]", env={"p": PointerType("f16", "Global")})
        assert len(rs) == 1 and rs[0].code == "E0303"
