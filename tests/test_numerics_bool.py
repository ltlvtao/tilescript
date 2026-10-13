"""R3 比较逻辑与布尔封闭测试（numerics/value-semantics 实现面，tasks 4.x）。

断言公共可观察结果：合法比较条件零拒绝、非布尔条件 E0605（不支持真值
语义）、比较操作数家族规则（混家族/跨 dtype/Tensor/待定型）、布尔封闭
集外使用 E0605（赋值绑定/实参/返回/下标/一元负）、UNKNOWN 让渡。
"""

import ast

from tilescript.numerics import check_module as numerics_check

_KERNEL_TMPL = '''\
import tis

@tis.state
class St:
    O_acc: Tensor[f32, (4,), Register]

@tis.kernel
def f(seq_len: int, scale: f32, v_f16: f16, D: comptime[int] = 64,
      BR: comptime[int] = 64, BC: comptime[int] = 64):
{body}
    return
'''


def _check(body: str):
    return numerics_check(ast.parse(_KERNEL_TMPL.format(body=body)))


def _codes(rejections):
    return [r.code for r in rejections]


class TestCompare:
    """tasks 4.1：比较与条件（spec R3 Scenario 1/2/4 + 操作数家族规则）。"""

    def test_legal_int_comparison_condition(self):
        """if seq_len > 0: 合法（int 家族 comptime/int 单向兼容混用，S1）。"""
        rs = _check("""\
    if seq_len > 0:
        x = seq_len + 1
""")
        assert rs == []

    def test_comptime_comparison_legal(self):
        """if BC > 60: 合法（全 comptime 比较）。"""
        rs = _check("""\
    if BC > 60:
        x = seq_len
""")
        assert rs == []

    def test_non_bool_condition_rejected(self):
        """if seq_len: E0605，建议显式比较（S2——不支持真值语义）。"""
        rs = _check("""\
    if seq_len:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]
        assert rs[0].category == "bool-closure"
        assert "真值" in rs[0].suggestion or "布尔" in rs[0].suggestion

    def test_while_condition_rejected(self):
        """while seq_len: 同构 E0605。"""
        rs = _check("""\
    while seq_len:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]

    def test_tensor_comparison_rejected(self):
        """if S == T:（Tensor）E0605，建议计算原语域后续 change（S4）。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    T = tis.zeros((4,), f32, Register)
    if S == T:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]
        assert "计算原语" in rs[0].suggestion or "逐元素" in rs[0].suggestion

    def test_mixed_family_comparison_rejected(self):
        """seq_len > scale 混家族比较 E0605。"""
        rs = _check("    x = seq_len > scale\n")
        assert _codes(rs) == ["E0605"]

    def test_cross_dtype_comparison_rejected(self):
        """scale > v_f16 家族内跨 dtype 比较 E0605。"""
        rs = _check("    x = scale > v_f16\n")
        assert _codes(rs) == ["E0605"]

    def test_same_dtype_scalar_comparison_legal(self):
        """同 dtype 标量比较合法（结果布尔作条件——条件语境封闭集内）。"""
        rs = _check("""\
    if scale > scale:
        x = seq_len
""")
        assert rs == []

    def test_pending_operand_comparison_rejected(self):
        """scale > 0.0：待定型常量作比较操作数 E0605（比较限已定型标量）。"""
        rs = _check("    x = scale > 0.0\n")
        assert _codes(rs) == ["E0605"]

    def test_unknown_condition_yields(self):
        """UNKNOWN 条件让渡（未绑定名数值段不报——类型段 E03xx 域）。"""
        rs = _check("""\
    if mystery_name:
        x = seq_len
""")
        assert rs == []


class TestClosure:
    """tasks 4.2：布尔使用封闭集（spec R3 Scenario 3 + 正文封闭集枚举）。"""

    def test_bool_assignment_rejected(self):
        """flag = a > b 赋值绑定 E0605（S3）。"""
        rs = _check("    flag = seq_len > 0\n")
        assert _codes(rs) == ["E0605"]
        assert "封闭集" in rs[0].suggestion

    def test_bool_literal_assignment_rejected(self):
        """flag = True 字面量绑定 E0605（True/False 亦是布尔值）。"""
        rs = _check("    flag = True\n")
        assert _codes(rs) == ["E0605"]

    def test_bool_as_primitive_arg_rejected(self):
        """以 a > b 为原语实参 E0605（S3 后半）。"""
        rs = _check("    A = tis.full((2,), seq_len > 0, i32, Register)\n")
        assert _codes(rs) == ["E0605"]

    def test_bool_as_state_ctor_arg_rejected(self):
        """布尔值作状态类构造实参 E0605（非 tis 调用实参位）。"""
        rs = _check("    st = St(O_acc=seq_len > 0)\n")
        assert _codes(rs) == ["E0605"]

    def test_bool_return_rejected(self):
        """布尔值作返回值 E0605（封闭集外）。"""
        rs = _check("    return seq_len > 0\n")
        assert _codes(rs) == ["E0605"]

    def test_bool_subscript_rejected(self):
        """布尔值作下标表达式 E0605。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    v = S[seq_len > 0]
""")
        assert _codes(rs) == ["E0605"]

    def test_bool_unary_neg_rejected(self):
        """-(a > b) 布尔一元负 E0605。"""
        rs = _check("    x = -(seq_len > 0)\n")
        assert _codes(rs) == ["E0605"]

    def test_bool_as_arith_operand_in_condition(self):
        """条件内算术用布尔：if (a > b) + 1 > 0: E0605。"""
        rs = _check("""\
    if (seq_len > 0) + 1 > 0:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]

    def test_logical_ops_require_bool_operands(self):
        """and/or/not 操作数须布尔：seq_len and BC → E0605。"""
        rs = _check("""\
    if seq_len > 0 and seq_len:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]

    def test_not_non_bool_rejected(self):
        """not seq_len E0605。"""
        rs = _check("""\
    if not seq_len:
        x = seq_len
""")
        assert _codes(rs) == ["E0605"]

    def test_legal_logical_condition(self):
        """if a > 0 and b > 60: 合法（逻辑运算操作数布尔、结果布尔）。"""
        rs = _check("""\
    if seq_len > 0 and BC > 60:
        x = seq_len
""")
        assert rs == []

    def test_not_bool_legal(self):
        """if not (seq_len > 0): 合法。"""
        rs = _check("""\
    if not (seq_len > 0):
        x = seq_len
""")
        assert rs == []
