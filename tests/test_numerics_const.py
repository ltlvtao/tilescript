"""R2/R4 待定型常量与 dtype 位置绑定 + R5 折叠 + R6 cast 测试（tasks 3.x/5.x）。

断言公共可观察结果：E0606 错误码与恢复建议（值域/实际值、dtype 语境、
E4M3 无穷缺失）、位置③再绑定定型和拒绝、comptime 折叠值联动（full value
位值域裁决证明折叠发生）、cast 返回类型派生参与后续算术。
"""

import ast

from tilescript.numerics import check_module as numerics_check

_KERNEL_TMPL = '''\
import tis

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


class TestPending:
    """tasks 3.1：待定型常量——裸绑定/算术操作数位/位置③（R2 S4/S5 + R4 S4/S5）。"""

    def test_float_literal_bare_binding_rejected(self):
        """x = 1.5 裸绑定 E0606（R4 Scenario 4）。"""
        rs = _check("    x = 1.5\n")
        assert _codes(rs) == ["E0606"]
        assert rs[0].category == "const-binding"
        assert "dtype" in rs[0].suggestion

    def test_inf_bare_binding_rejected(self):
        """x = inf 裸绑定 E0606（R2 Scenario 4——与 float 字面量同构）。"""
        rs = _check("    x = inf\n")
        assert _codes(rs) == ["E0606"]
        assert "inf" in rs[0].suggestion or "具名常量" in rs[0].suggestion

    def test_negated_float_bare_binding_rejected(self):
        """x = -1.5 一元负形态裸绑定 E0606。"""
        rs = _check("    x = -1.5\n")
        assert _codes(rs) == ["E0606"]

    def test_neg_inf_bare_binding_rejected(self):
        """x = -inf 裸绑定 E0606。"""
        rs = _check("    x = -inf\n")
        assert _codes(rs) == ["E0606"]

    def test_pending_times_int_family_rejected(self):
        """待定型 × int 家族 E0601（无 dtype 语境不可定型）。"""
        rs = _check("    x = 1.5 + seq_len\n")
        assert _codes(rs) == ["E0601"]

    def test_rebind_float_to_f32_legal(self):
        """x 已绑 f32；x = 1.5 按已绑定 dtype 定型（R4 Scenario 5）。"""
        rs = _check("    x = scale\n    x = 1.5\n")
        assert rs == []

    def test_rebind_inf_to_f32_legal(self):
        """x 已绑 f32；x = inf 合法（IEEE 浮点无穷）。"""
        rs = _check("    x = scale\n    x = inf\n")
        assert rs == []

    def test_rebind_float_to_int_rejected(self):
        """x 已绑 int；x = 1.5 → E0606（位置③ int 家族拒）。"""
        rs = _check("    x = seq_len\n    x = 1.5\n")
        assert _codes(rs) == ["E0606"]

    def test_rebind_int_literal_to_f32_rejected(self):
        """x 已绑 f32；x = 4（int 字面量）→ E0606（int 常量绑定限原语参数位）。"""
        rs = _check("    x = scale\n    x = 4\n")
        assert _codes(rs) == ["E0606"]

    def test_unary_neg_preserves_tensor(self):
        """-P 保持 dtype/shape（R2 Scenario 5）零拒绝。"""
        rs = _check("""\
    P = tis.zeros((4, 8), f32, Register)
    N = -P
""")
        assert rs == []

    def test_unary_neg_comptime_fold_negative(self):
        """-BC 折叠取负（COMPTIME 值联动：-BC - 65 = -129 越界 i8）。"""
        rs = _check("    A = tis.full((1,), -BC, i8, Register)\n")
        assert rs == []
        rs = _check("    A = tis.full((1,), -BC - 65, i8, Register)\n")
        assert _codes(rs) == ["E0606"]


class TestFullValue:
    """tasks 3.2：tis.full value 位常量绑定（R4 S1/2/3/6/7 + R2 S1/2/3）。"""

    def test_int_literal_in_range_legal(self):
        """full((4,), 0, i32) 合法（R4 Scenario 1）。"""
        rs = _check("    A = tis.full((4,), 0, i32, Register)\n")
        assert rs == []

    def test_int_literal_out_of_range_rejected(self):
        """full((4,), 300, i8) E0606 报值域与实际值（R4 Scenario 2）。"""
        rs = _check("    A = tis.full((4,), 300, i8, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "300" in rs[0].suggestion and "i8" in rs[0].suggestion

    def test_float_literal_to_float_dtype_legal(self):
        """full((2,), 3.14159, f16) 合法（roundTiesToEven 行为承诺，S3）。"""
        rs = _check("    A = tis.full((2,), 3.14159, f16, Register)\n")
        assert rs == []

    def test_float_literal_overflow_not_compile_rejected(self):
        """full((2,), 1e5, f16) 值饱和到 inf 但编译期不拒（R6 运行期承诺）。"""
        rs = _check("    A = tis.full((2,), 1e5, f16, Register)\n")
        assert rs == []

    def test_int_to_float_dtype_legal(self):
        """整到浮：full((2,), 3, f32) 合法。"""
        rs = _check("    A = tis.full((2,), 3, f32, Register)\n")
        assert rs == []

    def test_float_literal_to_int_dtype_rejected(self):
        """full((4,), 1.5, i32) E0606（浮点常量限浮点 dtype）。"""
        rs = _check("    A = tis.full((4,), 1.5, i32, Register)\n")
        assert _codes(rs) == ["E0606"]

    def test_runtime_int_name_rejected(self):
        """full((4,), n, f32)（n 运行期 int）E0606（R4 Scenario 6）。"""
        rs = _check("    A = tis.full((4,), seq_len, f32, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "宿主" in rs[0].suggestion or "运行期" in rs[0].suggestion

    def test_cross_dtype_scalar_rejected(self):
        """full((2,), v, f16)（v 为 f32 标量）E0606（R4 Scenario 7）。"""
        rs = _check("    A = tis.full((2,), scale, f16, Register)\n")
        assert _codes(rs) == ["E0606"]

    def test_same_dtype_scalar_legal(self):
        """full((2,), v_f16, f16) 同 dtype 标量合法（负例）。"""
        rs = _check("    A = tis.full((2,), v_f16, f16, Register)\n")
        assert rs == []

    def test_inf_to_i32_rejected(self):
        """full((4,), inf, i32) E0606 无穷无整数表示（R2 Scenario 2）。"""
        rs = _check("    A = tis.full((4,), inf, i32, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "无穷" in rs[0].suggestion

    def test_inf_to_f8e4m3_rejected(self):
        """full((2,), inf, f8e4m3) E0606 E4M3 无穷表示缺失（R2 Scenario 3）。"""
        rs = _check("    A = tis.full((2,), inf, f8e4m3, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "f8e4m3" in rs[0].suggestion or "E4M3" in rs[0].suggestion

    def test_inf_to_f16_legal(self):
        """full((2,), inf, f16) 合法（IEEE 标准精度浮点）。"""
        rs = _check("    A = tis.full((2,), inf, f16, Register)\n")
        assert rs == []

    def test_neg_inf_to_f32_legal(self):
        """full((BR,), -inf, f32) 合法（R2 Scenario 1——负无穷位模式承诺）。"""
        rs = _check("    A = tis.full((BR,), -inf, f32, Register)\n")
        assert rs == []

    def test_neg_inf_to_f8e4m3_rejected(self):
        """-inf 同样拒 f8e4m3（一元负形态承载）。"""
        rs = _check("    A = tis.full((2,), -inf, f8e4m3, Register)\n")
        assert _codes(rs) == ["E0606"]


class TestFold:
    """tasks 5.1：comptime 折叠与 cdiv 值（R5 Scenario 1/2/3）。"""

    def test_cdiv_all_comptime_folds(self):
        """cdiv(128, 64) 值 2——值域联动证明（300/1 折叠 300 越界 i8）。"""
        rs = _check("    A = tis.full((1,), tis.cdiv(128, 64), i8, Register)\n")
        assert rs == []
        rs = _check("    A = tis.full((1,), tis.cdiv(300, 1), i8, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "300" in rs[0].suggestion

    def test_cdiv_ceil_semantics(self):
        """ceil(a/b) 数学向上取整：cdiv(7, 2) = 4（i8 值域联动 257/64=5）。"""
        rs = _check("    A = tis.full((1,), tis.cdiv(7, 2) - 4, i32, Register)\n")
        assert rs == []

    def test_cdiv_runtime_not_folded(self):
        """cdiv(seq_len, BC) → 运行期 int（Scenario 2），算术参与零拒绝。"""
        rs = _check("    x = tis.cdiv(seq_len, BC) + 1\n")
        assert rs == []

    def test_bc_arith_folds(self):
        """BC*2-1 = 127（Scenario 3）：绑 i32 合法。"""
        rs = _check("    A = tis.full((1,), BC * 2 - 1, i32, Register)\n")
        assert rs == []

    def test_fold_value_range_connection(self):
        """R5×R4 连接：BC*4+68 = 324 绑 i8 → E0606（折叠 324 越界）。"""
        rs = _check("    A = tis.full((1,), BC * 4 + 68, i8, Register)\n")
        assert _codes(rs) == ["E0606"]
        assert "324" in rs[0].suggestion

    def test_cdiv_zero_divisor_silent(self):
        """cdiv(1, 0) b 值 0 让渡不报（E0505 已拒，本段不重复）。"""
        rs = _check("    x = tis.cdiv(1, 0)\n")
        assert rs == []


class TestCast:
    """tasks 5.2：cast 间接承载（R6——返回类型派生 + 运行期承诺不上卡）。"""

    def test_cast_result_dtype_participates(self):
        """cast(S, f16) 结果 dtype f16：× f32 标量 → E0602（派生证明）。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    T = tis.cast(S, f16)
    U = T * scale
""")
        assert _codes(rs) == ["E0602"]
        assert "f16" in rs[0].suggestion and "f32" in rs[0].suggestion

    def test_cast_shape_scope_carried(self):
        """cast 承接 shape/scope：cast((4,8) f32→f16) + zeros((4,8) f16) 合法。"""
        rs = _check("""\
    S = tis.zeros((4, 8), f32, Register)
    T = tis.cast(S, f16)
    A = tis.zeros((4, 8), f16, Register)
    U = T + A
""")
        assert rs == []

    def test_cast_shape_mismatch_detected(self):
        """cast 后 shape 参与：cast((4,)) 结果 + (8,) f16 → E0603（承接证明）。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    T = tis.cast(S, f16)
    A = tis.zeros((8,), f16, Register)
    U = T + A
""")
        assert _codes(rs) == ["E0603"]

    def test_cast_runtime_overflow_not_compile_rejected(self):
        """cast 运行期溢出不上卡：饱和规则是行为承诺，编译期零拒绝（注记）。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    T = tis.cast(S, f8e4m3)
""")
        assert rs == []
