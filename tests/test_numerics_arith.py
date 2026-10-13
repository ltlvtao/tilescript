"""R1 算术操作数与结果规则测试（numerics/value-semantics 实现面，tasks 2.x）。

断言公共可观察结果：E0601–E0604 错误码、四要素（category/suggestion）、
R7 合并（同表达式同码多操作数一条列全部位置）、增强赋值等价展开
（R1 Scenario 10）、算术赋值再绑定等价（design D2 承载面 + R7 Scenario 2
双命中择一只报 E0303）。
"""

import ast

from tilescript.numerics import check_module as numerics_check

_KERNEL_TMPL = '''\
import tis

@tis.kernel
def f(seq_len: int, scale: f32, D: comptime[int] = 64,
      BR: comptime[int] = 64, BC: comptime[int] = 64):
{body}
    return
'''


def _check(body: str, extra_params: str = ""):
    source = _KERNEL_TMPL.format(body=body)
    if extra_params:
        source = source.replace("BC: comptime[int] = 64):",
                                f"BC: comptime[int] = 64,{extra_params}):")
    return numerics_check(ast.parse(source))


def _codes(rejections):
    return [r.code for r in rejections]


class TestFamily:
    """tasks 2.1：家族与 dtype/scope 裁决（spec R1 Scenario 3–8 + R7 合并）。"""

    def test_cross_dtype_rejected(self):
        """家族内跨 dtype E0602，建议 tis.cast（Scenario 4）。"""
        rs = _check("""\
    A = tis.zeros((4, 4), f16, Register)
    B = tis.zeros((4, 4), f32, Register)
    C = A + B
""")
        assert _codes(rs) == ["E0602"]
        assert rs[0].category == "cross-dtype-arith"
        assert "tis.cast" in rs[0].suggestion

    def test_mixed_family_int_scalar_rejected(self):
        """int 标量 × f32 标量混家族 E0601（Scenario 5）。"""
        rs = _check("    x = seq_len * scale\n")
        assert _codes(rs) == ["E0601"]
        assert rs[0].category == "mixed-family-arith"
        assert "混家族" in rs[0].suggestion

    def test_int_literal_operand_not_exempted(self):
        """int 字面量作算术操作数 E0601，建议 float 字面量（Scenario 6）。"""
        rs = _check("""\
    S = tis.zeros((4, 4), f32, Register)
    T = S * 2
""")
        assert _codes(rs) == ["E0601"]
        assert "2.0" in rs[0].suggestion  # 建议 float 字面量形态

    def test_cross_scope_rejected_reports_both(self):
        """跨 scope E0603 报两侧 scope（Scenario 7）。"""
        rs = _check("""\
    R = tis.zeros((4,), f32, Register)
    S = tis.zeros((4,), f32, Shared)
    T = R + S
""")
        assert _codes(rs) == ["E0603"]
        assert rs[0].category == "tensor-incompatible"
        suggestion = rs[0].suggestion
        assert "Register" in suggestion and "Shared" in suggestion

    def test_int_family_division_rejected(self):
        """int 家族 `/` E0604，建议 tis.cdiv（Scenario 8）。"""
        rs = _check("    x = seq_len / BC\n")
        assert _codes(rs) == ["E0604"]
        assert "cdiv" in rs[0].suggestion

    def test_all_comptime_division_rejected(self):
        """全 comptime `/` 同样 E0604（折叠仅 + - *）。"""
        rs = _check("    x = BC / 2\n")
        assert _codes(rs) == ["E0604"]

    def test_int_dtype_division_rejected(self):
        """整型 dtype `/` E0604（浮点 dtype `/` 合法对照在同例断言外）。"""
        rs = _check("""\
    A = tis.zeros((4,), i32, Register)
    B = tis.zeros((4,), i32, Register)
    C = A / B
""")
        assert _codes(rs) == ["E0604"]

    def test_float_dtype_division_legal(self):
        """f32 `/` 合法（仅浮点 dtype 允许）。"""
        rs = _check("""\
    A = tis.zeros((4,), f32, Register)
    B = tis.zeros((4,), f32, Register)
    C = A / B
""")
        assert rs == []

    def test_runtime_int_times_comptime_is_int(self):
        """j*BC 非全 comptime → 运行期 int（Scenario 3：无违规零拒绝）。"""
        rs = _check("    x = seq_len * BC\n    y = seq_len + 1\n")
        assert rs == []

    def test_same_expr_multiple_operands_merged(self):
        """a+b+c 中 a、c 均混家族 → 一条 E0601 列两个位置（R7 Scenario 3）。"""
        rs = _check("""\
    A = tis.zeros((4,), f32, Register)
    T = A + seq_len
    T = A + seq_len + BC
""")
        # 第一条 T 单命中；第二条 (A + seq_len) + BC 外层再命中 BC——帧内同码合并
        assert _codes(rs) == ["E0601", "E0601"]
        merged = rs[1]
        assert "违规操作数" in merged.suggestion
        assert merged.suggestion.count("行") >= 2  # 列出全部违规操作数位置

    def test_different_codes_in_one_expr_both_reported(self):
        """同表达式不同码（不同位置）各报各条（R7 合并仅同码）。"""
        rs = _check("""\
    A = tis.zeros((4,), f16, Register)
    B = tis.zeros((4,), f32, Register)
    T = A + B + seq_len
""")
        # 内层 A+B 跨 dtype E0602；外层 (A+B)+seq_len 混家族 E0601（近似取 dtype 侧）
        codes = _codes(rs)
        assert "E0602" in codes and "E0601" in codes

    def test_bool_operand_rejected(self):
        """布尔值作算术操作数 E0605（R3 封闭集——判定序首）。"""
        rs = _check("    x = (seq_len > 0) + 1\n")
        assert _codes(rs) == ["E0605"]
        assert rs[0].category == "bool-closure"


class TestBroadcast:
    """tasks 2.2 前半：广播派生与 shape 判定（spec R1 Scenario 1/2/9）。"""

    def test_scalar_tensor_broadcast_legal(self):
        """Tensor × 同 dtype 标量广播（Scenario 1）。"""
        rs = _check("""\
    S = tis.zeros((4, 8), f32, Register)
    T = S * scale
""")
        assert rs == []

    def test_none_broadcast_expansion_legal(self):
        """alpha[:, None] * O_acc：dim1 的 1 扩展（Scenario 2）。"""
        rs = _check("""\
    alpha = tis.zeros((4,), f32, Register)
    acc = tis.zeros((4, 8), f32, Register)
    T = alpha[:, None] * acc
""")
        assert rs == []

    def test_broadcast_incompatible_rejected(self):
        """dim0 不等且均非 1 → E0603 报两侧 shape（Scenario 9）。"""
        rs = _check("""\
    A = tis.zeros((64, 32), f32, Register)
    B = tis.zeros((128, 32), f32, Register)
    C = A + B
""")
        assert _codes(rs) == ["E0603"]
        assert "64" in rs[0].suggestion or "shape" in rs[0].suggestion

    def test_fold_equivalent_dims_not_misreported(self):
        """ConstDim(64) vs 32*2 fold 等价——不误报（tasks 2.2 原文例）。"""
        rs = _check("""\
    A = tis.zeros((64,), f32, Register)
    B = tis.zeros((32 * 2,), f32, Register)
    C = A + B
""")
        assert rs == []

    def test_prefix_dim_preserved_legal(self):
        """不同 ndim：尾维对齐相等、长侧前缀维保留（(8,) + (4, 8)）。"""
        rs = _check("""\
    A = tis.zeros((8,), f32, Register)
    B = tis.zeros((4, 8), f32, Register)
    C = A + B
""")
        assert rs == []

    def test_unknown_dim_yields_silently(self):
        """seq_len 运行期维参与广播：SymbolDim 等值保留，零拒绝。"""
        rs = _check("""\
    A = tis.zeros((seq_len, 8), f32, Register)
    B = tis.zeros((seq_len, 8), f32, Register)
    C = A + B
""")
        assert rs == []


class TestAugAssign:
    """tasks 2.2 后半：增强赋值等价展开（spec R1 Scenario 10 + 增强面）。"""

    def test_aug_scalar_legal(self):
        """S *= scale 等价展开合法（Scenario 10）。"""
        rs = _check("""\
    S = tis.zeros((4, 8), f32, Register)
    S *= scale
""")
        assert rs == []

    def test_aug_int_literal_rejected(self):
        """S *= 2 → E0601（int 常量操作数不豁免）。"""
        rs = _check("""\
    S = tis.zeros((4, 8), f32, Register)
    S *= 2
""")
        assert _codes(rs) == ["E0601"]

    def test_aug_float_literal_legal(self):
        """S *= 2.0 合法（float 字面量按 S 的 dtype 定型，R4 位置②）。"""
        rs = _check("""\
    S = tis.zeros((4, 8), f32, Register)
    S *= 2.0
""")
        assert rs == []

    def test_aug_int_family_legal(self):
        """x *= 2（x:int）int 家族合法。"""
        rs = _check("    x = seq_len\n    x *= 2\n")
        assert rs == []

    def test_aug_int_division_rejected(self):
        """x /= 2（x:int）→ E0604。"""
        rs = _check("    x = seq_len\n    x /= 2\n")
        assert _codes(rs) == ["E0604"]

    def test_aug_cross_dtype_reports_e0602(self):
        """S_i8 *= scale_f32：展开跨 dtype E0602（等价成立保留单命中）。"""
        rs = _check("""\
    S = tis.zeros((4,), i8, Register)
    S *= scale
""")
        assert _codes(rs) == ["E0602"]

    def test_aug_unequivalent_reports_e0303(self):
        """x: f32 标量；x *= Tensor → 展开结果 Tensor 不等价 → 恰一条 E0303。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    x = scale
    x *= S
""")
        assert _codes(rs) == ["E0303"]
        assert rs[0].stage == "type-system"


class TestRebinding:
    """tasks 2.3：算术赋值再绑定等价（design D2——R7 授权面）。"""

    def test_unequivalent_single_hit_e0303(self):
        """x:int；x = alpha*scale（f32×f32）→ 恰一条 E0303（无 E06xx）。"""
        rs = _check("    x = seq_len\n    alpha2 = scale\n    x = alpha2 * scale\n")
        assert _codes(rs) == ["E0303"]
        assert rs[0].stage == "type-system"
        assert rs[0].order == 4
        assert rs[0].category == "assign"

    def test_double_hit_reports_only_e0303(self):
        """x:int；x = f32_t * f16_t → 只报 E0303 不报 E0602（R7 Scenario 2）。"""
        rs = _check("""\
    a = scale
    b = tis.full((4,), 1.5, f16, Register)
    b0 = b[0]
    x = seq_len
    x = scale * b0
""")
        assert _codes(rs) == ["E0303"]

    def test_equivalent_rebinding_keeps_e06xx(self):
        """等价再绑定保留 E06xx 单命中：T = T * 2 → 恰一条 E0601。"""
        rs = _check("""\
    T = tis.zeros((4, 8), f32, Register)
    T = T * 2
""")
        assert _codes(rs) == ["E0601"]

    def test_equivalent_clean_rebinding_legal(self):
        """等价且无运算违规：T = T + S 零拒绝。"""
        rs = _check("""\
    T = tis.zeros((4, 8), f32, Register)
    S = tis.zeros((4, 8), f32, Register)
    T = T + S
""")
        assert rs == []

    def test_unary_source_e0303(self):
        """一元负源再绑定不等价：x:int；x = -S → 恰一条 E0303。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    x = seq_len
    x = -S
""")
        assert _codes(rs) == ["E0303"]

    def test_non_arith_source_rebinding_yields(self):
        """非算术源（tis.* 调用）再绑定 M1 让渡：数值段零拒绝（residual 1）。"""
        rs = _check("""\
    S = tis.zeros((4,), f32, Register)
    T = tis.cast(S, f16)
    x = seq_len
    x = T
""")
        assert rs == []
