"""计算原语检查测试（compute-ops spec：E0408 调用结构与操作数/参数值域）。

4.1 面：R1×5——六原语参数集、dot mma/pad 关键字专属、reduce 未知关键字/
缺失必选、transpose 超量。操作数契约（4.2）、E0402/E0403（4.3）、语境
专用值（4.4）、其余原语（4.5）、段内序与确定性（4.6）增量接入。

入口走公共面 primitives.check_module（接受例为后续面的前瞻合法构造）。
"""

from tilescript.frontend import carrier
from tilescript.primitives import check_module

_PARAMS = (
    "aS: Tensor[f16, (16, 16), Shared], "      # dot A (M,K)
    "bS: Tensor[f16, (16, 16), Shared], "      # dot B (K,N)
    "cR: Tensor[f32, (16, 16), Register], "    # dot C (M,N)
    "aR: Tensor[f16, (16, 16), Register], "    # dot A Register 形态
    "sR: Tensor[f32, (64, 8), Register], "     # reduce x
    "n: int, "
    "BR: comptime[int]"
)


def _run(body: str, params=_PARAMS, target="nvidia_h200"):
    source = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        f"def k({params}):\n"
        f"{body}"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return check_module(tree, target)


def _stmt(line: str):
    return f"    {line}\n"


class TestCallStructure:
    """R1：参数集结构（E0408 param-set）。"""

    def test_dot_omitted_keywords_accepted(self):
        """Scenario 1：dot(A, B, C) 省略 mma/pad 接受（Auto/Error；
        16/16/16 被 h200 Auto 选择 (16,8,16) 整除）。"""
        assert _run(_stmt("q = tis.dot(aS, bS, cR)")) == []

    def test_dot_keyword_only_positional_rejected(self):
        """Scenario 2：mma 按第四位置实参 → E0408 注明 A/B/C 三个与
        关键字专属。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, tis.MMA(16, 8, 16))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "param-set"
        assert "A/B/C" in rs[0].suggestion and "关键字" in rs[0].suggestion

    def test_dot_explicit_mma_pad_accepted(self):
        """R2 Scenario 1 前置：显式 mma=tis.MMA(16,8,16) + pad=Error 接受
        （在 h200 支持列表且整除）。"""
        assert _run(_stmt(
            "q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 8, 16),"
            " pad=tis.PadPolicy.Error)"
        )) == []

    def test_reduce_unknown_keyword_rejected(self):
        """Scenario 3：keepdims 不在参数集 → E0408 报合法参数集。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=1, op=Max, keepdims=True)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "param-set"
        assert "x/axis/op/scope" in rs[0].suggestion

    def test_reduce_missing_required_rejected(self):
        """Scenario 4：缺必选 op → E0408 注明必选且值域 Sum/Max。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=1)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "param-set"
        assert "op" in rs[0].suggestion and "Sum/Max" in rs[0].suggestion

    def test_transpose_too_many_positional_rejected(self):
        """Scenario 5：transpose 两位置实参 → E0408 只接受一个。"""
        rs = _run(_stmt("t = tis.transpose(sR, 0, 1)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "param-set" and "位置实参" in rs[0].suggestion

    def test_reduce_positional_full_form_accepted(self):
        """R1 正文：reduce(S, 1, Max, Block) 位置全形态等价关键字
        （h200 支持 Block）。"""
        assert _run(_stmt("r = tis.reduce(sR, 1, Max, Block)")) == []

    def test_dot_keyword_form_of_positional_accepted(self):
        """可按位置参数的关键字形式：dot(A=aS, B=bS, C=cR) 等价。"""
        assert _run(_stmt("q = tis.dot(A=aS, B=bS, C=cR)")) == []


class TestDotOperands:
    """R2 操作数契约（E0408）：dtype/scope/rank/常量维/K 相容/C 形状、
    MMA 构造与 pad 值域形态。"""

    def test_comptime_symbol_dims_accepted(self):
        """Scenario 2：A/B 为 comptime 符号维（BR/BC）接受——E0403 无数值
        不触发（D5 域，4.3 负例再固定）。"""
        params = ("aB: Tensor[f16, (BR, BC), Shared], "
                  "bB: Tensor[f16, (BC, BR), Shared], "
                  "cB: Tensor[f32, (BR, BR), Register], "
                  "BR: comptime[int], BC: comptime[int]")
        assert _run(_stmt("q = tis.dot(aB, bB, cB)"), params=params) == []

    def test_operand_dtype_rejected(self):
        """Scenario 3：A 为 f32 / B 为 f32 → E0408 注明 f16。"""
        params = _PARAMS + ", a32: Tensor[f32, (16, 16), Shared]," \
                           " b32: Tensor[f32, (16, 16), Shared]"
        rs = _run(_stmt("q = tis.dot(a32, bS, cR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-dtype" and "f16" in rs[0].suggestion
        rs = _run(_stmt("q = tis.dot(aS, b32, cR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-dtype"

    def test_c_dtype_and_scope_rejected(self):
        """C 面：dtype f16 或 scope Shared → E0408 注明 f32/Register。"""
        params = _PARAMS + ", c16: Tensor[f16, (16, 16), Register]," \
                           " cS: Tensor[f32, (16, 16), Shared]"
        rs = _run(_stmt("q = tis.dot(aS, bS, c16)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-dtype" and "f32" in rs[0].suggestion
        rs = _run(_stmt("q = tis.dot(aS, bS, cS)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-scope" and "Register" in rs[0].suggestion

    def test_operand_global_scope_rejected(self):
        """A/B scope 值域 Shared|Register：Global 直传 → E0408。"""
        params = _PARAMS + ", g16: Tensor[f16, (16, 16), Global]"
        rs = _run(_stmt("q = tis.dot(g16, bS, cR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-scope"
        assert "Shared/Register" in rs[0].suggestion

    def test_operand_rank_rejected(self):
        """(M,K)/(K,N)/(M,N) 二维契约：一维 A → E0408。"""
        params = _PARAMS + ", a1: Tensor[f16, (16,), Shared]"
        rs = _run(_stmt("q = tis.dot(a1, bS, cR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-shape" and "二维" in rs[0].suggestion

    def test_k_incompatible_rejected(self):
        """Scenario 5：A 的 K=32 与 B 的 K=16 → E0408 K 相容面。"""
        params = _PARAMS + ", aK: Tensor[f16, (16, 32), Shared]"
        rs = _run(_stmt("q = tis.dot(aK, bS, cR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "k-incompatible" and "K" in rs[0].suggestion

    def test_c_mn_shape_rejected(self):
        """C 必须为 (M, N)：C (32,16) 对 A/B 的 (16,16) → E0408。"""
        params = _PARAMS + ", cMN: Tensor[f32, (32, 16), Register]"
        rs = _run(_stmt("q = tis.dot(aS, bS, cMN)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mn-incompatible"

    _SLICE_KERNEL = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(j: int, n: int, a: Tensor[f16, (128, 16), Shared],"
        " b: Tensor[f16, (16, 16), Shared],"
        " c: Tensor[f32, (16, 16), Register]):\n"
        "{body}"
        "    return\n"
    )

    def test_runtime_derived_dim_rejected(self):
        """Scenario 6：A 的 shape 含运行期派生维（a[0:n, :] 的 n 派生）→
        E0408 编译期常量维契约。"""
        source = self._SLICE_KERNEL.format(
            body="    q = tis.dot(a[0:n, :], b, c)\n")
        tree, rejection = carrier.parse(source)
        assert rejection is None
        rs = check_module(tree, "nvidia_h200")
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-dim-const"
        assert "编译期常量维" in rs[0].suggestion

    def test_folded_constant_slice_dim_accepted(self):
        """(j+1)*16-j*16 折叠为纯常数 16 → 编译期可知，接受
        （E0403 整除仍按 D5 窄域不触发——4.3 固定）。"""
        source = self._SLICE_KERNEL.format(
            body="    q = tis.dot(a[j*16:(j+1)*16, :], b, c)\n")
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert check_module(tree, "nvidia_h200") == []

    def test_mma_runtime_int_rejected(self):
        """MMA 构造：运行期 int 实参 / 两实参 / float 实参 → E0408。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(n, 8, 16))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-form" and "comptime" in rs[0].suggestion
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 8))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-form"
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(1.5, 8, 16))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-form"

    def test_mma_form_domain(self):
        """mma 槽形态：非 MMA 构造非 Auto → E0408 值域面。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=RowMajor)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-form" and "Auto" in rs[0].suggestion

    def test_pad_domain_rejected(self):
        """pad 封闭四值：集合外属性 / 非 PadPolicy 形态 → E0408。"""
        rs = _run(_stmt(
            "q = tis.dot(aS, bS, cR, pad=tis.PadPolicy.Zero)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "pad-value"
        assert "PadPolicy" in rs[0].suggestion
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, pad=RowMajor)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "pad-value"

    def test_unknown_operand_yields(self):
        """任一操作数 UNKNOWN（produce 形参 buf）→ 整体让渡零拒绝。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(b: Tensor[f16, (16, 16), Shared],"
            " c: Tensor[f32, (16, 16), Register]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "\n"
            "    @pipe.produce\n"
            "    def p(j: int, buf):\n"
            "        q = tis.dot(buf, b, c)\n"
            "\n"
            "    return\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert check_module(tree, "nvidia_h200") == []


class TestMmaSupportAndAlignment:
    """R2 硬件面（E0402/E0403）：MMA 支持列表（两目标差异）、Auto 选择
    整除、pad 策略、D5 可判定域。"""

    def test_explicit_mma_not_supported_rejected(self):
        """显式 (16,16,16) 不在 h200 列表 → E0402 报全列表（登记顺序）。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 16, 16))"))
        assert [r.code for r in rs] == ["E0402"]
        assert rs[0].category == "mma-unsupported"
        assert "(16, 8, 16)" in rs[0].suggestion \
            and "(16, 8, 32)" in rs[0].suggestion
        assert (rs[0].suggestion.index("(16, 8, 16)")
                < rs[0].suggestion.index("(16, 8, 32)"))

    def test_explicit_mma_supported_accepted(self):
        """列表内形状两目标：h200 (16,8,32)（K=32）/ ascend (16,16,16)。"""
        params = (_PARAMS + ", aK2: Tensor[f16, (16, 32), Shared],"
                  " bK2: Tensor[f16, (32, 16), Shared]")
        assert _run(_stmt("q = tis.dot(aK2, bK2, cR, mma=tis.MMA(16, 8, 32))"),
                    params=params) == []
        assert _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 16, 16))"),
                    target="ascend_910b") == []

    def test_two_targets_differ(self):
        """hal「跨 HAL 行为不变面」：同一显式形状 (16,8,16) 在 h200 接受、
        在 ascend_910b → E0402 报其列表 (16, 16, 16)。"""
        assert _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 8, 16))")) == []
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 8, 16))"),
                  target="ascend_910b")
        assert [r.code for r in rs] == ["E0402"]
        assert "(16, 16, 16)" in rs[0].suggestion

    def test_auto_unaligned_error_rejected(self):
        """h200 Auto (16,8,16)：M=17 不整除且默认 pad=Error → E0403
        最近对齐建议 32。"""
        params = (_PARAMS + ", a17: Tensor[f16, (17, 16), Shared],"
                  " c17: Tensor[f32, (17, 16), Register]")
        rs = _run(_stmt("q = tis.dot(a17, bS, c17)"), params=params)
        assert [r.code for r in rs] == ["E0403"]
        assert rs[0].category == "mma-unaligned"
        assert "32" in rs[0].suggestion and "pad" in rs[0].suggestion

    def test_unaligned_with_policies_accepted(self):
        """不整除 + pad=PadZero/Mask/Split → 显式边界处理，通过。"""
        params = (_PARAMS + ", a17: Tensor[f16, (17, 16), Shared],"
                  " c17: Tensor[f32, (17, 16), Register]")
        for policy in ("PadZero", "Mask", "Split"):
            assert _run(_stmt(f"q = tis.dot(a17, bS, c17,"
                              f" pad=tis.PadPolicy.{policy})"),
                        params=params) == []

    def test_explicit_mma_alignment(self):
        """显式形状的整除判定：K=33 对 (16,8,32) → E0403 建议 64。"""
        params = (_PARAMS + ", a33: Tensor[f16, (16, 33), Shared],"
                  " b33: Tensor[f16, (33, 16), Shared]")
        rs = _run(_stmt("q = tis.dot(a33, b33, cR, mma=tis.MMA(16, 8, 32))"),
                  params=params)
        assert [r.code for r in rs] == ["E0403"]
        assert "64" in rs[0].suggestion

    def test_comptime_symbol_dims_skip_e0403(self):
        """D5 可判定域负例：M 为 comptime 符号 BR（无数值）→ E0403 不触发
        （K/N 常量维正常整除判定）。"""
        params = ("aB: Tensor[f16, (BR, 16), Shared], "
                  "bS2: Tensor[f16, (16, 16), Shared], "
                  "cB: Tensor[f32, (BR, 16), Register], "
                  "BR: comptime[int]")
        assert _run(_stmt("q = tis.dot(aB, bS2, cB)"), params=params) == []

    def test_explicit_comptime_mma_yields(self):
        """显式 mma 含 comptime 名（数值不可判定）：E0402/E0403 让渡。"""
        assert _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(BR, 8, 16))")) == []


class TestSpecialValueContext:
    """R2 末 Scenario：MMA/PadPolicy 值仅限 dot 的 mma=/pad= 关键字实参位
    （design D7 独立扫描）；其余任何出现 → E0408。"""

    def test_mma_assignment_rejected(self):
        rs = _run(_stmt("m = tis.MMA(16, 8, 16)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-context" and "mma" in rs[0].suggestion

    def test_padpolicy_assignment_rejected(self):
        rs = _run(_stmt("p = tis.PadPolicy.Mask"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "pad-context" and "pad" in rs[0].suggestion

    def test_mma_as_call_arg_rejected(self):
        """其他原语实参位（cast 的 x）→ E0408（cast 本身让渡零叠加）。"""
        rs = _run(_stmt("q = tis.cast(tis.MMA(16, 8, 16), f16)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-context"

    def test_mma_in_arithmetic_rejected(self):
        """算术操作数（BinOp right）→ E0408。"""
        rs = _run(_stmt("x = 1 + tis.MMA(16, 8, 16)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-context"

    def test_padpolicy_in_compare_rejected(self):
        """比较条件操作数位（D7 明文面；If 条件不经遍历推断，无先拒）→
        E0408。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k():\n"
            "    if tis.PadPolicy.Mask == tis.PadPolicy.Error:\n"
            "        pass\n"
            "    return\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        rs = check_module(tree, "nvidia_h200")
        assert [r.code for r in rs] == ["E0408", "E0408"]  # 比较两侧各一节点
        assert all(r.category == "pad-context" for r in rs)
        assert [r.line for r in rs] == [5, 5]

    def test_mma_in_dot_c_position_rejected(self):
        """MMA 传 dot 的 C 位置实参：dot 操作数让渡，语境扫描承载拒绝。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, tis.MMA(16, 8, 16))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "mma-context"

    def test_legal_keyword_position_single_report(self):
        """合法语境（mma= 关键字位）：语境扫描零叠加（E0402 面单条）。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 16, 16))"))
        assert [r.code for r in rs] == ["E0402"]

    def test_hit_coordination_single_report(self):
        """位置传 MMA：param-set 已报（dot hit），语境扫描不叠加。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, tis.MMA(16, 8, 16))"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "param-set"


class TestReduce:
    """R3：reduce 契约（E0408）——Register x、axis comptime 界内、op 二值、
    scope 三值 + HAL 支持面。"""

    def test_scopes_accepted_on_h200(self):
        """Scenario 1/7：Warp 与 Block 均在 h200 支持列表，接受。"""
        assert _run(_stmt("r = tis.reduce(sR, axis=1, op=Max, scope=Warp)")) == []
        assert _run(_stmt("r = tis.reduce(sR, 1, Max, Block)")) == []

    def test_x_global_rejected(self):
        """Scenario 2：x 跨 scope（Global）→ E0408 建议先转移到 Register。"""
        params = _PARAMS + ", gl8: Tensor[f32, (64, 8), Global]"
        rs = _run(_stmt("r = tis.reduce(gl8, axis=1, op=Sum)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "x-scope" and "Register" in rs[0].suggestion \
            and "load" in rs[0].suggestion

    def test_axis_runtime_int_rejected(self):
        """Scenario 3：axis 为运行期 int 名 → E0408 注明 comptime[int]。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=n, op=Sum)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "axis-form" and "comptime" in rs[0].suggestion

    def test_axis_out_of_range_rejected(self):
        """Scenario 4：axis=2 对 rank 2 → E0408 报值域 [0, 2)。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=2, op=Sum)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "axis-range" and "[0, 2)" in rs[0].suggestion

    def test_op_domain_rejected(self):
        """Scenario 5：op=Prod → E0408 二值域。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=1, op=Prod)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "op-value" and "Sum/Max" in rs[0].suggestion

    def test_scope_unsupported_on_ascend_rejected(self):
        """Scenario 6：ascend_910b 不支持 Warp → E0408 报目标支持清单
        （Block）。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=1, op=Max, scope=Warp)"),
                  target="ascend_910b")
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "scope-unsupported" and "Block" in rs[0].suggestion

    def test_scope_domain_rejected(self):
        """scope 值域三值外（Grid）→ E0408。"""
        rs = _run(_stmt("r = tis.reduce(sR, axis=1, op=Max, scope=Grid)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "scope-value" and "Auto/Warp/Block" in rs[0].suggestion

    def test_comptime_axis_yields(self):
        """axis=comptime 名（无数值）：界内让渡、接受（去维结果 UNKNOWN）。"""
        assert _run(_stmt("r = tis.reduce(sR, axis=BR, op=Sum)")) == []


class TestMaximum:
    """R4：maximum 严格同 dtype/shape/Register。"""

    def test_identical_operands_accepted(self):
        params = _PARAMS + ", e1: Tensor[f32, (64, 8), Register]," \
                           " e2: Tensor[f32, (64, 8), Register]"
        assert _run(_stmt("m = tis.maximum(e1, e2)"), params=params) == []

    def test_dtype_mismatch_rejected(self):
        params = _PARAMS + ", e1: Tensor[f32, (64, 8), Register]," \
                           " e16: Tensor[f16, (64, 8), Register]"
        rs = _run(_stmt("m = tis.maximum(e1, e16)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-mismatch" and "dtype" in rs[0].suggestion

    def test_shape_mismatch_rejected(self):
        params = _PARAMS + ", e1: Tensor[f32, (64, 8), Register]," \
                           " ev: Tensor[f32, (64,), Register]"
        rs = _run(_stmt("m = tis.maximum(e1, ev)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-mismatch" and "shape" in rs[0].suggestion

    def test_scope_mismatch_rejected(self):
        params = _PARAMS + ", e1: Tensor[f32, (64, 8), Register]," \
                           " esh: Tensor[f32, (64, 8), Shared]"
        rs = _run(_stmt("m = tis.maximum(e1, esh)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-scope" and "Register" in rs[0].suggestion

    def test_folded_equivalent_shape_accepted(self):
        """数学等价 shape（seq_len*2 与 seq_len+seq_len）按维度相容语义接受。"""
        params = (_PARAMS + ", f1: Tensor[f32, (seq_len*2,), Register],"
                  " f2: Tensor[f32, (seq_len+seq_len,), Register]")
        assert _run(_stmt("m = tis.maximum(f1, f2)"), params=params) == []


class TestUnaryMath:
    """R4/R5：exp/log 浮点 Register Tensor。"""

    def test_float_dtypes_accepted(self):
        params = (_PARAMS + ", h16: Tensor[f16, (64,), Register],"
                  " hbf: Tensor[bf16, (64,), Register],"
                  " hf8: Tensor[f8e4m3, (64,), Register]")
        for x in ("sR", "h16", "hbf", "hf8"):
            assert _run(_stmt(f"y = tis.exp({x})"), params=params) == []
            assert _run(_stmt(f"y = tis.log({x})"), params=params) == []

    def test_integer_dtype_rejected(self):
        params = _PARAMS + ", i8r: Tensor[i8, (64,), Register]"
        rs = _run(_stmt("y = tis.exp(i8r)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "dtype-float" and "浮点" in rs[0].suggestion
        rs = _run(_stmt("y = tis.log(i8r)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "dtype-float"

    def test_global_scope_rejected(self):
        params = _PARAMS + ", gf: Tensor[f32, (64,), Global]"
        rs = _run(_stmt("y = tis.exp(gf)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "x-scope" and "Register" in rs[0].suggestion


class TestTranspose:
    """R5：transpose 恰二维、Shared|Register。"""

    def test_two_dim_scopes_accepted(self):
        params = _PARAMS + ", t2r: Tensor[f16, (8, 16), Register]," \
                           " t2s: Tensor[f32, (8, 16), Shared]"
        assert _run(_stmt("t = tis.transpose(t2r)"), params=params) == []
        assert _run(_stmt("t = tis.transpose(t2s)"), params=params) == []

    def test_three_dim_rejected(self):
        params = _PARAMS + ", t3: Tensor[f32, (2, 3, 4), Register]"
        rs = _run(_stmt("t = tis.transpose(t3)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "x-shape" and "二维" in rs[0].suggestion

    def test_global_scope_rejected(self):
        params = _PARAMS + ", t2g: Tensor[f16, (8, 16), Global]"
        rs = _run(_stmt("t = tis.transpose(t2g)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "x-scope" and "Shared/Register" in rs[0].suggestion


class TestStageOrderAndDeterminism:
    """R7：段内序（E0408 操作数/值域先于 E0402/E0403）+ 位置升序 +
    Auto 重复编译一致。"""

    def test_e0408_beats_e0402_same_call(self):
        """pad 形态违规（E0408）+ mma 不在支持列表（E0402）→ 只报 E0408。"""
        rs = _run(_stmt("q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 16, 16),"
                        " pad=RowMajor)"))
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "pad-value"
        # 操作数面（E0408）同先于 E0402。
        params = _PARAMS + ", a32: Tensor[f32, (16, 16), Shared]"
        rs = _run(_stmt("q = tis.dot(a32, bS, cR, mma=tis.MMA(16, 16, 16))"),
                  params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "operand-dtype"

    def test_e0408_beats_e0403_same_call(self):
        """不整除（E0403）+ pad 形态违规（E0408）→ 只报 E0408。"""
        params = (_PARAMS + ", a17: Tensor[f16, (17, 16), Shared],"
                  " c17: Tensor[f32, (17, 16), Register]")
        rs = _run(_stmt("q = tis.dot(a17, bS, c17, pad=RowMajor)"),
                  params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "pad-value"

    def test_multi_position_ascending(self):
        """跨 checker 多拒绝：位置升序（行 5 E0402、行 6 E0406）。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            f"def k({_PARAMS}):\n"
            "    q = tis.dot(aS, bS, cR, mma=tis.MMA(16, 16, 16))\n"
            "    tis.load(gl, sh, mode=Fast)\n"
            "    return\n"
        ).replace(", n: int, BR: comptime[int]",
                  ", n: int, BR: comptime[int]"
                  ", gl: Tensor[f16, (64,), Global], sh: Tensor[f16, (64,), Shared]")
        tree, rejection = carrier.parse(source)
        assert rejection is None
        rs = check_module(tree, "nvidia_h200")
        assert [r.code for r in rs] == ["E0402", "E0406"]
        assert [r.line for r in rs] == [5, 6]

    def test_nested_violation_reported_once(self):
        """嵌套 tis.* 违规恰一条：前置全量推断已记录，槽位二次推断
        不重放（路由层按调用节点缓存结果——同节点检查只发生一次）。"""
        params = _PARAMS + ", i8r: Tensor[i8, (64,), Register]"
        rs = _run(_stmt("r = tis.maximum(tis.exp(i8r), sR)"), params=params)
        assert [r.code for r in rs] == ["E0408"]
        assert rs[0].category == "dtype-float"

    def test_auto_repeated_compilation_identical(self):
        """Auto 选择确定性：同输入同目标重复编译逐条一致（含拒绝面）。"""
        params = (_PARAMS + ", a17: Tensor[f16, (17, 16), Shared],"
                  " c17: Tensor[f32, (17, 16), Register]")
        rs = _run(_stmt("q = tis.dot(a17, bS, c17)\n"
                        "    r = tis.reduce(sR, axis=1, op=Max, scope=Warp)"),
                  params=params, target="ascend_910b")
        assert [r.code for r in rs] == ["E0403", "E0408"]
        rs2 = _run(_stmt("q = tis.dot(a17, bS, c17)\n"
                         "    r = tis.reduce(sR, axis=1, op=Max, scope=Warp)"),
                   params=params, target="ascend_910b")
        assert [r.to_dict() for r in rs] == [r.to_dict() for r in rs2]
