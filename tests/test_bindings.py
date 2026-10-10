"""绑定检查 E0303 测试（type-system spec R6 单类型不变量 + R7 绑定四类位置）。

Scenario 映射：
- 单类型不变量违规被拒绝（R6 L152——int 后绑 Tensor）
- dtype 不匹配的赋值被拒绝（R7 L178——建议 tis.cast）
- scope 不匹配的赋值被拒绝并指向显式原语（R7 L183）
- comptime 值绑定 int 位置被接受（R7 L188——显式调用 j: int 形参传 64）
- 状态构造实参类型不匹配被拒绝（R7 L193——AttnState(m=...) 定位实参）
让渡面负例（design D4）随行。
"""

from tilescript.frontend import carrier
from tilescript.frontend.report import finalize
from tilescript.typecheck import annotations, bindings, state_fields, symbols


def _run(body: str, kernel_params_decl="", prelude=""):
    """构造单 kernel + produce/consume 模块并运行设备函数绑定遍历。

    body 写在 kernel 体内；kernel_params_decl 是 kernel 形参声明文本。
    返回 finalize 后的拒绝清单。
    """
    source = (
        "import tis\n"
        "\n"
        f"{prelude}"
        "\n"
        "@tis.kernel\n"
        f"def kernel({kernel_params_decl}):\n"
        f"{body}"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    comptime = symbols.kernel_comptime_names(tree)
    registry, rej = state_fields.check(tree, comptime_syms=comptime)
    rej += annotations.check(tree, comptime_syms=comptime, registry=registry)
    rej += bindings.check_device_functions(tree, registry, comptime)
    return finalize(rej)


def _indented(*lines):
    return "".join(f"    {line}\n" for line in lines)


_KERNEL_PARAMS = (
    "a: Tensor[f16, (64,), Register], "
    "b: Tensor[f32, (64,), Register], "
    "sh: Tensor[f16, (64,), Shared], "
    "n: int"
)


class TestAssignBindings:
    def test_single_type_invariant_violation_rejected(self):
        """Scenario R6 L152：首次绑定 int 标量后再绑 Tensor → E0303 定位再绑定。"""
        rs = _run(_indented(
            "n = 5",
            "n = b",
        ), kernel_params_decl=_KERNEL_PARAMS)
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "assign" and rs[0].line == 7  # 再绑定行
        assert "Tensor[f32, (64), Register]" in rs[0].suggestion

    def test_dtype_mismatch_suggests_cast(self):
        """Scenario R7 L178：f16 值赋 f32 目标 → 建议 tis.cast。"""
        rs = _run(_indented("b = a"), kernel_params_decl=_KERNEL_PARAMS)
        assert [r.code for r in rs] == ["E0303"] and rs[0].category == "assign"
        assert "tis.cast" in rs[0].suggestion
        assert "f16" in rs[0].suggestion and "f32" in rs[0].suggestion

    def test_scope_mismatch_suggests_move_primitive(self):
        """Scenario R7 L183：Register 值赋 Shared 目标 → 建议显式移动原语。"""
        rs = _run(_indented("sh = a"), kernel_params_decl=_KERNEL_PARAMS)
        assert [r.code for r in rs] == ["E0303"]
        assert "移动原语" in rs[0].suggestion

    def test_shape_mismatch_and_equivalent_binding_accepted(self):
        same = "d: Tensor[f16, (64,), Register]"
        wider = "w: Tensor[f16, (64, 64), Register]"
        rs = _run(_indented(
            "d = a",   # 等价绑定：通过
            "w = a",   # shape 不匹配 → E0303 建议核对维度
        ), kernel_params_decl=_KERNEL_PARAMS + ", " + same + ", " + wider)
        assert [r.code for r in rs] == ["E0303"]
        assert "维度" in rs[0].suggestion

    def test_subscript_target_binding_checked(self):
        """Subscript 目标：目标侧派生类型 × 源类型。"""
        t64 = "t64: Tensor[f32, (64, 64), Register]"
        rs = _run(_indented(
            "t64[0] = a",  # 目标 (64,) f32 × 源 (64,) f16 → dtype 差异
        ), kernel_params_decl=_KERNEL_PARAMS + ", " + t64)
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "assign" and "tis.cast" in rs[0].suggestion

    def test_attribute_target_binding_checked(self):
        """Attribute 目标：字段声明类型 × 源类型（R5 异类/异型互赋面）。"""
        prelude = (
            "@tis.state\n"
            "class AttnState:\n"
            "    m: Tensor[f32, (64,), Register]\n"
            "\n"
        )
        rs = _run(_indented(
            "st = AttnState(m=b)",
            "st.m = a",  # 字段 f32 × 源 f16 → E0303
        ), kernel_params_decl="a: Tensor[f16, (64,), Register], "
                              "b: Tensor[f32, (64,), Register]",
            prelude=prelude)
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "assign"

    def test_state_nominal_mismatch_reported_both_sides(self):
        """R5 异类状态互赋：两侧类名进报告（名义等价失败）。"""
        prelude = (
            "@tis.state\n"
            "class A:\n"
            "    x: Tensor[f32, (64,), Register]\n"
            "\n"
            "@tis.state\n"
            "class B:\n"
            "    x: Tensor[f32, (64,), Register]\n"
            "\n"
        )
        rs = _run(_indented(
            "sa = A(x=b)",
            "sb = B(x=b)",
            "sb = sa",
        ), kernel_params_decl="b: Tensor[f32, (64,), Register]", prelude=prelude)
        assert [r.code for r in rs] == ["E0303"]
        assert "A" in rs[0].suggestion and "B" in rs[0].suggestion


class TestUnknownRelinquishment:
    def test_unknown_binding_never_reports(self):
        """让渡负例：UNKNOWN 源/目标不报（x = tis.zeros(...) 后 x = 5）。"""
        rs = _run(_indented(
            "x = tis.zeros((64,))",
            "x = 5",
            "u = x[0]",
            "y = u[1:3]",
        ))
        assert rs == []

    def test_unknown_dim_propagation_not_reported(self):
        """UNKNOWN 边界切片 → 未知维整体 UNKNOWN：再绑已知 Tensor 不报。"""
        rs = _run(_indented(
            "x = tis.zeros((64,))",
            "q = a[x : x + 16]",  # 边界 UNKNOWN → 未知维
            "q = a",              # 目标含未知维 → 按 UNKNOWN 参与绑定，跳过
        ), kernel_params_decl=_KERNEL_PARAMS)
        assert rs == []

    def test_loop_and_with_variables_unknown(self):
        rs = _run(_indented(
            "for i in range(n):",
            "    i = 5",
            "    a = i",
        ))
        assert rs == []


class TestReturnAndCalls:
    """return×返回注解与第 4 类调用实参绑定——全部经 kernel 内嵌套
    `@pipe.*` 真实形态走公共管线（模块级 @tis.* 形态被语法段拒绝）。"""

    def _pipeline(self, body_inner: str, call_stmts: str, kernel_params="n: int"):
        from tilescript import pipeline
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            f"def k({kernel_params}):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "\n"
            f"{body_inner}"
            "\n"
            f"{call_stmts}"
            "    return\n"
        )
        return pipeline.compile_stages(source, target="nvidia_h200")

    def test_return_checked_against_annotation(self):
        """return × 返回注解（嵌套 consume 形态）。"""
        from tilescript import pipeline
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "\n"
            "    @pipe.consume\n"
            "    def epilogue(j: int, buf) -> Tensor[f32, (64,), Register]:\n"
            "        return a\n"
            "\n"
            "    return\n"
        )
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"] and rs[0].category == "return"
        assert "tis.cast" in rs[0].suggestion

    def test_return_without_annotation_not_checked(self):
        rs = self._pipeline(
            "    @pipe.produce\n"
            "    def p(j: int, buf):\n"
            "        return 5\n",
            "",
        )
        assert rs == []

    def test_state_ctor_kwarg_mismatch_rejected(self):
        """Scenario R7 L193：构造关键字实参类型×字段声明类型，定位实参。"""
        from tilescript import pipeline
        source = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class AttnState:\n"
            "    m: Tensor[f32, (64,), Register]\n"
            "    l: Tensor[f32, (64,), Register]\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register],"
            " b: Tensor[f32, (64,), Register]):\n"
            "    st = AttnState(m=a, l=b)  # m 字段 f32 × 实参 a f16 → E0303\n"
            "    return\n"
        )
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "state-ctor-arg"
        assert (rs[0].line, rs[0].col) == (10, 22)  # 关键字实参值节点

    def test_annotated_call_args_checked_through_pipeline(self):
        """D7 第 4 类：显式调用嵌套 produce——comptime 值 → int 位接受（R7 L188）。"""
        body = (
            "    @pipe.produce\n"
            "    def inner(j: int, buf):\n"
            "        ...\n"
        )
        rs = self._pipeline(body, "    inner(64)\n    inner(j=n)\n")
        assert rs == []
        rs = self._pipeline(
            body,
            "    x = 5\n"
            "    inner(x)\n"      # comptime_int → int：单向兼容接受
            "    inner(a)\n",     # Tensor → int 形参：拒绝
            kernel_params="n: int, a: Tensor[f16, (64,), Register]",
        )
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "call-arg" and "标量种类" in rs[0].suggestion

    def test_runtime_int_to_comptime_position_rejected(self):
        """R2 Scenario：运行期 int 值绑定 comptime[int] 位置 → E0303（反向不兼容）。"""
        body = (
            "    @pipe.produce\n"
            "    def inner(j: comptime[int], buf):\n"
            "        ...\n"
        )
        assert self._pipeline(body, "    inner(64)\n") == []      # comptime 常量接受
        rs = self._pipeline(body, "    inner(n)\n")              # 运行期 int 拒绝
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "call-arg" and "comptime" in rs[0].suggestion

    def test_unannotated_call_and_position_ctor_not_checked(self):
        """负例：无注解形参位不查；状态构造位置实参不查（proposal 非目标）。"""
        from tilescript import pipeline
        source = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    m: Tensor[f32, (64,), Register]\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "\n"
            "    @pipe.produce\n"
            "    def inner(j, buf):\n"
            "        ...\n"
            "\n"
            "    inner(a, a)\n"     # 形参无注解：不查
            "    st = S(a)\n"       # 构造位置实参：不查
            "    return\n"
        )
        assert pipeline.compile_stages(source, target="nvidia_h200") == []

    def test_explicit_call_legality_not_reported(self):
        """负例：produce/consume 显式调用合法性归 E0502——M1 本段不报调用本身。"""
        rs = self._pipeline(
            "    @pipe.produce\n"
            "    def inner(j: int):\n"
            "        ...\n",
            "    inner(5)\n",
        )
        assert rs == []
