"""E0302 注解解析器测试（type-system spec R1「类型表达式结构规则」×4 + R2 shape 组件×2）。

Scenario 映射：
- 合法类型表达式通过结构检查（L12——Tensor 三参/Pointer 两参/Tensor 四参 layout）
- 未知 dtype 被拒绝（L17）
- 参数顺序错误被拒绝（L22）
- 集合外 scope 被拒绝（L27）
- 浮点 shape 组件被拒绝（L45）
- dtype 标量注解被接受（L55）/ 动态维度（L40，随 2.2 符号判定）
"""

import ast

from tilescript.frontend import carrier
from tilescript.typecheck import annotations


def _ann(annotation: str, comptime=(), runtime=()):
    """构造单注解表达式源并解析（kernel 签名形态经 E0104 已过；直测解析器本体）。"""
    source = (
        "import tis\n"
        "\n"
        "@tis.state\n"
        "class S:\n"
        f"    x: {annotation}\n"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    field = tree.body[1].body[0]
    return annotations.parse_annotation(
        field.annotation,
        comptime_syms=frozenset(comptime), runtime_syms=frozenset(runtime),
    )


class TestLegalForms:
    def test_three_legal_forms_accepted(self):
        """Scenario L12：Tensor 三参 / Pointer 两参 / Tensor 四参 RowMajor。"""
        t, r = _ann("Tensor[f32, (64, D), Register]", comptime=("D",))
        assert r is None and t.dtype == "f32" and t.scope == "Register"
        assert t.dims == (annotations.ConstDim(64), annotations.SymbolDim("D", True))
        p, r = _ann("Pointer[f16, Global]")
        assert r is None and isinstance(p, annotations.PointerType)
        assert p.dtype == "f16" and p.scope == "Global"
        t4, r = _ann("Tensor[f16, (8, D), Global, RowMajor]", comptime=("D",))
        assert r is None and t4.layout == "RowMajor"

    def test_scalar_annotations_accepted(self):
        """Scenario L55：dtype 标量注解；int 与 comptime[int]。"""
        t, r = _ann("f32")
        assert r is None and t.kind == "f32"
        t, r = _ann("int")
        assert r is None and t.kind == "int"
        t, r = _ann("comptime[int]")
        assert r is None and t.kind == "comptime_int"

    def test_dynamic_dim_accepted(self):
        """Scenario L40：运行期 int 标量为 shape 维度（动态维度合法）。"""
        t, r = _ann("Tensor[f16, (seq_len, D), Global]", runtime=("seq_len",), comptime=("D",))
        assert r is None
        assert t.dims[0] == annotations.SymbolDim("seq_len", False)

    def test_derived_shape_component(self):
        """算术派生 shape 组件：全 comptime→comptime 派生；含运行期→运行期派生。"""
        t, r = _ann("Tensor[f32, (BR*2, D), Register]", comptime=("BR", "D"))
        assert r is None and t.dims[0].comptime is True and t.dims[0].op == "*"
        t, r = _ann("Tensor[f32, (bm*BR, D), Register]",
                    runtime=("bm",), comptime=("BR", "D"))
        assert r is None and t.dims[0].comptime is False


class TestE0302Rejections:
    def test_unknown_dtype_rejected(self):
        """Scenario L17：f64 不在封闭集，建议列六种。"""
        t, r = _ann("Tensor[f64, (64, 64), Register]")
        assert t is None and r.code == "E0302" and r.category == "unknown-dtype"
        for d in ("f16", "bf16", "f32", "f8e4m3", "i8", "i32"):
            assert d in r.suggestion

    def test_param_order_rejected(self):
        """Scenario L22：Tensor[Register, (64,), f32]——dtype 位收到 scope 名。"""
        t, r = _ann("Tensor[Register, (64,), f32]")
        assert t is None and r.code == "E0302" and r.category == "invalid-param-order"
        assert "dtype" in r.suggestion

    def test_unknown_scope_rejected(self):
        """Scenario L27：Distributed 不在 scope 封闭集，建议列三值。"""
        t, r = _ann("Tensor[f32, (64,), Distributed]")
        assert t is None and r.code == "E0302" and r.category == "unknown-scope"
        assert all(s in r.suggestion for s in ("Global", "Shared", "Register"))

    def test_float_shape_component_rejected(self):
        """Scenario L45：(2.5, D)——浮点 shape 组件。"""
        t, r = _ann("Tensor[f32, (2.5, D), Register]", comptime=("D",))
        assert t is None and r.code == "E0302" and r.category == "invalid-shape-component"

    def test_param_count_and_form_rejected(self):
        """参数数量（Tensor 两参/Pointer 三参）与形式（未知下标名/非注解形态/layout 值）。"""
        assert _ann("Tensor[f32, (64,)]")[1].category == "invalid-param-count"
        assert _ann("Pointer[f16, Global, Register]")[1].category == "invalid-param-count"
        assert _ann("Vector[f32, (64,)]")[1].category == "invalid-annotation-form"
        assert _ann("comptime[str]")[1].category == "invalid-annotation-form"
        assert _ann("Tensor[f16, (8,), Global, ColumnMajor]")[1].category == "invalid-layout"

    def test_bool_shape_component_rejected(self):
        """布尔不是整数常量组件（bool 非 int——E0302 shape 组件）。"""
        t, r = _ann("Tensor[f32, (True, D), Register]", comptime=("D",))
        assert t is None and r.category == "invalid-shape-component"

    def test_position_one_based(self):
        """E0302 报告定位注解表达式起点（col 1 起算）。"""
        source = "import tis\n\n@tis.state\nclass S:\n    x: Tensor[f64, (64,), Register]\n"
        tree, _ = carrier.parse(source)
        field = tree.body[1].body[0]
        t, r = annotations.parse_annotation(field.annotation)
        assert (r.line, r.col) == (5, 8)


class TestCheckPositions:
    """2.3 三类检查位置接入：kernel 签名 / @tis.state 字段（state_fields）/
    produce/consume 形参与返回注解（含状态类名查注册表）。"""

    def _check(self, source):
        from tilescript.typecheck import state_fields, symbols
        tree, rejection = carrier.parse(source)
        assert rejection is None, f"载体意外拒绝：{rejection}"
        comptime = symbols.kernel_comptime_names(tree)
        registry, rejections = state_fields.check(tree, comptime_syms=comptime)
        rejections = rejections + annotations.check(tree, comptime_syms=comptime,
                                                    registry=registry)
        return rejections

    def test_kernel_signature_e0302(self):
        src = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def kernel(x: Tensor[f64, (64,), Global]):\n"
            "    ...\n"
        )
        rs = self._check(src)
        assert [r.code for r in rs] == ["E0302"]
        assert rs[0].category == "unknown-dtype"

    def test_kernel_state_class_name_rejected(self):
        """kernel 形参注解为状态类名 → E0302（状态类仅 consume 签名合法）。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    a: Tensor[f32, (64,), Register]\n"
            "\n"
            "\n"
            "@tis.kernel\n"
            "def kernel(st: S):\n"
            "    ...\n"
        )
        rs = self._check(src)
        assert [r.code for r in rs] == ["E0302"]
        assert rs[0].category == "invalid-annotation-form"

    def test_consume_signature_annotations_checked(self):
        """produce/consume 形参与返回注解过 E0302；None 返回跳过；状态类名接受。"""
        good = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    a: Tensor[f32, (64,), Register]\n"
            "\n"
            "\n"
            "@tis.consume\n"
            "def epilogue(st: S, out: Pointer[f32, Global]) -> None:\n"
            "    ...\n"
        )
        assert self._check(good) == []
        bad = (
            "import tis\n"
            "\n"
            "@tis.produce\n"
            "def p(x: Tensor[f32, (2.5,), Global]) -> Tensor[i64, (64,), Register]:\n"
            "    ...\n"
        )
        rs = self._check(bad)
        assert [r.code for r in rs] == ["E0302", "E0302"]
        assert [r.category for r in rs] == ["invalid-shape-component", "unknown-dtype"]

    def test_unannotated_param_and_none_return_skipped(self):
        """无注解形参与无返回注解不查（D7：无注解形参位不查）。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.produce\n"
            "def p(x, y: Tensor[f32, (64,), Global]):\n"
            "    ...\n"
        )
        assert self._check(src) == []

    def test_plain_function_annotations_not_checked(self):
        """三类位置之外的普通函数注解不查（M1 边界，design D5）。"""
        src = (
            "import tis\n"
            "\n"
            "def helper(x: Whatever[f64]):\n"
            "    ...\n"
        )
        assert self._check(src) == []


class TestNestedPipelineSignatures:
    """代码审查 cycle 1 Major 1：kernel 内嵌套 `@pipe.produce/consume` 签名
    注解是 R1 明文辖域（语法层不检查注解类别，类型层是其唯一防线）——
    经类型段直测验证（第四段接入后载体形态归执行段契约，见 _nested）。"""

    def _nested(self, inner_sig: str):
        """嵌套签名载体走类型段直测（第四段接入后显式调用/Pipeline 缺参
        形态归 E0502 执行段承载，类型段单面直测保持焦点）。"""
        from tilescript import typecheck
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
            f"{inner_sig}"
            "    return\n"
        )
        return typecheck.check_module(ast.parse(source))

    def test_nested_param_annotation_rejected(self):
        """嵌套 produce 形参注解 f64 → E0302 unknown-dtype（审查探针 1）。"""
        rs = self._nested(
            "    @pipe.produce\n"
            "    def fetch(j: int, buf: Tensor[f64, (64,), Register]):\n"
            "        ...\n"
        )
        assert [r.code for r in rs] == ["E0302"]
        assert rs[0].category == "unknown-dtype"

    def test_nested_return_annotation_rejected(self):
        """嵌套 consume 返回注解 Distributed → E0302 unknown-scope。"""
        rs = self._nested(
            "    @pipe.consume\n"
            "    def epilogue(st: S, buf)"
            " -> Tensor[f32, (64,), Distributed]:\n"
            "        ...\n"
        )
        assert [r.code for r in rs] == ["E0302"]
        assert rs[0].category == "unknown-scope"

    def test_nested_legal_signature_with_state_class_accepted(self):
        """合法嵌套签名（状态类名形参 + Pointer + 返回注解）零拒绝。"""
        rs = self._nested(
            "    @pipe.consume\n"
            "    def epilogue(st: S, buf,"
            " out: Pointer[f32, Global]) -> Tensor[f32, (64,), Register]:\n"
            "        ...\n"
        )
        assert rs == []
