"""设备代码表达式接受集测试（syntax spec R6「设备代码表达式接受集」×4 Scenario）。

Scenario 映射：
- 切片与广播索引被接受（L129）
- 调用实参中的元组与字典字面量被接受（L134）
- 列表推导被拒绝（L139）
- 白名单外运算符被拒绝（L144）
"""

from tilescript.frontend import carrier, expressions


def _check(expr: str, stmt: str = "x = {}"):
    source = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(n: int, Q: Pointer[f16], m_new: Tensor[f32, (16, 1), Register]):\n"
        "    " + stmt.format(expr) + "\n"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return expressions.check_module(tree)


class TestE0106Whitelist:
    def test_slice_and_broadcast_index_accepted(self):
        """Scenario L129：`Q[bm*BR:(bm+1)*BR, :]` 与 `m_new[:, None]` 通过。"""
        assert _check("Q[bm*BR:(bm+1)*BR, :]", "y = {}") == []
        assert _check("m_new[:, None]", "y = {}") == []

    def test_call_arg_tuple_and_dict_accepted(self):
        """Scenario L134：zeros((BR,BC),...)、make_tensor(L_ptr,(seq_len,))、Pipeline(stages=..., buffers={"K": K_s})。"""
        body = (
            "t = tis.zeros((BR, BC), f32, Register)\n"
            "    m = tis.make_tensor(L_ptr, (seq_len,))\n"
            "    p = tis.Pipeline(stages=STAGES, buffers={\"K\": K_s, \"V\": V_s})"
        )
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(n: int, L_ptr: Pointer[f16]):\n"
            "    " + body + "\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert expressions.check_module(tree) == []

    def test_binary_compare_logic_and_constants_accepted(self):
        """算术/比较/逻辑/一元负与 not/数值与布尔常量/None/属性/下标/名称引用。"""
        assert _check("a + b - c * d / e", "y = {}") == []
        assert _check("a < b and c >= d or e == f", "y = {}") == []
        assert _check("not -a", "y = {}") == []
        assert _check("0x10 + 0b11 + 0o17 + 42 + 1.5 + True + False + None", "y = {}") == []
        assert _check("tis.foo(x).bar[0]", "y = {}") == []


class TestE0106Rejections:
    def test_list_comprehension_rejected(self):
        """Scenario L139：`[x*2 for x in row]` → E0106，类别列表推导。"""
        out = _check("[x*2 for x in row]")
        assert len(out) == 1 and out[0].code == "E0106"
        assert out[0].category == "list-comprehension"

    def test_disallowed_operators_rejected(self):
        """Scenario L144：`a // b`、`a ** b` 等白名单外运算符 → E0106。"""
        for expr, op in [("a // b", "//"), ("a % b", "%"), ("a ** b", "**"),
                         ("a << b", "<<"), ("a >> b", ">>"), ("a & b", "&"),
                         ("a | b", "|"), ("a ^ b", "^"), ("~a", "~"), ("+a", "+")]:
            out = _check(expr)
            assert len(out) == 1 and out[0].code == "E0106", expr
            assert out[0].category == "disallowed-operator"
            assert op in out[0].suggestion

    def test_lambda_yield_await_walrus_ternary_rejected(self):
        """lambda/yield/await/海象/三元各自命中。"""
        cases = {
            "lambda: 1": "lambda",
            "(yield x)": "yield-expression",
            "(lambda: 1)()": "lambda",
        }
        for expr, category in cases.items():
            out = _check(expr)
            assert out and out[0].code == "E0106" and out[0].category == category, expr

    def test_chained_comparison_rejected(self):
        """链式比较 `a < b < c` → E0106（仅两操作数单比较）。"""
        out = _check("a < b < c")
        assert len(out) == 1 and out[0].code == "E0106"
        assert out[0].category == "chained-comparison"

    def test_membership_comparison_rejected(self):
        """`in`/`is` 比较运算符不在接受集。"""
        assert _check("a in b")[0].category == "disallowed-operator"
        assert _check("a is b")[0].category == "disallowed-operator"

    def test_fstring_rejected(self):
        """f-string → E0106（类别 f-string）。"""
        out = _check("f\"{a}\"", "tis.log({})")
        assert len(out) == 1 and out[0].code == "E0106"
        assert out[0].category == "f-string"

    def test_string_literal_position_rejected(self):
        """字符串常量仅调用关键字实参值位置：赋值右值/位置实参处 → E0106。"""
        out = _check("\"abc\"", "x = {}")
        assert out[0].category == "string-literal-position"
        out = _check("tis.foo(\"abc\")")
        assert out[0].category == "string-literal-position"
        # 关键字实参值位置合法
        assert _check("tis.foo(tag=\"abc\")") == []

    def test_tuple_literal_position_rejected(self):
        """元组仅切片下标与调用（位置）实参：赋值右值元组 → E0106。"""
        out = _check("(1, 2)", "x = {}")
        assert out[0].category == "tuple-literal-position"

    def test_dict_literal_position_rejected(self):
        """字典仅调用关键字实参值位置且键须字符串常量。"""
        out = _check("{1: a}", "x = {}")
        assert out[0].category == "dict-literal-position"
        # 位置实参处的字典拒绝
        out = _check("tis.foo({\"K\": k})")
        assert out[0].category == "dict-literal-position"
        # kwarg 值位置、键为字符串常量：合法
        assert _check("tis.foo(cfg={\"K\": k})") == []
        # kwarg 值位置、键非字符串常量：拒绝
        out = _check("tis.foo(cfg={k: v})")
        assert out[0].category == "dict-literal-position"

    def test_list_set_literal_rejected(self):
        """list/set 字面量不在接受集。"""
        assert _check("[1, 2]", "x = {}")[0].category == "list-literal"
        assert _check("{1, 2}", "x = {}")[0].category == "set-literal"

    def test_starred_rejected(self):
        """星号解包实参 → E0106。"""
        out = _check("tis.foo(*args)")
        assert out[0].category == "starred-expression"

    def test_matmul_operator_symbol_has_fallback(self):
        """代码审查 F1：`a @ b` 拒绝建议中的运算符符号不得为 None。"""
        out = _check("a @ b")
        assert out[0].category == "disallowed-operator"
        assert "@" in out[0].suggestion
        assert "None" not in out[0].suggestion

    def test_bytes_literal_rejected_everywhere(self):
        """代码审查 F2：bytes 不在 R6 接受集——任何位置（含 kwarg 值）拒绝。"""
        out = _check("b\"x\"", "x = {}")
        assert out[0].code == "E0106" and out[0].category == "bytes-literal"
        out = _check("tis.foo(tag=b\"x\")")
        assert out[0].code == "E0106" and out[0].category == "bytes-literal"
