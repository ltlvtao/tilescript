"""kernel 签名规则测试（syntax spec R4「kernel 函数签名规则」×4 Scenario）。

Scenario 映射：
- 参数缺类型注解被拒绝（L72）
- comptime 默认值被接受（L77）
- 运行期参数带默认值被拒绝（L82）
- comptime 默认值非常量字面量被拒绝（L87）
"""

from tilescript.frontend import carrier, signature


def _check(params: str):
    source = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        f"def k({params}):\n"
        "    pass\n"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return signature.check(tree)


class TestE0104:
    def test_missing_annotation_rejected(self):
        """Scenario L72：`def k(src, n: int)` 的 src → E0104 定位 + 合法注解清单。"""
        out = _check("src, n: int")
        assert len(out) == 1 and out[0].code == "E0104"
        assert out[0].category == "missing-annotation"
        assert out[0].line == 4
        assert "Pointer" in out[0].suggestion and "comptime[int]" in out[0].suggestion

    def test_annotation_forms_accepted(self):
        """注解形式封闭集全形态通过（Pointer/Tensor/dtype/int/comptime[int]）。"""
        assert _check("a: Pointer[f16], b: Tensor[f32, (16,), Register], c: f16") == []
        assert _check("a: bf16, b: f8e4m3, c: i8, d: i32, n: int") == []
        assert _check("a: Pointer[Tensor[f16, (16,), Shared]]") == []

    def test_invalid_annotation_rejected(self):
        """注解形式集合外（float、Tensor 无参数化）→ E0104。"""
        out = _check("a: float")
        assert len(out) == 1 and out[0].code == "E0104"
        assert out[0].category == "invalid-annotation"
        out = _check("a: Tensor")
        assert len(out) == 1 and out[0].code == "E0104"

    def test_comptime_int_literal_default_accepted(self):
        """Scenario L77：`D: comptime[int] = 64` 通过（int 字面量，任意进制同理）。"""
        assert _check("D: comptime[int] = 64") == []
        assert _check("D: comptime[int] = 0x40") == []
        assert _check("D: comptime[int] = 0b110") == []
        assert _check("D: comptime[int] = 0o17") == []

    def test_comptime_bool_default_accepted(self):
        """布尔常量默认值同样合法。"""
        assert _check("FLAG: comptime[int] = True") == []
        assert _check("FLAG: comptime[int] = False") == []

    def test_non_comptime_default_rejected(self):
        """Scenario L82：`scale: f32 = 1.0` → E0104，说明默认值仅限 comptime。"""
        out = _check("scale: f32 = 1.0")
        assert len(out) == 1 and out[0].code == "E0104"
        assert out[0].category == "default-on-non-comptime"
        assert "comptime" in out[0].suggestion

    def test_comptime_constant_expression_default_rejected(self):
        """Scenario L87：`D: comptime[int] = 64 * 1024` → E0104（仅单个字面量）。"""
        out = _check("D: comptime[int] = 64 * 1024")
        assert len(out) == 1 and out[0].code == "E0104"
        assert out[0].category == "invalid-comptime-default"
        assert "字面量" in out[0].suggestion

    def test_comptime_float_default_rejected(self):
        """float 字面量不是 int 字面量或布尔 → E0104。"""
        out = _check("D: comptime[int] = 1.5")
        assert len(out) == 1 and out[0].code == "E0104"
        assert out[0].category == "invalid-comptime-default"

    def test_multiple_params_each_reported(self):
        """收集全部：两个违规参数两条拒绝，按位置升序。"""
        out = _check("a, b: float")
        assert [r.category for r in out] == ["missing-annotation", "invalid-annotation"]

    def test_non_entry_function_not_checked(self):
        """非入口函数（produce 嵌套等）签名不属本 Requirement（execution 承载）。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(n: int):\n"
            "    @pipe.produce\n"
            "    def body(x):\n"
            "        pass\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert signature.check(tree) == []
