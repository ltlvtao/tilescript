"""顶层结构白名单测试（syntax spec R2「模块顶层结构白名单」×2 Scenario）。

Scenario 映射：
- 顶层裸语句被拒绝（L39）
- 顶层 kernel 定义被接受（L44——函数体由语句接受集检查，此处只测顶层面零命中）
"""

from tilescript.frontend import carrier, top_level


def _check(source: str):
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return top_level.check(tree)


class TestE0102:
    def test_top_level_bare_assign_rejected(self):
        """Scenario L39（赋值形态）：定位 + 类别 + 恢复建议（顶层仅三类结构）。"""
        out = _check("import tis\nx = 1\n")
        assert len(out) == 1 and out[0].code == "E0102"
        assert out[0].line == 2 and out[0].col == 1
        assert out[0].category == "top-level-assignment"
        assert "顶层" in out[0].suggestion

    def test_top_level_expression_statement_rejected(self):
        """Scenario L39（表达式调用语句形态）。"""
        out = _check("import tis\ntis.foo()\n")
        assert len(out) == 1 and out[0].code == "E0102"
        assert out[0].category == "top-level-expression"

    def test_top_level_bare_class_rejected(self):
        """裸类（无 @tis.state）不在顶层白名单 → E0102。"""
        out = _check("import tis\nclass S:\n    pass\n")
        assert len(out) == 1 and out[0].code == "E0102"
        assert out[0].category == "top-level-class"

    def test_top_level_state_class_accepted(self):
        """@tis.state 类在白名单内。"""
        out = _check("import tis\n\n\n@tis.state\nclass S:\n    O: Tensor[f32, (16,), Register]\n")
        assert out == []

    def test_top_level_kernel_accepted(self):
        """Scenario L44：@tis.kernel 顶层函数 → 顶层检查通过。"""
        out = _check(
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(src: Pointer[f16], n: int):\n"
            "    pass\n"
        )
        assert out == []

    def test_top_level_import_forms_accepted(self):
        """import 语句按类别接受（名字不管控，design D11）。"""
        out = _check("import tis\nfrom math import ceil\nimport a.b.c\n")
        assert out == []
