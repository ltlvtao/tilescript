"""设备入口装饰器识别测试（syntax spec R3「设备入口装饰器识别」×3 Scenario）。

Scenario 映射：
- 未知装饰器被拒绝（L53）
- pipe.produce 嵌套函数被接受（L58）
- 入口装饰器用于嵌套函数被拒绝（L63）
"""

from tilescript.frontend import carrier, decorators


def _check(source: str):
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return decorators.check(tree)


_KERNEL = (
    "import tis\n"
    "\n"
    "@tis.kernel\n"
    "def k(Q: Pointer[f16], n: int):\n"
    "    pass\n"
)


class TestE0103:
    def test_unknown_tis_decorator_rejected(self):
        """Scenario L53：@tis.autotune 集合外 → E0103，含装饰器名称/位置/清单。"""
        out = _check(
            "import tis\n"
            "\n"
            "@tis.autotune\n"
            "def k(Q: Pointer[f16]):\n"
            "    pass\n"
        )
        assert len(out) == 1 and out[0].code == "E0103"
        assert out[0].category == "unknown-decorator"
        assert out[0].line == 3
        assert "tis.autotune" in out[0].suggestion
        assert "produce" in out[0].suggestion  # 已识别集合进入建议

    def test_bare_name_decorator_rejected(self):
        """裸名 @foo 同样集合外 → E0103。"""
        out = _check("import tis\n\n@foo\ndef k(n: int):\n    pass\n")
        assert len(out) == 1 and out[0].code == "E0103"
        assert out[0].category == "unknown-decorator"

    def test_pipe_produce_nested_accepted(self):
        """Scenario L58：@pipe.produce 嵌套函数（实例名任意）→ 识别通过。"""
        out = _check(
            _KERNEL.replace(
                "    pass\n",
                "    @pipe.produce\n"
                "    def body(x: f16) -> None:\n"
                "        pass\n",
            )
        )
        assert out == []

    def test_arbitrary_instance_consume_accepted(self):
        """consume 同理，实例名任意（按属性名识别）。"""
        out = _check(
            _KERNEL.replace(
                "    pass\n",
                "    @p2.consume\n"
                "    def body(x: f16) -> None:\n"
                "        pass\n",
            )
        )
        assert out == []

    def test_entry_decorator_on_nested_function_rejected(self):
        """Scenario L63：设备代码内 @tis.kernel 嵌套函数 → 类别不匹配 E0103。"""
        out = _check(
            _KERNEL.replace(
                "    pass\n",
                "    @tis.kernel\n"
                "    def inner(x: f16) -> None:\n"
                "        pass\n",
            )
        )
        assert len(out) == 1 and out[0].code == "E0103"
        assert out[0].category == "decorator-target-mismatch"

    def test_produce_on_top_level_function_rejected(self):
        """对称面：produce/consume 仅设备代码内——顶层函数使用 → 类别不匹配。"""
        out = _check("import tis\n\n@pipe.produce\ndef k(n: int):\n    pass\n")
        assert len(out) == 1 and out[0].code == "E0103"
        assert out[0].category == "decorator-target-mismatch"

    def test_state_decorator_on_function_rejected(self):
        """tis.state 仅模块顶层类——用于函数 → 类别不匹配。"""
        out = _check("import tis\n\n@tis.state\ndef k(n: int):\n    pass\n")
        assert len(out) == 1 and out[0].code == "E0103"
        assert out[0].category == "decorator-target-mismatch"

    def test_valid_entry_and_state_accepted(self):
        """合法组合（顶层 state 类 + kernel 入口）零命中。"""
        out = _check(
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    O: Tensor[f32, (16,), Register]\n"
            "\n"
            "@tis.kernel\n"
            "def k(n: int):\n"
            "    pass\n"
        )
        assert out == []
