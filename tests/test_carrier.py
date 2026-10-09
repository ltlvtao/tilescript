"""载体检查测试（syntax spec R1「源码载体为 Python 3.10 语法基线」×4 Scenario）。

Scenario 映射：
- 非 Python 语法被拒绝并定位（L15）
- module 伪代码语法被拒绝（L20）
- 高于基线的 Python 语法被拒绝（L25——冻结节点表路径）
- 合法 Python 3.10 源文本通过载体检查（L30）
"""

from tilescript.frontend import carrier


class TestE0101:
    def test_non_python_syntax_rejected_with_position(self):
        """Scenario L15：`def f( {` → E0101 + 行列 + 原始解析错误描述 + 恢复建议。"""
        tree, rejection = carrier.parse("def f( {")
        assert tree is None
        assert rejection is not None
        assert rejection.code == "E0101"
        assert rejection.line >= 1 and rejection.col >= 1
        assert rejection.category == "syntax-error"
        assert rejection.suggestion  # 非空恢复建议
        # 原始解析错误描述随建议进入报告面（四要素）——此处断言建议提及 Python 3.10 载体。
        assert "Python 3.10" in rejection.suggestion

    def test_module_pseudocode_rejected(self):
        """Scenario L20：首行 `module flash_attention:` → E0101，建议说明模块即源文件本身。"""
        source = "module flash_attention:\nimport tis\n"
        tree, rejection = carrier.parse(source)
        assert tree is None
        assert rejection.code == "E0101"
        assert rejection.line == 1
        assert "源文件" in rejection.suggestion
        assert rejection.category == "module-pseudocode"

    def test_newer_python_syntax_rejected(self):
        """Scenario L25：3.11+ 语法（except*）→ E0101（冻结节点表后检查），建议指出基线 3.10。"""
        source = (
            "try:\n"
            "    pass\n"
            "except* ValueError:\n"
            "    pass\n"
        )
        tree, rejection = carrier.parse(source)
        assert tree is None
        assert rejection.code == "E0101"
        assert rejection.category == "newer-python-syntax"
        assert "Python 3.10" in rejection.suggestion

    def test_newer_type_alias_syntax_rejected(self):
        """3.12 `type X = int`（冻结节点表 TypeAlias）→ E0101。"""
        tree, rejection = carrier.parse("type Alias = int\n")
        assert tree is None
        assert rejection.code == "E0101"
        assert rejection.category == "newer-python-syntax"

    def test_valid_python310_passes(self):
        """Scenario L30：合法 Python 3.10 源文本 → 无 E0101，进入后续结构检查。"""
        source = "import tis\n\n\n@tis.kernel\ndef k(src: Pointer[f16], n: int):\n    pass\n"
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert tree is not None
        assert tree.__class__.__name__ == "Module"

    def test_no_dep_recursion_error_is_syntax_rejection(self):
        """极端不可解析输入（深层括号）不被实现崩溃吞掉：仍以 E0101 报告或通过（等价 ast 语义）。"""
        tree, rejection = carrier.parse("x = " + "(" * 200 + "1" + ")" * 200)
        # 只要不抛非 SyntaxError 异常即符合载体契约。
        assert (rejection is None) == (tree is not None) or rejection.code == "E0101"
