"""状态类类体结构测试（syntax spec R7「状态类类体结构」×3 Scenario）。

Scenario 映射：
- 纯字段状态类被接受（L153）
- 状态类包含方法被拒绝（L158）
- 字段带右值被拒绝（L163，建议指向 pipe.run(init=...)）
"""

from tilescript.frontend import carrier, state_class


def _check(body: str):
    source = f"import tis\n\n@tis.state\nclass S:\n{body}"
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return state_class.check(tree)


class TestE0107:
    def test_pure_field_state_class_accepted(self):
        """Scenario L153：只含带注解无右值字段声明 → 通过（类型规则归 type-system）。"""
        out = _check(
            "    O_acc: Tensor[f32, (16, 128), Register]\n"
            "    m: Tensor[f32, (16,), Register]\n"
            "    step: i32\n"
        )
        assert out == []

    def test_method_rejected(self):
        """Scenario L158：类体内方法定义 → E0107 定位。"""
        out = _check("    def reset(self):\n        pass\n")
        assert len(out) == 1 and out[0].code == "E0107"
        assert out[0].category == "class-method"
        assert out[0].line == 5

    def test_field_with_value_rejected(self):
        """Scenario L163：`m: Tensor[...] = tis.zeros(...)` → E0107，建议 pipe.run(init=...)。"""
        out = _check("    m: Tensor[f32, (16,), Register] = tis.zeros((16,), f32, Register)\n")
        assert len(out) == 1 and out[0].code == "E0107"
        assert out[0].category == "field-with-value"
        assert "pipe.run(init=...)" in out[0].suggestion

    def test_unannotated_and_other_statements_rejected(self):
        """无注解赋值/表达式语句/pass 等其他结构同样拒绝。"""
        assert _check("    m = 0\n")[0].category == "unannotated-field"
        assert _check("    tis.foo()\n")[0].category == "other-class-body"
        assert _check("    pass\n")[0].category == "other-class-body"

    def test_non_state_class_not_checked(self):
        """非 @tis.state 类（不可能合法出现在顶层，但嵌套时）不属本 Requirement。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    O: Tensor[f32, (16,), Register]\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert state_class.check(tree) == []

    def test_multiple_violations_all_collected(self):
        """收集全部：方法 + 带右值字段两条拒绝，位置升序。"""
        out = _check(
            "    def reset(self):\n"
            "        pass\n"
            "    m: Tensor[f32, (16,), Register] = tis.zeros((16,), f32, Register)\n"
        )
        assert [r.category for r in out] == ["class-method", "field-with-value"]
