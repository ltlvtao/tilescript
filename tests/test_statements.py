"""设备代码语句接受集测试（syntax spec R5「设备代码语句接受集」×5 Scenario）。

Scenario 映射：
- 白名单语句组合被接受（L98）
- while 循环被拒绝并给出替代（L103）
- match 语句被拒绝（L108）
- 注解式局部赋值被拒绝（L113，带右值/无右值两形态）
- 白名单语句含被拒表达式只报表达式错误（L118）
"""

from tilescript.frontend import carrier, statements


def _check(body: str, decorator: str = "@tis.kernel", signature: str = "def k(n: int):"):
    source = f"import tis\n\n{decorator}\n{signature}\n{body}"
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return statements.check(tree)


class TestE0105Whitelist:
    def test_whitelist_statement_combination_accepted(self):
        """Scenario L98：白名单语句组合通过（赋值/增强赋值/表达式/for-range/for-tile_iter/if/return/pass）。"""
        out = _check(
            "    acc = tis.zeros((16,), f32, Register)\n"
            "    acc += x\n"
            "    tis.sync()\n"
            "    for r in range(4):\n"
            "        if r > 1:\n"
            "            acc = acc + 1\n"
            "        else:\n"
            "            pass\n"
            "    for tile in tis.tile_iter():\n"
            "        acc = acc * 2\n"
            "    return\n"
        )
        assert out == []

    def test_with_warp_group_accepted(self):
        """with 限 tis.warp_group(...) 调用（含多上下文项）。"""
        out = _check(
            "    with tis.warp_group(role=\"producer\", warps=2) as pg:\n"
            "        pass\n"
        )
        assert out == []

    def test_produce_consume_nested_function_accepted(self):
        """produce/consume 装饰的嵌套函数定义是白名单语句类别。"""
        out = _check(
            "    @pipe.produce\n"
            "    def body(x: f16) -> None:\n"
            "        pass\n"
        )
        assert out == []


class TestE0105Rejections:
    def test_while_rejected_with_alternative(self):
        """Scenario L103：while → E0105，类别 while、建议改写 for range/tile_iter。"""
        out = _check("    while n > 0:\n        n = n - 1\n")
        assert len(out) == 1 and out[0].code == "E0105"
        assert out[0].category == "while-loop"
        assert out[0].line == 5 and out[0].col == 5
        assert "range" in out[0].suggestion and "tile_iter" in out[0].suggestion

    def test_while_subtree_short_circuited(self):
        """design D3 裁决：E0105 命中语句的子树不再深入（体内 match 不另报）。"""
        out = _check("    while n > 0:\n        match n:\n"
                     "            case 0:\n"
                     "                pass\n")
        assert len(out) == 1 and out[0].code == "E0105"
        assert out[0].category == "while-loop"

    def test_match_rejected(self):
        """Scenario L108：match → E0105，类别 match。"""
        out = _check("    match n:\n        case 0:\n            pass\n")
        assert len(out) == 1 and out[0].code == "E0105"
        assert out[0].category == "match-statement"

    def test_annotated_local_assignment_rejected(self):
        """Scenario L113：`x: f32 = 0.0` 与无右值 `x: f32` 两形态均拒，建议无注解赋值。"""
        for form in ("    x: f32 = 0.0\n", "    x: f32\n"):
            out = _check(form)
            assert len(out) == 1 and out[0].code == "E0105", form
            assert out[0].category == "annotated-assignment"
            assert "推断" in out[0].suggestion

    def test_rejected_category_each_reported(self):
        """拒绝清单其余类别逐项命中（break/continue/try/raise/assert/del/global/nonlocal/设备内 import/类定义/普通嵌套函数）。"""
        cases = {
            "    break\n": "break-statement",
            "    continue\n": "continue-statement",
            "    try:\n        pass\n    except Exception:\n        pass\n": "try-statement",
            "    raise ValueError()\n": "raise-statement",
            "    assert n > 0\n": "assert-statement",
            "    del n\n": "del-statement",
            "    global g\n": "global-declaration",
            "    nonlocal q\n": "nonlocal-declaration",
            "    import math\n": "device-import",
            "    from math import ceil\n": "device-import",
            "    class Inner:\n        pass\n": "class-definition",
            "    def helper(x: f16):\n        pass\n": "function-definition",
        }
        for body, category in cases.items():
            out = _check(body)
            assert out and out[0].code == "E0105" and out[0].category == category, (body, out)

    def test_multiple_target_and_augassign_operator_rejected(self):
        """链式赋值/解包赋值/白名单外增强赋值运算符 → E0105。"""
        assert _check("    a = b = 1\n")[0].category == "chained-assignment"
        assert _check("    a, b = 1, 2\n")[0].category == "unpacking-assignment"
        assert _check("    a //= 2\n")[0].category == "augmented-assignment-operator"
        assert _check("    a %= 2\n")[0].category == "augmented-assignment-operator"

    def test_for_iterable_and_with_context_restricted(self):
        """for 可迭代表达式限 range/tis.tile_iter；with 上下文限 tis.warp_group。"""
        assert _check("    for i in items:\n        pass\n")[0].category == "invalid-for-iterable"
        assert _check("    for i in tis.foo():\n        pass\n")[0].category == "invalid-for-iterable"
        assert _check("    with something as x:\n        pass\n")[0].category == "invalid-with-context"
        assert _check("    with open(\"f\") as x:\n        pass\n")[0].category == "invalid-with-context"


class TestE0105SubexpressionRule:
    def test_whitelist_statement_with_rejected_expression_reports_only_e0106(self):
        """Scenario L118：`f = lambda: 1` → 只报 E0106，不另报 E0105。"""
        out = _check("    f = lambda: 1\n")
        assert len(out) == 1 and out[0].code == "E0106"
        assert out[0].category == "lambda"

    def test_expressions_in_nested_bodies_checked(self):
        """嵌套体（for/if 内）的被拒子表达式同样按 E0106 收集。"""
        out = _check("    for r in range(4):\n"
                     "        if r > 1:\n"
                     "            y = [x for x in row]\n")
        assert len(out) == 1 and out[0].code == "E0106"
        assert out[0].category == "list-comprehension"
