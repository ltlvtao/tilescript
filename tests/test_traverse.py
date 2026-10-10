"""设备函数遍历骨架测试（design D3——原语段 traverse）。

公共行为面：
- 同构遍历（kernel/produce/consume 体；与类型段 bindings 相同的作用域规则）
  下合法片段零拒绝、防御断言不触发；
- `tis.*` Call 改道原语检查器（stub 承载）：结果类型进入 env，
    经后续调用的实参推断间接可观察；
- 嵌套 produce/consume 进入/退出语境跟踪（E0407 Async 判定的数据来源）；
- 防御断言：类型段让渡缝隙外漂移（如 Pointer 下标——类型段必拒形态
  直调进入）在遍历收尾即暴露。
"""

import pytest

from tilescript.frontend import carrier
from tilescript.primitives import traverse
from tilescript.typecheck.types import ConstDim, TensorType, desc


class _StubChecker:
    """记录改道调用与实参推断结果；zeros 固定返回 Register f16 (64,)。"""

    def __init__(self, result=None):
        self.calls: "list[tuple]" = []  # (原语名, [(实参desc, ...)], 语境)
        self.result = result or TensorType("f16", (64,), "Register")

    def check_call(self, node, inferencer):
        args_desc = tuple(desc(inferencer.infer(a)) for a in node.args)
        self.calls.append((node.func.attr, args_desc,
                           inferencer.in_produce_consume))
        return self.result


def _walk(source: str, checker=None) -> _StubChecker:
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    checker = checker or _StubChecker()
    traverse.walk_device_functions(tree, registry={}, comptime_syms=frozenset(),
                                   checker=checker)
    return checker


class TestTraverseBasics:
    def test_legal_fragment_zero_rejections(self):
        """合法片段（无 tis.* 与合法 tis.* 混合）遍历零拒绝零异常。"""
        checker = _walk(
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register]):\n"
            "    x = tis.zeros((64,), f16)\n"
            "    y = x\n"
            "    for i in range(8):\n"
            "        y = x\n"
            "    tis.store(x, a)\n"
            "    return\n"
        )
        assert len(checker.calls) == 2  # zeros + store 均改道

    def test_primitive_result_type_flows_into_env(self):
        """原语结果类型绑定 env：后续调用的实参推断可见（间接可观察）。"""
        checker = _walk(
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register]):\n"
            "    x = tis.zeros((64,), f16)\n"
            "    tis.store(x, a)\n"
            "    return\n"
        )
        zeros_args, store_args = checker.calls[0][1], checker.calls[1][1]
        assert len(zeros_args) == 2  # shape 元组 + dtype 名：均 UNKNOWN 让渡
        assert store_args[0] == "Tensor[f16, (64), Register]"  # x = zeros 结果


class TestNestedContext:
    def test_produce_consume_context_entered_and_exited(self):
        """嵌套 produce/consume 体内语境 True、退出恢复 False（kernel 顶层）。"""
        checker = _walk(
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (64,), Register]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "    tis.load(a, a)\n"
            "\n"
            "    @pipe.produce\n"
            "    def p(j: int, buf):\n"
            "        tis.load(a, a)\n"
            "\n"
            "    tis.load(a, a)\n"
            "    return\n"
        )
        loads = [c for c in checker.calls if c[0] == "load"]
        assert [c[2] for c in loads] == [False, True, False]

    def test_nested_scope_params_and_restoration(self):
        """嵌套函数形参进入 env、退出恢复外层快照（作用域不泄漏）。"""
        shared = TensorType("f32", (ConstDim(8),), "Shared")

        class _Checker(_StubChecker):
            def check_call(self, node, inferencer):
                if node.func.attr == "store":
                    arg = node.args[0]
                    # buf 形参让渡 UNKNOWN；嵌套内覆盖名 p 外恢复
                    self.calls.append(("seen", (inferencer.infer(arg),)))
                    return self.result
                return super().check_call(node, inferencer)

        checker = _Checker()
        _walk(
            checker=checker,
            source=(
                "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(buf: Tensor[f32, (8,), Shared]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "    tis.store(buf, buf)\n"
            "\n"
            "    @pipe.produce\n"
            "    def p(j: int, buf):\n"
            "        tis.store(buf, buf)\n"
            "\n"
            "    tis.store(buf, buf)\n"
                "    return\n"
            ),
        )
        seen = [c[1][0] for c in checker.calls if c[0] == "seen"]
        assert len(seen) == 3 and seen[0] == shared and seen[2] == shared
        assert "未知" in desc(seen[1])  # 嵌套形参无注解 buf → UNKNOWN


class TestDefensiveAssertion:
    def test_drift_surfaces_as_assertion(self):
        """类型段必拒形态（Pointer 下标 E0303）直调进入 → 收尾防御断言暴露。"""
        checker = _StubChecker()
        with pytest.raises(AssertionError, match="漂移"):
            _walk(
                "import tis\n"
                "\n"
                "@tis.kernel\n"
                "def k(p: Pointer[f16, Global]):\n"
                "    x = p[0]\n"
                "    return\n"
            )
