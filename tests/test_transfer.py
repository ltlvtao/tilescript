"""转移检查 E0301 测试（type-system spec R4「作用域转移矩阵」）。

Scenario 映射：
- Global→Shared 的 load 被接受（R4）
- Global→Global 被拒绝并建议合法目标清单（R4×2）
负例：UNKNOWN/非 Tensor 实参不报（E0406 辖域）、copy/move 格不报（E0404 辖域）。
"""

from tilescript.frontend import carrier
from tilescript.frontend.report import finalize
from tilescript.typecheck import annotations, bindings, state_fields, symbols


def _run(body: str, kernel_params_decl=""):
    source = (
        "import tis\n"
        "\n"
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


_PARAMS = (
    "K: Tensor[f16, (64, 64), Global], "
    "S: Tensor[f16, (64, 64), Shared], "
    "R: Tensor[f16, (64, 64), Register], "
    "K2: Tensor[f16, (64, 64), Global], "
    "n: int"
)


class TestTransferMatrix:
    def test_load_global_to_shared_accepted(self):
        """Scenario：Global→Shared 的 load 合法。"""
        assert _run("    tis.load(K, S)\n", _PARAMS) == []

    def test_legal_cells_accepted(self):
        """矩阵全部合法格：load 三格 + store 三格。"""
        assert _run(
            "    tis.load(K, R)\n    tis.load(S, R)\n"
            "    tis.store(S, K)\n    tis.store(R, K)\n    tis.store(R, S)\n",
            _PARAMS) == []

    def test_global_to_global_rejected(self):
        """Scenario：load(K, K2) Global→Global → E0301，建议列合法目标清单。"""
        rs = _run("    tis.load(K, K2)\n", _PARAMS)
        assert [r.code for r in rs] == ["E0301"]
        r = rs[0]
        assert r.category == "global-to-global"
        assert (r.line, r.col) == (6, 5)  # load 调用起点
        for frag in ("Shared", "Register", "Global"):
            assert frag in r.suggestion

    def test_store_global_to_global_rejected(self):
        rs = _run("    tis.store(K, K2)\n", _PARAMS)
        assert [r.code for r in rs] == ["E0301"]


class TestJurisdictionBoundaries:
    def test_unknown_args_not_reported(self):
        """UNKNOWN 实参（tis.* 结果）不报。"""
        assert _run(
            "    x = tis.zeros((64, 64))\n"
            "    tis.load(K, x)\n"
            "    tis.load(x, S)\n",
            _PARAMS) == []

    def test_non_tensor_args_not_reported(self):
        """非 Tensor 实参（标量）不报——E0406 辖域（后续段）。"""
        assert _run("    tis.load(K, n)\n", _PARAMS) == []

    def test_copy_move_not_reported(self):
        """copy/move 格（Shared→Shared、Register→Register）不报——E0404 辖域。"""
        assert _run("    tis.copy(S, S)\n    tis.move(R, R)\n", _PARAMS) == []

    def test_missing_args_not_reported(self):
        """实参不足（形态错误）不报——E0406 辖域。"""
        assert _run("    tis.load(K)\n", _PARAMS) == []

    def test_transfer_inside_flow(self):
        """流式位置：循环内/条件分支内的 load 检查使用当时 env。"""
        rs = _run(
            "    for i in range(n):\n"
            "        tis.load(K, K2)\n",
            _PARAMS)
        assert [r.code for r in rs] == ["E0301"]
