"""执行结构段形态环境测试（design D3：注解归约各支 + comptime 口径 + buffers 三分）。

形态环境为 execution 包公共面（shape_env 模块级函数）；粗形态不含 dtype/dims
（本段无此判定面），StateShape 带类名（nominal 比较用）。
"""

import ast

from tilescript.execution import shape_env
from tilescript.typecheck import state_fields


def _annot(source: str):
    tree = ast.parse(f"x: {source} = None")
    return tree.body[0].annotation


def _registry():
    tree = ast.parse(
        "@tis.state\nclass AttnState:\n    m: Tensor[f32, (16,), Register]\n")
    registry, _ = state_fields.check(tree)
    return registry


class TestAnnotationShapes:
    def test_tensor_by_scope(self):
        """Tensor 注解按 scope 归约（Shared/Register/Global 三支）。"""
        assert shape_env.from_annotation(
            _annot("Tensor[f16, (16,), Shared]"), frozenset()) == shape_env.SHARED_TENSOR
        assert shape_env.from_annotation(
            _annot("Tensor[f16, (16,), Register]"), frozenset()) == shape_env.REGISTER_TENSOR
        assert shape_env.from_annotation(
            _annot("Tensor[f16, (16,), Global]"), frozenset()) == shape_env.GLOBAL_TENSOR

    def test_scalar_kinds(self):
        """标量种类：int / comptime[int] / dtype 名三支。"""
        assert shape_env.from_annotation(_annot("int"), frozenset()) == shape_env.INT
        assert shape_env.from_annotation(
            _annot("comptime[int]"), frozenset()) == shape_env.COMPTIME_INT
        assert shape_env.from_annotation(_annot("f32"), frozenset()) == shape_env.DTYPE_SCALAR

    def test_state_class(self):
        """状态类注解 → StateShape(类名)（registry 供类名集）。"""
        shape = shape_env.from_annotation(_annot("AttnState"), frozenset(), _registry())
        assert shape == shape_env.StateShape("AttnState")

    def test_unannotated_and_unparseable(self):
        """无注解与解析失败（E0302 已报短路不见；防御）→ UNKNOWN。"""
        assert shape_env.from_annotation(None, frozenset()) == shape_env.UNKNOWN
        assert shape_env.from_annotation(
            _annot("Tensor[f64, (16,), Shared]"), frozenset()) == shape_env.UNKNOWN


class TestComptimeIntForm:
    """comptime[int] 形态口径（stages/warps/block_idx/barrier_id 共用，design D4）。"""

    def test_constant(self):
        node = ast.parse("2", mode="eval").body
        assert shape_env.comptime_int_form(node, frozenset()) == ("const", 2)

    def test_comptime_name_defers(self):
        node = ast.parse("STAGES", mode="eval").body
        assert shape_env.comptime_int_form(node, frozenset({"STAGES"})) == ("name", None)

    def test_runtime_name_rejected(self):
        node = ast.parse("n", mode="eval").body
        assert shape_env.comptime_int_form(node, frozenset()) is None

    def test_other_form_rejected(self):
        node = ast.parse("1.5", mode="eval").body
        assert shape_env.comptime_int_form(node, frozenset()) is None


class TestBuffersVerdict:
    """buffers 值判定三分（design D3：确定 Shared 接受 / 确定非 Shared 拒 / UNKNOWN 让渡）。"""

    def test_shared_accepted(self):
        assert shape_env.buffers_verdict(shape_env.SHARED_TENSOR) == "accept"

    def test_definite_non_shared_rejected(self):
        for shape in (shape_env.REGISTER_TENSOR, shape_env.GLOBAL_TENSOR,
                      shape_env.INT, shape_env.COMPTIME_INT, shape_env.DTYPE_SCALAR,
                      shape_env.PIPELINE):
            assert shape_env.buffers_verdict(shape) == "reject", shape

    def test_unknown_deferred(self):
        for shape in (shape_env.UNKNOWN, shape_env.UNKNOWN_TENSOR):
            assert shape_env.buffers_verdict(shape) == "defer", shape
