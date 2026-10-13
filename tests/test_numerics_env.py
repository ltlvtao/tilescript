"""数值段基座与段编排测试（numerics/value-semantics 实现面，tasks 1.1/1.2）。

公共行为断言：numerics.check_module(tree)（无 target——跨 HAL 行为不变面）
返回排序去重后的 Rejection 清单（stage="numerics"、order=段内序 1–6）。
合法用例取 FLASH 结构子集扩展数值面（算术/比较/常量/cdiv）；扫描只读
AST，不触既有四段（同输入四段结果不变）。
"""

import ast

from tilescript.numerics import check_module as numerics_check

# FLASH 结构子集（§7）+ 数值面扩展：算术赋值/比较条件/comptime 折叠/cdiv。
_LEGAL_NUMERICS = '''\
import tis

@tis.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]

@tis.kernel
def flash(K_ptr: Pointer[f16, Global], seq_len: int, scale: f32,
          D: comptime[int] = 64, BR: comptime[int] = 64,
          BC: comptime[int] = 64, STAGES: comptime[int] = 2):
    K = tis.make_tensor(K_ptr, (seq_len, D))
    K_s = tis.alloc_shared((BC, D), f16)
    S = tis.zeros((BR, D), f32, Register)
    T = S * scale
    steps = tis.cdiv(seq_len, BC)
    base = BR * 2 - 1
    if seq_len > 0:
        T = T + S
    pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s})

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K[j * BC:(j + 1) * BC, :], buf.K, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        return AttnState(O_acc=tis.zeros((BR, D), f32, Register))

    st = pipe.run(range(tis.cdiv(seq_len, BC)),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))
    return
'''


def _check(source: str):
    return numerics_check(ast.parse(source))


def _codes(rejections):
    return [r.code for r in rejections]


class TestEnvBase:
    """tasks 1.1：数值类型环境基座（env 起点/委托/tis 表/遍历同构）。"""

    def test_legal_module_zero_rejections(self):
        """合法模块（原语调用 + 算术赋值 + 比较 + 折叠）数值段零拒绝。"""
        assert _check(_LEGAL_NUMERICS) == []

    def test_repeat_compile_identical(self):
        """重复编译拒绝清单逐条一致（确定性）。"""
        first = _check(_LEGAL_NUMERICS)
        second = _check(_LEGAL_NUMERICS)
        assert [r.to_dict() for r in first] == [r.to_dict() for r in second]

    def test_scan_does_not_touch_existing_stages(self):
        """同输入四段结果不变（数值段只读 AST，不触既有段检查）。"""
        from tilescript import execution, primitives, typecheck
        from tilescript.frontend import check_module as syntax_check
        tree = ast.parse(_LEGAL_NUMERICS)
        assert syntax_check(_LEGAL_NUMERICS) == []
        assert typecheck.check_module(tree) == []
        assert primitives.check_module(tree, "nvidia_h200") == []
        assert execution.check_module(tree, "nvidia_h200") == []
        assert numerics_check(tree) == []  # 数值段执行后
        assert typecheck.check_module(tree) == []  # 既有段结果不变
        assert execution.check_module(tree, "nvidia_h200") == []

    def test_kernel_without_numerics_face_zero_rejections(self):
        """纯结构模块（无数值面）零拒绝：env 起点与遍历不产生误报。"""
        source = '''\
import tis

@tis.kernel
def empty(K_ptr: Pointer[f16, Global], seq_len: int,
          BC: comptime[int] = 64):
    K_s = tis.alloc_shared((BC,), f16)
    return
'''
        assert _check(source) == []


class TestStageWiring:
    """tasks 1.2：段编排（段名/段内 order/finalize 收口/无 target 签名）。"""

    def test_stage_name(self):
        from tilescript.numerics import STAGE_NAME
        assert STAGE_NAME == "numerics"

    def test_check_module_signature_no_target(self):
        """check_module(tree) 无 target——数值语义跨 HAL 行为不变（hal spec）。"""
        import inspect
        assert "target" not in inspect.signature(numerics_check).parameters

    def test_finalize_orders_by_position(self):
        """多条拒绝按源码位置升序（finalize 收口，tasks 6.1 前置形态）。"""
        source = '''\
import tis

@tis.kernel
def f(seq_len: int, S: Tensor[f32, (4,), Register]):
    T = S * seq_len
    x = 1.5
    return
'''
        rejections = _check(source)
        assert _codes(rejections) == ["E0601", "E0606"]
        lines = [r.line for r in rejections]
        assert lines == sorted(lines)
        for r in rejections:
            assert r.stage == "numerics"

    def test_rejections_carry_four_elements(self):
        """四要素：错误码/行列/类别/恢复建议（to_dict 序列化面）。"""
        source = '''\
import tis

@tis.kernel
def f(seq_len: int):
    x = 1.5
    return
'''
        rejections = _check(source)
        assert _codes(rejections) == ["E0606"]
        r = rejections[0].to_dict()
        assert set(r) == {"code", "line", "col", "category", "suggestion"}
        assert "float" in r["suggestion"] or "浮点" in r["suggestion"]


class TestReportContract:
    """tasks 6.1：数值段报告契约（R7——排序/段内 order/同码合并/重复一致/择一）。"""

    _TWO_HITS = '''\
import tis

@tis.kernel
def f(seq_len: int, S: Tensor[f32, (4,), Register]):
    x = 1.5
    T = S * seq_len
    return
'''

    def test_position_sort_dominates_code_family(self):
        """多条 E06xx 按位置升序：前 E0606（第 6 行）后 E0601（第 7 行）——
        位置主导排序，非按码族。"""
        rejections = _check(self._TWO_HITS)
        assert _codes(rejections) == ["E0606", "E0601"]
        lines = [r.line for r in rejections]
        assert lines == sorted(lines)

    def test_same_position_intra_stage_order_first_wins(self):
        """同位置多码取段内 order 最小（E0601=1 先于 E0606=6）。

        M1 语法下 E0601/E0606 检查节点不相交（真实同位双命中不可构造），
        段内优先由 finalize 的 order tiebreak 机制承载——直测该公共行为
        （同 tests/test_pipeline.py test_same_position_type_rules_first_wins
        先例）。
        """
        from tilescript.frontend.report import Rejection, finalize
        from tilescript.numerics.value_env import CATEGORIES, ORDERS

        def _r(code):
            return Rejection(code=code, line=6, col=4, category=CATEGORIES[code],
                             suggestion="s", order=ORDERS[code], stage="numerics")

        assert finalize([_r("E0606"), _r("E0601")]) == [_r("E0601")]

    def test_same_code_merged_single_entry(self):
        """同表达式同码多操作数合并一条列全部（R7——帧收口）。"""
        source = '''\
import tis

@tis.kernel
def f(seq_len: int, D: comptime[int] = 64, BR: comptime[int] = 64,
      BC: comptime[int] = 64):
    A = tis.zeros((4,), f16, Register)
    T = A + seq_len + BC
    return
'''
        rejections = _check(source)
        assert _codes(rejections) == ["E0601"]
        assert "违规操作数" in rejections[0].suggestion
        assert rejections[0].suggestion.count("行") >= 2  # 列出全部违规操作数位置

    def test_repeat_compilation_identical_with_rejections(self):
        """拒绝非空模块重复编译逐条一致（确定性，R7）。"""
        first = [r.to_dict() for r in _check(self._TWO_HITS)]
        second = [r.to_dict() for r in _check(self._TWO_HITS)]
        assert first == second and len(first) == 2

    def test_double_hit_reports_earliest_stage_e0303(self):
        """同位置跨段双命中（E0303×E0602）只报最早段 E0303（择一机制，
        R7 Scenario 2 报告面——stage 口径与类别随最早段）。"""
        source = '''\
import tis

@tis.kernel
def f(seq_len: int, scale: f32, D: comptime[int] = 64, BR: comptime[int] = 64,
      BC: comptime[int] = 64):
    b = tis.full((4,), 1.5, f16, Register)
    b0 = b[0]
    x = seq_len
    x = scale * b0
    return
'''
        rejections = _check(source)
        assert _codes(rejections) == ["E0303"]
        assert rejections[0].stage == "type-system"
        assert rejections[0].category == "assign"
