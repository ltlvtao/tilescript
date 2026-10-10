"""执行结构段 E0504/E0501 面（warp_group 参数契约与语境）测试。

公共行为断言同 test_execution_ops.py：check_module(tree, target) 返回
排序去重后的 Rejection 清单（stage="execution-structure"）。
"""

import ast

from tilescript.execution import check_module as execution_check

# warp_group 合法配对基线（spec「合法 producer 与 consumer 配对」Scenario）。
_WG_BASE = '''\
import tis

@tis.kernel
def wg_kernel(seq_len: int, BR: comptime[int] = 64):
    K_s = tis.alloc_shared((BR, 16), f16)

    with tis.warp_group(role="producer", warps=2) as pg:
        tis.barrier()
    with tis.warp_group(role="consumer", warps=6) as cg:
        tis.barrier()
    tis.warp_group_sync(pg, cg, barrier_id=0)
'''


def _check(source: str, target: str = "nvidia_h200"):
    return execution_check(ast.parse(source), target)


def _single(rejections, code, category=None):
    """断言恰一条拒绝并返回它（公共四要素面）。"""
    assert len(rejections) == 1, [r.to_dict() for r in rejections]
    r = rejections[0]
    assert r.code == code
    assert r.stage == "execution-structure"
    if category is not None:
        assert r.category == category
    return r


class TestWarpGroupContract:
    """E0504 warp_group 参数契约（R「warp_group 上下文契约」）。"""

    def test_legal_pairing_accepted(self):
        assert _check(_WG_BASE) == []

    def test_role_out_of_domain_rejected(self):
        r = _single(_check(_WG_BASE.replace('role="producer"', 'role="dma"')),
                    "E0504")
        assert r.category == "wg-role"
        assert "producer/consumer" in r.suggestion

    def test_role_missing_rejected(self):
        r = _single(_check(_WG_BASE.replace('role="producer", warps=2',
                                            'warps=2')), "E0504")
        assert r.category == "wg-role"

    def test_warps_zero_rejected(self):
        r = _single(_check(_WG_BASE.replace("warps=2", "warps=0")), "E0504")
        assert r.category == "wg-warps"
        assert "不小于 1" in r.suggestion

    def test_warps_runtime_name_rejected(self):
        r = _single(_check(_WG_BASE.replace("warps=2", "warps=seq_len")), "E0504")
        assert r.category == "wg-warps"

    def test_call_in_expression_position_rejected(self):
        source = _WG_BASE.replace(
            "    K_s = tis.alloc_shared((BR, 16), f16)",
            "    K_s = tis.alloc_shared((BR, 16), f16)\n"
            '    wg_local = tis.warp_group(role="producer", warps=2)')
        r = _single(_check(source), "E0504")
        assert r.category == "wg-call-position"
        assert "with" in r.suggestion

    def test_binding_as_value_rejected(self):
        source = _WG_BASE.replace(
            "    tis.warp_group_sync(pg, cg, barrier_id=0)",
            "    y = pg\n    tis.warp_group_sync(pg, cg, barrier_id=0)")
        r = _single(_check(source), "E0504")
        assert r.category == "warp-binding-escape"

    def test_sync_same_binding_rejected(self):
        r = _single(_check(_WG_BASE.replace("tis.warp_group_sync(pg, cg,",
                                            "tis.warp_group_sync(pg, pg,")),
                    "E0504")
        assert r.category == "sync-binding"
        assert "不同" in r.suggestion

    def test_sync_non_binding_arg_rejected(self):
        r = _single(_check(_WG_BASE.replace("tis.warp_group_sync(pg, cg,",
                                            "tis.warp_group_sync(pg, K_s,")),
                    "E0504")
        assert r.category == "sync-binding"

    def test_sync_one_positional_arg_rejected(self):
        r = _single(_check(_WG_BASE.replace(
            "tis.warp_group_sync(pg, cg, barrier_id=0)",
            "tis.warp_group_sync(pg, barrier_id=0)")), "E0504")
        assert r.category == "sync-binding"

    def test_sync_barrier_id_missing_rejected(self):
        r = _single(_check(_WG_BASE.replace(
            "tis.warp_group_sync(pg, cg, barrier_id=0)",
            "tis.warp_group_sync(pg, cg)")), "E0504")
        assert r.category == "sync-barrier-id"

    def test_sync_barrier_id_negative_rejected(self):
        r = _single(_check(_WG_BASE.replace("barrier_id=0", "barrier_id=-1")),
                    "E0504")
        assert r.category == "sync-barrier-id"

    def test_sync_in_value_position_rejected(self):
        source = _WG_BASE.replace(
            "    tis.warp_group_sync(pg, cg, barrier_id=0)",
            "    s = tis.warp_group_sync(pg, cg, barrier_id=0)")
        r = _single(_check(source), "E0504")
        assert r.category == "sync-value"

    def test_sync_in_nested_body_rejected(self):
        source = _WG_BASE.replace(
            "    tis.warp_group_sync(pg, cg, barrier_id=0)",
            "    tis.warp_group_sync(pg, cg, barrier_id=0)\n\n"
            "    def late_fn(x: int):\n"
            "        tis.warp_group_sync(pg, cg, barrier_id=0)")
        r = _single(_check(source), "E0504")
        assert r.category == "sync-position"

    def test_sync_before_with_rejected(self):
        source = _WG_BASE.replace(
            "    K_s = tis.alloc_shared((BR, 16), f16)",
            "    K_s = tis.alloc_shared((BR, 16), f16)\n"
            "    tis.warp_group_sync(pg, cg, barrier_id=0)")
        r = _single(_check(source), "E0504")
        assert r.category == "sync-position"


class TestWarpGroupContext:
    """E0501 warp_group 语境（producer 计算原语/顶层 WarpGroup barrier）。"""

    def test_producer_dot_rejected(self):
        source = _WG_BASE.replace(
            "        tis.barrier()\n    with tis.warp_group(role=\"consumer\"",
            "        tis.dot(a, b, c)\n    with tis.warp_group(role=\"consumer\"")
        r = _single(_check(source), "E0501")
        assert r.category == "wg-compute-prim"
        assert "producer" in r.suggestion

    def test_producer_reduce_rejected(self):
        source = _WG_BASE.replace(
            "        tis.barrier()\n    with tis.warp_group(role=\"consumer\"",
            "        tis.reduce(x, 1, Sum)\n    with tis.warp_group(role=\"consumer\"")
        r = _single(_check(source), "E0501")
        assert r.category == "wg-compute-prim"

    def test_consumer_dot_accepted(self):
        source = _WG_BASE.replace(
            "        tis.barrier()\n    tis.warp_group_sync",
            "        tis.dot(a, b, c)\n    tis.warp_group_sync")
        assert _check(source) == []

    def test_top_level_warp_group_barrier_rejected(self):
        source = _WG_BASE.replace(
            "    K_s = tis.alloc_shared((BR, 16), f16)",
            "    K_s = tis.alloc_shared((BR, 16), f16)\n"
            "    tis.barrier(scope=WarpGroup)")
        r = _single(_check(source), "E0501")
        assert r.category == "barrier-scope"
        assert "warp_group" in r.suggestion

    def test_warp_group_barrier_in_body_accepted(self):
        source = _WG_BASE.replace(
            "    with tis.warp_group(role=\"producer\", warps=2) as pg:\n"
            "        tis.barrier()",
            "    with tis.warp_group(role=\"producer\", warps=2) as pg:\n"
            "        tis.barrier(scope=WarpGroup)")
        assert _check(source) == []

    def test_producer_memory_prims_accepted(self):
        source = _WG_BASE.replace(
            "    with tis.warp_group(role=\"producer\", warps=2) as pg:\n"
            "        tis.barrier()",
            "    with tis.warp_group(role=\"producer\", warps=2) as pg:\n"
            "        tis.load(K_s, K_s, mode=Sync)\n"
            "        tis.store(K_s, K_s)\n"
            "        tis.barrier()")
        assert _check(source) == []

    def test_legal_pairing_zero_rejections(self):
        """汇合可见性为运行时承诺：合法配对静态零拒绝（间接承载）。"""
        assert _check(_WG_BASE) == []
