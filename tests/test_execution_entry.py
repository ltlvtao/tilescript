"""E0506 入口 HAL 支持面测试（execution/entry_hal 实现面）。

公共行为断言：check_module(tree, target) 返回排序去重后的 Rejection
清单（stage="execution-structure"、段内 order=6）。两目标分化面：
ascend_910b（persistent_kernel=false）拒 / nvidia_h200（true）接受；
装饰器行同位置语法段拒绝由管线短路承载（design D7）。
"""

import ast

from tilescript.execution import check_module as execution_check
from tilescript.frontend import check_module as syntax_check

_PK_BASE = '''\
import tis

@tis.persistent_kernel
def pk_kernel(K_ptr: Pointer[f16, Global], D: comptime[int] = 64):
    K_s = tis.alloc_shared((D, 16), f16)
    return
'''


def _check(source: str, target: str):
    return execution_check(ast.parse(source), target)


class TestEntryHal:
    """E0506 入口 HAL 支持面（hal spec「入口 HAL 支持面」3S）。"""

    def test_ascend_rejects_persistent_kernel_entry(self):
        """ascend_910b 拒 persistent_kernel 入口：四要素齐全（装饰器行）。"""
        r = _check(_PK_BASE, "ascend_910b")
        assert len(r) == 1, [x.to_dict() for x in r]
        r = r[0]
        assert r.code == "E0506"
        assert r.stage == "execution-structure"
        assert r.category == "entry-unsupported"
        assert r.line == 3                       # 入口装饰器行
        assert "tis.kernel" in r.suggestion      # 替代路径
        assert "目标" in r.suggestion            # 违反规则（支持状态 false）

    def test_h200_accepts_same_entry(self):
        """nvidia_h200 接受同一入口（E0506 零命中；体深检查为 residual 2）。"""
        assert _check(_PK_BASE, "nvidia_h200") == []

    def test_decorator_line_syntax_rejection_single_report(self):
        """装饰器行同位置语法段拒绝：管线最早段规则只报 E0103 一条。

        E0506 判定存在（同目标下顶层函数形态命中），但语法段非空时
        执行段不跑——hal spec Scenario 3 由短路结构性承载（design D7）。
        """
        source = '''\
import tis

@tis.persistent_kernel
class pk_kernel:
    pass
'''
        syntax = syntax_check(source)
        # 语法段输出（含类类别违规 E0102 于类行）；装饰器行（line 3）恰一条 E0103
        dec_line = [r for r in syntax if r.line == 3]
        assert len(dec_line) == 1 and dec_line[0].code == "E0103"
        # 管线编排（最早段短路；第四段接入前模拟 9.2 编排语义）
        pipeline_out = syntax if syntax else _check(source, "ascend_910b")
        assert not any(r.code == "E0506" for r in pipeline_out)   # E0506 不出现
        assert len([r for r in pipeline_out if r.line == 3]) == 1

    def test_repeat_compilation_identical(self):
        """同一输入重复编译 E0506 清单逐条一致。"""
        a = _check(_PK_BASE, "ascend_910b")
        b = _check(_PK_BASE, "ascend_910b")
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
