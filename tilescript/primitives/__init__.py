"""原语契约段检查管线（primitives/memory-ops + primitives/compute-ops 实现面）。

编排（design D1/D8/D9）：state_fields 复跑取注册表（输入与类型段一致、
零拒绝——段间短路下的确定性重放）；traverse 同构遍历（tis.* Call 改道
checker）；拒绝经 frontend.report.finalize（位置升序、order tiebreak）。
target 为 HAL 依赖检查（E0402/E0403/reduce scope）的唯一数据源。
"""

from ..frontend import report
from ..typecheck import state_fields, symbols
from . import compute_ops, memory_ops, traverse

STAGE_NAME = "primitive-contract"


class _RoutedChecker:
    """按 tis.* 名路由到存取/计算 checker（traverse 改道唯一回调点）。"""

    def __init__(self, memory: memory_ops.MemoryOpsChecker,
                 compute: compute_ops.ComputeOpsChecker):
        self.memory = memory
        self.compute = compute

    def check_call(self, node, inc):
        if node.func.attr in compute_ops._PRIM_SPECS:
            return self.compute.check_call(node, inc)
        return self.memory.check_call(node, inc)


def check_module(tree, target: str) -> "list[report.Rejection]":
    """对 AST 运行原语契约段检查，返回排序去重后的拒绝清单。"""
    comptime = symbols.kernel_comptime_names(tree)
    registry, _ = state_fields.check(tree, comptime_syms=comptime)
    memory = memory_ops.MemoryOpsChecker()
    compute = compute_ops.ComputeOpsChecker(target)
    checker = _RoutedChecker(memory, compute)
    traverse.walk_device_functions(tree, registry, comptime, checker)
    memory_ops.scan_value_positions(tree, memory)  # E0407 值位置独立扫描
    compute_ops.scan_special_value_contexts(tree, compute)  # D7 语境专用扫描
    return report.finalize(memory.rejections + compute.rejections)
