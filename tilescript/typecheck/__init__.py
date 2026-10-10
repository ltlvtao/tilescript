"""类型系统段检查管线（language/type-system 实现面）。

检查器按段内顺序收集（E0302→E0304→E0301→E0303；E0301/E0303 由
bindings 设备函数遍历经 infer/transfer 承载），报告契约（收集全部、
位置升序、同位置段内 order tiebreak、重复一致）复用 frontend.report。
"""

from tilescript.frontend import report

from . import annotations, bindings, state_fields, symbols

STAGE_NAME = "type-system"


def check_module(tree) -> "list[report.Rejection]":
    """对 AST 运行类型系统段检查，返回排序去重后的拒绝清单。"""
    comptime = symbols.kernel_comptime_names(tree)
    registry, rejections = state_fields.check(tree, comptime_syms=comptime)
    rejections.extend(annotations.check(tree, comptime_syms=comptime,
                                        registry=registry))
    rejections.extend(bindings.check_device_functions(tree, registry, comptime))
    return report.finalize(rejections)
