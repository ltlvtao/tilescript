"""E0506 入口 HAL 支持面（hal/capability-descriptions「入口 HAL 支持面」）。

模块级扫描（design D1 编排首步，段内顺位最后）：persistent_kernel=false
目标拒 @tis.persistent_kernel（四要素：错误码/装饰器行位置/违反规则/
恢复建议）。目标能力唯一数据源 hal.capability(target)（design D8）。

扫描面为模块顶层函数（spec 主体「装饰的入口函数」）；装饰位置违规
形态（类/嵌套位）由语法段 E0103 先拒（管线短路结构性承载，design
D7）。入口体深检查不查（design residual 2——遍历面仅 @tis.kernel
同构）。
"""

import ast

from ..frontend.top_level import _has_tis_decorator
from ..hal import capability


def scan(tree, target, reject) -> "list":
    """模块级入口装饰器支持面扫描（E0506；reject 为 Rejection 构造回调）。"""
    if capability(target).persistent_kernel:
        return []
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _has_tis_decorator(node, "persistent_kernel"):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Attribute) and dec.attr == "persistent_kernel"
                    and isinstance(dec.value, ast.Name) and dec.value.id == "tis"):
                out.append(reject(
                    dec, "E0506", "entry-unsupported",
                    f"目标 {target} 的 persistent_kernel 支持状态为 "
                    "false；@tis.persistent_kernel 入口在该目标上不可用。"
                    "恢复建议：改用 tis.kernel 入口，或更换支持该特性的"
                    "目标。"))
    return out
