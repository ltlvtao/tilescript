"""五段检查管线编排（design D9）。

syntax 在前；语法段存在任一拒绝时后续段 MUST NOT 执行（R8 delta 段间
短路）。已实现段为 syntax 与 type-system，其余三段由后续 change 实现。
CLI 只依赖本模块（design D1：编排归 pipeline，入口不触各段内部）。
"""

from . import frontend, typecheck
from .frontend import carrier

# 本 change 后已实现段（toolchain/cli R3 段清单的唯一来源）。
IMPLEMENTED_STAGES = (frontend.STAGE_NAME, typecheck.STAGE_NAME)
ALL_STAGES = frontend.ALL_STAGES


def compile_stages(source: str) -> "list[frontend.report.Rejection]":
    """对模块源文本依序运行已实现检查段。

    语法段非空即短路返回（E0101 单条特例亦在其内）；否则重解析载体进入
    类型系统段。返回排序去重后的拒绝清单（空 = 已实现段零命中）。
    """
    syntax_rejections = frontend.check_module(source)
    if syntax_rejections:
        return syntax_rejections
    tree, _ = carrier.parse(source)  # E0101 已短路；此处 tree 必有效
    return typecheck.check_module(tree)
