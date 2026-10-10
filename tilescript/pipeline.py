"""五段检查管线编排（design D9）。

syntax → type-system → primitive-contract 依序；前序段存在任一拒绝时
后续段 MUST NOT 执行（R8 delta 段间短路）。execution-structure 与
numerics 由后续 change 实现。target（登记表成员）为 HAL 依赖检查的
数据输入（design D8；原语契约段 E0402/E0403/reduce scope 消费）。
CLI 只依赖本模块（design D1：编排归 pipeline，入口不触各段内部）。
"""

from . import frontend, hal, primitives, typecheck
from .frontend import carrier

# 本 change 后已实现段（toolchain/cli R3 段清单的唯一来源）。
IMPLEMENTED_STAGES = (frontend.STAGE_NAME, typecheck.STAGE_NAME,
                      primitives.STAGE_NAME)
ALL_STAGES = frontend.ALL_STAGES


def compile_stages(source: str, *, target: str) -> "list[frontend.report.Rejection]":
    """对模块源文本依序运行已实现检查段（target 为登记表成员，CLI 透传）。

    语法段非空即短路返回（E0101 单条特例亦在其内）；否则重解析载体进入
    类型系统段；类型段非空同样短路（原语契约段 MUST NOT 执行）。返回
    排序去重后的拒绝清单（空 = 已实现段零命中）。
    """
    if target not in hal.REGISTERED_TARGETS:
        raise ValueError(f"未登记目标：{target}")  # CLI 入口已挡；内部防御
    syntax_rejections = frontend.check_module(source)
    if syntax_rejections:
        return syntax_rejections
    tree, _ = carrier.parse(source)  # E0101 已短路；此处 tree 必有效
    type_rejections = typecheck.check_module(tree)
    if type_rejections:
        return type_rejections
    return primitives.check_module(tree, target)
