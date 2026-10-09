"""语法段检查管线（language/syntax-acceptance-set 实现面）。

七检查器按错误码数字序执行（E0101→E0107），报告契约（收集全部、位置升序、
同位置最早检查器 tiebreak、重复编译一致）由 report.finalize 承载（design D3）。
"""

from . import (
    carrier,
    decorators,
    expressions,
    report,
    signature,
    state_class,
    statements,
    top_level,
)

# 五段检查管线的段名封闭集（toolchain/cli R3：两数组合并恰为五值）。
STAGE_NAME = "syntax"
ALL_STAGES = (
    "syntax",
    "type-system",
    "primitive-contract",
    "execution-structure",
    "numerics",
)


def check_module(source: str) -> "list[report.Rejection]":
    """对模块源文本运行语法段检查，返回排序去重后的拒绝清单。

    E0101（无法建立 AST）走单条特例立即返回（syntax spec「语法拒绝报告契约」）。
    """
    tree, carrier_rejection = carrier.parse(source)
    if carrier_rejection is not None:
        return [carrier_rejection]

    rejections: list[report.Rejection] = []
    rejections.extend(top_level.check(tree))
    rejections.extend(decorators.check(tree))
    rejections.extend(signature.check(tree))
    rejections.extend(statements.check(tree))
    rejections.extend(state_class.check(tree))
    # expressions 由 statements 在白名单语句内递归调用（短路规则，design D3），
    # 此处不再整树扫描，避免对被拒语句子树重复报告。
    return report.finalize(rejections)
