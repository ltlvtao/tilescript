"""E0301 转移非法格（type-system spec R4「作用域转移矩阵」）。

只裁矩阵唯一非法格 Global→Global（design D8）：tis.load/tis.store 的
src（实参 0）与 dst（实参 1）两侧均 Global 时拒绝。copy/move 格误用归
E0404、非 Tensor/UNKNOWN 实参与实参形态错误归 E0406 与后续段——本模块
均 MUST NOT 报告。参数集复用 primitives/memory-ops R1（src 0 / dst 1）。
"""

import ast

from tilescript.frontend.report import Rejection

from .types import TensorType, treats_as_unknown

_ORDER = 3  # design D9：E0302=1/E0304=2/E0301=3/E0303=4

_SUGGESTION = (
    "作用域转移矩阵不含 Global→Global 格（无承载原语）；"
    "load 合法目标：Shared、Register（Global→Shared / Global→Register / "
    "Shared→Register），store 合法源：Shared、Register（Shared→Global / "
    "Register→Global / Register→Shared）。跨 Global 搬运请经 Shared 或 "
    "Register 中转。"
)


def is_data_move_call(node) -> bool:
    """tis.load / tis.store 调用识别（E0301 检查入口，infer 在 Call 推断时调用）。"""
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "tis"
            and node.attr in ("load", "store"))


def check_call(node, inc):
    """对 tis.load/tis.store 调用裁决转移格；非法格追加 E0301 到 inc.rejections。"""
    if len(node.args) < 2:
        return  # 实参形态错误归 E0406（design D8），本段不裁
    source = inc.infer(node.args[0])
    target = inc.infer(node.args[1])
    if treats_as_unknown(source) or treats_as_unknown(target):
        return  # UNKNOWN 让渡（D4）
    if not (isinstance(source, TensorType) and isinstance(target, TensorType)):
        return  # 非 Tensor 实参归 E0406 与后续段（design D8）
    if source.scope == "Global" and target.scope == "Global":
        inc.rejections.append(Rejection(
            code="E0301", line=node.lineno, col=node.col_offset + 1,
            category="global-to-global", suggestion=_SUGGESTION, order=_ORDER,
            stage="type-system",
        ))
