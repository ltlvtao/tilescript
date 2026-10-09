"""拒绝报告契约（syntax spec「语法拒绝报告契约」+ toolchain/cli R2 序列化）。

- Rejection：四要素（错误码/位置/类别/恢复建议）+ 排序元数据（order=检查管线序，
  跨段 tiebreak 预留段序随后续段实现生效，design D3/D10）。
- finalize：收集全部 → 稳定排序 (line, col, order) → 同位置去重（只留检查管线
  最早一条）。「同位置并列时按确定性收集序」由 Python 稳定排序自然承载。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rejection:
    code: str          # 错误码，如 "E0105"
    line: int          # 1 起始（toolchain/cli R2 冻结）
    col: int           # 1 起始
    category: str      # 被拒绝类别的稳定机器可读 slug（小写 ASCII+连字符，design D6）
    suggestion: str    # 人类可读恢复建议（中文，与 spec 恢复建议句对应）
    order: int         # 检查管线序（本 change：1=E0101 … 7=E0107；非错误码数值序）
    stage: str = field(default="syntax")  # 段名（跨段 tiebreak 预留）

    def to_dict(self) -> dict:
        """v1 冻结五字段，字段序固定（R2：序列化不得改变字段集与顺序）。"""
        return {
            "code": self.code,
            "line": self.line,
            "col": self.col,
            "category": self.category,
            "suggestion": self.suggestion,
        }


def finalize(rejections: "list[Rejection]") -> "list[Rejection]":
    """排序 + 同位置 tiebreak：位置升序；同 (line, col) 只保留检查管线序（order）
    最小的条目集合——跨检查阶段去重（spec「只报最早检查阶段」），同 order 并列
    全保留且按收集序（稳定排序，design D10 确定性）。
    """
    ordered = sorted(rejections, key=lambda r: (r.line, r.col, r.order))
    min_order: dict = {}
    for r in ordered:
        min_order.setdefault((r.line, r.col), r.order)
    return [r for r in ordered if r.order == min_order[(r.line, r.col)]]
