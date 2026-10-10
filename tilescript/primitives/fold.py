"""Dim → 整数系数符号多项式规范化（design D5——E0405 折叠支 (d) 专用）。

fold 把类型段的 Dim（ConstDim/SymbolDim/DerivedDim）规范化为
「单项式 → 整系数」多项式：单项式是排序后的符号名元组，`()` 为常数项，
空 dict 表示 0。两维折叠结果相等 ⇔ 维长多项式等价。

可折叠域（超域一律返回 None，调用方不产生相容结论）：
- `+`/`-`/`*`/一元负：多项式代数（分配律展开、同类项合并、零系数清理）；
- `//`/`%`：双 ConstDim 纯常量域整数求值（除数为 0 或含符号操作数 → None）；
- `/`：真除法永不折叠 → None；
- `slice`：维差分 `upper - lower`（Inferencer 切片维的记录形态）；
- UnknownDim / 未知 op：None。

折叠仅用于 E0405 逐维长度相容判定，不产生类型层等价结论
（type-system「无数值折叠」承诺不动，design D5）。
"""

from ..typecheck.types import ConstDim, DerivedDim, SymbolDim


def fold(dim) -> "dict[tuple[str, ...], int] | None":
    """Dim → 规范化多项式；不可折叠返回 None。"""
    if isinstance(dim, ConstDim):
        return {(): dim.value}
    if isinstance(dim, SymbolDim):
        return {(dim.name,): 1}
    if not isinstance(dim, DerivedDim):
        return None  # UnknownDim 等：不可折叠
    if dim.op == "-" and len(dim.operands) == 1:  # 一元负（单操作数形态）
        return _neg(fold(dim.operands[0]))
    if dim.op == "slice":
        upper, lower = fold(dim.operands[1]), fold(dim.operands[0])
        return _sub(upper, lower)
    if dim.op in ("+", "-", "*"):
        left, right = fold(dim.operands[0]), fold(dim.operands[1])
        if dim.op == "+":
            return _add(left, right)
        if dim.op == "-":
            return _sub(left, right)
        return _mul(left, right)
    if dim.op in ("//", "%"):
        # 双 ConstDim 纯常量域：两侧折叠均为纯常数多项式才可整数求值。
        left, right = fold(dim.operands[0]), fold(dim.operands[1])
        if _constant(left) is None or _constant(right) is None:
            return None
        if _constant(right) == 0:
            return None  # 除数为 0：不可折叠
        value = (_constant(left) // _constant(right) if dim.op == "//"
                 else _constant(left) % _constant(right))
        return {(): value}
    return None  # "/" 与未知 op


def constant(dim):
    """多项式为纯常数时返回其值，否则 None（E0405 支 (d)/E0403 可判定域用）。"""
    return _constant(fold(dim))


# ---- 多项式代数（None 传播；零系数清理保证规范形唯一） ----------------------

def _constant(poly):
    if poly is None or set(poly) - {()}:
        return None
    return poly.get((), 0)


def _normalize(poly):
    return {m: c for m, c in poly.items() if c != 0} or {}


def _neg(poly):
    if poly is None:
        return None
    return {m: -c for m, c in poly.items()}


def _add(a, b):
    if a is None or b is None:
        return None
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0) + c
    return _normalize(out)


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    if a is None or b is None:
        return None
    out: "dict[tuple[str, ...], int]" = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(sorted((*ma, *mb)))  # 单项式合并：符号名排序拼接
            out[m] = out.get(m, 0) + ca * cb
    return _normalize(out)
