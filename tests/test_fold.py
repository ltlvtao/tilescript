"""多项式折叠基座测试（design D5——E0405 折叠支 (d) 专用规范化）。

fold 把 Dim 规范化为「单项式 → 整系数」多项式（单项式 = 排序符号名元组）：
- `+`/`-`/`*`/一元负做多项式代数；`//`/`%` 双 ConstDim 纯常量域整数求值；
- `/` → None；含符号操作数的 `//`/`%` 与除数 0 → None；
- `slice` 维取 upper - lower 差分。

折叠仅服务 E0405 相容判定，不产生类型层等价结论（类型段无数值折叠不动）。
"""

from tilescript.primitives.fold import fold
from tilescript.typecheck.types import ConstDim, DerivedDim, SymbolDim


def _sym(name):
    """运行期符号维（折叠代数与 comptime 性无关——只做结构代数）。"""
    return SymbolDim(name, False)


def test_slice_fold_bm_br():
    """tasks 2.1 例 1：`bm*BR:(bm+1)*BR` 差分折叠 → BR。"""
    lower = DerivedDim("*", (_sym("bm"), SymbolDim("BR", True)), False)
    upper = DerivedDim("*",
                       (DerivedDim("+", (_sym("bm"), ConstDim(1)), False),
                        SymbolDim("BR", True)), False)
    slice_dim = DerivedDim("slice", (lower, upper), False)
    assert fold(slice_dim) == {("BR",): 1}  # == fold(SymbolDim("BR"))


def test_derived_difference_folds_bc():
    """tasks 2.1 例 2：`(bm+1)*BC - bm*BC` → BC。"""
    left = DerivedDim("*",
                      (DerivedDim("+", (_sym("bm"), ConstDim(1)), False),
                       _sym("BC")), False)
    right = DerivedDim("*", (_sym("bm"), _sym("BC")), False)
    assert fold(DerivedDim("-", (left, right), False)) == {("BC",): 1}


def test_constant_difference_folds():
    """tasks 2.1 例 3：`64-0` → 64（纯常量差分）。"""
    dim = DerivedDim("-", (ConstDim(64), ConstDim(0)), True)
    assert fold(dim) == {(): 64}


def test_floor_div_pure_constants():
    """tasks 2.1 例 4：`(128//8,)` → 16（双 ConstDim 整数求值）。"""
    dim = DerivedDim("//", (ConstDim(128), ConstDim(8)), True)
    assert fold(dim) == {(): 16}
    assert fold(DerivedDim("%", (ConstDim(130), ConstDim(8)), True)) == {(): 2}


def test_floor_div_zero_divisor_none():
    """tasks 2.1 例 5：`x//0` 与常量除 0 → None。"""
    sym_div = DerivedDim("//", (ConstDim(128), ConstDim(0)), True)
    assert fold(sym_div) is None
    # 含符号操作数的 // / % → None（纯常量域外不可判定）
    assert fold(DerivedDim("//", (_sym("x"), ConstDim(8)), False)) is None
    assert fold(DerivedDim("%", (ConstDim(128), _sym("y")), False)) is None


def test_true_division_not_foldable():
    """tasks 2.1 例 6：含 `/` 的维不可折叠（真除法 → None）。"""
    dim = DerivedDim("/", (ConstDim(128), ConstDim(8)), True)
    assert fold(dim) is None
    nested = DerivedDim("slice", (ConstDim(0),
                                  DerivedDim("/", (ConstDim(16), ConstDim(2)), True)), True)
    assert fold(nested) is None


def test_basic_forms_and_algebra():
    """基元形态与零系数清理：常量/符号/一元负/相消为零。"""
    assert fold(ConstDim(64)) == {(): 64}
    assert fold(SymbolDim("BR", True)) == {("BR",): 1}
    assert fold(DerivedDim("-", (_sym("x"), ConstDim(3)), False)) == {("x",): 1, (): -3}
    neg = DerivedDim("-", (_sym("x"),), False)  # 一元负
    assert fold(neg) == {("x",): -1}
    assert fold(DerivedDim("-", (_sym("x"), _sym("x")), False)) == {}  # 相消 → 0
