"""E0404/E0405/E0406/E0407 存取原语检查（primitives/memory-ops 实现面）。

八原语参数集与值域（E0406）、转移格承载（E0404）、数据维度（E0405）、
语境（E0407）。段内 order（design D9）：E0404=1/E0405=2/E0406=3/E0407=4。

同一调用首个命中即止（R8「同位置多规则只报段内首个」）；判定序 =
实参表达式全量推断（嵌套 tis.* 检查不因外层违规丢失）→ 结构（参数集，
不可解析即短路后续判定）→ E0404 → E0405 → 实参类别/值域（E0406）→
E0407 语境。违规调用结果 UNKNOWN（拒绝已记录，design D4）。

非本域 `tis.*` 名（Pipeline/Layout/kernel 等）让渡返回 UNKNOWN、零拒绝。
"""

import ast

from ..frontend.report import Rejection
from ..typecheck.types import (
    DTYPES,
    UNKNOWN,
    ConstDim,
    DerivedDim,
    PointerType,
    StateType,
    SymbolDim,
    TensorType,
    UnknownDim,
    desc,
    treats_as_unknown,
)
from . import results
from .fold import fold

# 段内 order（design D9）。
_ORDERS = {"E0404": 1, "E0405": 2, "E0406": 3, "E0407": 4}

# 参数集（R1）：槽名/必选标志；顺序即声明序（位置实参按此绑定）。
_PRIM_SPECS = {
    "load": (("src", True), ("dst", True), ("mode", False)),
    "store": (("src", True), ("dst", True)),
    "barrier": (("scope", False),),
    "make_tensor": (("ptr", True), ("shape", True)),
    "alloc_shared": (("shape", True), ("dtype", True), ("layout", False)),
    "zeros": (("shape", True), ("dtype", True), ("scope", False)),
    "full": (("shape", True), ("value", True), ("dtype", True), ("scope", False)),
    "cast": (("x", True), ("dtype", True)),
}

_LOAD_CELLS = {("Global", "Shared"), ("Global", "Register"), ("Shared", "Register")}
_STORE_CELLS = {("Shared", "Global"), ("Register", "Global"), ("Register", "Shared")}
_UNCARRIED_CELLS = {("Shared", "Shared"): "copy", ("Register", "Register"): "move"}


class MemoryOpsChecker:
    """tis.* 存取原语检查器（traverse 改道回调；rejections 侧收集）。"""

    def __init__(self, comptime: "frozenset[str]" = frozenset()):
        self.comptime = comptime  # kernel comptime[int] 名集（R5 值域判定输入）
        self.rejections: "list[Rejection]" = []
        self.hit_calls: "set[int]" = set()  # 已产生拒绝的调用节点（E0407 协调）

    # ---- 改道入口 ---------------------------------------------------------

    def check_call(self, node: ast.Call, inc):
        name = node.func.attr
        if name not in _PRIM_SPECS:
            return UNKNOWN  # 非本域 tis.* 名：让渡
        # 实参表达式全量推断在前（嵌套 tis.* 检查不因外层违规丢失）。
        for arg in node.args:
            inc.infer(arg)
        for kw in node.keywords:
            inc.infer(kw.value)
        slots, err = _bind_args(node, name)
        if err is not None:
            self._reject(node, "param-set", err)
            return UNKNOWN
        return getattr(self, f"_check_{name}")(node, slots, inc)

    # ---- load / store -----------------------------------------------------

    def _check_load(self, node, slots, inc):
        return self._check_move(node, slots, inc, is_load=True)

    def _check_store(self, node, slots, inc):
        return self._check_move(node, slots, inc, is_load=False)

    def _check_move(self, node, slots, inc, is_load):
        src_t = inc.infer(slots["src"])
        dst_t = inc.infer(slots["dst"])
        # 段内序（design D9 裁定序）：E0404 → E0405 → E0406（类别/值域）。
        both_tensor = (isinstance(src_t, TensorType)
                       and isinstance(dst_t, TensorType))
        if both_tensor and not self._check_transfer_cell(node, src_t, dst_t,
                                                         is_load):
            return UNKNOWN
        if both_tensor and not self._check_dims(node, src_t, dst_t):
            return UNKNOWN
        for slot, t in (("src", src_t), ("dst", dst_t)):
            if not isinstance(t, TensorType) and not treats_as_unknown(t):
                self._reject(
                    node, "arg-kind",
                    f"{'load' if is_load else 'store'} 的 {slot} 必须为 Tensor；"
                    f"实际 {desc(t)}。Pointer 先以 tis.make_tensor 建立 Tensor "
                    "视图，标量/状态类值核对数据容器。")
                return UNKNOWN
        if is_load and "mode" in slots and not _is_name(slots["mode"],
                                                        ("Sync", "Async")):
            self._reject(node, "mode-value",
                         "load 的 mode 值域为 Sync/Async（默认 Sync）。")
            return UNKNOWN
        if (is_load and _is_name(slots.get("mode"), ("Async",))
                and not inc.in_produce_consume):
            self._reject(node, "async-context",
                         "Async 加载只能出现在 Pipeline produce/consume 嵌套"
                         "函数体内；移入 produce/consume 或改用 mode=Sync"
                         "（完成保证由 Pipeline 契约承载）。", code="E0407")
            return UNKNOWN
        return UNKNOWN  # load/store 不产生值（合法形态为表达式语句，E0407 承载）

    # ---- barrier ----------------------------------------------------------

    def _check_barrier(self, node, slots, inc):
        if "scope" in slots and not _is_name(slots["scope"], ("Block", "WarpGroup")):
            self._reject(node, "scope-value",
                         "barrier 的 scope 值域为 Block/WarpGroup（默认 Block）。")
            return UNKNOWN
        return UNKNOWN  # 不产生值

    # ---- make_tensor / alloc_shared / zeros / full ------------------------

    def _check_make_tensor(self, node, slots, inc):
        ptr_t = inc.infer(slots["ptr"])
        if not treats_as_unknown(ptr_t) and not isinstance(ptr_t, PointerType):
            self._reject(node, "ptr-kind",
                         f"make_tensor 的 ptr 必须为 Pointer；实际 {desc(ptr_t)}。")
            return UNKNOWN
        if isinstance(ptr_t, PointerType) and ptr_t.scope != "Global":
            self._reject(node, "ptr-scope",
                         f"make_tensor 只接受 Global scope 的 Pointer；"
                         f"实际 Pointer[{ptr_t.dtype}, {ptr_t.scope}]。")
            return UNKNOWN
        dims = self._shape_dims(slots["shape"], inc)
        if dims is None:
            return UNKNOWN  # shape 形态/组件违规已记 E0406
        if treats_as_unknown(ptr_t):
            return UNKNOWN  # ptr 让渡（buf.* 等）：结果 UNKNOWN
        return results.make_tensor_result(ptr_t, dims)

    def _check_alloc_shared(self, node, slots, inc):
        dims = self._shape_dims(slots["shape"], inc)
        dtype = _dtype_name(slots["dtype"])
        if dtype is None:
            self._reject(node, "dtype-value",
                         f"alloc_shared 的 dtype 必须为六种封闭集之一"
                         f"（{'/'.join(DTYPES)}）。")
            return UNKNOWN
        if "layout" in slots and not _layout_ok(slots["layout"], self, node):
            return UNKNOWN
        if dims is None:
            return UNKNOWN
        return results.alloc_shared_result(dtype, dims)

    def _check_zeros(self, node, slots, inc):
        return self._check_alloc_like(node, slots, inc)

    def _check_full(self, node, slots, inc):
        value_t = inc.infer(slots["value"])
        if isinstance(value_t, (TensorType, PointerType, StateType)):
            self._reject(node, "value-form",
                         f"full 的 value 必须为标量表达式；实际 {desc(value_t)}。")
            return UNKNOWN
        return self._check_alloc_like(node, slots, inc)

    def _check_alloc_like(self, node, slots, inc):
        """zeros/full 共体：shape 组件、dtype 值域、scope 值域（Global 拒）。"""
        dims = self._shape_dims(slots["shape"], inc)
        dtype = _dtype_name(slots["dtype"])
        if dtype is None:
            self._reject(node, "dtype-value",
                         f"dtype 必须为六种封闭集之一（{'/'.join(DTYPES)}）。")
            return UNKNOWN
        if "scope" in slots:
            scope_node = slots["scope"]
            if _is_name(scope_node, ("Global",)):
                self._reject(node, "scope-value",
                             "zeros/full 的 scope 值域为 Register/Shared；"
                             "Global 张量必须经 tis.make_tensor 从指针建立。")
                return UNKNOWN
            if not _is_name(scope_node, ("Register", "Shared")):
                self._reject(node, "scope-value",
                             "zeros/full 的 scope 值域为 Register/Shared"
                             "（默认 Register）。")
                return UNKNOWN
            scope = scope_node.id
        else:
            scope = "Register"
        if dims is None:
            return UNKNOWN
        return results.alloc_result(dtype, dims, scope)

    # ---- cast -------------------------------------------------------------

    def _check_cast(self, node, slots, inc):
        x_t = inc.infer(slots["x"])
        dtype = _dtype_name(slots["dtype"])
        if not treats_as_unknown(x_t) and not isinstance(x_t, TensorType):
            self._reject(node, "arg-kind",
                         f"cast 只接受 Tensor；实际 {desc(x_t)}。")
            return UNKNOWN
        if dtype is None:
            self._reject(node, "dtype-value",
                         f"cast 的 dtype 必须为六种封闭集之一（{'/'.join(DTYPES)}）。")
            return UNKNOWN
        if treats_as_unknown(x_t):
            return UNKNOWN
        return results.cast_result(x_t, dtype)

    # ---- E0404 / E0405（3.2 / 3.3 增量接入） -------------------------------

    def _check_transfer_cell(self, node, src_t, dst_t, is_load) -> bool:
        """R2 转移格承载：True = 通过/不适用；False = 已记 E0404。

        矩阵非法格（Global→Global）归类型段 E0301（管线下短路不达本段；
        直调零 E04xx），本段 MUST NOT 重复报告。
        """
        combo = (src_t.scope, dst_t.scope)
        own = _LOAD_CELLS if is_load else _STORE_CELLS
        if combo in own:
            return True
        opposite = _STORE_CELLS if is_load else _LOAD_CELLS
        if combo in opposite:
            kind = "store" if is_load else "load"
            self._reject(node, "transfer-cell",
                         f"scope 组合 {src_t.scope}→{dst_t.scope} 的操作类别为 "
                         f"{kind}，应使用 tis.{kind}。", code="E0404")
            return False
        if combo in _UNCARRIED_CELLS:
            kind = _UNCARRIED_CELLS[combo]
            self._reject(node, "transfer-cell",
                         f"scope 组合 {src_t.scope}→{dst_t.scope} 为 {kind} 类别，"
                         "当前无承载原语；同 scope 数据路径由普通赋值承载。",
                         code="E0404")
            return False
        return True  # 矩阵非法格：E0301 归类型段

    def _check_dims(self, node, src_t, dst_t) -> bool:
        """R3 数据维度契约：True = 通过/让渡；False = 已记 E0405。

        dtype 先于逐维（首个命中即止）；含 UNKNOWN 维的 Tensor 整体让渡
        （design D11 residual 1）。
        """
        if src_t.dtype != dst_t.dtype:
            self._reject(node, "dtype-mismatch",
                         f"数据维度契约：dtype 不同（{src_t.dtype} 与 "
                         f"{dst_t.dtype}）；src {desc(src_t)}，dst {desc(dst_t)}；"
                         "使用 tis.cast 显式转换后再移动。", code="E0405")
            return False
        if len(src_t.dims) != len(dst_t.dims):
            self._reject(node, "dim-incompatible",
                         f"数据维度契约：维度数不同（{len(src_t.dims)} 与 "
                         f"{len(dst_t.dims)}）；src {desc(src_t)}，"
                         f"dst {desc(dst_t)}。", code="E0405")
            return False
        if any(isinstance(d, UnknownDim) for d in (*src_t.dims, *dst_t.dims)):
            return True  # UNKNOWN 维让渡
        for i, (d1, d2) in enumerate(zip(src_t.dims, dst_t.dims)):
            if not _dim_compatible(d1, d2):
                self._reject(node, "dim-incompatible",
                             f"数据维度契约：第 {i} 维不相容"
                             f"（{_dim_text(d1)} 与 {_dim_text(d2)}，无相容支）；"
                             f"src {desc(src_t)}，dst {desc(dst_t)}。",
                             code="E0405")
                return False
        return True

    # ---- 辅助 -------------------------------------------------------------

    def _shape_dims(self, node, inc):
        """shape 实参元组 → Dim 元组；形态/组件违规记 E0406 并返回 None。"""
        if not isinstance(node, ast.Tuple):
            self._reject(node, "shape-component",
                         "shape 必须为元组，每维为 int 标量表达式或 "
                         "comptime[int] 常量。")
            return None
        dims = []
        for comp in node.elts:
            dim = results.infer_shape_component(comp, inc.env)
            if dim is None:
                self._reject(node, "shape-component",
                             "shape 组件必须为整数类别表达式（int 标量或 "
                             "comptime[int] 常量）；浮点/非整数组件不合法。")
                return None
            dims.append(dim)
        return tuple(dims)

    def _reject(self, node, category, suggestion, code="E0406"):
        self.hit_calls.add(id(node))  # R8：同调用后续 E0407 值位置不再叠加
        self.rejections.append(Rejection(
            code=code, line=node.lineno, col=node.col_offset + 1,
            category=category, suggestion=suggestion, order=_ORDERS[code],
            stage="primitive-contract",
        ))


# ---- 实参槽绑定与值域判定 -------------------------------------------------

def _bind_args(node: ast.Call, name: str):
    """位置+关键字实参 → {槽名: AST 节点}；结构违规返回 (None, 错误说明)。

    spec R1：未知关键字 / 位置超量 / 缺必选 → E0406。重复绑定（位置与
    关键字同槽）与 **kwargs 展开同为参数集封闭性违规（undefined 细节的
    一贯延伸，落 E0406）。
    """
    spec = _PRIM_SPECS[name]
    names = [s[0] for s in spec]
    if len(node.args) > len(spec):
        return None, (f"{name} 只接受 {len(spec)} 个位置实参"
                      f"（{_param_set_line(name)}）；实际传入 "
                      f"{len(node.args)} 个。")
    slots = {}
    for i, arg in enumerate(node.args):
        slots[names[i]] = arg
    for kw in node.keywords:
        if kw.arg is None:
            return None, f"{name} 不接受 **kwargs 展开；{_param_set_line(name)}。"
        if kw.arg not in names:
            return None, (f"未知关键字实参 {kw.arg}；{name} 的合法参数集为 "
                          f"{_param_set_line(name)}。")
        if kw.arg in slots:
            return None, (f"实参 {kw.arg} 同时按位置与关键字传递；"
                          f"{name} 的合法参数集为 {_param_set_line(name)}。")
        slots[kw.arg] = kw.value
    missing = [n for n, required in spec if required and n not in slots]
    if missing:
        return None, (f"缺少必选实参 {'/'.join(missing)}；{name} 的合法参数集为 "
                      f"{_param_set_line(name)}。")
    return slots, None


def _param_set_line(name: str) -> str:
    return "/".join(n for n, _ in _PRIM_SPECS[name])


def _dim_compatible(d1, d2) -> bool:
    """R3 逐维长度相容（专用判定，不产生类型层等价结论）。

    可折叠域：多项式等价（涵盖支 (a) 双常量等值、(b) 同名符号、(d) 派生
    折叠为常量、(c) 的数学等价超集）；不可折叠域（含 `/` 等返回 None）：
    结构等价兜底（支 (c) 原义）。
    """
    f1, f2 = fold(d1), fold(d2)
    if f1 is not None and f2 is not None:
        return f1 == f2
    return d1 == d2


def _dim_text(dim) -> str:
    """维度拒绝报告的组件文本（常量值/符号名/派生结构）。"""
    if isinstance(dim, ConstDim):
        return f"编译期常量 {dim.value}"
    if isinstance(dim, SymbolDim):
        return f"{'comptime' if dim.comptime else '运行期'} {dim.name}"
    if isinstance(dim, DerivedDim):
        parts = " ".join(_dim_text(o) for o in dim.operands)
        return f"派生维（{parts}）"
    return "未知维度"


def _is_name(node, ids: tuple) -> bool:
    return isinstance(node, ast.Name) and node.id in ids


def _dtype_name(node):
    """dtype 槽 → 封闭集内名字；集合外/非名字形态返回 None。"""
    if isinstance(node, ast.Name) and node.id in DTYPES:
        return node.id
    return None


def _layout_ok(node, checker, call_node) -> bool:
    """layout 值域：RowMajor | tis.Layout.swizzled(xor=非负 comptime[int])。"""
    if isinstance(node, ast.Name) and node.id == "RowMajor":
        return True
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "swizzled"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "Layout"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "tis"):
        xor = None
        if len(node.args) == 1:
            xor = node.args[0]
        for kw in node.keywords:
            if kw.arg == "xor":
                xor = kw.value
        if xor is not None and _comptime_nonneg(xor, checker.comptime):
            return True
        checker._reject(call_node, "layout-value",
                        "swizzled 的 xor 必须为非负 comptime[int]。")
        return False
    checker._reject(call_node, "layout-value",
                    "layout 值域为 RowMajor（默认）或 "
                    "tis.Layout.swizzled(xor=非负 comptime[int])。")
    return False


def _comptime_nonneg(node, comptime) -> bool:
    """xor 实参：非负 int 字面量（可带一元负判定为负即违规）或
    comptime[int] 名（值约束域在名上，让渡为接受面）；运行期 int 名
    值非静态可知 → 违规。"""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value >= 0
    if isinstance(node, ast.Name):
        return node.id in comptime
    return False


# ---- E0407 值位置独立扫描（design D6：封闭集 = 赋值右侧/调用实参/return 值）--

_VALUELESS_PRIMS = ("load", "store", "barrier")


def scan_value_positions(tree: ast.Module, checker: MemoryOpsChecker):
    """无值原语（load/store/barrier）值位置扫描（R4/R7 值位置 Scenario）。

    同调用已产生更早段内序拒绝（E0404/E0405/E0406）时跳过（R8 只报首个）；
    封闭集外位置（If/While 条件、赋值 target 侧等）不裁（spec undefined，
    design D6 收窄——后续补洞 change 裁决）。
    """
    parents: "dict[ast.AST, ast.AST]" = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tis"
                and node.func.attr in _VALUELESS_PRIMS):
            continue
        if id(node) in checker.hit_calls:
            continue  # 同调用更早段内序命中已报告
        position = _value_position(node, parents)
        if position is None:
            continue
        checker._reject(node, "value-position",
                        f"{node.func.attr} 不产生值，不能用于{position}；"
                        "合法使用形态为表达式语句。", code="E0407")


def _value_position(node, parents) -> "str | None":
    """沿父链判定值位置：命中返回位置描述，合法/封闭集外返回 None。"""
    cur = node
    while True:
        parent = parents.get(cur)
        if parent is None:
            return None
        if isinstance(parent, ast.Expr):
            return None  # 表达式语句根：合法形态
        if isinstance(parent, ast.Call):
            return "调用实参位置"
        if isinstance(parent, ast.Return):
            return "return 值位置"
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if cur is parent.value:
                return "赋值右侧"
            return None  # target 侧：封闭集外不裁
        if isinstance(parent, ast.stmt):
            return None  # If/While/For 等条件与语句位：undefined 不裁（D6）
        cur = parent
