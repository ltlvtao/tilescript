"""数值类型环境与表达式推断总入口（design D2/D3）。

数值域节点自载（Constant / 具名常量 inf / BinOp / Compare / BoolOp /
UnaryOp / tis.* 调用 / Subscript 下标位），其余委托 `typecheck.infer.
Inferencer`（共享 env 引用）；**类型域拒绝零重放**：委托内部产生的
E0303 落在 `_delegate.rejections` 不进 sink——数值段 env 比类型段精确
（算术结果有类型），委托路径重放可能产生类型段当时因 UNKNOWN 让渡而
未报的 E0303，type-system「UNKNOWN 参与绑定不产生 E0303」是冻结行为，
数值段不得改变其输出面。

数值域内部形态（`BOOL` 哨兵、`PendingConst` 待定型常量）定义于
`constants`（依赖方向：value_env → {binary_ops, constants}，
binary_ops → constants，无环）。

R7 合并帧（`_Frame`）：同一顶层算术/比较/逻辑表达式内，表达式级违规
（E0601–E0606——含 tis.full value 常量绑定）按码聚合；收帧时**每码恰一
条**（不同码各报各条——按段内序），同码多操作数合并为一条并列出全部
违规操作数位置（R7「同表达式多个操作数同类违规合并」——不按码排除）；
「同位置多码取段内首个」由 report.finalize 的 (line, col, order) 去重
承载。E0303 与语句级绑定拒绝（裸绑定/再绑定/注解绑定）为位置级，
不经帧。
"""

import ast
import math

from ..frontend import report
from ..primitives import results as prim_results
from ..primitives.compute_ops import COMPUTE_PRIMS
from ..primitives.compute_ops import _bind_args as _bind_compute
from ..primitives.memory_ops import _bind_args as _bind_memory
from ..typecheck.infer import _mismatch_suggestion
from ..typecheck.types import (
    COMPTIME_INT,
    DTYPES,
    INT,
    UNKNOWN,
    PointerType,
    ScalarType,
    TensorType,
    treats_as_unknown,
)
from . import binary_ops, constants
from .constants import BOOL, PendingConst  # noqa: F401 —— 数值域形态 re-export

# 段内 order（spec R7：E0601=1 → E0606=6；E0303 择一构造取类型段口径 4）。
ORDERS = {"E0601": 1, "E0602": 2, "E0603": 3, "E0604": 4, "E0605": 5, "E0606": 6}
E0303_ORDER = 4  # type-system 段 E0303 口径（infer._ORDER 同值）

# E06xx 违规类别 slug（稳定机器可读；R7 四要素的类别位）。
CATEGORIES = {
    "E0601": "mixed-family-arith",
    "E0602": "cross-dtype-arith",
    "E0603": "tensor-incompatible",
    "E0604": "int-division",
    "E0605": "bool-closure",
    "E0606": "const-binding",
}


class _Frame:
    """R7 合并帧：顶层表达式根 + 操作数级违规按码聚合（收集序确定）。"""

    __slots__ = ("root", "hits")

    def __init__(self, root):
        self.root = root
        self.hits = {}  # code -> [(节点, 文案)]

    def hit(self, code, node, note):
        self.hits.setdefault(code, []).append((node, note))

    def emit(self, inc):
        """收帧：每码恰一条（段内序）；同码多操作数合并列全部位置。"""
        for code in sorted(self.hits, key=lambda c: ORDERS[c]):
            hits = self.hits[code]
            if len(hits) == 1:
                node, note = hits[0]
                inc.reject(node, code, CATEGORIES[code], f"{note}。")
            else:
                first_node, first_note = hits[0]
                operands = "；".join(
                    f"第 {n.lineno} 行第 {n.col_offset + 1} 列"
                    for n, _ in hits)
                inc.reject(first_node, code, CATEGORIES[code],
                           f"{first_note}；违规操作数：{operands}。")


class NumericsInferencer:
    """数值域推断器：数值域节点自载 + 其余委托 Inferencer（零重放）。

    sink：E06xx/E0303 拒绝收集回调（遍历器 rejections 列表 append）。
    values：comptime 名 → 值（kernel 形参默认值起点 + 局部 comptime 绑定）。
    """

    def __init__(self, env, registry, functions, comptime_syms, values, sink):
        from ..typecheck.infer import Inferencer
        self._delegate = Inferencer(env, registry, functions, comptime_syms)
        self.env = env
        self.registry = registry
        self.comptime_syms = comptime_syms
        self.functions = functions
        self.values = dict(values)  # 名 → comptime 整数值（不改动调用方表）
        self.sink = sink
        self.const_values: "dict[int, int]" = {}  # id(节点) → 折叠值
        self._frame: "_Frame | None" = None

    # ---- 拒绝构造 ---------------------------------------------------------

    def reject(self, node, code, category, suggestion, *, order=None,
               stage=None):
        self.sink(report.Rejection(
            code=code, line=node.lineno, col=node.col_offset + 1,
            category=category, suggestion=suggestion,
            order=order if order is not None else ORDERS.get(code, 1),
            stage=stage if stage is not None else "numerics"))

    def reject_e0303(self, node, source, target):
        """算术赋值再绑定不等价（R7 授权面；类型段口径 stage/order/category）。"""
        self.sink(report.Rejection(
            code="E0303", line=node.lineno, col=node.col_offset + 1,
            category="assign", suggestion=_mismatch_suggestion(source, target),
            order=E0303_ORDER, stage="type-system"))

    def report(self, code, node, note):
        """操作数级违规入口：帧活跃进帧（R7 合并），否则直接构造单条。"""
        if self._frame is not None:
            self._frame.hit(code, node, note)
        else:
            self.reject(node, code, CATEGORIES[code], f"{note}。")

    # ---- 表达式入口 -------------------------------------------------------

    def infer(self, node):
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            if node.id == "inf" and node.id not in self.env \
                    and node.id not in self.comptime_syms:
                return PendingConst("inf", math.inf)  # 预置名（env 遮蔽优先）
            return self._delegate.infer(node)
        if isinstance(node, ast.BinOp):
            return self.framed(node, lambda: binary_ops.arith(node, self))
        if isinstance(node, ast.Compare):
            return self.framed(node, lambda: binary_ops.compare(node, self))
        if isinstance(node, ast.BoolOp):
            return self.framed(node, lambda: binary_ops.boolop(node, self))
        if isinstance(node, ast.UnaryOp):
            return binary_ops.unary(node, self)
        if isinstance(node, ast.Call):
            return self._infer_call(node)
        if isinstance(node, ast.Subscript):
            self._check_indices(node)
            return self._delegate.infer(node)
        return self._delegate.infer(node)  # Attribute 等：委托

    def framed(self, root, compute):
        """顶层数值域表达式开帧；子表达式违规进外层帧（嵌套合并）。

        公开入口：语句层等价展开路径（AugAssign 值）与 infer 分派共用，
        保证非 infer 入口的表达式同样承载 R7 同帧合并。
        """
        if self._frame is not None:
            return compute()
        self._frame = _Frame(root)
        try:
            result = compute()
        finally:
            frame = self._frame
            self._frame = None
        frame.emit(self)
        return result

    def _constant(self, node):
        value = node.value
        if type(value) is int:
            self.const_values[id(node)] = value
            return ScalarType(COMPTIME_INT)
        if type(value) is float:
            return PendingConst("float", value)
        if type(value) is bool:
            return BOOL
        return UNKNOWN  # None/str：让渡

    # ---- 下标位（R3 封闭集外使用 + 数值域检查） ---------------------------

    def _check_indices(self, node):
        """下标/切片边界表达式：数值域检查（算术违规）+ BOOL 逃逸（R3——
        下标位在布尔封闭集外）。"""
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) \
            else (node.slice,)
        for item in items:
            if isinstance(item, ast.Slice):
                for part in (item.lower, item.upper, item.step):
                    if part is not None:
                        self._check_index_expr(part)
            else:
                self._check_index_expr(item)

    def _check_index_expr(self, expr):
        if self.infer(expr) is BOOL:
            self.reject(expr, "E0605", CATEGORIES["E0605"],
                        "布尔值仅限条件语境封闭集；不能作为下标表达式")

    # ---- 调用：tis.* 自载（D3 返回类型表）+ 其余委托 ----------------------

    def _infer_call(self, node):
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "tis"):
            return self._tis_result(node)
        # 非 tis 调用（状态类构造/带注解设备函数/内建）：委托路径。实参的
        # 布尔逃逸（R3 封闭集——调用实参位）数值段自载检查；类型域 E0303
        # 由委托内部 check_binding 产生、落 _delegate.rejections（零重放）。
        for arg in (*node.args, *(kw.value for kw in node.keywords)):
            if self.infer(arg) is BOOL:
                self.reject(arg, "E0605", CATEGORIES["E0605"],
                            "布尔值仅限条件语境封闭集（if/elif 条件、逻辑"
                            "运算操作数）；不能作为调用实参")
        return self._delegate.infer(node)

    def _tis_result(self, node):
        """tis.* 调用数值段返回类型（D3 表；参数契约由前置段保证）。

        实参先全量推断（嵌套数值域检查不因外层形态丢失）；布尔逃逸与
        full value 常量绑定（R4 位置①）在此承载。结构域原语与不可判定
        形态一律 UNKNOWN 让渡。
        """
        attr = node.func.attr
        arg_nodes = (*node.args, *(kw.value for kw in node.keywords))
        values = [self.infer(a) for a in arg_nodes]
        for value, arg_node in zip(values, arg_nodes):
            if value is BOOL:
                self.reject(arg_node, "E0605", CATEGORIES["E0605"],
                            "布尔值仅限条件语境封闭集；不能作为原语实参")
        if attr in ("alloc_shared", "zeros", "full", "make_tensor", "cast",
                    "dot", "reduce", "maximum", "exp", "log", "transpose"):
            # 计算原语的参数集表在 compute_ops、其余在 memory_ops（两表
            # 同名 _PRIM_SPECS；按公共 COMPUTE_PRIMS 清单分派——清单扩展
            # 的同步义务由 compute_ops 侧承载）。
            bind = _bind_compute if attr in COMPUTE_PRIMS else _bind_memory
            slots, err = bind(node, attr)
            if err is not None:
                return UNKNOWN  # 参数集违规：原语段已拒（防御让渡）
            return getattr(self, f"_prim_{attr}")(node, slots)
        if attr == "cdiv":
            return self._prim_cdiv(node)
        if attr == "block_idx":
            return ScalarType(INT)
        return UNKNOWN  # Pipeline/produce/consume/run/load/store/barrier/
        # warp_group/warp_group_sync/tile_iter/kernel 装饰器域等：结构域让渡

    # ---- D3 结果构造（复用 primitives/results 构造器，不重实现） ----------

    def _prim_alloc_shared(self, node, slots):
        dims = self._shape_dims(slots["shape"])
        dtype = _dtype_of(slots["dtype"])
        if dims is None or dtype is None:
            return UNKNOWN
        return prim_results.alloc_shared_result(dtype, dims)

    def _prim_zeros(self, node, slots):
        return self._alloc_like(slots)

    def _prim_full(self, node, slots):
        # R4 位置①：value 常量绑定裁决（标量形态由 E0406 先行裁决）。
        constants.full_value_verdict(slots["value"], slots["dtype"], self)
        return self._alloc_like(slots)

    def _alloc_like(self, slots):
        dims = self._shape_dims(slots["shape"])
        dtype = _dtype_of(slots["dtype"])
        if dims is None or dtype is None:
            return UNKNOWN
        scope = "Register"
        scope_node = slots.get("scope")
        if isinstance(scope_node, ast.Name) and scope_node.id in ("Register",
                                                                 "Shared"):
            scope = scope_node.id
        return prim_results.alloc_result(dtype, dims, scope)

    def _prim_make_tensor(self, node, slots):
        ptr = self.infer(slots["ptr"])
        dims = self._shape_dims(slots["shape"])
        if not isinstance(ptr, PointerType) or dims is None:
            return UNKNOWN
        return prim_results.make_tensor_result(ptr, dims)

    def _prim_cast(self, node, slots):
        x = self.infer(slots["x"])
        dtype = _dtype_of(slots["dtype"])
        if not isinstance(x, TensorType) or treats_as_unknown(x) \
                or dtype is None:
            return UNKNOWN
        return prim_results.cast_result(x, dtype)

    def _prim_dot(self, node, slots):
        a, b = self.infer(slots["A"]), self.infer(slots["B"])
        if not (isinstance(a, TensorType) and isinstance(b, TensorType)):
            return UNKNOWN
        return prim_results.dot_result(a.dims[0], b.dims[1])

    def _prim_reduce(self, node, slots):
        x = self.infer(slots["x"])
        axis = slots["axis"]
        if not isinstance(x, TensorType) or treats_as_unknown(x):
            return UNKNOWN
        if not (isinstance(axis, ast.Constant) and type(axis.value) is int):
            return UNKNOWN  # comptime 名 axis：去维不可判定（原语段同口径）
        return prim_results.reduce_result(x, axis.value)

    def _prim_maximum(self, node, slots):
        # maximum(a, b)：二元逐元素（槽名 a/b，compute_ops 参数集表）；
        # 结果承接可判定侧（近似取左——a 可判定优先）。
        for slot in ("a", "b"):
            t = self.infer(slots[slot])
            if isinstance(t, TensorType) and not treats_as_unknown(t):
                return prim_results.elementwise_result(t)
        return UNKNOWN

    def _prim_exp(self, node, slots):
        return self._unary_math(slots)

    def _prim_log(self, node, slots):
        return self._unary_math(slots)

    def _unary_math(self, slots):
        x = self.infer(slots["x"])
        if not isinstance(x, TensorType) or treats_as_unknown(x):
            return UNKNOWN
        return prim_results.elementwise_result(x)

    def _prim_transpose(self, node, slots):
        x = self.infer(slots["x"])
        if not isinstance(x, TensorType) or treats_as_unknown(x):
            return UNKNOWN
        return prim_results.transpose_result(x)

    def _prim_cdiv(self, node):
        """tis.cdiv（R5）：全 comptime 折叠 ceil(a/b)；否则运行期 int。

        b 值为 0 让渡不报（E0505 已拒，本段不重复报告）；实参形态违规亦
        归 E0505——防御性按非折叠处理。
        """
        if len(node.args) != 2 or node.keywords:
            return ScalarType(INT)
        a = constants.fold_value(node.args[0], self)
        b = constants.fold_value(node.args[1], self)
        if a is not None and b is not None and b != 0:
            value = -(-a // b)  # ceil(a/b)（数学向上取整）
            self.const_values[id(node)] = value
            return ScalarType(COMPTIME_INT)
        return ScalarType(INT)

    def _shape_dims(self, node):
        """shape 实参 → Dim 元组（复用 infer_shape_component；异常形态让渡）。"""
        if not isinstance(node, ast.Tuple):
            return None
        dims = []
        for comp in node.elts:
            dim = prim_results.infer_shape_component(comp, self.env)
            if dim is None:
                return None
            dims.append(dim)
        return tuple(dims)


def _dtype_of(node):
    if isinstance(node, ast.Name) and node.id in DTYPES:
        return node.id
    return None
