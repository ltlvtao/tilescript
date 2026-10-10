"""E0408/E0402/E0403 计算原语检查（primitives/compute-ops 实现面）。

六原语（dot/reduce/maximum/exp/log/transpose）参数集结构（E0408）、
操作数契约、MMA 支持面（E0402）、整除与 pad 策略（E0403）。

段内 order（design D9）：E0408=5/E0402=6/E0403=7。判定序 = 结构
（E0408 param-set，不可解析即短路）→ 操作数/参数值域（E0408）→ E0402
（MMA 支持面）→ E0403（整除与 pad）；实参表达式全量推断在前（嵌套
tis.* 检查不因外层违规丢失）。HAL 依赖面（E0402 列表、E0403 形状选择、
reduce scope 支持面）以 target 能力描述为唯一数据源（design D8）。

dot 的 mma/pad 为关键字专属槽（不占位置槽位）；违规调用结果 UNKNOWN
（拒绝已记录，design D4）。非本域 tis.* 名让渡返回 UNKNOWN、零拒绝。
"""

import ast

from .. import hal
from ..frontend.report import Rejection
from ..typecheck.types import (
    UNKNOWN,
    ConstDim,
    DerivedDim,
    SymbolDim,
    TensorType,
    UnknownDim,
    desc,
    treats_as_unknown,
)
from . import results
from .fold import fold
from .memory_ops import _dim_compatible, _dim_text, _is_name

# 段内 order（design D9）。
_ORDERS = {"E0408": 5, "E0402": 6, "E0403": 7}

# 参数集（R1）：槽名/必选标志；顺序即声明序（位置实参按此绑定）。
_PRIM_SPECS = {
    "dot": (("A", True), ("B", True), ("C", True)),
    "reduce": (("x", True), ("axis", True), ("op", True), ("scope", False)),
    "maximum": (("a", True), ("b", True)),
    "exp": (("x", True),),
    "log": (("x", True),),
    "transpose": (("x", True),),
}

# 关键字专属槽（R1：dot 的 mma/pad 不占位置槽位）。
_KEYWORD_ONLY = {"dot": ("mma", "pad")}

# 缺失必选报文的值域注记（R1 Scenario 4：reduce 缺 op 注明值域）。
_REQUIRED_VALUE_DOMAINS = {"reduce": {"op": "Sum/Max"}}

# pad 封闭四值（R2）。
_PAD_VALUES = ("Error", "PadZero", "Mask", "Split")

# mma 槽选择哨兵：Auto / 数值不可判定（comptime 名构造让渡）。
AUTO = "auto"
OPAQUE = "opaque"

# 浮点 dtype 集（exp/log 契约，R4）。
_FLOAT_DTYPES = ("f16", "bf16", "f32", "f8e4m3")


class ComputeOpsChecker:
    """tis.* 计算原语检查器（traverse 改道回调；rejections 侧收集）。"""

    def __init__(self, target: str):
        self.target = target  # HAL 依赖检查唯一数据源（design D8）
        self.rejections: "list[Rejection]" = []
        self.hit_calls: "set[int]" = set()  # 已产生拒绝的调用节点（语境扫描协调）

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

    # ---- 逐原语判定（4.3–4.5 增量接入） ------------------------------------

    def _check_dot(self, node, slots, inc):
        a_t = inc.infer(slots["A"])
        b_t = inc.infer(slots["B"])
        c_t = inc.infer(slots["C"])
        for slot, t in (("A", a_t), ("B", b_t), ("C", c_t)):
            if not treats_as_unknown(t) and not isinstance(t, TensorType):
                self._reject(node, "arg-kind",
                             f"dot 的 {slot} 必须为 Tensor；实际 {desc(t)}。")
                return UNKNOWN
        if any(treats_as_unknown(t) for t in (a_t, b_t, c_t)):
            return UNKNOWN  # 任一操作数 UNKNOWN：整体让渡
        shapes = {"A": ("M,K", "f16", ("Shared", "Register")),
                  "B": ("K,N", "f16", ("Shared", "Register")),
                  "C": ("M,N", "f32", ("Register",))}
        for slot, t in (("A", a_t), ("B", b_t), ("C", c_t)):
            shape, dtype, scopes = shapes[slot]
            if len(t.dims) != 2:
                self._reject(node, "operand-shape",
                             f"dot 的 {slot} 必须为二维 Tensor（{shape}）；"
                             f"实际 {desc(t)}。")
                return UNKNOWN
            if t.dtype != dtype:
                self._reject(node, "operand-dtype",
                             f"dot 的 {slot} 必须为 {dtype} Tensor；"
                             f"实际 {desc(t)}。")
                return UNKNOWN
            if t.scope not in scopes:
                self._reject(node, "operand-scope",
                             f"dot 的 {slot} 的 scope 必须为 {'/'.join(scopes)}；"
                             f"实际 {t.scope}。")
                return UNKNOWN
        if any(isinstance(d, UnknownDim) for t in (a_t, b_t, c_t)
               for d in t.dims):
            return UNKNOWN  # UNKNOWN 维整体让渡（design D11 residual 1 同构）
        for slot, t in (("A", a_t), ("B", b_t), ("C", c_t)):
            bad = next((d for d in t.dims if not _dim_comptime_known(d)), None)
            if bad is not None:
                self._reject(node, "operand-dim-const",
                             f"dot 的 {slot} 的 shape 各维必须为编译期常量维"
                             f"（MMA 形状整除检查要求编译期可知）；"
                             f"{_dim_text(bad)} 不可判定。")
                return UNKNOWN
        if not _dim_compatible(a_t.dims[1], b_t.dims[0]):
            self._reject(node, "k-incompatible",
                         f"dot 的 K 维不相容：A 的 {_dim_text(a_t.dims[1])} 与 "
                         f"B 的 {_dim_text(b_t.dims[0])} 无相容支。")
            return UNKNOWN
        if not (_dim_compatible(a_t.dims[0], c_t.dims[0])
                and _dim_compatible(b_t.dims[1], c_t.dims[1])):
            self._reject(node, "mn-incompatible",
                         f"dot 的 C 必须为 (M, N) 形状：C {desc(c_t)} 与 "
                         f"A {desc(a_t)}、B {desc(b_t)} 的 M/N 不相容。")
            return UNKNOWN
        mma = (self._mma_form_ok(slots["mma"], node, inc)
               if "mma" in slots else AUTO)
        if mma is None:
            return UNKNOWN
        pad = (self._pad_form_ok(slots["pad"], node)
               if "pad" in slots else "Error")
        if pad is None:
            return UNKNOWN
        # E0402（R2）：显式形状支持面——数值可判定域（comptime 名构造让渡）。
        shape = None if mma is OPAQUE else (
            hal.auto_mma(self.target) if mma is AUTO else mma)
        if shape is not None and shape not in hal.capability(self.target).mma_shapes:
            self._reject(node, "mma-unsupported",
                         f"显式 mma 形状 ({shape[0]}, {shape[1]}, {shape[2]}) 不在"
                         f"目标 {self.target} 的支持列表；支持："
                         f"{hal.mma_shapes_line(self.target)}。", code="E0402")
            return UNKNOWN
        # E0403（R2）：所选形状整除 × pad 策略——D5 窄域（仅 ConstDim 数值
        # 参与；comptime 符号/派生维不触发）。
        if shape is not None and pad == "Error":
            for label, dim, size in (("M", a_t.dims[0], shape[0]),
                                     ("N", b_t.dims[1], shape[1]),
                                     ("K", a_t.dims[1], shape[2])):
                if isinstance(dim, ConstDim) and dim.value % size != 0:
                    aligned = -(-dim.value // size) * size
                    self._reject(node, "mma-unaligned",
                                 f"MMA 形状 ({shape[0]}, {shape[1]}, {shape[2]})"
                                 f"不整除 {label} 维 {dim.value} 且 pad=Error；"
                                 f"最近对齐 {aligned}，或改用 pad=PadZero/Mask/"
                                 "Split 显式处理边界。", code="E0403")
                    return UNKNOWN
        return results.dot_result(a_t.dims[0], b_t.dims[1])

    def _check_reduce(self, node, slots, inc):
        x_t = inc.infer(slots["x"])
        if not treats_as_unknown(x_t) and not isinstance(x_t, TensorType):
            self._reject(node, "arg-kind",
                         f"reduce 的 x 必须为 Tensor；实际 {desc(x_t)}。")
            return UNKNOWN
        if treats_as_unknown(x_t):
            return UNKNOWN
        if x_t.scope != "Register":
            self._reject(node, "x-scope",
                         f"reduce 的 x 必须为 Register Tensor（当前 "
                         f"{x_t.scope}）；先以 tis.load 转移到 Register 再规约。")
            return UNKNOWN
        axis_node = slots["axis"]
        axis_val = None  # int 字面量值；comptime 名 → None（界内让渡）
        if isinstance(axis_node, ast.Constant) and type(axis_node.value) is int:
            axis_val = axis_node.value
        elif not (isinstance(axis_node, ast.Name)
                  and axis_node.id in inc.comptime_syms):
            self._reject(node, "axis-form",
                         "reduce 的 axis 必须为 comptime[int]（int 字面量或 "
                         "comptime 常量名）；不接受运行期 int 或其他类型。")
            return UNKNOWN
        if axis_val is not None and not 0 <= axis_val < len(x_t.dims):
            self._reject(node, "axis-range",
                         f"reduce 的 axis 值域为 [0, {len(x_t.dims)})；"
                         f"实际 {axis_val}。")
            return UNKNOWN
        if not _is_name(slots["op"], ("Sum", "Max")):
            self._reject(node, "op-value",
                         "reduce 的 op 值域为 Sum/Max。")
            return UNKNOWN
        if "scope" in slots:
            scope_node = slots["scope"]
            if not _is_name(scope_node, ("Auto", "Warp", "Block")):
                self._reject(node, "scope-value",
                             "reduce 的 scope 值域为 Auto/Warp/Block"
                             "（默认 Auto）。")
                return UNKNOWN
            if (scope_node.id != "Auto"
                    and not hal.language_scope_supported(self.target,
                                                        scope_node.id)):
                self._reject(node, "scope-unsupported",
                             f"目标 {self.target} 不支持 reduce "
                             f"scope={scope_node.id}；支持："
                             f"{hal.supported_scopes_line(self.target)}。")
                return UNKNOWN
        if axis_val is None:
            return UNKNOWN  # comptime 名 axis：去维不可判定
        return results.reduce_result(x_t, axis_val)

    def _check_maximum(self, node, slots, inc):
        a_t = inc.infer(slots["a"])
        b_t = inc.infer(slots["b"])
        if any(treats_as_unknown(t) for t in (a_t, b_t)):
            return UNKNOWN
        for slot, t in (("a", a_t), ("b", b_t)):
            if not isinstance(t, TensorType):
                self._reject(node, "arg-kind",
                             f"maximum 的 {slot} 必须为 Tensor；实际 {desc(t)}。")
                return UNKNOWN
        if (a_t.dtype != b_t.dtype
                or not all(_dim_compatible(d1, d2)
                           for d1, d2 in zip(a_t.dims, b_t.dims))
                or len(a_t.dims) != len(b_t.dims)):
            self._reject(node, "operand-mismatch",
                         f"maximum 要求 a/b 严格同 dtype 同 shape："
                         f"{desc(a_t)} 与 {desc(b_t)}。")
            return UNKNOWN
        if a_t.scope != "Register" or b_t.scope != "Register":
            self._reject(node, "operand-scope",
                         f"maximum 的 a/b 必须为 Register Tensor；实际 "
                         f"{a_t.scope} 与 {b_t.scope}。")
            return UNKNOWN
        return results.elementwise_result(a_t)

    def _check_exp(self, node, slots, inc):
        return self._check_unary_math(node, slots, inc, "exp")

    def _check_log(self, node, slots, inc):
        return self._check_unary_math(node, slots, inc, "log")

    def _check_unary_math(self, node, slots, inc, name):
        x_t = inc.infer(slots["x"])
        if not treats_as_unknown(x_t) and not isinstance(x_t, TensorType):
            self._reject(node, "arg-kind",
                         f"{name} 的 x 必须为 Tensor；实际 {desc(x_t)}。")
            return UNKNOWN
        if treats_as_unknown(x_t):
            return UNKNOWN
        if x_t.scope != "Register":
            self._reject(node, "x-scope",
                         f"{name} 的 x 必须为 Register Tensor（当前 "
                         f"{x_t.scope}）；先以 tis.load 转移到 Register。")
            return UNKNOWN
        if x_t.dtype not in _FLOAT_DTYPES:
            self._reject(node, "dtype-float",
                         f"{name} 的 x 必须为浮点 dtype Tensor"
                         f"（{'/'.join(_FLOAT_DTYPES)}）；实际 {x_t.dtype}。")
            return UNKNOWN
        return results.elementwise_result(x_t)

    def _check_transpose(self, node, slots, inc):
        x_t = inc.infer(slots["x"])
        if not treats_as_unknown(x_t) and not isinstance(x_t, TensorType):
            self._reject(node, "arg-kind",
                         f"transpose 的 x 必须为 Tensor；实际 {desc(x_t)}。")
            return UNKNOWN
        if treats_as_unknown(x_t):
            return UNKNOWN
        if len(x_t.dims) != 2:
            self._reject(node, "x-shape",
                         f"transpose 只接受二维 Tensor；实际 {desc(x_t)}。")
            return UNKNOWN
        if x_t.scope not in ("Shared", "Register"):
            self._reject(node, "x-scope",
                         f"transpose 的 x 的 scope 必须为 Shared/Register；"
                         f"实际 {x_t.scope}。")
            return UNKNOWN
        return results.transpose_result(x_t)

    # ---- 辅助 -------------------------------------------------------------

    def _mma_form_ok(self, node, call_node, inc):
        """mma 槽形态 → AUTO | (m, n, k) 数值元组 | OPAQUE（无数值让渡）| None（拒）。"""
        if _is_name(node, ("Auto",)):
            return AUTO
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "MMA"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tis"):
            self._reject(call_node, "mma-form",
                         "mma 值为 tis.MMA(m, n, k) 构造或 Auto"
                         "（默认取目标支持列表第一项）。")
            return None
        if len(node.args) != 3 or node.keywords:
            self._reject(call_node, "mma-form",
                         "MMA 构造必须为三个 comptime[int] 实参（m, n, k）。")
            return None
        if not all(_comptime_int_node(a, inc) for a in node.args):
            self._reject(call_node, "mma-form",
                         "MMA 的 m/n/k 必须为 comptime[int]（int 字面量或 "
                         "comptime 常量名）；不接受运行期 int 或其他类型。")
            return None
        if all(isinstance(a, ast.Constant) for a in node.args):
            return tuple(a.value for a in node.args)
        return OPAQUE  # comptime 名构造：数值不可判定（E0402/E0403 让渡）

    def _pad_form_ok(self, node, call_node):
        """pad 槽形态 → 四值名 | None（拒）。"""
        if (isinstance(node, ast.Attribute) and node.attr in _PAD_VALUES
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "PadPolicy"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "tis"):
            return node.attr
        self._reject(call_node, "pad-value",
                     "pad 值域为封闭四值 tis.PadPolicy.Error/PadZero/Mask/"
                     "Split（默认 Error）。")
        return None

    def _reject(self, node, category, suggestion, code="E0408"):
        self.hit_calls.add(id(node))  # R8：同调用只报段内首条
        self.rejections.append(Rejection(
            code=code, line=node.lineno, col=node.col_offset + 1,
            category=category, suggestion=suggestion, order=_ORDERS[code],
            stage="primitive-contract",
        ))


# ---- 实参槽绑定 -----------------------------------------------------------

def _bind_args(node: ast.Call, name: str):
    """位置+关键字实参 → {槽名: AST 节点}；结构违规返回 (None, 错误说明)。

    R1：未知关键字 / 位置实参数量超出可按位置传递参数总长度（dot 为
    A/B/C 三个，mma/pad 关键字专属）/ 缺失必选 → E0408。重复绑定与
    **kwargs 展开同为参数集封闭性违规。
    """
    spec = _PRIM_SPECS[name]
    names = [n for n, _ in spec]
    kwonly = _KEYWORD_ONLY.get(name, ())
    if len(node.args) > len(names):
        if kwonly:
            return None, (
                f"{name} 可按位置传递的实参只有 {'/'.join(names)} 共 "
                f"{len(names)} 个；{'/'.join(kwonly)} 必须以关键字传递；"
                f"实际传入 {len(node.args)} 个位置实参。")
        return None, (f"{name} 只接受 {len(names)} 个位置实参"
                      f"（{_param_set_line(name)}）；实际传入 "
                      f"{len(node.args)} 个。")
    slots = {}
    for i, arg in enumerate(node.args):
        slots[names[i]] = arg
    for kw in node.keywords:
        if kw.arg is None:
            return None, f"{name} 不接受 **kwargs 展开；{_param_set_line(name)}。"
        if kw.arg not in names and kw.arg not in kwonly:
            return None, (f"未知关键字实参 {kw.arg}；{name} 的合法参数集为 "
                          f"{_param_set_line(name)}。")
        if kw.arg in slots:
            return None, (f"实参 {kw.arg} 同时按位置与关键字传递；"
                          f"{name} 的合法参数集为 {_param_set_line(name)}。")
        slots[kw.arg] = kw.value
    missing = [n for n, required in spec if required and n not in slots]
    if missing:
        domains = _REQUIRED_VALUE_DOMAINS.get(name, {})
        parts = "/".join(
            f"{n}（值域 {domains[n]}）" if n in domains else n for n in missing)
        return None, (f"缺少必选实参 {parts}；{name} 的合法参数集为 "
                      f"{_param_set_line(name)}。")
    return slots, None


def _param_set_line(name: str) -> str:
    return "/".join([n for n, _ in _PRIM_SPECS[name]]
                    + list(_KEYWORD_ONLY.get(name, ())))


def _dim_comptime_known(dim) -> bool:
    """dot shape 的编译期可知域（R2 常量维契约）。

    ConstDim / comptime 符号与派生维（types 层 comptime 标志）均编译期
    可知；运行期维仅当 fold 折叠出纯常数（如 (j+1)*16-j*16 → 16）可知。
    E0403 整除判定另按 D5 窄域（仅 ConstDim 数值）——两层域不相交使用。
    """
    if isinstance(dim, ConstDim):
        return True
    if isinstance(dim, (SymbolDim, DerivedDim)):
        if dim.comptime:
            return True
        folded = fold(dim)
        return folded is not None and set(folded) == {()}
    return False


def _comptime_int_node(node, inc) -> bool:
    """MMA 构造实参：int 字面量或 comptime 常量名（运行期 int 名拒）。"""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return True
    return isinstance(node, ast.Name) and node.id in inc.comptime_syms


# ---- MMA/PadPolicy 实参语境专用扫描（design D7：合法位 = dot 的 mma=/pad= 值）--

def scan_special_value_contexts(tree: ast.Module, checker: "ComputeOpsChecker"):
    """tis.MMA(...) 与 tis.PadPolicy.<值> 的语境扫描（R2 末 Scenario）。

    合法位置唯一：tis.dot 调用的 mma=/pad= 关键字实参 value。其余任何
    出现（赋值绑定/其他调用实参/算术比较操作数/return 值）→ E0408。
    沿父链最近 Call 已产生拒绝（如位置传 MMA 的 param-set）时跳过
    （R8 同调用只报首个）。
    """
    parents: "dict[ast.AST, ast.AST]" = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not _is_mma_node(node) and not _is_padpolicy_node(node):
            continue
        if _legal_dot_keyword_value(node, parents):
            continue
        if _enclosing_call_hit(node, parents, checker):
            continue
        if _is_mma_node(node):
            checker._reject(node, "mma-context",
                            "tis.MMA(...) 只能作为 tis.dot 的 mma= 关键字实参"
                            "使用，不能绑定、传参或参与运算。")
        else:
            checker._reject(node, "pad-context",
                            "tis.PadPolicy.<值> 只能作为 tis.dot 的 pad= 关键字"
                            "实参使用，不能绑定、传参或参与运算。")


def _is_mma_node(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "MMA"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tis")


def _is_padpolicy_node(node) -> bool:
    return (isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "PadPolicy"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "tis")


def _legal_dot_keyword_value(node, parents) -> bool:
    """父链两层内：keyword(mma=/pad=) 包装且其载体为 tis.dot 调用。"""
    parent = parents.get(node)
    if not (isinstance(parent, ast.keyword) and parent.arg in ("mma", "pad")
            and parent.value is node):
        return False
    grand = parents.get(parent)
    return (isinstance(grand, ast.Call) and isinstance(grand.func, ast.Attribute)
            and grand.func.attr == "dot"
            and isinstance(grand.func.value, ast.Name)
            and grand.func.value.id == "tis")


def _enclosing_call_hit(node, parents, checker) -> bool:
    """沿父链最近的 tis.* 调用已产生段内更早拒绝（R8 只报首个）。"""
    cur = node
    while cur is not None:
        parent = parents.get(cur)
        if isinstance(parent, ast.Call):
            return id(parent) in checker.hit_calls
        cur = parent
    return False
