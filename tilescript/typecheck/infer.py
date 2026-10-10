"""表达式类型推断（type-system spec R6「表达式结果类型规则」）。

让渡面（design D4）：结果类型未由本段定义的表达式一律 UNKNOWN，UNKNOWN 参与
任何绑定不产生 E0303；切片/下标边界的 UNKNOWN 操作数传播为未知维，含未知维
的 Tensor 整体按 UNKNOWN 参与绑定（equivalent 恒 False，绑定层跳过）。

本模块产生的 E0303：下标基对象已知非 Tensor（R6 下标规则）、状态类构造
关键字实参×字段类型、带注解设备函数调用实参×形参注解（D7 第 3/4 类）。
赋值/return 绑定（D7 第 1/2 类）由 bindings 遍历器基于本模块推断结果裁决。
"""

import ast

from tilescript.frontend.report import Rejection

from .transfer import check_call, is_data_move_call
from .types import (
    COMPTIME_INT,
    INT,
    UNKNOWN,
    ConstDim,
    DerivedDim,
    ScalarType,
    StateType,
    SymbolDim,
    TensorType,
    UnknownDim,
    desc,
    equivalent,
    treats_as_unknown,
)

_ORDER = 4  # design D9：E0302=1/E0304=2/E0301=3/E0303=4


class Inferencer:
    """表达式类型推断器。

    env：变量名 → 类型（外部遍历器流式维护，本对象持同一引用）。
    registry：状态类注册表（构造调用识别与实参检查）。
    functions：设备代码函数名 → {"params": [(形参名, 注解类型 | None)],
    "returns": 返回注解类型 | None}（D7 第 4 类实参绑定）。
    """

    def __init__(self, env, registry=None, functions=None, comptime_syms=frozenset()):
        self.env = env
        self.registry = registry if registry is not None else {}
        self.functions = functions if functions is not None else {}
        self.comptime_syms = comptime_syms
        self.rejections: "list[Rejection]" = []

    # ---- 表达式入口 -------------------------------------------------------

    def infer(self, node):
        if isinstance(node, ast.Constant):
            if type(node.value) is int:  # 四进制 int 字面量（bool 排除）
                return ScalarType(COMPTIME_INT)
            return UNKNOWN  # float/bool/None/str：让渡（R6 常量段）
        if isinstance(node, ast.Name):
            return self.env.get(node.id, UNKNOWN)  # 未绑定名让渡
        if isinstance(node, ast.Attribute):
            return self._infer_attribute(node)
        if isinstance(node, ast.Subscript):
            return self._infer_subscript(node)
        if isinstance(node, ast.Call):
            return self._infer_call(node)
        return UNKNOWN  # 算术/比较/逻辑/一元/IfExp/组合字面量：让渡（D4）

    # ---- 属性访问 ---------------------------------------------------------

    def _infer_attribute(self, node):
        base = self.infer(node.value)
        if isinstance(base, StateType):
            for name, ftype in base.fields:
                if name == node.attr:
                    return ftype  # 字段访问 → 声明类型（R6）
        return UNKNOWN  # pipe.*/buf.* 属性、未知基、非状态类基：让渡

    # ---- 下标与切片 -------------------------------------------------------

    def _infer_subscript(self, node):
        base = self.infer(node.value)
        if treats_as_unknown(base):
            return UNKNOWN
        if not isinstance(base, TensorType):
            self._reject(node, "subscript-base",
                         "下标基对象必须为 Tensor；标量、状态类或 Pointer "
                         "不能使用下标，请核对基对象类型。")
            return UNKNOWN
        items = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        dims: "list" = []
        consumed = 0
        for item in items:
            if isinstance(item, ast.Slice):
                dims.append(self._slice_dim(base.dims[consumed] if consumed < len(base.dims)
                                            else None, item))
                consumed += 1
            elif isinstance(item, ast.Constant) and item.value is None:
                dims.append(ConstDim(1))  # None 广播：新增常量 1 维（R6）
            else:
                consumed += 1  # 整数下标消去该维（值已知性不影响消维）
        dims.extend(base.dims[consumed:])
        return TensorType(base.dtype, tuple(dims), base.scope)

    def _slice_dim(self, original, item):
        if item.step is not None:
            return UnknownDim()  # step 切片：M1 让渡
        if item.lower is None and item.upper is None:
            return original  # `:` 全取保留原维（R6）
        lower = self._infer_dim(item.lower)
        upper = self._infer_dim(item.upper)
        if lower is None or upper is None:
            return UnknownDim()  # 缺省边界不可静态派生
        if isinstance(lower, UnknownDim) or isinstance(upper, UnknownDim):
            return UnknownDim()  # UNKNOWN 组件传播（D4）
        if isinstance(lower, ConstDim) and isinstance(upper, ConstDim):
            return ConstDim(upper.value - lower.value)
        return DerivedDim("slice", (lower, upper),
                          _dim_comptime(lower) and _dim_comptime(upper))

    def _infer_dim(self, node):
        """切片/下标边界表达式 → Dim；None 表示彻底不可判定（调用方产未知维）。"""
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return ConstDim(node.value)
            return UnknownDim()  # float 边界等：传播
        if isinstance(node, ast.Name):
            bound = self.env.get(node.id, UNKNOWN)
            if bound == ScalarType(COMPTIME_INT):
                return SymbolDim(node.id, True)
            if bound == ScalarType(INT):
                return SymbolDim(node.id, False)
            return UnknownDim()  # UNKNOWN/非标量/未绑定：传播
        if isinstance(node, ast.BinOp) and type(node.op).__name__ in _BINOPS:
            left = self._infer_dim(node.left)
            right = self._infer_dim(node.right)
            if left is None or right is None:
                return UnknownDim()
            if isinstance(left, UnknownDim) or isinstance(right, UnknownDim):
                return UnknownDim()
            return DerivedDim(_BINOPS[type(node.op).__name__], (left, right),
                              _dim_comptime(left) and _dim_comptime(right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._infer_dim(node.operand)
            if operand is None or isinstance(operand, UnknownDim):
                return UnknownDim()
            return DerivedDim("-", (operand,), _dim_comptime(operand))
        return UnknownDim()

    # ---- 调用 -------------------------------------------------------------

    def _infer_call(self, node):
        if not isinstance(node.func, ast.Name):
            # tis.load/tis.store：E0301 转移格裁决（结果类型仍让渡）。
            if is_data_move_call(node.func):
                check_call(node, self)
            return UNKNOWN  # tis.*/pipe.* 等属性调用：让渡（R7 豁免）
        name = node.func.id
        if name in self.registry:
            # 状态类构造调用：结果类型为该状态类（R5）；关键字实参×字段类型。
            self._check_ctor_args(name, node)
            return self.registry[name]
        if name in self.functions:
            # 带源码层形参注解的设备函数调用：实参×形参注解（D7 第 4 类）。
            self._check_call_args(name, node)
            returns = self.functions[name].get("returns")
            return returns if returns is not None else UNKNOWN
        return UNKNOWN  # 未知名/内建：让渡

    def _check_ctor_args(self, class_name, node):
        fields = dict(self.registry[class_name].fields)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in fields:
                continue  # **kwargs/未知关键字：M1 让渡
            target = fields[keyword.arg]
            source = self.infer(keyword.value)
            self.check_binding(keyword.value, source, target, "state-ctor-arg")

    def _check_call_args(self, name, node):
        params = dict(self.functions[name]["params"])
        # 位置实参按序对应形参；无注解形参位不查（D7）。
        for i, arg in enumerate(node.args):
            if i < len(self.functions[name]["params"]):
                target = self.functions[name]["params"][i][1]
                if target is not None:
                    self.check_binding(arg, self.infer(arg), target, "call-arg")
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg in params:
                target = params[keyword.arg]
                if target is not None:
                    self.check_binding(keyword.value, self.infer(keyword.value),
                                        target, "call-arg")

    # ---- 绑定不匹配报告（bindings 遍历器复用） -----------------------------

    def check_binding(self, node, source, target, category):
        if treats_as_unknown(source) or treats_as_unknown(target):
            return  # UNKNOWN 参与任何绑定不报（D4）
        if _compatible(source, target):
            return
        self._reject(node, category, _mismatch_suggestion(source, target))

    def _reject(self, node, category, suggestion):
        self.rejections.append(Rejection(
            code="E0303", line=node.lineno, col=node.col_offset + 1,
            category=category, suggestion=suggestion, order=_ORDER,
            stage="type-system",
        ))


_BINOPS = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "FloorDiv": "//", "Mod": "%"}


def _dim_comptime(dim) -> bool:
    if isinstance(dim, ConstDim):
        return True
    return dim.comptime


def _compatible(source, target) -> bool:
    """comptime[int] 值单向兼容 int 位置；其余须 equivalent。"""
    if (isinstance(source, ScalarType) and isinstance(target, ScalarType)
            and source.kind == COMPTIME_INT and target.kind == INT):
        return True
    return equivalent(source, target)


def _mismatch_suggestion(source, target) -> str:
    """两侧类型 + 差异组件建议映射（R7 恢复建议规则）。"""
    if isinstance(source, TensorType) and isinstance(target, TensorType):
        if source.dtype != target.dtype:
            advice = "dtype 不同，使用显式转换原语 tis.cast"
        elif source.scope != target.scope:
            advice = "scope 不同，使用显式移动原语承载跨 scope 移动"
        else:
            advice = "shape 不同，核对两侧维度"
    elif isinstance(source, StateType) or isinstance(target, StateType):
        advice = "状态类为名义类型，核对两侧状态类名"
    else:
        advice = "标量种类不同，核对标量种类（comptime[int] 值可绑定 int 位置，反向不可）"
    return (f"类型不匹配：实际 {desc(source)}，目标 {desc(target)}；{advice}。")
