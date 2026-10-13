"""算术与比较/逻辑裁决（R1 算术操作数与结果 / R3 比较逻辑与布尔封闭）。

判定序（design D4 + stable spec R2/R4 待定型语境条款；同节点多码只报段内
序首个——首个命中即止）：
1. 布尔操作数 → E0605（算术操作数位不在布尔封闭集）；
2. 一侧 int 家族、一侧 dtype 家族 → E0601（int 字面量/comptime 常量不
   豁免）；
3. 待定型常量 × int 家族或整型 dtype（i8/i32）→ E0606（R2/R4——待定型
   常量只在浮点 dtype 语境合法；inf 在 int 家族/整型位置无穷无整数
   表示）；待定型 inf × f8e4m3 → E0606（E4M3 无无穷表示）；
4. 两侧 dtype 家族 dtype 不同 → E0602；
5. `/`：int 家族或整型 dtype → E0604；
6. Tensor scope 不同 → E0603（报两侧 scope）；
7. Tensor shape 逐维广播不兼容 → E0603（报两侧 shape）；
8. 一侧 UNKNOWN → 全部让渡（结果 UNKNOWN）。

结果派生：int 家族 `+ - *` 全 comptime → COMPTIME_INT（值按 R5 折叠记入
侧表），否则 int；dtype 不变；标量⊗Tensor → 该 Tensor shape/scope；
Tensor⊗Tensor → 广播 shape 同 scope；待定型⊗浮点 dtype 家族 → 按该侧
定型（R4 位置②）。违规时结果类型按类别近似（混家族取 dtype 家族侧——
外层组合继续判定，同级其余 int 操作数可继续列出，R7 同码合并；跨
dtype/scope/shape 取左操作数），供再绑定等价与后续组合推断消费。

广播维相等判定 = `primitives.fold` 多项式等价（两维 fold 相等 ⇔ 维长
等价——ConstDim(64) 与 32*2 不误报）；相等保留取左维、一侧 1 扩展取
另一侧；UnknownDim/fold None 让渡该维（UnknownDim 传播）。
"""

import ast

from ..primitives.fold import constant as dim_constant
from ..primitives.fold import fold
from ..typecheck.types import (
    COMPTIME_INT,
    DTYPES,
    INT,
    UNKNOWN,
    ConstDim,
    ScalarType,
    TensorType,
    UnknownDim,
    desc,
    treats_as_unknown,
)
from .constants import PendingConst, fold_value
from .value_env import BOOL

_INT_DTYPES = ("i8", "i32")
_ARITH_OPS = ("Add", "Sub", "Mult", "Div")


def arith(node, inc):
    """BinOp 算术裁决（+ - * /；语法段已封闭其余运算符）。"""
    if type(node.op).__name__ not in _ARITH_OPS:
        return UNKNOWN  # 防御（语法段 E0106 已拒 // % 等）
    return arith_operands(node.left, node.right, node.op, node, inc)


def arith_operands(left_node, right_node, op, report_node, inc):
    """算术裁决核心：操作数为 AST 节点，或增强赋值展开的已绑定类型对象。

    report_node 为违规报告锚点（BinOp 节点或 AugAssign 语句）；操作数级
    违规经 inc.report（R7 帧合并/单条）。
    """
    left = left_node if _is_type(left_node) else inc.infer(left_node)
    right = right_node if _is_type(right_node) else inc.infer(right_node)
    op_name = type(op).__name__
    is_div = op_name == "Div"

    lf, rf = family(left), family(right)

    # 1. 布尔操作数（判定序首——命中即止本节点）。
    if lf == "bool" or rf == "bool":
        bad = left_node if lf == "bool" else right_node
        inc.report("E0605", _node_of(bad, report_node),
                   "布尔值仅限条件语境封闭集；不能作为算术操作数")
        return UNKNOWN

    # 7 前置：一侧 UNKNOWN（含未知维 Tensor）不参与成对判定。
    if lf == "unknown" or rf == "unknown":
        return UNKNOWN

    # 2. 混家族（int × dtype 家族）。
    if {lf, rf} == {"int", "dtype"}:
        int_node = left_node if lf == "int" else right_node
        dtype_t = right if rf == "dtype" else left
        note = ("混家族算术：int 家族值不能作 dtype 家族算术的操作数（另一侧"
                f"{desc(dtype_t)}）；当前版本不支持混家族算术，数值转换须由"
                "宿主侧完成或经 Tensor 显式 tis.cast，float 字面量（如 2.0）"
                "按另一侧 dtype 定型")
        inc.report("E0601", _node_of(int_node, report_node), note)
        return dtype_t  # 近似取 dtype 家族侧（R7 同码合并承载）

    # 3. 待定型 × int 家族（R2/R4：待定型常量只在浮点 dtype 语境合法）。
    if {lf, rf} == {"int", "pending"}:
        pend_node = left_node if lf == "pending" else right_node
        pend = left if lf == "pending" else right
        inc.report("E0606", _node_of(pend_node, report_node),
                   _pending_context_note(pend, "int 家族位置"))
        return UNKNOWN

    # int 家族 ⊗ int 家族：全 comptime 折叠（R5），`/` 拒（E0604）。
    if lf == "int" and rf == "int":
        if is_div:
            inc.report("E0604", report_node,
                       "除法 / 不接受 int 家族操作数；整除使用 tis.cdiv")
            return left
        if left.kind == COMPTIME_INT and right.kind == COMPTIME_INT:
            _record_fold(left_node, right_node, op_name, report_node, inc)
            return ScalarType(COMPTIME_INT)
        return ScalarType(INT)

    # 待定型 ⊗ 待定型：spec 未定义（两侧均无 dtype 语境）——让渡。
    if lf == "pending" and rf == "pending":
        return UNKNOWN

    # 待定型 ⊗ dtype 家族（R4 位置②）：语境裁决——整型 dtype / inf×f8e4m3
    # 拒（E0606），浮点 dtype 按该侧定型（合法）。
    if lf == "pending" or rf == "pending":
        pend = left if lf == "pending" else right
        pend_node = left_node if lf == "pending" else right_node
        dtype_t = right if rf == "dtype" else left
        d = dtype_of(dtype_t)
        if d in _INT_DTYPES:
            inc.report("E0606", _node_of(pend_node, report_node),
                       _pending_context_note(pend, f"整型 dtype（{d}）位置"))
            return dtype_t  # 近似取 dtype 侧（外层组合继续判定）
        if pend.tag == "inf" and d == "f8e4m3":
            inc.report("E0606", _node_of(pend_node, report_node),
                       "无穷常量不能出现在 f8e4m3 位置（OCP FP8 E4M3 无"
                       "无穷表示，max finite ±448）；使用 f16/bf16/f32 或"
                       "有限常量")
            return dtype_t
        return _combine(dtype_t, dtype_t, op, report_node, inc, is_div)

    # dtype 家族 ⊗ dtype 家族。
    if lf == "dtype" and rf == "dtype":
        return _combine(left, right, op, report_node, inc, is_div)

    return UNKNOWN  # StateType/Pointer 参与算术：M1 让渡（residual 2）


def _combine(left, right, op, report_node, inc, is_div):
    """dtype 家族两侧组合：dtype → `/` 整型 → scope → shape（判定序）。"""
    ld, rd = dtype_of(left), dtype_of(right)
    if ld != rd:
        inc.report("E0602", report_node,
                   f"家族内跨 dtype 算术：{ld} 与 {rd}；使用显式转换原语 "
                   "tis.cast 对齐 dtype（本语言不做隐式 dtype 提升）")
        return left  # 近似取左操作数（再绑定等价按左类别判定）
    if is_div and ld in _INT_DTYPES:
        inc.report("E0604", report_node,
                   f"除法 / 不接受整型 dtype（{ld}）操作数；标量整除使用 "
                   "tis.cdiv，Tensor 整除零承载暂无原语路径")
        return left
    # 标量 ⊗ 标量 → 标量。
    if not isinstance(left, TensorType) and not isinstance(right, TensorType):
        return ScalarType(ld)
    # 标量 ⊗ Tensor → 该 Tensor（shape/scope 承接）。
    if not isinstance(left, TensorType):
        return right
    if not isinstance(right, TensorType):
        return left
    # Tensor ⊗ Tensor：scope → shape。
    if left.scope != right.scope:
        inc.report("E0603", report_node,
                   f"Tensor scope 不一致：{left.scope} 与 {right.scope}"
                   f"（{desc(left)} 与 {desc(right)}）；核对 scope 或经显式"
                   "移动原语（tis.load/tis.store）对齐")
        return left
    dims = _broadcast_dims(left, right, report_node, inc)
    return TensorType(ld, dims, left.scope)


def _broadcast_dims(left, right, report_node, inc):
    """右对齐逐维广播：相等保留取左 / 一侧 1 扩展 / 否则 E0603。"""
    dims_l, dims_r = left.dims, right.dims
    n = max(len(dims_l), len(dims_r))
    out = []
    for i in range(1, n + 1):
        dl = dims_l[-i] if i <= len(dims_l) else None
        dr = dims_r[-i] if i <= len(dims_r) else None
        if dl is None:
            out.append(dr)  # 长侧前缀维直接保留
            continue
        if dr is None:
            out.append(dl)
            continue
        if isinstance(dl, UnknownDim) or isinstance(dr, UnknownDim):
            out.append(UnknownDim())  # 该维让渡（residual 5）
            continue
        if _dim_equal(dl, dr):
            out.append(dl)  # 相等保留取左（comptime 性由该维承载）
            continue
        if _dim_is_one(dr):
            out.append(dl)
            continue
        if _dim_is_one(dl):
            out.append(dr)
            continue
        if fold(dl) is None or fold(dr) is None:
            out.append(UnknownDim())  # 不可折叠派生维（如 / 派生）：该维让渡
            continue
        inc.report("E0603", report_node,
                   f"shape 广播不兼容：{desc(left)} 与 {desc(right)}；"
                   "核对操作数维度，或经切片与 None 广播调整形状至兼容")
        out.append(dl)
    return tuple(reversed(out))  # 循环自尾维起（右对齐），结果反转回首维序


def _dim_equal(a, b) -> bool:
    if a == b:
        return True
    fa, fb = fold(a), fold(b)
    return fa is not None and fb is not None and fa == fb


def _dim_is_one(dim) -> bool:
    if isinstance(dim, ConstDim) and dim.value == 1:
        return True
    return dim_constant(dim) == 1


def _pending_context_note(pend, context: str) -> str:
    """待定型常量出现在非浮点 dtype 语境的 E0606 建议文案（R2/R4）。"""
    if pend.tag == "inf":
        return (f"无穷常量不能出现在{context}（无穷无整数表示）；与浮点 "
                "dtype 家族值（f16/bf16/f32）运算按该 dtype 定型，或经 "
                "tis.full 构造")
    return (f"float 字面量只在浮点 dtype 语境合法，不能出现在{context}；"
            "与浮点 dtype 家族值运算后按该 dtype 定型，或经 tis.full 构造")


def _record_fold(left_node, right_node, op_name, report_node, inc):
    """全 comptime 运算折叠值记入侧表（R5 整数精确算术；AugAssign 展开
    的类型操作数无节点——跳过记录，种类推导不受影响）。"""
    if not (isinstance(left_node, ast.AST) and isinstance(right_node, ast.AST)):
        return
    if not isinstance(report_node, ast.BinOp):
        return
    a = fold_value(left_node, inc)
    b = fold_value(right_node, inc)
    if a is None or b is None:
        return
    value = {"Add": a + b, "Sub": a - b, "Mult": a * b}.get(op_name)
    if value is not None:
        inc.const_values[id(report_node)] = value


def unary(node, inc):
    """一元运算：USub（一元负）/ Not（逻辑非）。"""
    operand = inc.infer(node.operand)
    if isinstance(node.op, ast.Not):
        if operand is BOOL:
            return BOOL
        if family(operand) == "unknown":
            return UNKNOWN
        inc.report("E0605", node,
                   "not 的操作数必须为布尔值（比较/逻辑运算结果或 "
                   "True/False 字面量）")
        return UNKNOWN
    if isinstance(node.op, ast.USub):
        return unary_neg(node, node.operand, operand, inc)
    return UNKNOWN  # UAdd/Invert：语法段封闭外（防御）


def unary_neg(node, operand_node, operand, inc):
    """一元负：int 家族折叠取负 / dtype 家族同 dtype shape / 待定型符号承载。"""
    f = family(operand)
    if f == "bool":
        inc.report("E0605", node, "布尔值不能作一元负操作数（条件语境专用）")
        return UNKNOWN
    if f == "unknown":
        return UNKNOWN
    if f == "pending":
        return PendingConst(operand.tag, -operand.value)  # -inf 符号承载
    if f == "int":
        if operand.kind == COMPTIME_INT:
            value = fold_value(operand_node, inc)
            if value is not None:
                inc.const_values[id(node)] = -value
            return ScalarType(COMPTIME_INT)
        return ScalarType(INT)
    return operand  # dtype 家族（标量/Tensor）：同 dtype 同 shape（R2）


def compare(node, inc):
    """比较裁决（仅两操作数单比较——语法段已拒链式）：同家族标量 → 布尔。"""
    left = inc.infer(node.left)
    right = inc.infer(node.comparators[0]) if node.comparators else UNKNOWN
    lf, rf = family(left), family(right)
    if lf == "unknown" or rf == "unknown":
        return UNKNOWN  # 操作数让渡
    for side, t in (("左", left), ("右", right)):
        if t is BOOL:
            inc.report("E0605", node,
                       f"比较操作的{side}操作数为布尔值；布尔值仅限条件语境"
                       "封闭集，不能作比较操作数")
            return UNKNOWN
    if isinstance(left, TensorType) or isinstance(right, TensorType):
        inc.report("E0605", node,
                   f"Tensor 不能参与比较（{desc(left)} 与 {desc(right)}）；"
                   "逐元素比较归计算原语域后续 change")
        return UNKNOWN
    if lf == "pending" or rf == "pending":
        inc.report("E0605", node,
                   "待定型常量（float 字面量/inf）不能作比较操作数；"
                   "先经 tis.full 或与已定型值运算确定 dtype")
        return UNKNOWN
    if lf == "int" and rf == "int":
        return BOOL  # int 家族两标量（comptime/int 单向兼容混用）
    if lf == "dtype" and rf == "dtype":
        if dtype_of(left) == dtype_of(right):
            return BOOL  # dtype 家族同 dtype 两标量
        inc.report("E0605", node,
                   f"比较操作数必须为同家族同 dtype 标量：{desc(left)} 与 "
                   f"{desc(right)}；核对两侧标量 dtype")
        return UNKNOWN
    inc.report("E0605", node,
               f"比较操作数必须为同家族标量：{desc(left)} 与 {desc(right)}；"
               "混家族比较须先显式转换")
    return UNKNOWN


def boolop(node, inc):
    """and/or：操作数须布尔；结果布尔（任一 UNKNOWN 让渡）。"""
    unknown = False
    for value_node in node.values:
        t = inc.infer(value_node)
        if t is BOOL:
            continue
        if family(t) == "unknown":
            unknown = True
            continue
        inc.report("E0605", value_node,
                   f"逻辑运算（{type(node.op).__name__}）的操作数必须为"
                   "布尔值；显式比较（如 x > 0）产生布尔值")
    if unknown:
        return UNKNOWN
    return BOOL


# ---- 家族辅助（遍历器共用） -------------------------------------------------


def family(t) -> str:
    """操作数家族：int / dtype / pending / bool / unknown / other。"""
    if t is BOOL:
        return "bool"
    if isinstance(t, PendingConst):
        return "pending"
    if treats_as_unknown(t):
        return "unknown"
    if isinstance(t, ScalarType):
        if t.kind in (COMPTIME_INT, INT):
            return "int"
        if t.kind in DTYPES:
            return "dtype"
        return "other"
    if isinstance(t, TensorType):
        return "dtype"
    return "other"  # StateType/PointerType


def dtype_of(t):
    if isinstance(t, TensorType):
        return t.dtype
    if isinstance(t, ScalarType) and t.kind in DTYPES:
        return t.kind
    return None


def _is_type(x):
    return not isinstance(x, ast.AST)


def _node_of(x, fallback):
    return x if isinstance(x, ast.AST) else fallback
