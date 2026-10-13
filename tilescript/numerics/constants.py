"""常量绑定裁决与折叠值（R2 待定型 / R4 dtype 位置绑定 / R5 comptime 折叠）。

数值域内部形态（不进类型段类型世界，R3「条件语境专用值」条款）：
- `BOOL`：布尔哨兵单例（合法 Compare/BoolOp/not 结果、True/False 字面量）；
- `PendingConst`：待定型常量（float 字面量 / inf；tag 区分——f8e4m3 对
  float 字面量合法、对 inf 拒绝，R2/R4）。

折叠值侧表（R5，design D7）：类型对象无值槽（types.py 冻结表示），值经
NumericsInferencer 侧表承载——`const_values`（AST 节点 → 值）与 `values`
（名 → 值，起点 kernel comptime 形参默认值）。`fold_value` 是唯一求值
入口（整数精确算术；`/` 永不折叠）。
"""

import ast
from dataclasses import dataclass

from ..typecheck.symbols import (
    _is_comptime_int_annotation,
    find_kernel,
)
from ..typecheck.types import DTYPES, INT, ScalarType

# 整型 dtype 补码值域（R4 值域裁决）。
INT_RANGES = {"i8": (-128, 127), "i32": (-(2 ** 31), 2 ** 31 - 1)}
# IEEE 754 标准精度浮点 dtype：inf 可绑定（f8e4m3 无穷表示缺失，R2）。
IEEE_FLOAT_DTYPES = ("f16", "bf16", "f32")
FLOAT_DTYPES = ("f16", "bf16", "f32", "f8e4m3")


class Bool:
    """布尔值（数值段内部形态；R3——不进类型段封闭类型世界）。"""

    def __repr__(self):  # pragma: no cover - 调试可读性
        return "BOOL"


BOOL = Bool()


@dataclass(frozen=True)
class PendingConst:
    """待定型常量：tag='float'（字面量，带十进制值）/ 'inf'（±inf）。"""

    tag: str
    value: float


def kernel_comptime_values(tree: ast.Module) -> dict:
    """kernel comptime[int] 形参默认值 → {名: int}（值表起点）。

    E0104 只接受单个 int 字面量与 `True`/`False` 默认值（常量表达式形式
    被语法段拒）；布尔默认值按 R3 取整数值（True→1/False→0）。无默认值
    或非 comptime[int] 注解不记。
    """
    kernel = find_kernel(tree)
    if kernel is None:
        return {}
    values = {}
    args = kernel.args
    positional = [*args.posonlyargs, *args.args]
    pairs = []
    if args.defaults:
        pairs.extend(zip(positional[-len(args.defaults):], args.defaults))
    pairs.extend((arg, default) for arg, default in
                 zip(args.kwonlyargs, args.kw_defaults) if default is not None)
    for arg, default in pairs:
        if (_is_comptime_int_annotation(arg.annotation)
                and isinstance(default, ast.Constant)
                and (type(default.value) is int
                     or isinstance(default.value, bool))):
            values[arg.arg] = int(default.value)
    return values


def fold_value(node, inc) -> "int | None":
    """AST 节点 → comptime 整数值；不可折叠返回 None（整数精确算术）。

    值来源：int 字面量 / comptime 名（inc.values，True/False 按 1/0 不经
    此路径——BOOL 先行裁决）/ 已折叠子表达式与 cdiv 结果（inc.const_values）。
    """
    if id(node) in inc.const_values:
        return inc.const_values[id(node)]  # 已折叠子表达式 / cdiv 结果
    if isinstance(node, ast.Constant):
        return node.value if type(node.value) is int else None
    if isinstance(node, ast.Name):
        return inc.values.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = fold_value(node.operand, inc)
        return None if value is None else -value
    if isinstance(node, ast.BinOp):
        op = type(node.op).__name__
        a = fold_value(node.left, inc)
        b = fold_value(node.right, inc)
        if a is None or b is None:
            return None
        if op == "Add":
            return a + b
        if op == "Sub":
            return a - b
        if op == "Mult":
            return a * b
        return None  # Div 等：不折叠（int 家族 `/` 是 E0604，合法路径不达）
    return None


def full_value_verdict(value_node, dtype_node, inc):
    """R4 位置①：tis.full 的 value 常量绑定裁决（违规 E0606）。

    value 的标量形态合法性由原语段 E0406 先行裁决；BOOL 实参逃逸已在
    调用层报 E0605。值类分派：int 家族编译期常量按目标 dtype 值域（浮点
    dtype 整到浮按 R6 舍入，不拒）；float 字面量限浮点 dtype；inf 限
    IEEE 标准精度浮点 dtype；运行期 int 拒（无原语路径）；dtype 标量须
    同 dtype（无隐式转换）。违规经 `inc.report`——帧活跃（外层表达式）
    时与同码违规合并为一条（R7 同表达式同类违规合并）。
    """
    if not (isinstance(dtype_node, ast.Name) and dtype_node.id in DTYPES):
        return  # dtype 实参形态违规：原语段 E0406 已拒
    dtype = dtype_node.id

    # int 家族编译期常量（字面量/一元负/comptime 表达式）。
    value = fold_value(value_node, inc)
    if value is not None:
        if dtype in INT_RANGES:
            lo, hi = INT_RANGES[dtype]
            if not lo <= value <= hi:
                inc.report("E0606", value_node,
                           f"常量值 {value} 超出 {dtype} 补码值域 "
                           f"[{lo}, {hi}]；核对值域或改用更宽整型 dtype")
        return  # 浮点 dtype：整到浮 roundTiesToEven（R6 行为承诺），不拒

    t = inc.infer(value_node)
    if t == ScalarType(INT):
        inc.report("E0606", value_node,
                   "运行期 int 标量不能绑定 dtype 标量位置（运行期整型到"
                   " dtype 转换无原语路径）；由宿主侧传入目标 dtype 标量或"
                   "经 Tensor 显式 tis.cast 路径")
        return
    if isinstance(t, ScalarType) and t.kind in DTYPES:
        if t.kind != dtype:
            inc.report("E0606", value_node,
                       f"dtype 标量（{t.kind}）与目标 dtype（{dtype}）不同；"
                       "家族内无隐式转换，经显式 tis.cast 路径转换后取值")
        return  # 同 dtype 标量合法
    if isinstance(t, PendingConst):
        if dtype in FLOAT_DTYPES:
            if t.tag == "inf" and dtype not in IEEE_FLOAT_DTYPES:
                inc.report("E0606", value_node,
                           "无穷常量不能绑定 f8e4m3 位置（OCP FP8 E4M3 无"
                           "无穷表示，max finite ±448）；使用 f16/bf16/f32 "
                           "或有限常量")
            return  # float 字面量正确舍入 / inf 绑 IEEE 浮点（R2/R6 承诺）
        note = "无穷常量无整数表示" if t.tag == "inf" else "float 字面量"
        inc.report("E0606", value_node,
                   f"浮点常量不能绑定整型 dtype 位置（{note}）；整型位置"
                   "使用整型常量")
        return
    # Tensor/State/UNKNOWN/comptime 无值名：非标量让渡（E0406 域）或不可判定


def pending_rebinding_verdict(pending, bound, node, inc):
    """R4 位置③：待定型常量（float 字面量/inf）再绑定到已绑定变量。

    已绑定浮点 dtype 标量：float 字面量按该 dtype 定型（roundTiesToEven）、
    inf 绑 IEEE 标准精度（f8e4m3 拒）；int 家族/整型 dtype：E0606；非标量
    目标（Tensor/状态类）：让渡（非 dtype 标量语境）。
    """
    if not isinstance(bound, ScalarType):
        return  # 非标量目标：让渡
    if bound.kind in FLOAT_DTYPES:
        if pending.tag == "inf" and bound.kind not in IEEE_FLOAT_DTYPES:
            inc.reject(node, "E0606", "const-binding",
                       "无穷常量不能绑定 f8e4m3 位置（E4M3 无无穷表示，"
                       "max finite ±448）；使用有限常量")
        return  # 按已绑定 dtype 定型（R4 Scenario「再绑定按已绑定 dtype 定型」）
    kind_note = "无穷常量无整数表示" if pending.tag == "inf" \
        else "float 字面量需要浮点 dtype 语境"
    inc.reject(node, "E0606", "const-binding",
               f"浮点常量不能绑定 int 家族/整型 dtype 变量（{kind_note}）；"
               "经 tis.full 构造或绑定浮点 dtype 变量后运算")
