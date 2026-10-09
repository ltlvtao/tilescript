"""E0106 设备代码表达式接受集（syntax spec R6「设备代码表达式接受集」）。

上下文受限类别（design 裁决与 spec 位置列举对齐）：
- 元组字面量：仅切片下标与调用**位置**实参位置；
- 字典字面量：仅调用关键字实参值位置，且键 MUST 为字符串常量；
- 字符串常量：仅调用关键字实参值位置。

`check_module` 为独立入口（设备代码内全部语句的子表达式检查，不做语句类别
判定）；`direct_subexprs`/`check_expr` 供 statements 在白名单语句上协同调用
（短路规则：被 E0105 拒绝的语句不深入，design D3）。
"""

import ast

from .report import Rejection
from .top_level import iter_entry_functions

_ORDER = 6

# 二元运算符白名单：算术 + - * /
_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
# 比较运算符白名单（仅两操作数单比较）
_CMPOPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
# 一元白名单：- 与 not
_UNARYOPS = (ast.USub, ast.Not)

# 拒绝节点类别 slug 冻结表（design D6）。
_NODE_CATEGORY = {
    ast.Lambda: "lambda",
    ast.ListComp: "list-comprehension",
    ast.SetComp: "set-comprehension",
    ast.DictComp: "dict-comprehension",
    ast.GeneratorExp: "generator-expression",
    ast.JoinedStr: "f-string",
    ast.FormattedValue: "f-string",
    ast.Yield: "yield-expression",
    ast.YieldFrom: "yield-expression",
    ast.Await: "await-expression",
    ast.NamedExpr: "named-expression",
    ast.IfExp: "conditional-expression",
    ast.Starred: "starred-expression",
    ast.List: "list-literal",
    ast.Set: "set-literal",
}

_OP_SYMBOL = {
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**", ast.MatMult: "@",
    ast.LShift: "<<", ast.RShift: ">>",
    ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^",
    ast.Invert: "~", ast.UAdd: "+",
    ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not",
}


def check_module(tree: ast.Module) -> "list[Rejection]":
    """设备代码（顶层入口函数体，词法含嵌套）内全部语句的子表达式检查。"""
    rejections: list[Rejection] = []
    for node in iter_entry_functions(tree):
        rejections.extend(_check_stmts(node.body))
    return rejections


def direct_subexprs(stmt: ast.stmt):
    """白名单语句的直接子表达式（含上下文标记），供语句检查器协同调用。"""
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            yield t, "general"
        yield stmt.value, "general"
    elif isinstance(stmt, ast.AugAssign):
        yield stmt.target, "general"
        yield stmt.value, "general"
    elif isinstance(stmt, ast.Expr):
        yield stmt.value, "general"
    elif isinstance(stmt, ast.Return):
        if stmt.value is not None:
            yield stmt.value, "general"
    elif isinstance(stmt, ast.If):
        yield stmt.test, "general"
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        yield stmt.target, "general"
        yield stmt.iter, "general"
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        for item in stmt.items:
            yield item.context_expr, "general"


def check_expr(node: ast.expr, context: str = "general") -> "list[Rejection]":
    out: list[Rejection] = []
    _visit(node, context, out)
    return out


def _check_stmts(stmts: "list[ast.stmt]") -> "list[Rejection]":
    """递归全部语句树收集子表达式拒绝（不做语句类别判定——独立测试入口语义）。"""
    out: list[Rejection] = []
    for stmt in stmts:
        for expr, ctx in direct_subexprs(stmt):
            out.extend(check_expr(expr, ctx))
        for child in _child_stmts(stmt):
            out.extend(_check_stmts([child]))
    return out


def _child_stmts(stmt: ast.stmt) -> "list[ast.stmt]":
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return list(stmt.body) + list(stmt.orelse)
    if isinstance(stmt, ast.If):
        return list(stmt.body) + list(stmt.orelse)
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return list(stmt.body)
    if isinstance(stmt, ast.Try):
        return (list(stmt.body) + list(stmt.orelse) + list(stmt.finalbody)
                + [s for h in stmt.handlers for s in h.body])
    if isinstance(stmt, ast.Match):
        return [s for case in stmt.cases for s in case.body]
    return []


def _visit(node: ast.expr, ctx: str, out: "list[Rejection]") -> None:
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.Attribute):
        _visit(node.value, "general", out)
        return
    if isinstance(node, ast.Subscript):
        _visit(node.value, "general", out)
        _visit_slice(node.slice, out)
        return
    if isinstance(node, ast.Slice):
        _visit_slice(node, out)
        return
    if isinstance(node, ast.Tuple):
        if ctx not in ("call-arg", "slice"):
            out.append(_rej(node, "tuple-literal-position",
                            "元组字面量仅接受在切片下标与调用（位置）实参位置。"))
            return
        for elt in node.elts:
            _visit(elt, ctx, out)
        return
    if isinstance(node, ast.Dict):
        if ctx != "kwarg-value":
            out.append(_rej(node, "dict-literal-position",
                            "字典字面量仅接受在调用关键字实参值位置。"))
            return
        for key in node.keys:
            if key is None or not (isinstance(key, ast.Constant) and type(key.value) is str):
                out.append(_rej(node, "dict-literal-position",
                                "字典字面量的键 MUST 为字符串常量。"))
                return
        for value in node.values:
            _visit(value, "kwarg-value", out)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bytes):
            # bytes 不在 R6 接受集（数值常量/布尔/None/字符串常量之外）——
            # 任何位置（含关键字实参值）拒绝。
            out.append(_rej(node, "bytes-literal",
                            "bytes 字面量不在设备代码表达式接受集。"))
            return
        if isinstance(node.value, str) and ctx != "kwarg-value":
            out.append(_rej(node, "string-literal-position",
                            "字符串常量仅接受在调用关键字实参值位置。"))
            return
        return  # int/float/bool/None/str(kwarg 值) 无进一步限制
    if isinstance(node, ast.Call):
        _visit(node.func, "general", out)
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                out.append(_rej(arg, "starred-expression",
                                "星号解包实参不在表达式接受集。"))
            else:
                _visit(arg, "call-arg", out)
        for kw in node.keywords:
            if kw.arg is None:
                out.append(_rej(kw.value, "starred-expression",
                                "** 解包关键字实参不在表达式接受集。"))
            else:
                _visit(kw.value, "kwarg-value", out)
        return
    if isinstance(node, ast.BinOp):
        symbol = _OP_SYMBOL.get(type(node.op))
        if type(node.op) not in _BINOPS:
            out.append(_rej(node, "disallowed-operator",
                            f"运算符 {symbol} 不在表达式接受集"
                            "（算术仅 + - * /，比较仅 < <= > >= == !=，逻辑仅 and or）。"))
            return
        _visit(node.left, "general", out)
        _visit(node.right, "general", out)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            symbol = _OP_SYMBOL.get(type(node.op), type(node.op).__name__)
            out.append(_rej(node, "disallowed-operator",
                            f"一元运算符 {symbol} 不在表达式接受集（仅一元负 - 与 not）。"))
            return
        _visit(node.operand, "general", out)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _visit(value, "general", out)
        return
    if isinstance(node, ast.Compare):
        if len(node.ops) > 1:
            out.append(_rej(node, "chained-comparison",
                            "链式比较不在表达式接受集；仅接受两操作数单比较。"))
            return
        for op in node.ops:
            if type(op) not in _CMPOPS:
                symbol = _OP_SYMBOL.get(type(op), type(op).__name__)
                out.append(_rej(node, "disallowed-operator",
                                f"比较运算符 {symbol} 不在表达式接受集"
                                "（仅 < <= > >= == !=）。"))
                return
        _visit(node.left, "general", out)
        for comp in node.comparators:
            _visit(comp, "general", out)
        return
    category = _NODE_CATEGORY.get(type(node))
    if category is not None:
        out.append(_rej(node, category, _CATEGORY_SUGGESTION[category]))
        return
    out.append(_rej(node, "disallowed-expression",
                    "该表达式类别不在设备代码表达式接受集。"))


def _visit_slice(slice_node: ast.expr, out: "list[Rejection]") -> None:
    """切片下标：多值下标（Tuple）与 Slice 及其边界表达式（元组在此位置合法）。"""
    if isinstance(slice_node, ast.Tuple):
        for elt in slice_node.elts:
            _visit(elt, "slice", out)
        return
    if isinstance(slice_node, ast.Slice):
        for part in (slice_node.lower, slice_node.upper, slice_node.step):
            if part is not None:
                _visit(part, "general", out)
        return
    _visit(slice_node, "slice", out)


def _rej(node: ast.expr, category: str, suggestion: str) -> Rejection:
    return Rejection(
        code="E0106", line=node.lineno, col=node.col_offset + 1,
        category=category, suggestion=suggestion, order=_ORDER,
    )


_CATEGORY_SUGGESTION = {
    "lambda": "lambda 不在表达式接受集；请定义具名嵌套函数（produce/consume 装饰）。",
    "list-comprehension": "列表推导不在表达式接受集；请展开为 for 循环与显式赋值。",
    "set-comprehension": "集合推导不在表达式接受集；请展开为 for 循环与显式赋值。",
    "dict-comprehension": "字典推导不在表达式接受集；请展开为 for 循环与显式赋值。",
    "generator-expression": "生成器表达式不在表达式接受集；请展开为 for 循环。",
    "f-string": "f-string 与任意字符串格式化表达式不在表达式接受集。",
    "yield-expression": "yield 不在表达式接受集。",
    "await-expression": "await 不在表达式接受集。",
    "named-expression": "海象运算符不在表达式接受集；请拆分为独立赋值语句。",
    "conditional-expression": "条件（三元）表达式不在表达式接受集；请改写为 if/else 语句。",
    "starred-expression": "星号解包不在表达式接受集。",
    "list-literal": "列表字面量不在表达式接受集。",
    "set-literal": "集合字面量不在表达式接受集。",
}
