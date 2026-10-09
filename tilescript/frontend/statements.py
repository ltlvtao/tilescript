"""E0105 设备代码语句接受集（syntax spec R5「设备代码语句接受集」）。

设备代码 = 设备入口函数体与 produce/consume 嵌套函数体（词法包含）。
短路规则（design D3，spec L96 的反面裁决）：语句类别被拒时不深入其子树
（子语句与子表达式均不再报）；白名单语句才检查子表达式（按 E0106 报、
不另报 E0105）。
"""

import ast

from . import expressions
from .report import Rejection
from .top_level import _is_entry_function

_ORDER = 5

_ASSIGN_TARGETS = (ast.Name, ast.Attribute, ast.Subscript)
_AUG_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)

# 拒绝类别 slug 与恢复建议冻结表（design D6：「最接近的合法替代」）。
_REJECTED = {
    ast.While: ("while-loop",
                "while 循环不在设备代码语句接受集；请改写为 for ... in range(...) "
                "或 for ... in tis.tile_iter()。"),
    ast.Break: ("break-statement",
                "break 不在接受集；请以循环变量边界与 if 条件改写控制流。"),
    ast.Continue: ("continue-statement",
                   "continue 不在接受集；请以 if 条件包裹循环体余下部分。"),
    ast.Try: ("try-statement",
              "try/except 不在接受集；请以返回值或 if 分支承载错误路径。"),
    ast.Raise: ("raise-statement",
                "raise 不在接受集；设备代码不使用异常机制。"),
    ast.Assert: ("assert-statement",
                 "assert 不在接受集；请删除或改写为显式 if 分支。"),
    ast.Delete: ("del-statement", "del 不在接受集。"),
    ast.Global: ("global-declaration",
                 "global 声明不在接受集；请通过参数与返回值传递状态。"),
    ast.Nonlocal: ("nonlocal-declaration",
                   "nonlocal 声明不在接受集；请通过参数与返回值传递状态。"),
    ast.Match: ("match-statement",
                "match 语句不在接受集；请改写为 if/elif/else 链。"),
    ast.AnnAssign: ("annotated-assignment",
                    "注解式局部赋值不在接受集；局部类型由推断承载，"
                    "请使用无注解赋值（x = ...）。"),
    ast.Import: ("device-import",
                 "设备代码内 import 不在接受集；import 语句仅允许出现在模块顶层。"),
    ast.ImportFrom: ("device-import",
                     "设备代码内 import 不在接受集；import 语句仅允许出现在模块顶层。"),
    ast.ClassDef: ("class-definition",
                   "设备代码内类定义不在接受集；状态类请定义于模块顶层"
                   "并以 @tis.state 装饰。"),
    ast.AsyncFunctionDef: ("function-definition",
                           "设备代码内仅允许 produce/consume 装饰的嵌套函数定义。"),
    ast.AsyncFor: ("async-for-statement",
                   "async for 不在接受集；可迭代表达式仅 range(...) 或 tis.tile_iter()。"),
    ast.AsyncWith: ("async-with-statement",
                    "async with 不在接受集；上下文表达式仅 tis.warp_group(...)。"),
}


def check(tree: ast.Module) -> "list[Rejection]":
    rejections: list[Rejection] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_entry_function(node):
            rejections.extend(_check_body(node.body))
    return rejections


def _check_body(stmts: "list[ast.stmt]") -> "list[Rejection]":
    out: list[Rejection] = []
    for stmt in stmts:
        verdict = _classify(stmt)
        if verdict is not None:
            out.append(Rejection(
                code="E0105", line=stmt.lineno, col=stmt.col_offset + 1,
                category=verdict[0], suggestion=verdict[1], order=_ORDER,
            ))
            continue  # 短路：被拒语句的子树不再深入（design D3）
        if isinstance(stmt, ast.FunctionDef):
            # produce/consume 装饰的嵌套函数：其体仍是设备代码
            out.extend(_check_body(stmt.body))
            continue
        for expr, ctx in expressions.direct_subexprs(stmt):
            out.extend(expressions.check_expr(expr, ctx))
        out.extend(_check_body(expressions._child_stmts(stmt)))
    return out


def _classify(stmt: ast.stmt):
    """返回 (category, suggestion) 或 None（白名单）。"""
    rejected = _REJECTED.get(type(stmt))
    if rejected is not None:
        return rejected

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) > 1:
            return ("chained-assignment", "仅接受单目标赋值；请拆分为多条赋值语句。")
        if not isinstance(stmt.targets[0], _ASSIGN_TARGETS):
            return ("unpacking-assignment",
                    "解包赋值不在接受集；请对每个目标逐个赋值。")
        return None

    if isinstance(stmt, ast.AugAssign):
        if type(stmt.op) not in _AUG_OPS:
            return ("augmented-assignment-operator",
                    "增强赋值运算符仅限 + - * /；请改用普通赋值承载其他运算。")
        return None

    if isinstance(stmt, ast.FunctionDef):
        if _is_pipeline_nested(stmt):
            return None
        return ("function-definition",
                "设备代码内仅允许 produce/consume 装饰的嵌套函数定义；"
                "请为嵌套函数添加 @<pipeline 实例名>.produce 或 .consume 装饰器。")

    if isinstance(stmt, ast.For):
        if not _is_range_or_tile_iter(stmt.iter):
            return ("invalid-for-iterable",
                    "for 的可迭代表达式仅为 range(...) 或 tis.tile_iter() 调用。")
        return None

    if isinstance(stmt, ast.With):
        for item in stmt.items:
            if not _is_warp_group_call(item.context_expr):
                return ("invalid-with-context",
                        "with 的上下文表达式仅为 tis.warp_group(...) 调用。")
        return None

    if isinstance(stmt, (ast.Expr, ast.Return, ast.Pass, ast.If)):
        return None

    return ("unknown-statement", "该语句类别不在设备代码语句接受集。")


def _is_pipeline_nested(stmt: ast.FunctionDef) -> bool:
    return any(
        isinstance(d, ast.Attribute) and d.attr in ("produce", "consume")
        and isinstance(d.value, ast.Name)
        for d in stmt.decorator_list
    )


def _is_range_or_tile_iter(iter_expr: ast.expr) -> bool:
    if not isinstance(iter_expr, ast.Call):
        return False
    func = iter_expr.func
    if isinstance(func, ast.Name) and func.id == "range":
        return True
    return (isinstance(func, ast.Attribute) and func.attr == "tile_iter"
            and isinstance(func.value, ast.Name) and func.value.id == "tis")


def _is_warp_group_call(expr: ast.expr) -> bool:
    return (isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute) and expr.func.attr == "warp_group"
            and isinstance(expr.func.value, ast.Name) and expr.func.value.id == "tis")
