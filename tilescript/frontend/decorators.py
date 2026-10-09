"""E0103 设备入口装饰器识别（syntax spec R3「设备入口装饰器识别」）。

已识别集合：`tis.state`（仅模块顶层类）、`tis.kernel`/`tis.persistent_kernel`/
`tis.fused_kernel`（仅模块顶层函数）、`<实例名>.produce`/`<实例名>.consume`
（仅设备代码内嵌套函数；按属性名识别，实例名任意——spec L51）。
"""

import ast

from .report import Rejection

_ORDER = 3

_TIS_STATE = "state"
_TIS_ENTRIES = ("kernel", "persistent_kernel", "fused_kernel")
_PIPELINE_ATTRS = ("produce", "consume")

_KNOWN_SET = "已识别装饰器集合：@tis.state（模块顶层类）、@tis.kernel/@tis.persistent_kernel/@tis.fused_kernel（模块顶层函数）、@<pipeline 实例名>.produce/@<pipeline 实例名>.consume（设备代码内嵌套函数，按属性名识别、实例名任意）"


def check(tree: ast.Module) -> "list[Rejection]":
    rejections: list[Rejection] = []
    top_level_nodes = set(map(id, tree.body))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for deco in node.decorator_list:
            verdict = _classify(deco, node, id(node) in top_level_nodes)
            if verdict is not None:
                name, category = verdict
                if category == "unknown-decorator":
                    suggestion = f"装饰器 {name} 不在已识别集合内。{_KNOWN_SET}"
                else:
                    suggestion = (
                        f"装饰器 {name} 与目标类别不匹配。{_KNOWN_SET}"
                    )
                rejections.append(Rejection(
                    code="E0103", line=deco.lineno, col=deco.col_offset + 1,
                    category=category, suggestion=suggestion, order=_ORDER,
                ))
    return rejections


def _classify(deco: ast.expr, node: ast.AST, is_top_level: bool):
    """返回 (装饰器名称, 类别) 或 None（识别通过）。

    类别：unknown-decorator（集合外）/ decorator-target-mismatch（类别不匹配）。
    """
    if isinstance(deco, ast.Attribute) and isinstance(deco.value, ast.Name):
        attr, base = deco.attr, deco.value.id
        if base == "tis":
            if attr == _TIS_STATE:
                ok = isinstance(node, ast.ClassDef) and is_top_level
                return None if ok else ("tis.state", "decorator-target-mismatch")
            if attr in _TIS_ENTRIES:
                ok = isinstance(node, ast.FunctionDef) and is_top_level
                return None if ok else (f"tis.{attr}", "decorator-target-mismatch")
            # tis.<其他>：集合外
            return (f"tis.{attr}", "unknown-decorator")
        if attr in _PIPELINE_ATTRS:
            # produce/consume 属性装饰器：仅设备代码内（非顶层）函数
            ok = isinstance(node, ast.FunctionDef) and not is_top_level
            return None if ok else (f"{base}.{attr}", "decorator-target-mismatch")
        # 其他属性装饰器（a.b）：集合外
        return (f"{base}.{attr}", "unknown-decorator")
    # 裸名、调用式（@foo()）、嵌套属性（a.b.c）等：集合外
    return (_deco_name(deco), "unknown-decorator")


def _deco_name(deco: ast.expr) -> str:
    try:
        return ast.unparse(deco)
    except Exception:
        return "<装饰器>"
