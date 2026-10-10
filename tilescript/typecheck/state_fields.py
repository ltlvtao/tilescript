"""E0304 状态类字段类型约束（type-system spec R5「状态类」）与注册表构建。

三路径触发（design D6）：
- Tensor 但 scope 非 Register → state-field-scope
- Pointer → state-field-pointer
- 其余种类（标量/状态类）→ state-field-scalar（R5 delta 补全路径）

一注解一结果：E0302 解析失败的字段不产 E0304（同位置由 order tiebreak 亦满足）。
违规但解析成功的字段类型仍入注册表（后续绑定检查用声明类型）。
"""

import ast

from tilescript.frontend.report import Rejection
from tilescript.frontend.top_level import _has_tis_decorator

from .annotations import parse_annotation
from .types import PointerType, StateType, TensorType

_ORDER = 2  # design D9：E0302=1/E0304=2/E0301=3/E0303=4


def check(tree: ast.Module, comptime_syms=frozenset()):
    """全部 @tis.state 类 → (注册表, E0302/E0304 拒绝清单)。

    注册表：类名 → StateType（类体声明序字段）；同名类按首次注册
    （design D11 residual 2）。字段注解解析传入 registry（源码序渐进可见）：
    已注册状态类名字段构成「既非 Tensor 也非 Pointer」→ E0304（R5 delta）。
    """
    registry: "dict[str, StateType]" = {}
    rejections: "list[Rejection]" = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _has_tis_decorator(node, "state"):
            fields, field_rejections = _check_class(node, comptime_syms, registry)
            rejections.extend(field_rejections)
            if node.name not in registry:
                registry[node.name] = StateType(node.name, tuple(fields))
    return registry, rejections


def _check_class(cls: ast.ClassDef, comptime_syms, registry):
    fields: "list[tuple[str, object]]" = []
    rejections: "list[Rejection]" = []
    for stmt in cls.body:
        # E0107 已裁类体形态（仅无值 AnnAssign）；防御性跳过其他语句。
        if not (isinstance(stmt, ast.AnnAssign) and stmt.value is None
                and isinstance(stmt.target, ast.Name)):
            continue
        annotation, rejection = parse_annotation(stmt.annotation,
                                                 comptime_syms=comptime_syms,
                                                 registry=registry)
        if rejection is not None:
            rejections.append(rejection)  # E0302（order=1）；无类型不入表
            continue
        fields.append((stmt.target.id, annotation))
        category = _field_violation(annotation)
        if category is not None:
            rejections.append(_rej(stmt, category))
    return fields, rejections


def _field_violation(annotation):
    if isinstance(annotation, TensorType):
        return None if annotation.scope == "Register" else "state-field-scope"
    if isinstance(annotation, PointerType):
        return "state-field-pointer"
    return "state-field-scalar"


def _rej(stmt: ast.AnnAssign, category: str) -> Rejection:
    suggestions = {
        "state-field-scope":
            "状态类字段必须位于 Register；跨状态空间的数据（Global/Shared）"
            "须以 kernel 形参或显式移动原语承载，不进状态类。",
        "state-field-pointer":
            "状态类字段只接受 Register scope 的 Tensor；Pointer 不合法，"
            "指针须作为 kernel 形参传递。",
        "state-field-scalar":
            "状态类字段必须是 Tensor[..., Register]；标量或嵌套状态类不合法。",
    }
    return Rejection(
        code="E0304", line=stmt.annotation.lineno, col=stmt.annotation.col_offset + 1,
        category=category, suggestion=suggestions[category], order=_ORDER,
        stage="type-system",
    )
