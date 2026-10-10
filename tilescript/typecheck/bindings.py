"""E0303 绑定检查遍历器（type-system spec R6 单类型不变量 + R7 绑定位置）。

设备函数（kernel / produce / consume）体按语句序流式遍历：赋值（Name 目标
单类型不变量、Subscript/Attribute 目标侧派生×源）、AugAssign（算术结果
UNKNOWN，保留占位）、Return × 返回注解。构造调用与带注解调用实参（D7 第
3/4 类）在 infer 的 Call 推断中承载。

produce/consume 体作用域 = kernel 形参快照 + 自身形参（design D3）。
"""

import ast

from tilescript.frontend.top_level import _has_tis_decorator

from . import infer as infer_mod
from . import symbols
from .annotations import is_produce_consume, parse_annotation
from .types import UNKNOWN, treats_as_unknown


def check_device_functions(tree: ast.Module, registry, comptime_syms=frozenset()):
    """遍历全部设备函数体，返回 E0303 拒绝清单（未 finalize）。"""
    functions = symbols.annotated_functions(tree, registry, comptime_syms)
    kernel_env = symbols.kernel_params(tree, comptime_syms)
    rejections = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _has_tis_decorator(node, "kernel"):
            env = dict(kernel_env)
        elif is_produce_consume(node):
            # 外层快照（kernel 形参）+ 自身形参（含状态类/buffer UNKNOWN）。
            env = dict(kernel_env)
            for arg in symbols._params(node):
                env[arg.arg] = symbols._param_type(arg.annotation, comptime_syms,
                                                   registry)
        else:
            continue
        returns = functions.get(node.name, {}).get("returns")
        inc = infer_mod.Inferencer(env, registry, functions)
        _walk(node.body, inc, returns)
        rejections.extend(inc.rejections)
    return rejections


def _walk(stmts, inc, returns):
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            source = inc.infer(stmt.value)
            for target in stmt.targets:
                _bind_target(target, source, inc)
        elif isinstance(stmt, ast.AnnAssign):
            _ann_assign(stmt, inc)
        elif isinstance(stmt, ast.AugAssign):
            inc.infer(stmt.value)  # 算术结果 UNKNOWN：实际跳过（D7 占位）
        elif isinstance(stmt, ast.Return):
            if returns is not None and stmt.value is not None:
                inc.check_binding(stmt.value, inc.infer(stmt.value), returns,
                                  "return")
        elif isinstance(stmt, ast.Expr):
            inc.infer(stmt.value)  # 纯调用语句：触发构造/实参检查（结果弃置）
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _bind_loop_target(stmt.target, inc)
            _walk(stmt.body, inc, returns)
            _walk(stmt.orelse, inc, returns)
        elif isinstance(stmt, ast.While):
            _walk(stmt.body, inc, returns)
            _walk(stmt.orelse, inc, returns)
        elif isinstance(stmt, ast.If):
            _walk(stmt.body, inc, returns)
            _walk(stmt.orelse, inc, returns)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    _bind_walrus_like(item.optional_vars, inc)
            _walk(stmt.body, inc, returns)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody,
                          [h for h in stmt.handlers for h in h.body]
                          if stmt.handlers else []):
                _walk(block, inc, returns)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_produce_consume(stmt):
                _walk_nested(stmt, inc)
        # Pass/Break/Continue/Assert 等：无绑定动作。


def _bind_target(target, source, inc):
    """赋值目标绑定（D7 第 1 类）。"""
    if isinstance(target, ast.Name):
        bound = inc.env.get(target.id)
        if bound is None:
            inc.env[target.id] = source  # 首次绑定定类型（UNKNOWN 保持）
            return
        if treats_as_unknown(bound):
            return  # UNKNOWN 变量此后保持：已知值再绑定不升级、不回溯（D4）
        inc.check_binding(target, source, bound, "assign")
        # 单类型不变量：再绑定不改变已确定类型（检查拒绝后亦不改写）。
    elif isinstance(target, (ast.Subscript, ast.Attribute)):
        target_type = inc.infer(target)  # 目标侧派生/字段声明类型
        inc.check_binding(target, source, target_type, "assign")
    # Tuple/Starred 等：M1 让渡（语法段白名单外的形态不达此处）。


def _ann_assign(stmt, inc):
    """带注解局部绑定：注解尽力解析（三类检查位置之外，失败让渡），值×注解检查。"""
    if not isinstance(stmt.target, ast.Name):
        return
    annotation, rejection = parse_annotation(stmt.annotation)
    declared = UNKNOWN if rejection is not None else annotation
    if stmt.value is not None:
        inc.check_binding(stmt.value, inc.infer(stmt.value), declared, "assign")
    inc.env[stmt.target.id] = declared


def _walk_nested(fn, inc):
    """嵌套 produce/consume（`@pipe.produce` 等）：体作用域 = 外层快照 + 自身形参
    （design D3）；遍历后恢复外层 env（函数体绑定不泄漏）。"""
    snapshot = dict(inc.env)
    for arg in symbols._params(fn):
        inc.env[arg.arg] = symbols._param_type(arg.annotation, inc.comptime_syms,
                                               inc.registry)
    returns = None
    if fn.returns is not None and not _is_none_node(fn.returns):
        parsed, rejection = parse_annotation(fn.returns,
                                             comptime_syms=inc.comptime_syms,
                                             registry=inc.registry)
        returns = None if rejection is not None else parsed
    _walk(fn.body, inc, returns)
    inc.env.clear()
    inc.env.update(snapshot)


def _is_none_node(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _bind_loop_target(target, inc):
    """for-range/tile_iter 循环变量 → UNKNOWN（元素类型归 execution/numerics）。"""
    if isinstance(target, ast.Name):
        inc.env[target.id] = UNKNOWN
    elif isinstance(target, ast.Tuple):
        for element in target.elts:
            _bind_loop_target(element, inc)


def _bind_walrus_like(node, inc):
    """with 绑定名（warp_group 的 g 等）→ UNKNOWN。"""
    if isinstance(node, ast.Name):
        inc.env[node.id] = UNKNOWN
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            _bind_walrus_like(element, inc)
    elif isinstance(node, ast.Starred):
        _bind_walrus_like(node.value, inc)
