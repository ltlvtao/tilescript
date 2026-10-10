"""设备函数遍历骨架（design D3——原语段推断复用与作用域同构）。

结构与 `typecheck.bindings` 同构（语句序流式遍历、kernel/produce/consume
三类设备函数、嵌套 produce/consume 快照恢复），差异两点：
1. 不做 E0303 检查（类型段职责；段间短路保证其零命中）；
2. `tis.*` Call 改道原语检查器 `checker.check_call(node, inferencer)`
   ——合法给结果类型（results 表）、违规给 UNKNOWN（E04xx 已记录）。

防御性收尾断言（design D3）：每函数遍历结束断言 `Inferencer.rejections`
为空——非空即遍历与类型段 `bindings._walk` 发生漂移（同构性破坏，
属实现缺陷），开发期即暴露而非混入错段的 E03xx。

`PrimitiveInferencer.in_produce_consume`：E0407 Async 语境数据来源
（kernel 顶层 False / produce-consume 嵌套体内 True，design D6）。
"""

import ast

from ..frontend.top_level import _has_tis_decorator
from ..typecheck import infer as infer_mod
from ..typecheck import symbols
from ..typecheck.annotations import parse_annotation, is_produce_consume
from ..typecheck.types import UNKNOWN


class PrimitiveInferencer(infer_mod.Inferencer):
    """Inferencer 的原语段扩展：`tis.*` Call 改道 + 嵌套语境跟踪。"""

    def __init__(self, env, registry, functions, comptime_syms, checker):
        super().__init__(env, registry, functions, comptime_syms)
        self.checker = checker
        self.in_produce_consume = False  # kernel 顶层（design D6）

    def _infer_call(self, node):
        func = node.func
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "tis"):
            return self.checker.check_call(node, self)  # E04xx + 结果类型
        return super()._infer_call(node)  # 构造/设备函数调用/其余：类型段规则


def walk_device_functions(tree: ast.Module, registry, comptime_syms, checker):
    """同构遍历全部设备函数体（无 E0303；`tis.*` 改道 checker）。

    checker：原语检查器组合体，需提供 `check_call(node, inferencer) -> 类型`；
    E04xx 拒绝由 checker 侧收集（本函数不触碰拒绝清单——段内序与 finalize
    归 check_module，design D9）。
    """
    functions = symbols.annotated_functions(tree, registry, comptime_syms)
    kernel_env = symbols.kernel_params(tree, comptime_syms)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _has_tis_decorator(node, "kernel"):
            env = dict(kernel_env)
        elif is_produce_consume(node):
            # 外层快照（kernel 形参）+ 自身形参（与 bindings 同构）。
            env = dict(kernel_env)
            for arg in symbols._params(node):
                env[arg.arg] = symbols._param_type(arg.annotation, comptime_syms,
                                                   registry)
        else:
            continue
        inc = PrimitiveInferencer(env, registry, functions, comptime_syms, checker)
        _walk(node.body, inc)
        # 防御断言（design D3）：段间短路下 Inferencer 内嵌 E03xx 必零命中。
        assert not inc.rejections, (
            "原语段遍历与类型段 bindings 漂移（Inferencer 内嵌检查非零命中）："
            f"{[r.to_dict() for r in inc.rejections]}")


def _walk(stmts, inc):
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            source = inc.infer(stmt.value)  # tis.* 改道在 Call 推断内承载
            for target in stmt.targets:
                _bind_target(target, source, inc)
        elif isinstance(stmt, ast.AnnAssign):
            _ann_assign(stmt, inc)
        elif isinstance(stmt, ast.AugAssign):
            inc.infer(stmt.value)  # 触发改道；算术结果不产生绑定
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                inc.infer(stmt.value)
        elif isinstance(stmt, ast.Expr):
            inc.infer(stmt.value)  # 纯调用语句：tis.* 检查主形态
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _bind_loop_target(stmt.target, inc)
            _walk(stmt.body, inc)
            _walk(stmt.orelse, inc)
        elif isinstance(stmt, (ast.While, ast.If)):
            _walk(stmt.body, inc)
            _walk(stmt.orelse, inc)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    _bind_walrus_like(item.optional_vars, inc)
            _walk(stmt.body, inc)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody,
                          [h for h in stmt.handlers for h in h.body]
                          if stmt.handlers else []):
                _walk(block, inc)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_produce_consume(stmt):
                _walk_nested(stmt, inc)
        # Pass/Break/Continue/Assert 等：无绑定动作。


def _bind_target(target, source, inc):
    """赋值目标 env 绑定（单类型不变量同构；E0303 归类型段）。

    首绑定类型（UNKNOWN 保持）；已绑定不改写（UNKNOWN 不升级——类型段
    D4 同构；已确定类型不因再绑定变化）。
    """
    if isinstance(target, ast.Name):
        if target.id not in inc.env:
            inc.env[target.id] = source
    elif isinstance(target, (ast.Subscript, ast.Attribute)):
        inc.infer(target)  # 触发目标侧 tis.* 改道；基对象类型不改写


def _ann_assign(stmt, inc):
    """带注解局部绑定：注解尽力解析（失败让渡），env 无条件按注解覆盖。"""
    if not isinstance(stmt.target, ast.Name):
        return
    annotation, rejection = parse_annotation(stmt.annotation)
    declared = UNKNOWN if rejection is not None else annotation
    if stmt.value is not None:
        inc.infer(stmt.value)  # 触发改道；值×注解绑定检查归类型段
    inc.env[stmt.target.id] = declared


def _walk_nested(fn, inc):
    """嵌套 produce/consume：快照 + 自身形参 + 语境置位；遍历后双双恢复。"""
    snapshot = dict(inc.env)
    for arg in symbols._params(fn):
        inc.env[arg.arg] = symbols._param_type(arg.annotation, inc.comptime_syms,
                                               inc.registry)
    outer_context = inc.in_produce_consume
    inc.in_produce_consume = True
    _walk(fn.body, inc)
    inc.in_produce_consume = outer_context
    inc.env.clear()
    inc.env.update(snapshot)


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
