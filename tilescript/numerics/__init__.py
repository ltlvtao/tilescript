"""数值语义段检查管线（numerics/value-semantics 实现面，E06xx）。

段编排：kernel / 顶层 produce/consume 体按语句序遍历（与 type-system
bindings 遍历同构——嵌套 produce/consume 作用域快照恢复）；表达式推断经
`NumericsInferencer`（数值域自载 + 委托零重放）。报告契约（收集全部、
位置升序、同位置段内 order tiebreak、重复一致）复用 `frontend.report`。

数值段 env 比类型段精确（算术/比较结果有类型、for-range 循环变量为
int、tis.* 原语返回类型有表、comptime 折叠值）——增强只发生在数值段
内部 env 与值表，不改变既有四段输出面（零重放 + 只读 AST）。
"""

import ast

from ..frontend import report
from ..frontend.top_level import _has_tis_decorator
from ..typecheck import state_fields, symbols
from ..typecheck.annotations import is_produce_consume, parse_annotation
from ..typecheck.infer import _compatible
from ..typecheck.types import (
    COMPTIME_INT,
    INT,
    UNKNOWN,
    ScalarType,
    treats_as_unknown,
)
from . import binary_ops, constants
from .constants import BOOL, FLOAT_DTYPES, PendingConst
from .value_env import NumericsInferencer

STAGE_NAME = "numerics"


def check_module(tree: ast.Module) -> "list[report.Rejection]":
    """对 AST 运行数值语义段检查，返回排序去重后的拒绝清单。

    无 target——数值语义跨 HAL 行为不变（hal capability「跨 HAL 行为
    不变面」）。
    """
    comptime_syms = symbols.kernel_comptime_names(tree)
    # 注册表复用 state_fields 构建（拒绝弃置：原语前置段已报，数值段零重放）。
    registry, _ = state_fields.check(tree, comptime_syms=comptime_syms)
    functions = symbols.annotated_functions(tree, registry, comptime_syms)
    kernel_env = symbols.kernel_params(tree, comptime_syms)
    base_values = constants.kernel_comptime_values(tree)
    rejections: "list[report.Rejection]" = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _has_tis_decorator(node, "kernel"):
            env, values = dict(kernel_env), dict(base_values)
        elif is_produce_consume(node):
            # 外层快照（kernel 形参）+ 自身形参（design D3，与类型段同构）。
            env, values = dict(kernel_env), dict(base_values)
            for arg in symbols._params(node):
                env[arg.arg] = symbols._param_type(arg.annotation,
                                                   comptime_syms, registry)
        else:
            continue
        returns = functions.get(node.name, {}).get("returns")
        inc = NumericsInferencer(env, registry, functions, comptime_syms,
                                 values, rejections.append)
        _walk(node.body, inc, returns, rejections)
    return report.finalize(rejections)


# ---- 语句遍历（结构同构 typecheck.bindings._walk；数值域承载增量） -------


def _walk(stmts, inc, returns, rejections):
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            _assign(stmt, inc, rejections)
        elif isinstance(stmt, ast.AnnAssign):
            _ann_assign(stmt, inc)
        elif isinstance(stmt, ast.AugAssign):
            _aug_assign(stmt, inc, rejections)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None and inc.infer(stmt.value) is BOOL:
                inc.reject(stmt.value, "E0605", "bool-closure",
                           "布尔值仅限条件语境封闭集；不能作为返回值")
        elif isinstance(stmt, ast.Expr):
            inc.infer(stmt.value)  # 纯调用语句：触发数值域/实参检查
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _bind_loop_target(stmt.target, stmt.iter, inc)
            _walk(stmt.body, inc, returns, rejections)
            _walk(stmt.orelse, inc, returns, rejections)
        elif isinstance(stmt, ast.While):
            _check_test(stmt.test, inc)
            _walk(stmt.body, inc, returns, rejections)
            _walk(stmt.orelse, inc, returns, rejections)
        elif isinstance(stmt, ast.If):
            _check_test(stmt.test, inc)
            _walk(stmt.body, inc, returns, rejections)
            _walk(stmt.orelse, inc, returns, rejections)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    _bind_unknown(item.optional_vars, inc)
            _walk(stmt.body, inc, returns, rejections)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody,
                          [h for h in stmt.handlers for h in h.body]
                          if stmt.handlers else []):
                _walk(block, inc, returns, rejections)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_produce_consume(stmt):
                _walk_nested(stmt, inc, rejections)
        # Pass/Break/Continue/Assert 等：无数值域绑定动作


def _assign(stmt, inc, rejections):
    """赋值：首绑定常量形态裁决（BOOL/PendingConst）+ 再绑定 R4 位置③与
    算术源等价择一（design D2——R7 授权面）。"""
    n0 = len(rejections)
    source = inc.infer(stmt.value)
    merged = False  # 择一标志（多目标语句只移除一次 E06xx）
    for target in stmt.targets:
        if isinstance(target, ast.Name):
            bound = inc.env.get(target.id)
            if bound is None:
                _first_bind(target, source, stmt, inc)
            elif not treats_as_unknown(bound):
                merged = _rebind(target, source, bound, stmt, inc,
                                 rejections, n0, merged)
            elif (source is not BOOL
                    and not isinstance(source, PendingConst)):
                inc.env[target.id] = source  # 旧绑定 UNKNOWN：改写为源类型
                # （无数值域裁决面；BOOL/PendingConst 形态不经此路径）
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            target_type = inc.infer(target)  # 目标侧派生/字段声明类型
            if (not merged and _is_arith_source(stmt.value)
                    and not treats_as_unknown(target_type)
                    and not treats_as_unknown(source)
                    and not _compatible(source, target_type)):
                del rejections[n0:]
                inc.reject_e0303(target, source, target_type)
                merged = True
    # env 更新：首绑定由 _first_bind 承载；再绑定由 _rebind 裁决路径承载
    # （让渡/等价改写为源类型，拒绝路径保旧——见 _rebind docstring）。


def _first_bind(target, source, stmt, inc):
    """首绑定：常量形态源裁决（BOOL→E0605 封闭集、PendingConst→E0606 裸
    绑定），其余记录类型；comptime 折叠值入值表（R5）。"""
    if source is BOOL:
        inc.reject(target, "E0605", "bool-closure",
                   "布尔值仅限条件语境封闭集（if/elif 条件、逻辑运算操作"
                   "数、comptime 默认值）；不能绑定变量")
        inc.env[target.id] = UNKNOWN
        return
    if isinstance(source, PendingConst):
        kind = "具名常量 inf 需要明确的浮点 dtype 语境" if source.tag == "inf" \
            else "float 字面量需要明确的浮点 dtype 语境"
        inc.reject(target, "E0606", "const-binding",
                   f"裸绑定无 dtype 语境不可定型（{kind}）；经 tis.full 构造"
                   "或与已定型值运算定型")
        inc.env[target.id] = UNKNOWN
        return
    inc.env[target.id] = source
    if source == ScalarType(COMPTIME_INT):
        value = constants.fold_value(stmt.value, inc)
        if value is not None:
            inc.values[target.id] = value


def _rebind(target, source, bound, stmt, inc, rejections, n0, merged):
    """再绑定（目标已确定类型）：常量形态源按 R4 位置③裁决；算术源等价
    判定——不等价构造 E0303 并择一移除本赋值产生的 E06xx（R7 同位置双
    命中只报 E0303），等价则保留 E06xx（单命中各照报）。

    env 语义：裁决按值的实际流向（spec 操作数家族条款以值为根基）——
    让渡/等价路径改写 env 为源类型（非算术源让渡是「不拒」而非冻结旧
    绑定，下游检查不得按陈旧绑定虚构违规）；拒绝路径保旧绑定（绑定未
    成立，report-only 无代码生成）或定型到已绑定 dtype（PendingConst）。
    """
    if source is BOOL:
        inc.reject(target, "E0605", "bool-closure",
                   "布尔值仅限条件语境封闭集；不能绑定变量")
        inc.env[target.id] = UNKNOWN
        return merged
    if isinstance(source, PendingConst):
        constants.pending_rebinding_verdict(source, bound, target, inc)
        inc.env[target.id] = bound  # 按已绑定 dtype 定型（拒绝路径同保旧）
        return merged
    if (isinstance(bound, ScalarType) and bound.kind in FLOAT_DTYPES
            and isinstance(source, ScalarType)
            and source.kind in (COMPTIME_INT, INT)):
        inc.reject(target, "E0606", "const-binding",
                   "int 家族常量/标量不能绑定 dtype 标量变量（int 编译期"
                   "常量绑定限 tis.full value 原语参数位）；dtype 标量变量"
                   "再绑定须同 dtype 标量，或 float 字面量按已绑定 dtype 定型")
        return merged
    if merged or not _is_arith_source(stmt.value):
        inc.env[target.id] = source  # 非算术源再绑定：M1 让渡（residual 1）
        return merged
    if treats_as_unknown(source):
        inc.env[target.id] = source  # 运算让渡（如一侧 UNKNOWN）：无等价结论
        return merged
    if not _compatible(source, bound):
        del rejections[n0:]
        inc.reject_e0303(target, source, bound)
        inc.env[target.id] = UNKNOWN
        return True
    inc.env[target.id] = source  # 等价：算术结果类型（与已绑定等价）
    return merged


def _ann_assign(stmt, inc):
    """带注解局部绑定：注解即 dtype 语境——PendingConst 按注解 dtype 裁决
    （浮点注解定型和 f8e4m3 无穷缺失）、BOOL 封闭集；绑定等价归类型段。"""
    if not isinstance(stmt.target, ast.Name):
        return
    annotation, rejection = parse_annotation(stmt.annotation)
    declared = UNKNOWN if rejection is not None else annotation
    if stmt.value is not None:
        source = inc.infer(stmt.value)
        if source is BOOL:
            inc.reject(stmt.value, "E0605", "bool-closure",
                       "布尔值仅限条件语境封闭集；不能绑定变量")
        elif isinstance(source, PendingConst):
            constants.pending_rebinding_verdict(source, declared,
                                                stmt.target, inc)
    inc.env[stmt.target.id] = declared


def _aug_assign(stmt, inc, rejections):
    """增强算术赋值：等价展开裁决（R1——`S *= scale` 同 `S = S * scale`）。
    运算违规报 E06xx；展开后再绑定不等价构造 E0303 并择一（R7 同位置双
    命中只报 E0303）。"""
    n0 = len(rejections)
    if not isinstance(stmt.target, ast.Name):
        inc.infer(stmt.value)  # 非名目标：值表达式数值域检查仍承载
        return  # Subscript/Attribute 目标：M1 无承载（执行段形态域）
    bound = inc.env.get(stmt.target.id)
    if bound is None or treats_as_unknown(bound):
        inc.infer(stmt.value)  # 无绑定基线：仅运算面（值侧嵌套检查）
        return
    # 有基线路径：arith_operands 内部对 stmt.value 的单次推断承载全部数值
    # 域检查（嵌套算术/原语调用/BOOL 逃逸）——外部不得再预推断（同节点
    # 双重推断会产生同位置同码重复条目，违反 R7 每码恰一条）。值表达式
    # 开帧：非 infer 入口同样承载 R7 同表达式同码合并（300 越界与整型
    # dtype 语境的 E0606 合并为一条）。
    result = inc.framed(
        stmt.value,
        lambda: binary_ops.arith_operands(bound, stmt.value, stmt.op,
                                          stmt, inc))
    if treats_as_unknown(result):
        return
    if not _compatible(result, bound):
        del rejections[n0:]
        inc.reject_e0303(stmt.target, result, bound)


def _is_arith_source(node) -> bool:
    """算术赋值源（design D2 承载面）：BinOp 算术四则或一元负。"""
    if isinstance(node, ast.BinOp):
        return type(node.op).__name__ in ("Add", "Sub", "Mult", "Div")
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)


def _check_test(test, inc):
    """if/elif/while 条件：须布尔值（R3——本语言不支持真值语义）。"""
    t = inc.infer(test)
    if t is BOOL or treats_as_unknown(t):
        return
    inc.reject(test, "E0605", "bool-closure",
               "条件必须为布尔值（本语言不支持真值语义/truthiness）；"
               "显式比较（如 seq_len > 0）产生布尔值")


def _bind_loop_target(target, iter_node, inc):
    """for 循环变量：range(...) → int（数值段增强）；tile_iter/其他 →
    UNKNOWN（元素类型让渡，与类型段同构）。"""
    t = UNKNOWN
    if (isinstance(iter_node, ast.Call)
            and isinstance(iter_node.func, ast.Name)
            and iter_node.func.id == "range"):
        t = ScalarType(INT)
    _bind(target, t, inc)


def _bind(target, t, inc):
    if isinstance(target, ast.Name):
        inc.env[target.id] = t
    elif isinstance(target, ast.Tuple):
        for element in target.elts:
            _bind(element, t, inc)


def _bind_unknown(node, inc):
    """with 绑定名（warp_group 的 g 等）→ UNKNOWN（与类型段同构）。"""
    if isinstance(node, ast.Name):
        inc.env[node.id] = UNKNOWN
    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            _bind_unknown(element, inc)
    elif isinstance(node, ast.Starred):
        _bind_unknown(node.value, inc)


def _walk_nested(fn, inc, rejections):
    """嵌套 produce/consume：体作用域 = 外层快照 + 自身形参（design D3）；
    遍历后恢复外层 env 与值表（函数体绑定不泄漏）。"""
    env_snapshot = dict(inc.env)
    values_snapshot = dict(inc.values)
    for arg in symbols._params(fn):
        inc.env[arg.arg] = symbols._param_type(arg.annotation,
                                               inc.comptime_syms, inc.registry)
    returns = None
    if fn.returns is not None and not _is_none_node(fn.returns):
        parsed, rejection = parse_annotation(fn.returns,
                                             comptime_syms=inc.comptime_syms,
                                             registry=inc.registry)
        returns = None if rejection is not None else parsed
    _walk(fn.body, inc, returns, rejections)
    inc.env.clear()
    inc.env.update(env_snapshot)
    inc.values.clear()
    inc.values.update(values_snapshot)


def _is_none_node(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None
