"""执行结构段检查管线（execution/pipeline-structure 实现面，E0501–E0506）。

编排（design D1/D9）：E0506 模块级扫描（entry_hal，段内顺位最后）→
comptime/registry 复跑（与原语段同输入、零拒绝——段间短路下的确定性
重放）→ per-kernel 单遍结构扫描（KernelScanner：形态环境 D3 + Pipeline
实例表 D4 + warp_group 语境跟踪 D6）→ frontend.report.finalize 收口。
纯结构扫描，不重放类型推断（design D2——判定点只查形态环境）。
"""

import ast

from ..frontend import report
from ..frontend.top_level import _has_tis_decorator
from ..primitives.compute_ops import COMPUTE_PRIMS
from ..typecheck import state_fields, symbols
from . import builtins, entry_hal, pipeline_ops, warp_group
from . import shape_env

STAGE_NAME = "execution-structure"

# 段内 order（design D4，与 spec 段内 tiebreak 一致）：
# E0501=1 → E0502=2 → E0503=3 → E0504=4 → E0505=5 → E0506=6。
ORDERS = {"E0501": 1, "E0502": 2, "E0503": 3, "E0504": 4, "E0505": 5, "E0506": 6}


def _reject(node, code, category, suggestion) -> "report.Rejection":
    """E05xx Rejection 构造（段内 order 与 stage 统一收口）。"""
    return report.Rejection(
        code=code, line=node.lineno, col=node.col_offset + 1,
        category=category, suggestion=suggestion,
        order=ORDERS[code], stage=STAGE_NAME)


def check_module(tree: ast.Module, target: str) -> "list[report.Rejection]":
    """对 AST 运行执行结构段检查，返回排序去重后的拒绝清单。"""
    rejections: "list[report.Rejection]" = list(
        entry_hal.scan(tree, target, _reject))
    comptime = symbols.kernel_comptime_names(tree)
    registry, _ = state_fields.check(tree, comptime_syms=comptime)
    for node in tree.body:
        # AsyncFunctionDef 与 primitives.traverse 同口径（语法段对 async
        # def 全拒——E0103/E0105，管线下永不达本段；此处防御性同扫）。
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and _has_tis_decorator(node, "kernel"):
            scanner = KernelScanner(registry, comptime)
            scanner.scan_kernel(node)
            rejections.extend(scanner.rejections)
    return report.finalize(rejections)


class KernelScanner:
    """per-kernel 单遍结构扫描（design D2/D4/D6）。

    形态环境按语句序构建；嵌套 produce/consume 快照恢复（与 bindings/
    traverse 同构）；warp_group 语境（体外 None / 体内 producer|consumer）
    与 with 绑定名随 With 语句进出置位。
    """

    def __init__(self, registry, comptime):
        self.registry = registry
        self.comptime = comptime
        self.rejections: "list[report.Rejection]" = []
        self.env: "dict[str, object]" = {}       # 名 → 形态（shape_env）
        self.pipe: "pipeline_ops.PipelineInstance | None" = None  # M1 单实例
        self.nested_names: "dict[str, str]" = {}  # 嵌套函数名 → produce/consume
        self.warp_role = None                     # E0501 语境（None=体外）
        self.warp_lines: "list[int]" = []         # warp_group with 行号（源码序）
        self.warp_bindings: "dict[str, int]" = {}  # with 绑定名 → with 行号
        self.in_nested = None                     # None=kernel 顶层
        self.in_with = 0                          # with 体深度（0=非 with 体内）

    # ---- 编排 -------------------------------------------------------------

    def scan_kernel(self, fn):
        for arg in shape_env.params_of(fn):
            self.env[arg.arg] = shape_env.from_annotation(arg.annotation,
                                                          self.comptime)
        self.walk_stmts(fn.body)
        pipeline_ops.finish_kernel(self)

    def reject(self, node, code, category, suggestion):
        self.rejections.append(_reject(node, code, category, suggestion))

    # ---- 语句遍历（与 typecheck.bindings / primitives.traverse 同构）-------

    def walk_stmts(self, stmts):
        for stmt in stmts:
            self.walk_stmt(stmt)

    def walk_stmt(self, stmt):
        if isinstance(stmt, ast.Assign):
            shape = self._assign_value(stmt.value)
            for target in stmt.targets:
                self._bind_target(target, shape)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self.scan_expr(stmt.value)
            if isinstance(stmt.target, ast.Name):
                self.env[stmt.target.id] = shape_env.from_annotation(
                    stmt.annotation, self.comptime, self.registry)
        elif isinstance(stmt, ast.AugAssign):
            self.scan_expr(stmt.target)   # 读位值封闭（+= 读且写）
            self.scan_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self.scan_expr(stmt.value)
        elif isinstance(stmt, ast.Expr):
            if not warp_group.sync_statement(stmt.value, self):
                self.scan_expr(stmt.value)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._for_statement(stmt)
        elif isinstance(stmt, (ast.While, ast.If)):
            self.scan_expr(stmt.test)
            self.walk_stmts(stmt.body)
            self.walk_stmts(stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            self._with_statement(stmt)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody,
                          [s for h in stmt.handlers for s in h.body]
                          if stmt.handlers else []):
                self.walk_stmts(block)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pipeline_ops.enter_nested(stmt, self)
        # Pass/Break/Continue：无动作。

    def _assign_value(self, value):
        """赋值源形态（D3 来源 2）；range 绑定为合法首绑定位。"""
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id == "range"):
            return builtins.check_range(value, self)
        return self.scan_expr(value)

    def _bind_target(self, target, shape):
        """首绑定记录（单类型不变量同构：已绑定不改写）。"""
        if isinstance(target, ast.Name):
            if target.id not in self.env:
                self.env[target.id] = shape
            if shape == shape_env.PIPELINE:
                pipeline_ops.bind_name(self, target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element, shape_env.UNKNOWN)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, shape_env.UNKNOWN)
        elif isinstance(target, (ast.Subscript, ast.Attribute)):
            self.scan_expr(target)  # 目标侧属性访问检查承载

    def _for_statement(self, stmt):
        if (isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name)
                and stmt.iter.func.id == "range"):
            builtins.check_range(stmt.iter, self)  # for 可迭代：合法位
        else:
            self.scan_expr(stmt.iter)
        self._bind_target(stmt.target, shape_env.INT)  # 循环变量 → int
        self.walk_stmts(stmt.body)
        self.walk_stmts(stmt.orelse)

    def _with_statement(self, stmt):
        roles = []
        for item in stmt.items:
            if warp_group.is_warp_group_call(item.context_expr):
                roles.append(warp_group.check_with_item(item, stmt, self))
            else:
                self.scan_expr(item.context_expr)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, shape_env.UNKNOWN)
        outer = self.warp_role
        if roles:
            self.warp_role = roles[-1]  # 多 item 取最后语境（M1 单 item）
        self.in_with += 1   # with 体内非 kernel 顶层（E0504 sync 位置判定）
        self.walk_stmts(stmt.body)
        self.in_with -= 1
        self.warp_role = outer

    # ---- 表达式扫描（形态推断 + 值位置封闭 + tis.* 域分派）-------------------

    def scan_expr(self, node) -> object:
        """表达式结构扫描；返回形态（shape_env 枚举或 StateShape）。"""
        if isinstance(node, ast.Name):
            shape = self.env.get(node.id, shape_env.UNKNOWN)
            self._name_use_check(node, shape)
            return shape
        if isinstance(node, ast.Constant):
            return (shape_env.COMPTIME_INT if type(node.value) is int
                    else shape_env.UNKNOWN)
        if isinstance(node, ast.Attribute):
            return self._attribute(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        # 其余表达式（算术/比较/下标/容器等）：递归子表达式，形态 UNKNOWN。
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self.scan_expr(child)
        return shape_env.UNKNOWN

    def _name_use_check(self, node, shape):
        """值位置封闭（PIPELINE/RANGE_VALUE/BUFFER_PARAM/WARP_BINDING/嵌套函数名）。

        合法位（装饰器接收者 / run 调用 / for 可迭代 / buf 属性接收者 /
        sync 实参）均经特判路径到达，不走本 Name 检查——到达即值使用违规。
        """
        if shape == shape_env.PIPELINE:
            self.reject(node, "E0502", "pipeline-value-escape",
                        "Pipeline 值的合法使用位置为封闭集：produce/consume "
                        "装饰器接收者、run 方法调用接收者与首次赋值绑定；"
                        "不能作为原语实参、参与运算或条件判断。")
        elif shape == shape_env.RANGE_VALUE:
            self.reject(node, "E0502", "range-value-escape",
                        "range 可迭代值的合法使用位置为封闭集：for 语句可"
                        "迭代表达式与 pipe.run 第一实参；不能参与运算、作为"
                        "原语实参、返回或再绑定。")
        elif node.id in self.nested_names:
            self.reject(node, "E0502", "nested-fn-value",
                        "produce/consume 嵌套函数的唯一执行入口是 pipe.run "
                        "的迭代展开；名称不能作为值使用（赋值源/实参/返回值/"
                        "显式调用）。")
        elif shape == shape_env.BUFFER_PARAM:
            self.reject(node, "E0503", "buffer-escape",
                        "buffer 形参仅限所在嵌套函数体内作属性访问的接收者"
                        "（buf.<名>）；整对象不能作为赋值源、调用实参或"
                        "返回值。")
        elif shape == shape_env.WARP_BINDING:
            self.reject(node, "E0504", "warp-binding-escape",
                        "with 绑定名（as g）的合法使用位置仅为 "
                        "tis.warp_group_sync 的实参位。")

    def _attribute(self, node):
        value = node.value
        if (isinstance(value, ast.Name)
                and self.env.get(value.id) == shape_env.BUFFER_PARAM):
            return pipeline_ops.buf_attribute(node, self)  # E0503 键集
        if node.attr in ("run", "produce", "consume"):
            # 到达本路径 = 非装饰器（check 装饰特判）且非调用接收者（_call
            # 特判）——run/produce/consume 成员在非法位置被访问。
            recv = (self.env.get(value.id, shape_env.UNKNOWN)
                    if isinstance(value, ast.Name) else self.scan_expr(value))
            pipeline_ops.member_access_check(node, recv, self)
            return shape_env.UNKNOWN
        self.scan_expr(value)
        return shape_env.UNKNOWN

    def _call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "tis":
                return self._tis_call(node)
            if func.attr == "run":
                return pipeline_ops.check_run(node, self)
        elif isinstance(func, ast.Name):
            if func.id == "range":
                return builtins.check_range_use(node, self)
            if func.id in self.registry:  # 状态类构造 → StateShape（D3）
                self._scan_args(node)
                return shape_env.StateShape(func.id)
        self.scan_expr(func)
        self._scan_args(node)
        return shape_env.UNKNOWN

    def _tis_call(self, node):
        """tis.* 域分派（E0501 语境面/E0502 构造与 run/E0504/E0505；其余让渡）。"""
        attr = node.func.attr
        if attr == "Pipeline":
            return pipeline_ops.check_ctor(node, self)
        if attr in ("block_idx", "cdiv"):
            return builtins.check_call(node, self)
        if attr == "warp_group":
            return warp_group.call_outside_with(node, self)
        if attr == "warp_group_sync":
            return warp_group.sync_value_use(node, self)
        shapes = self._scan_args(node)
        if attr == "barrier":
            return warp_group.barrier_context(node, self)
        if attr in COMPUTE_PRIMS:
            return warp_group.compute_prim_context(node, shapes, self)
        return self._plain_prim_shape(attr, shapes)

    def _plain_prim_shape(self, attr, shapes):
        """非本域判定面的 tis.* 形态（D3 来源 2；参数契约归原语段）。"""
        if attr == "alloc_shared":
            return shape_env.SHARED_TENSOR
        if attr == "make_tensor":
            return shape_env.GLOBAL_TENSOR
        if attr in ("zeros", "full"):
            return shape_env.REGISTER_TENSOR
        if attr == "cast" and shapes:
            return shapes[0]  # 转发 x 形态
        return shape_env.UNKNOWN  # load/store/barrier 不产生值；其余让渡

    def _scan_args(self, node) -> "list[object]":
        """实参扫描 → 位置实参形态列表（关键字值同步扫描；单遍不重入）。"""
        shapes = [self.scan_expr(a) for a in node.args]
        for kw in node.keywords:
            self.scan_expr(kw.value)
        return shapes
