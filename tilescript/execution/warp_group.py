"""E0501 warp_group 语境与 E0504 参数契约判定面。

本模块承接 KernelScanner 的 warp_group 域分派（design D6）：with 语境
参数契约（check_with_item）、warp_group_sync 语句位识别与契约
（sync_statement/check_sync）、表达式位置 warp_group 调用
（call_outside_with）、barrier 语境（barrier_context）、六计算原语语境
（compute_prim_context）。warp_role 由 __init__._with_statement 随 With
语句进出置位（None=体外）。

with 绑定名（WARP_BINDING）的合法位特判：仅 sync 两位置实参位
（check_sync/sync_value_use 特判跳过值封闭）；其余值使用经
_name_use_check 以 E0504 拒。
"""

import ast

from . import shape_env

_ROLE_VALUES = ("producer", "consumer")


def is_warp_group_call(node) -> bool:
    """tis.warp_group(...) 调用节点识别（with 语境分派入口）。"""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warp_group"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tis")


def check_with_item(item, stmt, scanner):
    """with tis.warp_group(role=…, warps=…) 语境项（E0504 参数契约）。

    role 必选关键字、字符串值域 producer/consumer；warps 必选关键字、
    comptime[int] 且 ≥1（comptime 名让渡）。登记 with 行号与绑定名
    （sync 配对判定面），绑定名绑 WARP_BINDING。违规合并一条。
    """
    call = item.context_expr
    scanner.warp_lines.append(stmt.lineno)
    violations = []
    for arg in call.args:
        scanner.scan_expr(arg)
    if call.args:
        violations.append(("wg-args",
                           "warp_group 不接受位置实参；role/warps 必须以"
                           "关键字传递。"))
    slots = {}
    for kw in call.keywords:
        if kw.arg is None:
            scanner.scan_expr(kw.value)
            violations.append(("wg-args",
                               "warp_group 不接受 **kwargs 展开。"))
        elif kw.arg in ("role", "warps"):
            slots[kw.arg] = kw.value
        else:
            scanner.scan_expr(kw.value)
            violations.append(("wg-args",
                               f"未知关键字实参 {kw.arg}；warp_group 的合法"
                               "参数集为 role/warps。"))
    role = None
    if "role" not in slots:
        violations.append(("wg-role",
                           '缺少必选关键字实参 role（值域 producer/consumer）。'))
    else:
        v = slots["role"]
        if isinstance(v, ast.Constant) and v.value in _ROLE_VALUES:
            role = v.value
        else:
            scanner.scan_expr(v)
            violations.append(("wg-role",
                               "role 值域为 producer/consumer。"))
    if "warps" not in slots:
        violations.append(("wg-warps",
                           "缺少必选关键字实参 warps（comptime[int] 且"
                           "不小于 1）。"))
    else:
        form = shape_env.comptime_int_form(slots["warps"], scanner.comptime)
        if form is None:
            scanner.scan_expr(slots["warps"])
            violations.append(("wg-warps",
                               "warps 必须为 comptime[int]（int 字面量或 "
                               "comptime 常量名）且不小于 1。"))
        elif form[0] == "const" and form[1] < 1:
            violations.append(("wg-warps",
                               f"warps 必须为不小于 1 的 comptime[int]；"
                               f"实际 {form[1]}。"))
    if violations:
        scanner.reject(stmt, "E0504", violations[0][0],
                       "".join(text for _, text in violations))
    if isinstance(item.optional_vars, ast.Name):
        scanner.warp_bindings[item.optional_vars.id] = stmt.lineno
        scanner.env[item.optional_vars.id] = shape_env.WARP_BINDING
    elif item.optional_vars is not None:
        scanner._bind_target(item.optional_vars, shape_env.UNKNOWN)
    return role


def sync_statement(value, scanner) -> bool:
    """表达式语句位 tis.warp_group_sync 识别（E0504 唯一合法语句形态）。

    True = 识别并处理（check_sync 承载判定）；False = 非本域调用，
    走通用表达式扫描。
    """
    if _is_sync_call(value):
        check_sync(value, scanner)
        return True
    return False


def check_sync(node, scanner):
    """tis.warp_group_sync(g1, g2, barrier_id=c) 契约（E0504）。

    两位置实参为两个不同 warp_group 语句的绑定名；barrier_id 必选
    comptime 非负；表达式语句位 + kernel 顶层 + 两个 with 语句之后。
    违规合并一条；绑定名实参为合法位（跳过值封闭）。
    """
    violations = []
    if scanner.in_nested is not None or scanner.in_with:
        violations.append(("sync-position",
                           "warp_group_sync 只能以表达式语句形态出现在 "
                           "kernel 顶层函数体。"))
    elif len(scanner.warp_lines) < 2 or node.lineno <= scanner.warp_lines[-2]:
        violations.append(("sync-position",
                           "warp_group_sync 必须位于两个 warp_group with "
                           "语句之后（汇合两组线程）。"))
    if len(node.args) != 2:
        violations.append(("sync-binding",
                           "warp_group_sync 恰接受两个位置实参（两个 "
                           "warp_group with 绑定名）。"))
    names = []
    for arg in node.args:
        if isinstance(arg, ast.Name) and arg.id in scanner.warp_bindings:
            names.append(arg.id)
            continue   # 绑定名：sync 实参位（E0504 合法位）
        if isinstance(arg, ast.Name):
            violations.append(("sync-binding",
                               f"实参 {arg.id} 不是 warp_group with 绑定名。"))
        else:
            violations.append(("sync-binding",
                               "实参必须为 warp_group with 绑定名。"))
        scanner.scan_expr(arg)
    if len(names) == 2 and names[0] == names[1]:
        violations.append(("sync-binding",
                           "两个位置实参必须为两个不同 warp_group 语句的"
                           "绑定名；同一绑定不能汇合自身。"))
    bid = None
    for kw in node.keywords:
        if kw.arg == "barrier_id":
            bid = kw.value
        else:
            scanner.scan_expr(kw.value)
            violations.append(("sync-barrier-id",
                               f"未知关键字实参 {kw.arg}；warp_group_sync "
                               "只接受 barrier_id= 关键字。"))
    if bid is None:
        violations.append(("sync-barrier-id",
                           "barrier_id= 必选关键字实参缺失（comptime[int] "
                           "且非负）。"))
    else:
        form = shape_env.comptime_int_form(bid, scanner.comptime)
        if form is None:
            scanner.scan_expr(bid)
            violations.append(("sync-barrier-id",
                               "barrier_id 必须为非负 comptime[int]"
                               "（int 字面量或 comptime 常量名）。"))
        elif form[0] == "const" and form[1] < 0:
            violations.append(("sync-barrier-id",
                               f"barrier_id 必须为非负 comptime[int]；"
                               f"实际 {form[1]}。"))
    if violations:
        scanner.reject(node, "E0504", violations[0][0],
                       "".join(text for _, text in violations))


def call_outside_with(node, scanner):
    """表达式位置的 tis.warp_group(...) 调用（E0504 wg-call-position）。

    with 上下文表达式位经 is_warp_group_call 特判，不达本路径——
    到达即非 with 上下文。实参仍扫描（单遍完整性）。
    """
    for arg in node.args:
        scanner.scan_expr(arg)
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
    scanner.reject(node, "E0504", "wg-call-position",
                   "tis.warp_group(...) 只能作为 with 语句的上下文表达式"
                   "使用（with tis.warp_group(role=…, warps=…) as g:）。")
    return shape_env.UNKNOWN


def sync_value_use(node, scanner):
    """值位置的 tis.warp_group_sync(...) 调用（E0504 sync-value）。"""
    for arg in node.args:
        if isinstance(arg, ast.Name) and arg.id in scanner.warp_bindings:
            continue   # 绑定名合法位；不因调用位置违规而双报
        scanner.scan_expr(arg)
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
    scanner.reject(node, "E0504", "sync-value",
                   "warp_group_sync 不产生值；只能以表达式语句形态使用"
                   "（tis.warp_group_sync(g1, g2, barrier_id=c)）。")
    return shape_env.UNKNOWN


def barrier_context(node, scanner):
    """tis.barrier(...) 调用语境（E0501）。

    barrier(scope=WarpGroup) 只能出现在 warp_group with 体内；顶层
    出现以 E0501 拒（语境违规）。实参已由调用方扫描。
    """
    for kw in node.keywords:
        if (kw.arg == "scope" and isinstance(kw.value, ast.Name)
                and kw.value.id == "WarpGroup" and scanner.warp_role is None):
            scanner.reject(node, "E0501", "barrier-scope",
                           "barrier(scope=WarpGroup) 只能出现在 warp_group "
                           "with 体内；顶层请用默认 Block 作用域"
                           "（tis.barrier()），或移入 warp_group 体内。")
            break
    return shape_env.UNKNOWN


def compute_prim_context(node, shapes, scanner):
    """六计算原语语境（E0501 + D3 形态）。

    producer 角色体内只允许内存原语：计算原语以 E0501 拒。形态：
    dot/reduce/maximum/exp/log → REGISTER_TENSOR；transpose 转发第一
    实参形态。实参已由调用方扫描。
    """
    if scanner.warp_role == "producer":
        scanner.reject(node, "E0501", "wg-compute-prim",
                       "producer 角色体内只允许内存原语（tis.load/"
                       "tis.store/tis.barrier）、普通赋值与语句控制流；"
                       "计算原语请移至 consumer 角色体内。")
    if node.func.attr == "transpose":
        return shapes[0] if shapes else shape_env.UNKNOWN
    return shape_env.REGISTER_TENSOR


def _is_sync_call(node) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warp_group_sync"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tis")
