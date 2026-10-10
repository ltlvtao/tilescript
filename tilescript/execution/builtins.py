"""E0505 索引与整数内建（tis.block_idx / tis.cdiv / range）判定面。

合法位特判：range 的三个合法位（for 可迭代 / 赋值源 / run 第一实参）
均由调用方特判转 check_range；其余表达式位置达 check_range_use
（E0502 range 可迭代值位置封闭）。block_idx/cdiv 由 _tis_call 直接
分派（实参自扫）。种类裁决（_int_kind）：comptime（字面量/comptime
名/两 comptime 的 cdiv 结果）/ int（运行期）/ bad（确定非 int 形态
——Tensor、浮点等）；UNKNOWN 让渡（非本段判定面，类型段承载）。
"""

import ast

from . import shape_env


def _int_kind(expr, scanner):
    """实参标量种类裁决（E0505）。

    返回 ("comptime", v|None)（int 字面量带值；comptime 名/形态值让渡）、
    ("int", None)（运行期 int）、("bad", None)（确定非 int 形态——Tensor、
    浮点/字符串/布尔字面量、状态类等）或 None（UNKNOWN 让渡）。
    """
    if isinstance(expr, ast.Constant):
        if type(expr.value) is int:
            return ("comptime", expr.value)
        return ("bad", None)   # float/str/bool/None 等非 int 字面量
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)) \
            and isinstance(expr.operand, ast.Constant) \
            and type(expr.operand.value) is int:
        v = expr.operand.value
        return ("comptime", -v if isinstance(expr.op, ast.USub) else v)
    shape = scanner.scan_expr(expr)
    if shape == shape_env.COMPTIME_INT:
        return ("comptime", None)
    if shape == shape_env.INT:
        return ("int", None)
    if shape == shape_env.UNKNOWN:
        return None
    return ("bad", None)


def check_call(node, scanner):
    """tis.block_idx(...) / tis.cdiv(...)（E0505 参数契约 + D3 形态）。"""
    if node.func.attr == "block_idx":
        return _check_block_idx(node, scanner)
    return _check_cdiv(node, scanner)


def _check_block_idx(node, scanner):
    """block_idx(dim)：dim 恰一位置实参 comptime[int] 非负；→ 运行期 int。"""
    violations = []
    if len(node.args) != 1:
        for arg in node.args:
            scanner.scan_expr(arg)
        violations.append(("block-idx-dim",
                           "block_idx 恰接受一个位置实参 dim"
                           "（comptime[int] 且非负）。"))
    else:
        kind = _int_kind(node.args[0], scanner)
        if kind is not None:
            if kind[0] == "bad":
                violations.append(("block-idx-dim",
                                   "dim 必须为非负 comptime[int]"
                                   "（int 字面量或 comptime 常量名）。"))
            elif kind[0] == "int":
                violations.append(("block-idx-dim",
                                   "dim 必须为 comptime[int]（int 字面量或 "
                                   "comptime 常量名）；运行期 int 不能作为"
                                   "维号。"))
            elif kind[1] is not None and kind[1] < 0:
                violations.append(("block-idx-dim",
                                   f"dim 必须为非负 comptime[int]；"
                                   f"实际 {kind[1]}。"))
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
        violations.append(("block-idx-dim",
                           "block_idx 不接受关键字实参。"))
    if violations:
        scanner.reject(node, "E0505", violations[0][0],
                       "".join(text for _, text in violations))
    return shape_env.INT


def _check_cdiv(node, scanner):
    """cdiv(a, b)：恰两位置实参 int/comptime 种类；b 零值拒。

    结果种类：两实参均 comptime（或让渡时不保守判 comptime——任一
    让渡按 int 处理）→ comptime[int]，否则 int。
    """
    violations = []
    kinds = []
    if len(node.args) != 2:
        for arg in node.args:
            kinds.append(_int_kind(arg, scanner))
        violations.append(("cdiv-args",
                           "cdiv 恰接受两个位置实参"
                           "（int 或 comptime[int] 种类的标量）。"))
    else:
        for arg in node.args:
            kinds.append(_int_kind(arg, scanner))
    if len(kinds) == 2:
        if kinds[0] == ("bad", None) or kinds[1] == ("bad", None):
            violations.append(("cdiv-args",
                               "cdiv 实参必须为 int 或 comptime[int] 种类的"
                               "标量；不能是 Tensor、浮点或其他形态值。"))
        if kinds[1] is not None and kinds[1][0] == "comptime" \
                and kinds[1][1] == 0:
            violations.append(("cdiv-args",
                               "除数 b 不能为 comptime[int] 值 0。"))
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
        violations.append(("cdiv-args", "cdiv 不接受关键字实参。"))
    if violations:
        scanner.reject(node, "E0505", violations[0][0],
                       "".join(text for _, text in violations))
    both_comptime = (len(kinds) == 2 and all(k is not None and
                                             k[0] == "comptime"
                                             for k in kinds))
    if both_comptime:
        return shape_env.COMPTIME_INT   # cdiv 两 comptime → comptime（D3）
    return shape_env.INT


def check_range(node, scanner):
    """range(...) 合法位（for 可迭代 / 赋值源 / run 第一实参；→ RANGE_VALUE）。

    恰一位置实参 int/comptime 种类（E0505）。
    """
    violations = []
    if len(node.args) != 1:
        for arg in node.args:
            scanner.scan_expr(arg)
        violations.append(("range-args",
                           "range 恰接受一个位置实参"
                           "（int 或 comptime[int] 种类）。"))
    else:
        kind = _int_kind(node.args[0], scanner)
        if kind == ("bad", None):
            violations.append(("range-args",
                               "range 实参必须为 int 或 comptime[int] 种类的"
                               "标量；不能是 Tensor、浮点或其他形态值。"))
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
        violations.append(("range-args", "range 不接受关键字实参。"))
    if violations:
        scanner.reject(node, "E0505", violations[0][0],
                       "".join(text for _, text in violations))
    return shape_env.RANGE_VALUE


def check_range_use(node, scanner):
    """range(...) 非合法位调用（E0502 range 可迭代值位置封闭）。

    到达本路径 = 非 for 可迭代、非赋值源、非 run 第一实参（三合法位
    均特判转 check_range）——位置封闭违规；参数契约让渡（调用本身
    非法，E0505 无独立报告意义），实参仍扫描（单遍完整性）。
    """
    for arg in node.args:
        scanner.scan_expr(arg)
    for kw in node.keywords:
        scanner.scan_expr(kw.value)
    scanner.reject(node, "E0502", "range-value-escape",
                   "range 可迭代值的合法使用位置为封闭集：for 语句可迭代"
                   "表达式与 pipe.run 第一实参；不能参与运算、作为原语"
                   "实参、返回或再绑定。")
    return shape_env.RANGE_VALUE
