"""E0502 Pipeline 构造/签名/run 契约与 E0503 buffer 键集判定面。

本模块承接 KernelScanner 的 Pipeline 域分派（design D4/D5）：构造参数
集与值域（3.x 填）、produce/consume 嵌套函数遍历与签名（4.x 填）、
run 调用契约与生命周期（5.x 填）、buf.<名> 键集与逃逸（6.x 填）。
骨架阶段只维护 Pipeline 实例表与形态环境（登记/快照恢复/合法位特判），
判定分支随对应 task 填充。

合法位特判（不经 scan_expr 的 Name 值封闭）：produce/consume 装饰器
接收者（enter_nested 识别）、run 调用接收者（check_run）、run 第一
实参 range（check_run 转 builtins.check_range）。
"""

import ast

from . import builtins, shape_env


class PipelineInstance:
    """kernel 内 Pipeline 实例登记（M1 单实例封闭，design D4）。"""

    def __init__(self, node):
        self.node = node                  # 构造调用节点（零 run 收尾报文定位）
        self.name = None                  # 绑定名（首绑定后回填）
        self.buffer_keys: "list[str]" = []   # buffers 键登记序（E0503 报文）
        self.produce_line = None          # 已装饰标记（5.2 前置装饰判定）
        self.consume_line = None
        self.consume_state = None         # consume 返回注解形态（nominal 比较）
        self.run_count = 0                # run 单次性判定（5.2）
        self.run_nodes: "list[ast.Call]" = []


def check_ctor(node, scanner):
    """tis.Pipeline(...) 构造（E0502 构造契约）。

    判定序（构造位置 → 参数集封闭 → stages → buffers → 第二实例）；
    全部违规合并为一条报告并列出全部（报告契约：同一调用多实参同类
    违规同码合并）。实参表达式全量扫描在前（嵌套违规不因外层违规
    丢失）。M1 单实例：第二构造拒，实例表保持首构造。
    """
    violations = []   # (category, 文案)；报文 category 取首个
    if scanner.in_nested is not None:
        violations.append(("ctor-nested-position",
                           "构造调用只能出现在 kernel 顶层函数体；不能在 "
                           "produce/consume 嵌套函数体内构造。"))
    for arg in node.args:
        scanner.scan_expr(arg)
    if node.args:
        violations.append(("ctor-positional-arg",
                           f"构造不接受位置实参（实际 {len(node.args)} 个）；"
                           "stages/buffers 必须以关键字传递。"))
    slots = {}
    for kw in node.keywords:
        if kw.arg is None:
            scanner.scan_expr(kw.value)
            violations.append(("ctor-positional-arg",
                               "构造不接受 **kwargs 展开；合法参数集为 "
                               "stages/buffers。"))
            continue
        if kw.arg not in ("stages", "buffers"):
            scanner.scan_expr(kw.value)
            violations.append(("ctor-unknown-keyword",
                               f"未知关键字实参 {kw.arg}；合法参数集为 "
                               "stages/buffers。"))
            continue
        slots[kw.arg] = kw.value
    if "stages" not in slots:
        violations.append(("ctor-missing-arg", "缺少必选关键字实参 stages。"))
    if "buffers" not in slots:
        violations.append(("ctor-missing-arg", "缺少必选关键字实参 buffers。"))
    if "stages" in slots:
        form = shape_env.comptime_int_form(slots["stages"], scanner.comptime)
        if form is None:
            scanner.scan_expr(slots["stages"])
            violations.append(("ctor-stages-form",
                               "stages 必须为 comptime[int]（int 字面量或 "
                               "comptime 常量名）且不小于 1。"))
        elif form[0] == "const" and form[1] < 1:
            violations.append(("ctor-stages-range",
                               f"stages 必须为不小于 1 的 comptime[int]；"
                               f"实际 {form[1]}。"))
    keys = []
    if "buffers" in slots:
        b = slots["buffers"]
        if not isinstance(b, ast.Dict) or not b.keys:
            scanner.scan_expr(b)
            violations.append(("ctor-buffers-form",
                               'buffers 必须为非空字典字面量，如 '
                               'buffers={"K": K_s}。'))
        else:
            seen, dup = set(), []
            for k in b.keys:
                # 键句法（字符串常量/形如标识符）由语法段承载；此处只登记键集。
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.append(k.value)
                    if k.value in seen and k.value not in dup:
                        dup.append(k.value)
                    seen.add(k.value)
            if dup:
                violations.append(("ctor-dup-buffer-key",
                                   "buffers 键必须互不相同；重复键："
                                   + "/".join(dup) + "。"))
            for v in b.values:
                shape = scanner.scan_expr(v)
                if shape_env.buffers_verdict(shape) == "reject":
                    violations.append(("ctor-buffers-scope",
                                       "buffers 值必须为 Shared scope Tensor"
                                       "（tis.alloc_shared 产物）；出现非 "
                                       "Shared 形态值。"))
    if scanner.pipe is not None:
        violations.append(("ctor-second-instance",
                           "每个 kernel 函数体内至多构造一个 Pipeline 实例"
                           "（M1 封闭）；本 kernel 已在更早位置构造。"))
    if violations:
        scanner.reject(node, "E0502", violations[0][0],
                       "".join(text for _, text in violations))
    else:
        scanner.pipe = PipelineInstance(node)
        scanner.pipe.buffer_keys = keys
    return shape_env.PIPELINE


def bind_name(scanner, name):
    """构造结果首绑定回填实例名（M1 单实例；_bind_target 之后调用链使用）。"""
    if scanner.pipe is not None and scanner.pipe.name is None:
        scanner.pipe.name = name


def enter_nested(fn, scanner):
    """kernel 体内嵌套函数遍历（produce/consume 识别 + 快照恢复，design D5）。

    produce/consume 装饰识别按宽形态（Attribute attr ∈ {produce, consume}，
    接收者形态检查随 3.2/4.2 填）；形参按位置绑定（D3：i==0 int、
    i==1 buffer、i>=2 注解归约）；体内遍历后恢复外层环境。签名判定
    （形参数/注解形态/返回注解 nominal）随 4.1 填充。
    """
    role = None
    dec_node = None
    for dec in fn.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr in ("produce", "consume"):
            role, dec_node = dec.attr, dec
            break
    if role is not None:
        recv_is_pipeline = _check_decorator_receiver(dec_node, scanner)
        _check_signature(fn, role, scanner)
    outer_env = dict(scanner.env)
    outer_in_nested = scanner.in_nested
    # 嵌套函数体可见外层名（作用域嵌套，与 primitives/traverse 的
    # kernel_env 拷贝同构）；形参按位置覆盖绑定；体内新绑定不外泄。
    if role is not None:
        scanner.nested_names[fn.name] = role
        for i, arg in enumerate(shape_env.params_of(fn)):
            if i == 0:
                scanner.env[arg.arg] = shape_env.INT
            elif i == 1:
                scanner.env[arg.arg] = shape_env.BUFFER_PARAM
            else:
                scanner.env[arg.arg] = shape_env.from_annotation(
                    arg.annotation, scanner.comptime, scanner.registry)
        if recv_is_pipeline and scanner.pipe is not None:
            if role == "produce":
                if scanner.pipe.produce_line is not None:
                    scanner.reject(fn, "E0502", "dup-decorate",
                                   "每个 Pipeline 实例至多装饰一个 produce "
                                   "嵌套函数与一个 consume 嵌套函数。")
                scanner.pipe.produce_line = fn.lineno
            else:
                if scanner.pipe.consume_line is not None:
                    scanner.reject(fn, "E0502", "dup-decorate",
                                   "每个 Pipeline 实例至多装饰一个 produce "
                                   "嵌套函数与一个 consume 嵌套函数。")
                scanner.pipe.consume_line = fn.lineno
                ret = shape_env.from_annotation(fn.returns, scanner.comptime,
                                                scanner.registry)
                if isinstance(ret, shape_env.StateShape):
                    scanner.pipe.consume_state = ret
        scanner.in_nested = role
    else:
        for arg in shape_env.params_of(fn):
            scanner.env[arg.arg] = shape_env.from_annotation(
                arg.annotation, scanner.comptime, scanner.registry)
        scanner.in_nested = "plain"   # 任意嵌套函数体内均非 kernel 顶层
    scanner.walk_stmts(fn.body)
    scanner.env = outer_env
    scanner.in_nested = outer_in_nested


def check_run(node, scanner):
    """pipe.run(...) 调用（E0502 run 调用契约）。

    run 接收者是 PIPELINE 值的合法使用位；第一实参 range Call 特判转
    builtins.check_range（合法位，避免值封闭误报）；接收者非 Pipeline
    值以 E0502 拒（member-receiver）。参数契约（恰两实参：位置 0
    range + init= 状态类且与 consume 注解同类）与位置契约（仅 kernel
    顶层）违规合并一条。生命周期计数只计顶层调用（体内调用是独立
    位置违规，不入收尾判定）。
    """
    recv = node.func.value
    recv_shape = None
    recv_ok = False
    if isinstance(recv, ast.Name):
        recv_shape = scanner.env.get(recv.id, shape_env.UNKNOWN)
        recv_ok = recv_shape == shape_env.PIPELINE
        if not recv_ok:
            scanner.reject(node.func, "E0502", "member-receiver",
                           "run 属性的合法接收者仅为 Pipeline 值"
                           "（pipe = tis.Pipeline(...) 的绑定名）。")
    else:
        scanner.scan_expr(recv)
        scanner.reject(node.func, "E0502", "member-receiver",
                       "run 属性的合法接收者仅为 Pipeline 值"
                       "（pipe = tis.Pipeline(...) 的绑定名）。")
    if not recv_ok:
        # 接收者违规：调用契约让渡（同调用只报首个）；实参仍扫描。
        # run 第一实参位的 range Call 维持合法位特判（不按值位报逃逸）。
        for i, arg in enumerate(node.args):
            if (i == 0 and isinstance(arg, ast.Call)
                    and isinstance(arg.func, ast.Name) and arg.func.id == "range"):
                builtins.check_range(arg, scanner)
            else:
                scanner.scan_expr(arg)
        for kw in node.keywords:
            scanner.scan_expr(kw.value)
        return shape_env.UNKNOWN

    violations = []
    if scanner.in_nested is not None:
        violations.append(("run-position",
                           "run 调用只能出现在 kernel 顶层函数体。"))
    if len(node.args) != 1:
        violations.append(("run-arity",
                           "run 恰接受两实参：位置 0 为 range(...) 调用，"
                           "init= 为必选关键字实参。"))
    first = node.args[0] if node.args else None
    if first is not None and isinstance(first, ast.Call) \
            and isinstance(first.func, ast.Name) and first.func.id == "range":
        builtins.check_range(first, scanner)   # run 第一实参：合法位
    else:
        violations.append(("run-first-arg",
                           "run 第一实参必须为 range(...) 调用。"))
        for arg in node.args:
            scanner.scan_expr(arg)
    init = None
    init_shape = None
    for kw in node.keywords:
        if kw.arg == "init":
            init = kw.value
        else:
            scanner.scan_expr(kw.value)
            violations.append(("run-arity",
                               f"未知关键字实参 {kw.arg}；run 只接受位置 0 "
                               "range(...) 与 init= 关键字。"))
    if init is None:
        violations.append(("run-init", "init= 必选关键字实参缺失。"))
    else:
        init_shape = scanner.scan_expr(init)
        if not isinstance(init_shape, shape_env.StateShape):
            violations.append(("run-init",
                               "init= 实参必须为状态类类型的值"
                               "（@tis.state 类构造）。"))
        elif (scanner.pipe is not None
              and isinstance(scanner.pipe.consume_state, shape_env.StateShape)
              and init_shape.class_name != scanner.pipe.consume_state.class_name):
            violations.append(("run-init",
                               "init= 状态类必须与 consume 第三形参注解为"
                               "同一状态类；consume 注解 "
                               f"{scanner.pipe.consume_state.class_name}，"
                               f"init 为 {init_shape.class_name}。"))
    if violations:
        scanner.reject(node, "E0502", violations[0][0],
                       "".join(text for _, text in violations))
    if recv_shape == shape_env.PIPELINE and scanner.pipe is not None:
        p = scanner.pipe
        if scanner.in_nested is not None:
            pass   # 位置违规已报；不入顶层生命周期
        else:
            if p.run_count >= 1:
                scanner.reject(node, "E0502", "run-lifecycle",
                               "每个 Pipeline 实例恰调用一次 run；"
                               "第二次及以后的调用被拒绝。")
            else:
                missing = [name for name, line in
                           (("produce", p.produce_line),
                            ("consume", p.consume_line)) if line is None]
                if missing:
                    scanner.reject(node, "E0502", "run-lifecycle",
                                   "run 调用前必须已装饰 produce 与 "
                                   "consume 嵌套函数；尚未装饰："
                                   + "/".join(missing) + "。")
            p.run_count += 1
            p.run_nodes.append(node)
    # run 结果绑定形态 = STATE_CLASS（design D3-2/D4-4——st 字段访问两形态
    # 均让渡，M1 无第二消费位；违规时 init 形态缺失则 UNKNOWN）。
    return init_shape if isinstance(init_shape, shape_env.StateShape) \
        else shape_env.UNKNOWN


def buf_attribute(node, scanner):
    """buf.<名> 属性访问（E0503 键集；→ SHARED_TENSOR）。

    未注册名拒绝，报文按登记序列出已注册名；实例表缺失（构造未
    执行/违规）时让渡。
    """
    if scanner.pipe is not None and node.attr not in scanner.pipe.buffer_keys:
        registered = "/".join(scanner.pipe.buffer_keys) \
            if scanner.pipe.buffer_keys else "（无）"
        scanner.reject(node, "E0503", "buffer-key",
                       f"未注册的 buffer 名 {node.attr}；已注册名："
                       f"{registered}（Pipeline 构造 buffers 字面量的键）。")
    return shape_env.SHARED_TENSOR


def member_access_check(node, recv, scanner):
    """run/produce/consume 成员在值位置被访问（E0502；3.2）。

    到达本路径 = 非装饰器（enter_nested 特判）且非调用接收者（_call
    特判）。接收者非 Pipeline 值 → member-receiver；Pipeline 值成员
    作值取出 → pipeline-value-escape（封闭集外）。
    """
    if recv == shape_env.PIPELINE:
        scanner.reject(node, "E0502", "pipeline-value-escape",
                       "Pipeline 值的合法使用位置为封闭集：produce/consume "
                       "装饰器接收者、run 方法调用接收者与首次赋值绑定；"
                       "run/produce/consume 成员不能作为值取出。")
    else:
        scanner.reject(node, "E0502", "member-receiver",
                       "run/produce/consume 属性的合法接收者仅为 Pipeline "
                       "值（pipe = tis.Pipeline(...) 的绑定名）。")


def _check_decorator_receiver(dec, scanner) -> bool:
    """produce/consume 装饰器接收者形态检查（E0502 member-receiver）。

    合法位特判：装饰器接收者不走 scan_expr 值封闭；仅形态裁决。
    返回接收者是否 Pipeline 值（非 Pipeline 的装饰不登记到实例表）。
    """
    recv = dec.value
    if isinstance(recv, ast.Name):
        shape = scanner.env.get(recv.id, shape_env.UNKNOWN)
    else:
        shape = shape_env.UNKNOWN
        scanner.scan_expr(recv)
    if shape != shape_env.PIPELINE:
        scanner.reject(dec, "E0502", "member-receiver",
                       "produce/consume 装饰器的接收者必须为 Pipeline 值"
                       "（pipe = tis.Pipeline(...) 的绑定名）。")
        return False
    return True


def _check_signature(fn, role, scanner):
    """produce/consume 签名契约（E0502；形参按位置裁决，名称任意）。

    produce 恰两形参（int 注解 + 无注解 buffer）；consume 恰三形参
    （int + 无注解 + 状态类）且返回注解与第三形参同类（nominal）。
    签名面违规合并一条（报告契约同码合并）。
    """
    params = shape_env.params_of(fn)
    violations = []
    if role == "produce":
        if len(params) != 2:
            violations.append(("produce-arity",
                               "produce 嵌套函数必须恰为两形参"
                               "（j: int, buf）；实际 "
                               f"{len(params)} 个。"))
    else:
        if len(params) != 3:
            violations.append(("consume-arity",
                               "consume 嵌套函数必须恰为三形参"
                               "（j: int, buf, st: <状态类>）；实际 "
                               f"{len(params)} 个。"))
    if params:
        first = shape_env.from_annotation(params[0].annotation, scanner.comptime)
        if first != shape_env.INT:
            violations.append(("sig-first-param",
                               "第一形参必须带 int 注解（迭代索引 j）。"))
    if len(params) > 1 and params[1].annotation is not None:
        violations.append(("sig-buffer-param",
                           "第二形参（buffer 形参）必须无注解；键集由 "
                           "Pipeline 构造的 buffers 字面量定义。"))
    state = None
    if role == "consume" and len(params) > 2:
        state = shape_env.from_annotation(params[2].annotation,
                                          scanner.comptime, scanner.registry)
        if not isinstance(state, shape_env.StateShape):
            violations.append(("consume-state-param",
                               "第三形参必须带状态类类型注解（@tis.state 类）。"))
    if role == "consume":
        ret = shape_env.from_annotation(fn.returns, scanner.comptime,
                                        scanner.registry)
        if not isinstance(ret, shape_env.StateShape):
            violations.append(("consume-return",
                               "consume 必须带返回注解，且为状态类类型。"))
        elif isinstance(state, shape_env.StateShape) \
                and ret.class_name != state.class_name:
            violations.append(("consume-return",
                               "返回注解必须与第三形参注解为同一状态类"
                               f"（nominal）；状态形参 {state.class_name}，"
                               f"返回 {ret.class_name}。"))
    if violations:
        scanner.reject(fn, "E0502", violations[0][0],
                       "".join(text for _, text in violations))


def finish_kernel(scanner):
    """kernel 收尾判定：零 run（E0502；报文定位 Pipeline 构造行）。"""
    p = scanner.pipe
    if p is not None and p.run_count == 0:
        scanner.reject(p.node, "E0502", "run-lifecycle",
                       "每个 Pipeline 实例恰调用一次 run；"
                       "构造后未调用 run。")
