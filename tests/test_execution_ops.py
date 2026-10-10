"""执行结构段 E0502/E0503/E0505 面测试（execution/pipeline-structure 实现面）。

公共行为断言：check_module(tree, target) 返回排序去重后的 Rejection 清单
（stage="execution-structure"、order=段内序 1–6）。合法用例取 FLASH 结构
子集；每例直接构建源文本 → ast.parse → check_module。
"""

import ast

from tilescript.execution import check_module as execution_check

# FLASH 结构子集（§7）：合法 Pipeline 全链最小形态。
_FLASH_LIKE = '''\
import tis

@tis.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]

@tis.kernel
def flash(K_ptr: Pointer[f16, Global], seq_len: int,
          D: comptime[int] = 64, BR: comptime[int] = 64,
          BC: comptime[int] = 64, STAGES: comptime[int] = 2):
    K_s = tis.alloc_shared((BC, D), f16)
    pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s})

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K[j * BC:(j + 1) * BC, :], buf.K, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        return AttnState(O_acc=tis.zeros((BR, D), f32, Register))

    st = pipe.run(range(tis.cdiv(seq_len, BC)),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))
    return
'''


def _check(source: str, target: str = "nvidia_h200"):
    return execution_check(ast.parse(source), target)


def _codes(rejections):
    return [r.code for r in rejections]


def _single(rejections, code, category=None):
    """断言恰一条拒绝并返回它（公共四要素面）。"""
    assert len(rejections) == 1, [r.to_dict() for r in rejections]
    r = rejections[0]
    assert r.code == code
    assert r.stage == "execution-structure"
    if category is not None:
        assert r.category == category
    return r


class TestSkeleton:
    def test_legal_flash_fragment_zero_rejections(self):
        """骨架用例（tasks 2.2）：合法 FLASH 片段结构扫描零拒绝。"""
        assert _check(_FLASH_LIKE) == []

    def test_scan_does_not_touch_existing_stages(self):
        """同输入三段结果不变（结构扫描只读 AST，不触既有段检查）。"""
        from tilescript import primitives, typecheck
        from tilescript.frontend import check_module as syntax_check
        tree = ast.parse(_FLASH_LIKE)
        assert syntax_check(_FLASH_LIKE) == []
        assert typecheck.check_module(tree) == []
        assert primitives.check_module(tree, "nvidia_h200") == []
        assert execution_check(tree, "nvidia_h200") == []


# 构造面测试模板（tasks 3.1）：__PRELUDE__（可选语句）/ __CTOR__（构造实参）。
# 合法全链基线（构造 + produce/consume + run）——收尾零 run 判定（5.2）后
# 无 run 模板不再零拒绝；构造违规例因实例不登记而无收尾报文（面隔离）。
_CTOR_BASE = '''\
import tis

@tis.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]

@tis.kernel
def flash(K_ptr: Pointer[f16, Global], seq_len: int,
          D: comptime[int] = 64, BR: comptime[int] = 64,
          STAGES: comptime[int] = 2):
__PRELUDE__
    K_s = tis.alloc_shared((BR, D), f16)
    pipe = tis.Pipeline(__CTOR__)

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K_s, buf.K, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        return st

    st = pipe.run(range(seq_len),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))
    return
'''


class TestPipelineCtor:
    """E0502 Pipeline 构造参数集与值域（R「Pipeline 构造契约」Scenario 1/2/3）。"""

    def _ctor(self, ctor, prelude=""):
        source = _CTOR_BASE.replace("__PRELUDE__", prelude).replace("__CTOR__", ctor)
        return _check(source)

    def test_legal_ctor_accepted(self):
        """合法构造（stages comptime 名 + buffers Shared Tensor）零拒绝。"""
        assert self._ctor('stages=STAGES, buffers={"K": K_s}') == []

    def test_stages_zero_rejected(self):
        r = _single(self._ctor('stages=0, buffers={"K": K_s}'), "E0502")
        assert r.category == "ctor-stages-range"
        assert "不小于 1" in r.suggestion

    def test_stages_runtime_name_rejected(self):
        r = _single(self._ctor('stages=seq_len, buffers={"K": K_s}'), "E0502")
        assert r.category == "ctor-stages-form"
        assert "comptime" in r.suggestion

    def test_stages_missing_rejected(self):
        r = _single(self._ctor('buffers={"K": K_s}'), "E0502")
        assert r.category == "ctor-missing-arg"
        assert "stages" in r.suggestion

    def test_buffers_missing_rejected(self):
        r = _single(self._ctor('stages=STAGES'), "E0502")
        assert r.category == "ctor-missing-arg"
        assert "buffers" in r.suggestion

    def test_buffers_register_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={"K": r}',
                               prelude="    r = tis.zeros((BR, D), f32, Register)"),
                    "E0502")
        assert r.category == "ctor-buffers-scope"
        assert "alloc_shared" in r.suggestion

    def test_buffers_global_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={"K": Q}',
                               prelude="    Q = tis.make_tensor(K_ptr, (seq_len, D))"),
                    "E0502")
        assert r.category == "ctor-buffers-scope"

    def test_buffers_non_tensor_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={"K": seq_len}'), "E0502")
        assert r.category == "ctor-buffers-scope"

    def test_buffers_empty_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={}'), "E0502")
        assert r.category == "ctor-buffers-form"
        assert "非空" in r.suggestion

    def test_buffers_dup_key_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={"K": K_s, "K": K_s}'),
                    "E0502")
        assert r.category == "ctor-dup-buffer-key"
        assert "互不相同" in r.suggestion

    def test_positional_arg_rejected(self):
        r = _single(self._ctor('2'), "E0502")
        assert r.category == "ctor-positional-arg"

    def test_unknown_keyword_rejected(self):
        r = _single(self._ctor('stages=STAGES, buffers={"K": K_s}, extra=1'), "E0502")
        assert r.category == "ctor-unknown-keyword"
        assert "extra" in r.suggestion

    def test_second_instance_rejected(self):
        source = _CTOR_BASE.replace("__PRELUDE__", "").replace(
            "__CTOR__", 'stages=STAGES, buffers={"K": K_s}'
        ).replace("    pipe = tis.Pipeline",
                  '    pipe2 = tis.Pipeline(stages=STAGES, buffers={"K": K_s})\n'
                  "    pipe = tis.Pipeline")
        r = _single(_check(source), "E0502")
        assert r.category == "ctor-second-instance"
        assert "至多" in r.suggestion

    def test_combined_violations_single_report(self):
        """同码合并（报告契约）：stages 与 buffers 同时违规 → 一条列全部。"""
        r = _single(self._ctor('stages=0, buffers={"K": r}',
                               prelude="    r = tis.zeros((BR, D), f32, Register)"),
                    "E0502")
        assert "stages" in r.suggestion
        assert "buffers" in r.suggestion


class TestPipelineValue:
    """E0502 构造位置与 Pipeline 值使用封闭（R「Pipeline 构造契约」Scenario 4）。"""

    def _with_pipe(self, extra):
        """合法全链基线 + kernel 尾部注入语句（锚定末尾 return）。"""
        base = _CTOR_BASE.replace("__PRELUDE__", "").replace(
            "__CTOR__", 'stages=STAGES, buffers={"K": K_s}')
        i = base.rindex("    return")
        return _check(base[:i] + extra + "\n" + base[i:])

    def test_ctor_in_nested_rejected(self):
        base = _CTOR_BASE.replace("__PRELUDE__", "").replace(
            "__CTOR__", 'stages=STAGES, buffers={"K": K_s}')
        source = base.replace(
            "        tis.load(K_s, buf.K, mode=Async)",
            '        tis.load(K_s, buf.K, mode=Async)\n'
            '        p2 = tis.Pipeline(stages=STAGES, buffers={"K": K_s})')
        r = _single(_check(source), "E0502")
        assert r.category == "ctor-nested-position"

    def test_pipeline_as_load_arg_rejected(self):
        r = _single(self._with_pipe("    tis.load(K_s, pipe)"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_pipeline_arithmetic_rejected(self):
        r = _single(self._with_pipe("    y = pipe + 1"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_pipeline_in_condition_rejected(self):
        r = _single(self._with_pipe("    if pipe:\n        pass"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_pipeline_return_rejected(self):
        r = _single(self._with_pipe("    return pipe"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_pipeline_rebind_rejected(self):
        r = _single(self._with_pipe("    pipe2 = pipe"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_run_on_non_pipeline_rejected(self):
        r = _single(self._with_pipe("    K_s.run()"), "E0502")
        assert r.category == "member-receiver"

    def test_produce_member_as_value_rejected(self):
        r = _single(self._with_pipe("    x = pipe.produce"), "E0502")
        assert r.category == "pipeline-value-escape"

    def test_receiver_violation_defers_range_legal_position(self):
        """接收者违规早退路径：run 第一实参 range 合法位维持（cycle 1 minor
        ——range 不被按值位报 range-value-escape）。"""
        source = _SIG_BASE.replace("pipe.run(range(seq_len),", "K_s.run(range(seq_len),")
        rs = _check(source)
        assert [r.category for r in rs] == ["run-lifecycle", "member-receiver"]
        assert not any(r.category == "range-value-escape" for r in rs)

    def test_produce_decorator_on_non_pipeline_rejected(self):
        base = _CTOR_BASE.replace("__PRELUDE__", "").replace(
            "__CTOR__", 'stages=STAGES, buffers={"K": K_s}')
        i = base.rindex("    return")
        source = base[:i] + '''\
    @K_s.produce
    def fetch2(j: int, buf):
        pass

    return
'''
        r = _single(_check(source), "E0502")
        assert r.category == "member-receiver"
        assert "接收者" in r.suggestion


# 签名面测试模板（tasks 4.1）：合法 produce/consume + run 全链基线。
_SIG_BASE = '''\
import tis

@tis.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]

@tis.state
class OtherState:
    x: Tensor[f32, (BR,), Register]

@tis.kernel
def flash(K_ptr: Pointer[f16, Global], seq_len: int,
          D: comptime[int] = 64, BR: comptime[int] = 64,
          STAGES: comptime[int] = 2):
    K_s = tis.alloc_shared((BR, D), f16)
    pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s})

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K_s, buf.K, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        return st

    st = pipe.run(range(seq_len),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))
    return
'''


class TestProduceConsumeSig:
    """E0502 produce/consume 签名契约（R「produce 与 consume 嵌套函数契约」）。"""

    def _sig(self, old, new):
        assert old in _SIG_BASE
        return _check(_SIG_BASE.replace(old, new))

    def test_legal_signatures_accepted(self):
        assert _check(_SIG_BASE) == []

    def test_consume_missing_return_rejected(self):
        r = _single(self._sig("def attend(j: int, buf, st: AttnState) -> AttnState:",
                              "def attend(j: int, buf, st: AttnState):"),
                    "E0502")
        assert r.category == "consume-return"
        assert "返回注解" in r.suggestion

    def test_consume_return_other_class_rejected(self):
        # init 同步换 OtherState 构造（状态链一致），隔离签名面单报。
        source = _SIG_BASE.replace("-> AttnState:", "-> OtherState:").replace(
            "init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))",
            "init=OtherState(x=tis.zeros((BR,), f32, Register)))")
        r = _single(_check(source), "E0502")
        assert r.category == "consume-return"
        assert "同一状态类" in r.suggestion

    def test_consume_return_non_state_rejected(self):
        r = _single(self._sig("-> AttnState:", "-> int:"), "E0502")
        assert r.category == "consume-return"

    def test_consume_two_params_rejected(self):
        r = _single(self._sig("def attend(j: int, buf, st: AttnState)",
                              "def attend(j: int, buf)"), "E0502")
        assert r.category == "consume-arity"

    def test_consume_third_param_not_state_rejected(self):
        r = _single(self._sig("st: AttnState", "st: int"), "E0502")
        assert r.category == "consume-state-param"

    def test_consume_second_param_annotated_rejected(self):
        r = _single(self._sig(
            "def attend(j: int, buf, st: AttnState)",
            "def attend(j: int, buf: Tensor[f16, (BR, D), Shared], st: AttnState)"),
            "E0502")
        assert r.category == "sig-buffer-param"

    def test_produce_three_params_rejected(self):
        r = _single(self._sig("def fetch(j: int, buf):",
                              "def fetch(j: int, buf, st: AttnState):"), "E0502")
        assert r.category == "produce-arity"

    def test_produce_one_param_rejected(self):
        r = _single(self._sig("def fetch(j: int, buf):", "def fetch(j: int):"),
                    "E0502")
        assert r.category == "produce-arity"

    def test_produce_first_param_unannotated_rejected(self):
        r = _single(self._sig("def fetch(j: int, buf):", "def fetch(j, buf):"),
                    "E0502")
        assert r.category == "sig-first-param"

    def test_produce_first_param_f32_rejected(self):
        r = _single(self._sig("def fetch(j: int, buf):", "def fetch(j: f32, buf):"),
                    "E0502")
        assert r.category == "sig-first-param"

    def test_produce_buffer_param_annotated_rejected(self):
        r = _single(self._sig(
            "def fetch(j: int, buf):",
            "def fetch(j: int, buf: Tensor[f16, (BR, D), Shared]):"), "E0502")
        assert r.category == "sig-buffer-param"


class TestNestedFnClosure:
    """E0502 调用封闭（R Scenario 4/5：重复装饰/显式调用/名作值）。"""

    def _tail(self, extra):
        """kernel 尾部注入语句（锚定文件末尾 return，不触体内 return st）。"""
        i = _SIG_BASE.rindex("    return")
        return _check(_SIG_BASE[:i] + extra + "\n" + _SIG_BASE[i:])

    def test_dup_produce_rejected(self):
        source = _SIG_BASE[:_SIG_BASE.rindex("    return")] + '''\
    @pipe.produce
    def fetch2(j: int, buf):
        pass

    return
'''
        r = _single(_check(source), "E0502")
        assert r.category == "dup-decorate"
        assert "至多" in r.suggestion

    def test_dup_consume_rejected(self):
        source = _SIG_BASE[:_SIG_BASE.rindex("    return")] + '''\
    @pipe.consume
    def attend2(j: int, buf, st: AttnState) -> AttnState:
        return st

    return
'''
        r = _single(_check(source), "E0502")
        assert r.category == "dup-decorate"

    def test_explicit_call_rejected(self):
        r = _single(self._tail("    fetch(0, K_s)"), "E0502")
        assert r.category == "nested-fn-value"
        assert "pipe.run" in r.suggestion

    def test_name_as_assign_source_rejected(self):
        r = _single(self._tail("    f = fetch"), "E0502")
        assert r.category == "nested-fn-value"

    def test_name_as_arg_rejected(self):
        r = _single(self._tail("    tis.load(K_s, fetch)"), "E0502")
        assert r.category == "nested-fn-value"

    def test_name_as_return_rejected(self):
        r = _single(self._tail("    return fetch"), "E0502")
        assert r.category == "nested-fn-value"


# run 面测试模板（tasks 5.1/5.2）：_SIG_BASE 已是含 run 的合法全链。
_RUN_BASE = _SIG_BASE
_RUN_BLOCK = '''\
    st = pipe.run(range(seq_len),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))
'''


class TestRunContract:
    """E0502 run 参数与位置（R「run 调用契约」Scenario 1/2/3/4）。"""

    def _run(self, old, new):
        assert old in _RUN_BASE
        return _check(_RUN_BASE.replace(old, new))

    def test_legal_run_accepted(self):
        assert _check(_RUN_BASE) == []

    def test_first_arg_not_range_rejected(self):
        r = _single(self._run("pipe.run(range(seq_len),", "pipe.run(K_s,"), "E0502")
        assert r.category == "run-first-arg"

    def test_two_positional_args_rejected(self):
        source = _RUN_BASE.replace(
            "pipe.run(range(seq_len),\n                  init=AttnState("
            "O_acc=tis.zeros((BR, D), f32, Register)))",
            "pipe.run(range(seq_len), AttnState("
            "O_acc=tis.zeros((BR, D), f32, Register)))")
        r = _single(_check(source), "E0502")
        assert r.category == "run-arity"

    def test_init_missing_rejected(self):
        source = _RUN_BASE.replace(
            "pipe.run(range(seq_len),\n                  init=AttnState("
            "O_acc=tis.zeros((BR, D), f32, Register)))",
            "pipe.run(range(seq_len))")
        r = _single(_check(source), "E0502")
        assert r.category == "run-init"
        assert "init" in r.suggestion

    def test_init_not_state_rejected(self):
        r = self._run("init=AttnState(O_acc=tis.zeros((BR, D), f32, Register))",
                      "init=seq_len")
        r = _single(r, "E0502")
        assert r.category == "run-init"
        assert "状态类" in r.suggestion

    def test_init_other_class_rejected(self):
        source = _RUN_BASE.replace(
            "init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))",
            "init=OtherState(x=tis.zeros((BR,), f32, Register)))")
        r = _single(_check(source), "E0502")
        assert r.category == "run-init"
        assert "同一状态类" in r.suggestion

    def test_run_in_nested_rejected(self):
        source = _SIG_BASE.replace(
            "        tis.load(K_s, buf.K, mode=Async)",
            "        tis.load(K_s, buf.K, mode=Async)\n"
            "        pipe.run(range(seq_len),\n"
            "                  init=AttnState("
            "O_acc=tis.zeros((BR, D), f32, Register)))")
        r = _single(_check(source), "E0502")
        assert r.category == "run-position"

    def test_combined_first_and_init_single_report(self):
        """同码合并：第一实参非 range 且 init 缺失 → 一条列全部。"""
        source = _RUN_BASE.replace(
            "pipe.run(range(seq_len),\n                  init=AttnState("
            "O_acc=tis.zeros((BR, D), f32, Register)))",
            "pipe.run(K_s)")
        r = _single(_check(source), "E0502")
        assert "range" in r.suggestion
        assert "init" in r.suggestion


_RUN_LINE = _RUN_BLOCK


class TestRunLifecycle:
    """E0502 run 单次性与前置装饰（R「每实例恰一次/调用前已装饰」）。"""

    def test_zero_run_rejected_at_ctor_line(self):
        """零 run：收尾判定，报文定位 Pipeline 构造行。"""
        source = _SIG_BASE.replace("\n" + _RUN_BLOCK.rstrip() + "\n", "\n")
        r = _single(_check(source), "E0502")
        assert r.category == "run-lifecycle"
        assert r.line == 16   # pipe = tis.Pipeline(...) 行
        assert "未调用" in r.suggestion

    def test_second_run_rejected_at_second_site(self):
        """多 run：第二处起即时报。"""
        i = _SIG_BASE.rindex("    return")
        source = _SIG_BASE[:i] + _RUN_LINE + "    return\n"
        r = _single(_check(source), "E0502")
        assert r.category == "run-lifecycle"
        assert "恰调用一次" in r.suggestion

    def test_run_before_decorate_rejected(self):
        """run 先于装饰：调用点报（produce/consume 均未装饰）。"""
        no_run = _SIG_BASE.replace("\n" + _RUN_BLOCK.rstrip() + "\n", "\n")
        source = (no_run[:no_run.index("    @pipe.produce")]
                  + _RUN_LINE + "\n"
                  + no_run[no_run.index("    @pipe.produce"):])
        r = _single(_check(source), "E0502")
        assert r.category == "run-lifecycle"
        assert "装饰" in r.suggestion

    def test_run_with_only_consume_rejected(self):
        """只有 consume 装饰：报缺 produce。"""
        source = _SIG_BASE.replace('''\

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K_s, buf.K, mode=Async)
''', "\n")
        r = _single(_check(source), "E0502")
        assert r.category == "run-lifecycle"
        assert "produce" in r.suggestion

    def test_empty_iteration_form_legal(self):
        """空迭代形态（range 实参运行期值）静态零拒绝。"""
        source = _SIG_BASE.replace("range(seq_len)", "range(tis.cdiv(seq_len, BR))")
        assert _check(source) == []


class TestBufferNamespace:
    """E0503 键集与逃逸（R「buffer 命名空间」Scenario 1/2/3）。"""

    def test_registered_key_accepted(self):
        """buf.<注册名> 接受（键集承载，类型面无静态输出差异）。"""
        assert _check(_SIG_BASE) == []

    def test_unregistered_key_rejected_lists_registered(self):
        r = _single(_check(_SIG_BASE.replace("buf.K", "buf.Q")), "E0503")
        assert r.category == "buffer-key"
        assert "K" in r.suggestion
        assert "Q" in r.suggestion

    def test_registered_names_in_declaration_order(self):
        source = _SIG_BASE.replace(
            "    K_s = tis.alloc_shared((BR, D), f16)\n"
            '    pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s})',
            "    K_s = tis.alloc_shared((BR, D), f16)\n"
            "    V_s = tis.alloc_shared((BR, D), f16)\n"
            '    pipe = tis.Pipeline(stages=STAGES, '
            'buffers={"K": K_s, "V": V_s})'
        ).replace("buf.K", "buf.Q")
        r = _single(_check(source), "E0503")
        assert "K/V" in r.suggestion   # 登记序

    def test_buffer_escape_assign_rejected(self):
        source = _SIG_BASE.replace("        return st",
                                   "        x = buf\n        return st")
        r = _single(_check(source), "E0503")
        assert r.category == "buffer-escape"

    def test_buffer_escape_arg_rejected(self):
        source = _SIG_BASE.replace(
            "        tis.load(K_s, buf.K, mode=Async)",
            "        tis.load(K_s, buf, mode=Async)")
        r = _single(_check(source), "E0503")
        assert r.category == "buffer-escape"

    def test_buffer_escape_return_rejected(self):
        source = _SIG_BASE.replace("        return st", "        return buf")
        r = _single(_check(source), "E0503")
        assert r.category == "buffer-escape"


class TestAsyncStaticFace:
    """Async 完成保证与组管理不可见（R 静态面：让渡 + E0406 联动）。"""

    def test_wait_group_deferred_zero_rejections(self):
        """tis.wait_group 不是已定义原语：执行段让渡零拒绝（spec 固定负例）。"""
        i = _SIG_BASE.rindex("    return")
        source = _SIG_BASE[:i] + "    tis.wait_group(0)\n" + _SIG_BASE[i:]
        assert _check(source) == []

    def test_group_kwarg_deferred_in_execution_stage(self):
        """group= 由原语段 E0406 拒（先例不重复报）；执行段自身让渡零 E05xx。"""
        source = _SIG_BASE.replace(
            "tis.load(K_s, buf.K, mode=Async)",
            "tis.load(K_s, buf.K, mode=Async, group=0)")
        assert _check(source) == []
        from tilescript import primitives
        prim = primitives.check_module(ast.parse(source), "nvidia_h200")
        assert any(r.code == "E0406" for r in prim), [r.code for r in prim]

    def test_legal_async_pairing_zero_rejections(self):
        """合法 produce Async / consume 读配对零拒绝（完成保证为运行时承诺）。"""
        assert _check(_SIG_BASE) == []


# 索引内建测试模板（tasks 8.1/8.2）。
_IX_BASE = '''\
import tis

@tis.kernel
def ix_kernel(seq_len: int, D: comptime[int] = 64):
    K_s = tis.alloc_shared((D, 16), f16)
    bm = tis.block_idx(0)
    n = tis.cdiv(seq_len, D)
    return
'''


class TestIndexBuiltins:
    """E0505 索引与整数内建参数（R「索引与整数内建」Scenario 1/2/3）。"""

    def test_legal_calls_accepted(self):
        assert _check(_IX_BASE) == []

    def test_block_idx_negative_rejected(self):
        r = _single(_check(_IX_BASE.replace("block_idx(0)", "block_idx(-1)")),
                    "E0505")
        assert r.category == "block-idx-dim"
        assert "非负" in r.suggestion

    def test_block_idx_runtime_name_rejected(self):
        r = _single(_check(_IX_BASE.replace("block_idx(0)", "block_idx(seq_len)")),
                    "E0505")
        assert r.category == "block-idx-dim"
        assert "comptime" in r.suggestion

    def test_block_idx_comptime_name_deferred(self):
        """comptime 名 dim：值让渡（运行期值域非负由数值层承诺）。"""
        assert _check(_IX_BASE.replace("block_idx(0)", "block_idx(D)")) == []

    def test_cdiv_tensor_arg_rejected(self):
        r = _single(_check(_IX_BASE.replace("cdiv(seq_len, D)",
                                            "cdiv(K_s, D)")), "E0505")
        assert r.category == "cdiv-args"
        assert "标量" in r.suggestion

    def test_cdiv_float_arg_rejected(self):
        r = _single(_check(_IX_BASE.replace("cdiv(seq_len, D)",
                                            "cdiv(3.14, D)")), "E0505")
        assert r.category == "cdiv-args"

    def test_cdiv_zero_divisor_rejected(self):
        r = _single(_check(_IX_BASE.replace("cdiv(seq_len, D)",
                                            "cdiv(seq_len, 0)")), "E0505")
        assert r.category == "cdiv-args"
        assert "0" in r.suggestion

    def test_cdiv_one_arg_rejected(self):
        r = _single(_check(_IX_BASE.replace("cdiv(seq_len, D)",
                                            "cdiv(seq_len)")), "E0505")
        assert r.category == "cdiv-args"

    def test_cdiv_comptime_result_feeds_block_idx(self):
        """结果种类推导（两 comptime → comptime）：block_idx 消费链零拒绝。"""
        assert _check(_IX_BASE.replace("bm = tis.block_idx(0)",
                                       "bm = tis.block_idx(tis.cdiv(64, D))")) == []

    def test_cdiv_int_result_rejected_by_block_idx(self):
        """cdiv 运行期结果（int）不是 comptime：block_idx 拒。"""
        r = _single(_check(_IX_BASE.replace(
            "bm = tis.block_idx(0)",
            "bm = tis.block_idx(tis.cdiv(seq_len, D))")), "E0505")
        assert r.category == "block-idx-dim"

    def test_range_two_args_rejected(self):
        r = _single(_check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    for i in range(1, 10):\n        pass\n"
            "    bm = tis.block_idx(0)")), "E0505")
        assert r.category == "range-args"

    def test_range_tensor_arg_rejected(self):
        r = _single(_check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    for i in range(K_s):\n        pass\n"
            "    bm = tis.block_idx(0)")), "E0505")
        assert r.category == "range-args"


class TestRangeValueClosure:
    """range 可迭代值位置封闭（R Scenario 4，E0502 承载）。"""

    def test_bound_range_in_arithmetic_rejected(self):
        r = _single(_check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    x = range(10)\n"
            "    y = x + 1\n"
            "    bm = tis.block_idx(0)")), "E0502")
        assert r.category == "range-value-escape"
        assert "for" in r.suggestion

    def test_bound_range_as_prim_arg_rejected(self):
        r = _single(_check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    x = range(10)\n"
            "    tis.load(K_s, x)\n"
            "    bm = tis.block_idx(0)")), "E0502")
        assert r.category == "range-value-escape"

    def test_direct_range_call_in_value_position_rejected(self):
        r = _single(_check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    tis.load(range(10), K_s)\n"
            "    bm = tis.block_idx(0)")), "E0502")
        assert r.category == "range-value-escape"

    def test_for_iterable_position_accepted(self):
        assert _check(_IX_BASE.replace(
            "    bm = tis.block_idx(0)",
            "    for i in range(10):\n        pass\n"
            "    bm = tis.block_idx(0)")) == []

    def test_run_first_arg_position_accepted(self):
        assert _check(_SIG_BASE) == []   # run(range(seq_len), ...) 基线承载
