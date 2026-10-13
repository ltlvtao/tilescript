"""段管线编排测试（type-system spec R8 delta 段间短路 + R8 报告契约）。

Scenario 映射：
- 语法段存在拒绝时类型段不执行（R8 delta 新增）
- 同位置语法与类型双命中只报语法错误（R8，管线层复测——短路下的实例化）
- 多条类型拒绝被收集并排序（R8——E0302 第 8 行在前、E0303 第 20 行在后）
- 同位置多个类型规则命中只报段内首个（R8——E0304>E0303 段内 tiebreak）
- 重复编译类型拒绝清单一致（R8）
"""

from tilescript.frontend.report import Rejection, finalize
from tilescript import pipeline


def _r(code, line, col, order, category="x"):
    return Rejection(code=code, line=line, col=col, category=category,
                     suggestion="s", order=order, stage="type-system")


class TestInterStageShortCircuit:
    def test_syntax_rejection_suppresses_type_stage(self):
        """R8 delta 新 Scenario：E0106 在前则潜在 E0303 不出现。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (16,), Register], b: Tensor[f32, (16,), Register]):\n"
            "    x = [i for i in row]\n"   # L5：E0106（语法拒绝）
            "    b = a\n"                   # 若类型段执行将报 E0303
        )
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0106"]
        assert all(r.code.startswith("E01") for r in rs)

    def test_carrier_rejection_short_circuits(self):
        """E0101 单条特例同样短路类型段。"""
        rs = pipeline.compile_stages("def broken(:\n", target="nvidia_h200")
        assert [r.code for r in rs] == ["E0101"]

    def test_clean_module_zero_rejections(self):
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (16,), Register]):\n"
            "    x = tis.zeros((16,), f16, Register)\n"
            "    y = a\n"
            "    return\n"
        )
        assert pipeline.compile_stages(source, target="nvidia_h200") == []


class TestReportContract:
    def test_multiple_rejections_collected_and_sorted(self):
        """R8 Scenario：E0302（注解行）与 E0303（再绑定行）恰两条、位置升序。"""
        head = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (16,), Register], b: Tensor[f32, (16,), Register],\n"
            "       c: Tensor[f64, (16,), Register]):\n"  # E0302：未知 dtype
        )
        filler = "".join(f"    q{i} = {i}\n" for i in range(12))
        tail = "    b = a\n"  # E0303：f16 值绑 f32 目标
        rs = pipeline.compile_stages(head + filler + tail, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0302", "E0303"]
        assert rs[0].line < rs[1].line

    def test_same_position_type_rules_first_wins(self):
        """R8 Scenario：同位置 E0304 与 E0303 双命中只报段内 order 更早的 E0304。

        M1 语法下两组检查节点不相交（真实同位双命中不可构造），段内优先由
        finalize 的 order tiebreak 机制承载——此处直测该公共行为。
        """
        e0304 = _r("E0304", 5, 8, order=2)
        e0303 = _r("E0303", 5, 8, order=4)
        assert finalize([e0303, e0304]) == [e0304]

    def test_repeated_compilation_identical(self):
        """R8 Scenario：同一非法模块连续编译两次清单逐条一致。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    m: int\n"
            "\n"
            "@tis.kernel\n"
            "def k(a: Tensor[f16, (16,), Register], b: Tensor[f32, (16,), Register]):\n"
            "    st = S(m=a)\n"
            "    b = a\n"
        )
        first = [r.to_dict() for r in pipeline.compile_stages(source, target="nvidia_h200")]
        second = [r.to_dict() for r in pipeline.compile_stages(source, target="nvidia_h200")]
        assert first == second and len(first) == 3  # E0304 + ctor-arg E0303 + assign E0303


class TestPrimitiveStageWiring:
    """原语契约段接入（三段完整序 + 类型段非空短路，R8 delta 直测）。"""

    _TYPED_BROKEN = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(gl: Tensor[f16, (16,), Global], sh: Tensor[f16, (16,), Shared],"
        " b: Tensor[f32, (16,), Register]):\n"
        "    b = gl\n"              # E0303：f16 值绑 f32 目标
        "    tis.store(gl, sh)\n"   # 若原语段执行将报 E0404
    )

    _PRIMITIVE_BROKEN = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(gl: Tensor[f16, (16,), Global], sh: Tensor[f16, (16,), Shared]):\n"
        "    tis.store(gl, sh)\n"   # E0404：store 承载 load 格
        "    return\n"
    )

    def test_type_rejection_suppresses_primitive_stage(self):
        """类型段非空 → 原语段 MUST NOT 执行（潜在 E0404 不出现）。"""
        rs = pipeline.compile_stages(self._TYPED_BROKEN, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert all(r.code.startswith("E03") for r in rs)

    def test_primitive_rejection_reaches_report(self):
        """类型段零命中 → 原语段拒绝上报（E0404）。"""
        rs = pipeline.compile_stages(self._PRIMITIVE_BROKEN, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0404"]
        assert rs[0].stage == "primitive-contract"

    def test_three_stage_order(self):
        """三段序实例化：语法/类型/原语各自单独命中时逐段上报。"""
        # 语法段（E0106）
        rs = pipeline.compile_stages("def broken(:\n", target="nvidia_h200")
        assert all(r.code.startswith("E01") for r in rs)
        # 原语段（E0404，同 _PRIMITIVE_BROKEN）
        # 类型段（E0303，同 _TYPED_BROKEN）——上面两测已分别固定。

    def test_implemented_stages_five(self):
        """toolchain/cli R3 段清单唯一来源：五段登记（本 change 第五段接入）。"""
        assert pipeline.IMPLEMENTED_STAGES == (
            "syntax", "type-system", "primitive-contract",
            "execution-structure", "numerics")


# ---- 9.2 第四段接入（execution-structure）：接入 + 短路 + 报告契约（delta R 5S）----

_EXEC_BASE = (
    "import tis\n"
    "\n"
    "@tis.state\n"
    "class AttnState:\n"
    "    O_acc: Tensor[f32, (BR, D), Register]\n"
    "\n"
    "@tis.kernel\n"
    "def flash(K_ptr: Pointer[f16, Global], seq_len: int,\n"
    "          D: comptime[int] = 64, BR: comptime[int] = 64,\n"
    "          STAGES: comptime[int] = 2):\n"
    "    K_s = tis.alloc_shared((BR, D), f16)\n"
    "    pipe = tis.Pipeline(stages=STAGES, buffers={\"K\": K_s})\n"
    "\n"
    "    @pipe.produce\n"
    "    def fetch(j: int, buf):\n"
    "        tis.load(K_s, buf.K, mode=Async)\n"
    "\n"
    "    @pipe.consume\n"
    "    def attend(j: int, buf, st: AttnState) -> AttnState:\n"
    "        return st\n"
    "\n"
    "    st = pipe.run(range(seq_len),\n"
    "                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))\n"
    "    return\n"
)

_PK_ENTRY = (
    "import tis\n"
    "\n"
    "@tis.persistent_kernel\n"
    "def pk(K_ptr: Pointer[f16, Global], D: comptime[int] = 64):\n"
    "    K_s = tis.alloc_shared((D, 16), f16)\n"
    "    return\n"
)


class TestExecutionStageIntegration:
    """第四段接入面：段清单、合法源零拒绝、E0506 管线上报。"""

    def test_clean_flash_zero_rejections_four_stages(self):
        """合法 FLASH 全链四段零拒绝。"""
        assert pipeline.compile_stages(_EXEC_BASE, target="nvidia_h200") == []

    def test_e0506_reaches_pipeline_report(self):
        """E0506 经管线上报：ascend 拒 / h200 零拒绝。"""
        rs = pipeline.compile_stages(_PK_ENTRY, target="ascend_910b")
        assert [r.code for r in rs] == ["E0506"]
        assert rs[0].stage == "execution-structure"
        assert pipeline.compile_stages(_PK_ENTRY, target="nvidia_h200") == []


class TestExecutionShortCircuit:
    """delta 段间短路句：前序段任一非空即执行段 MUST NOT 执行。"""

    _PRIM_BROKEN = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(gl: Tensor[f16, (16, 16), Global], sh: Tensor[f16, (16, 16), Shared],\n"
        "      STAGES: comptime[int] = 2):\n"
        "    tis.load(gl, sh, mode=Fast)\n"   # E0406（语言层实参）
        "    pipe = tis.Pipeline(stages=STAGES, buffers={\"K\": sh})\n"
        "    pipe2 = tis.Pipeline(stages=STAGES, buffers={\"K\": sh})\n"  # 本应 E0502
        "    return\n"
    )

    _TYPED_BROKEN = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(gl: Tensor[f16, (16,), Global], sh: Tensor[f16, (16,), Shared],\n"
        "      b: Tensor[f32, (16,), Register], STAGES: comptime[int] = 2):\n"
        "    b = gl\n"   # E0303
        "    pipe = tis.Pipeline(stages=STAGES, buffers={\"K\": sh})\n"
        "    pipe2 = tis.Pipeline(stages=STAGES, buffers={\"K\": sh})\n"  # 本应 E0502
        "    return\n"
    )

    def test_primitive_rejection_suppresses_execution_stage(self):
        """delta Scenario：原语段 E0406 在 → 第 8 行本应 E0502 不出现。"""
        rs = pipeline.compile_stages(self._PRIM_BROKEN, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0406"]
        assert not any(r.code.startswith("E05") for r in rs)

    def test_type_rejection_suppresses_execution_stage(self):
        """类型段非空同样短路（E0303 在 → 潜在 E0502 不出现）。"""
        rs = pipeline.compile_stages(self._TYPED_BROKEN, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert not any(r.code.startswith(("E04", "E05")) for r in rs)

    def test_same_position_primitive_wins(self):
        """delta Scenario：同 load 调用 E0407（顶层 Async）与 E0502（pipe
        逃逸）双命中，只报管线更早的原语段一条（短路承载）。"""
        source = _EXEC_BASE.replace(
            "    st = pipe.run(range(seq_len),\n"
            "                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))\n",
            "    tis.load(pipe, K_s, mode=Async)\n"
            "    st = pipe.run(range(seq_len),\n"
            "                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))\n")
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0407"]
        assert not any(r.code.startswith("E05") for r in rs)


class TestExecutionReportContract:
    """delta R 5S 管线面：收集排序 / 同码合并 / 重复一致。"""

    _TWO_HITS = _EXEC_BASE.replace("buf.K, mode=Async", "buf.Q, mode=Async").replace(
        "    st = pipe.run(range(seq_len),",
        "    y = pipe\n"
        "    st = pipe.run(range(seq_len),")

    def test_multiple_execution_rejections_collected_and_sorted(self):
        """delta Scenario 1：E0503（buf.Q 行）与 E0502（y = pipe 行）恰两条、
        位置升序。"""
        rs = pipeline.compile_stages(self._TWO_HITS, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0503", "E0502"]
        assert rs[0].line < rs[1].line
        assert all(r.stage == "execution-structure" for r in rs)

    def test_same_call_single_code_merged(self):
        """delta Scenario 4：pipe.run(K_s) 同调用多实参违规合并一条列全部。"""
        source = _EXEC_BASE.replace(
            "    st = pipe.run(range(seq_len),\n"
            "                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))\n",
            "    st = pipe.run(K_s)\n")
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert len(rs) == 1 and rs[0].code == "E0502"
        assert "range" in rs[0].suggestion and "init" in rs[0].suggestion

    def test_repeat_compilation_identical(self):
        """delta Scenario 5：同一非法模块连续编译两次逐条一致。"""
        a = pipeline.compile_stages(self._TWO_HITS, target="nvidia_h200")
        b = pipeline.compile_stages(self._TWO_HITS, target="nvidia_h200")
        assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


# ---- 第五段接入（numerics）：接入 + 短路（delta 新 Scenario）----------------

_NUM_FLASH = (
    "import tis\n"
    "\n"
    "@tis.state\n"
    "class AttnState:\n"
    "    O_acc: Tensor[f32, (BR, D), Register]\n"
    "\n"
    "@tis.kernel\n"
    "def flash(K_ptr: Pointer[f16, Global], seq_len: int, scale: f32,\n"
    "          D: comptime[int] = 64, BR: comptime[int] = 64,\n"
    "          BC: comptime[int] = 64, STAGES: comptime[int] = 2):\n"
    "    K = tis.make_tensor(K_ptr, (seq_len, D))\n"
    "    K_s = tis.alloc_shared((BC, D), f16)\n"
    "    S = tis.zeros((BR, D), f32, Register)\n"
    "    T = S * scale\n"
    "    steps = tis.cdiv(seq_len, BC)\n"
    "    base = BR * 2 - 1\n"
    "    if seq_len > 0:\n"
    "        T = T + S\n"
    "    pipe = tis.Pipeline(stages=STAGES, buffers={\"K\": K_s})\n"
    "\n"
    "    @pipe.produce\n"
    "    def fetch(j: int, buf):\n"
    "        tis.load(K[j * BC:(j + 1) * BC, :], buf.K, mode=Async)\n"
    "\n"
    "    @pipe.consume\n"
    "    def attend(j: int, buf, st: AttnState) -> AttnState:\n"
    "        return AttnState(O_acc=tis.zeros((BR, D), f32, Register))\n"
    "\n"
    "    st = pipe.run(range(tis.cdiv(seq_len, BC)),\n"
    "                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register)))\n"
    "    return\n"
)

_NUM_BROKEN = (
    "import tis\n"
    "\n"
    "@tis.kernel\n"
    "def f(seq_len: int, scale: f32, D: comptime[int] = 64,\n"
    "      BR: comptime[int] = 64, BC: comptime[int] = 64):\n"
    "    A = tis.zeros((4,), f16, Register)\n"
    "    B = tis.zeros((4,), f32, Register)\n"
    "    C = A + B\n"          # E0602（四段干净 → 数值段上报）
    "    return\n"
)

_EXEC_BROKEN = (
    "import tis\n"
    "\n"
    "@tis.state\n"
    "class AttnState:\n"
    "    O_acc: Tensor[f32, (BR, D), Register]\n"
    "\n"
    "@tis.kernel\n"
    "def flash(K_ptr: Pointer[f16, Global], seq_len: int, scale: f32,\n"
    "          D: comptime[int] = 64, BR: comptime[int] = 64,\n"
    "          STAGES: comptime[int] = 2):\n"
    "    K_s = tis.alloc_shared((BR, D), f16)\n"
    "    pipe = tis.Pipeline(stages=STAGES, buffers={\"K\": K_s})\n"
    "    pipe2 = tis.Pipeline(stages=STAGES, buffers={\"K\": K_s})\n"  # E0502
    "    T = K_s * 0.5\n"     # 数值面表达式（短路下数值段不执行）
    "    return\n"
)


class TestNumericsStageIntegration:
    """tasks 6.2：第五段接入面（段清单/合法源零拒绝/E06xx 管线上报）。"""

    def test_clean_flash_numerics_five_stages_zero(self):
        """合法 FLASH 数值面五段零拒绝。"""
        assert pipeline.compile_stages(_NUM_FLASH, target="nvidia_h200") == []

    def test_e0602_reaches_pipeline_report(self):
        """四段干净 → E0602 经管线上报（stage=numerics）。"""
        rs = pipeline.compile_stages(_NUM_BROKEN, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0602"]
        assert rs[0].stage == "numerics"

    def test_numerics_target_independent(self):
        """E06xx 无目标分化：两登记目标输出逐条一致（check_module 无 target）。"""
        h200 = [r.to_dict() for r in pipeline.compile_stages(_NUM_BROKEN,
                                                             target="nvidia_h200")]
        ascend = [r.to_dict() for r in pipeline.compile_stages(_NUM_BROKEN,
                                                               target="ascend_910b")]
        assert h200 == ascend


class TestNumericsShortCircuit:
    """delta 新 Scenario：前序段任一非空即数值段 MUST NOT 执行。"""

    def test_execution_rejection_suppresses_numerics(self):
        """E0502 在 → 数值面不执行（无任何 E06xx 出现）。"""
        rs = pipeline.compile_stages(_EXEC_BROKEN, target="nvidia_h200")
        assert rs and all(r.code == "E0502" for r in rs)
        assert not any(r.code.startswith("E06") for r in rs)

    def test_type_rejection_suppresses_numerics(self):
        """类型段非空同样短路（E0303 在 → 潜在 E0602 不出现）。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def f(gl: Tensor[f16, (4,), Global], b: Tensor[f32, (4,), Register],\n"
            "      D: comptime[int] = 64, BR: comptime[int] = 64,\n"
            "      BC: comptime[int] = 64):\n"
            "    b = gl\n"       # E0303：f16 形参值绑 f32 注解目标（类型段）
            "    C = gl + b\n"   # 若数值段执行将报 E0602（f16+f32）
            "    return\n"
        )
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert not any(r.code.startswith("E06") for r in rs)


class TestNumericsReportContract:
    """tasks 6.1 管线面：择一双命中经管线只报最早段（R7 Scenario 2）。"""

    _DOUBLE_HIT = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def f(seq_len: int, scale: f32, D: comptime[int] = 64,\n"
        "      BR: comptime[int] = 64, BC: comptime[int] = 64):\n"
        "    b = tis.full((4,), 1.5, f16, Register)\n"
        "    b0 = b[0]\n"
        "    x = seq_len\n"
        "    x = scale * b0\n"   # 数值段双命中（E0602+E0303）→ 择一恰一条 E0303
        "    return\n"
    )

    def test_double_hit_earliest_stage_through_pipeline(self):
        """前四段干净 → 数值段择一收口：清单恰一条 E0303（stage 口径随
        最早段 type-system，无 E06xx 同现）。"""
        rs = pipeline.compile_stages(self._DOUBLE_HIT, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].stage == "type-system"
        assert not any(r.code.startswith("E06") for r in rs)
