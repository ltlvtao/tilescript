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

    def test_implemented_stages_three(self):
        """toolchain/cli R3 段清单唯一来源：三段登记。"""
        assert pipeline.IMPLEMENTED_STAGES == (
            "syntax", "type-system", "primitive-contract")
