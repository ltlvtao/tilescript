"""集成样例：veps §7 FlashAttention 源码的两段实例化（syntax + type-system）。

syntax spec Scenario L100 引用 veps/design.md §7；该源码以源文件本身为
模块——`module flash_attention:` 行是文档组织伪代码，按 E0101 拒绝，集成
样例须去除（上一 change tasks 6.1）。本 change（tasks 8.1/8.2）：类型段
已知面（签名注解、状态字段、st.* 字段访问、构造与 return 绑定）等价、
让渡面（tis.*/算术/pipe.run）UNKNOWN 跳过——两段零拒绝；已知面注入
E0303 走 CLI rejected。
"""

import json

from tilescript import pipeline
from tilescript.cli import main
from tilescript.frontend import check_module

# veps/design.md §7（2026-10-09 对照誊录，仅去除首行 module 伪代码）。
FLASH_ATTENTION = '''\
@tis.state
class AttnState:
    O_acc: Tensor[f32, (BR, D), Register]
    m:     Tensor[f32, (BR,), Register]
    l:     Tensor[f32, (BR,), Register]

@tis.kernel
def flash_attn_fwd(
    Q_ptr: Pointer[f16, Global], K_ptr: Pointer[f16, Global],
    V_ptr: Pointer[f16, Global], O_ptr: Pointer[f16, Global],
    L_ptr: Pointer[f32, Global],
    seq_len: int, scale: f32,
    D: comptime[int] = 64, BR: comptime[int] = 64,
    BC: comptime[int] = 64, STAGES: comptime[int] = 2
):
    bm = tis.block_idx(0)
    Q = tis.make_tensor(Q_ptr, (seq_len, D)); K = tis.make_tensor(K_ptr, (seq_len, D))
    V = tis.make_tensor(V_ptr, (seq_len, D)); O = tis.make_tensor(O_ptr, (seq_len, D))
    L = tis.make_tensor(L_ptr, (seq_len,))

    Q_s = tis.alloc_shared((BR, D), f16, layout=tis.Layout.swizzled(xor=0b11100))
    K_s = tis.alloc_shared((BC, D), f16, layout=tis.Layout.swizzled(xor=0b11100))
    V_s = tis.alloc_shared((BC, D), f16, layout=tis.Layout.swizzled(xor=0b11100))

    tis.load(Q[bm*BR:(bm+1)*BR, :], Q_s, mode=Sync)
    tis.barrier()

    pipe = tis.Pipeline(stages=STAGES, buffers={"K": K_s, "V": V_s})

    @pipe.produce
    def fetch(j: int, buf):
        tis.load(K[j*BC:(j+1)*BC, :], buf.K, mode=Async)
        tis.load(V[j*BC:(j+1)*BC, :], buf.V, mode=Async)

    @pipe.consume
    def attend(j: int, buf, st: AttnState) -> AttnState:
        S = tis.dot(Q_s, tis.transpose(buf.K), tis.zeros((BR, BC), f32, Register),
                   mma=tis.MMA(16, 8, 16), pad=tis.PadPolicy.Error)
        S = S * scale
        m_ij  = tis.reduce(S, axis=1, op=Max)
        m_new = tis.maximum(st.m, m_ij)
        alpha = tis.exp(st.m - m_new)
        P     = tis.exp(S - m_new[:, None])
        l_new = alpha * st.l + tis.reduce(P, axis=1, op=Sum)
        O_new = alpha[:, None] * st.O_acc + tis.dot(tis.cast(P, f16), buf.V,
                                                   tis.zeros((BR, D), f32, Register))
        return AttnState(O_acc=O_new, m=m_new, l=l_new)

    st = pipe.run(range(tis.cdiv(seq_len, BC)),
                  init=AttnState(O_acc=tis.zeros((BR, D), f32, Register),
                                 m=tis.full((BR,), -inf, f32, Register),
                                 l=tis.zeros((BR,), f32, Register)))

    tis.store(tis.cast(st.O_acc / st.l[:, None], f16), O[bm*BR:(bm+1)*BR, :])
    tis.store(tis.log(st.l) + st.m, L[bm*BR:(bm+1)*BR])
'''


class TestFlashAttentionSyntaxStage:
    def test_syntax_stage_zero_rejections(self):
        """§7 源码（去除 module 伪代码行）语法段零拒绝。"""
        assert check_module(FLASH_ATTENTION) == []

    def test_module_pseudocode_line_alone_rejected(self):
        """对照：照搬 module 伪代码行则 E0101（集成样例去除该行的理由）。"""
        out = check_module("module flash_attention:\n" + FLASH_ATTENTION)
        assert len(out) == 1 and out[0].code == "E0101"
        assert out[0].category == "module-pseudocode"


class TestInjection:
    def test_injected_violations_collected_and_sorted(self, tmp_path, capsys):
        """注入 while（前）与列表推导（后）→ 恰两条、位置升序、exit 1。"""
        source = FLASH_ATTENTION.replace(
            "    bm = tis.block_idx(0)\n",
            "    bm = tis.block_idx(0)\n"
            "    while bm > 0:\n"
            "        bm = bm - 1\n",
        ).replace(
            "    tis.barrier()\n",
            "    tis.barrier()\n"
            "    junk = [i for i in row]\n",
        )
        src = tmp_path / "injected.tis"
        src.write_text(source, encoding="utf-8")

        capsys.readouterr()
        code = main(["compile", str(src), "--target", "nvidia_h200"])
        out = capsys.readouterr().out

        assert code == 1
        payload = json.loads(out)
        assert payload["status"] == "rejected"
        assert [r["code"] for r in payload["rejections"]] == ["E0105", "E0106"]
        assert payload["rejections"][0]["category"] == "while-loop"
        assert payload["rejections"][1]["category"] == "list-comprehension"


class TestFlashAttentionThreeStages:
    def test_three_stages_zero_rejections(self):
        """三段（syntax/type/primitive）零拒绝：load 切片折叠支 (bm+1)*BR-
        bm*BR → BR 贯通、dot 的 buf.* 操作数 UNKNOWN 让渡、算术/pipe 让渡。"""
        assert pipeline.compile_stages(FLASH_ATTENTION, target="nvidia_h200") == []

    def test_dot_known_face_comptime_dims(self):
        """dot 已知面（tasks 6.1）：buf.K → K_s 后非让渡路径——E0402 列表
        命中 (16,8,16)、E0403 全 comptime 符号维不触发（D5）。"""
        source = FLASH_ATTENTION.replace(
            "S = tis.dot(Q_s, tis.transpose(buf.K),"
            " tis.zeros((BR, BC), f32, Register),",
            "S = tis.dot(Q_s, tis.transpose(K_s),"
            " tis.zeros((BR, BC), f32, Register),",
        )
        assert pipeline.compile_stages(source, target="nvidia_h200") == []

    def test_known_face_equivalent_no_rejection(self):
        """已知面收紧回归：构造实参与字段声明同型（st.m/st.l 直传）不报。"""
        source = FLASH_ATTENTION.replace(
            "        return AttnState(O_acc=O_new, m=m_new, l=l_new)",
            "        return AttnState(O_acc=O_new, m=st.m, l=st.l)",
        )
        assert pipeline.compile_stages(source, target="nvidia_h200") == []

    def test_cli_end_to_end_incomplete_three_stages(self, tmp_path, capsys):
        """CLI 端到端：exit 0 + incomplete 三段清单（原语段接入后）。"""
        src = tmp_path / "flash.tis"
        src.write_text(FLASH_ATTENTION, encoding="utf-8")
        capsys.readouterr()
        code = main(["compile", str(src), "--target", "nvidia_h200"])
        payload = json.loads(capsys.readouterr().out)
        assert code == 0
        assert payload["status"] == "incomplete"
        assert payload["implemented_stages"] == [
            "syntax", "type-system", "primitive-contract"]
        assert set(payload["pending_stages"]) == {
            "execution-structure", "numerics"}

    def test_nested_consume_return_binding_checked(self):
        """嵌套 consume 体在检查面内：return 值类型×返回注解可违规。"""
        source = FLASH_ATTENTION.replace(
            "        return AttnState(O_acc=O_new, m=m_new, l=l_new)",
            "        return 5",
        )
        rs = pipeline.compile_stages(source, target="nvidia_h200")
        assert [r.code for r in rs] == ["E0303"]
        assert rs[0].category == "return"

    def test_relinquished_face_negative_regression(self):
        """让渡面负例：zeros 后绑 int、UNKNOWN 边界切片不报（tasks 8.2）。"""
        source = FLASH_ATTENTION.replace(
            "    bm = tis.block_idx(0)\n",
            "    x = tis.zeros((16,), f32, Register)\n"
            "    x = 5\n"
            "    y = Q[x : x + 16]\n",
        )
        assert pipeline.compile_stages(source, target="nvidia_h200") == []


class TestKnownFaceE0303ThroughCli:
    def test_ctor_dtype_mismatch_rejected_via_cli(self, tmp_path, capsys):
        """注入已知面 E0303（构造字段 dtype 不匹配）→ rejected JSON + exit 1。"""
        source = FLASH_ATTENTION.replace(
            "    seq_len: int, scale: f32,",
            "    seq_len: int, scale: f32, P_r: Tensor[f16, (BR,), Register],",
        ).replace(
            "        return AttnState(O_acc=O_new, m=m_new, l=l_new)",
            "        return AttnState(O_acc=O_new, m=P_r, l=l_new)",
        )
        src = tmp_path / "dtype.tis"
        src.write_text(source, encoding="utf-8")
        capsys.readouterr()
        code = main(["compile", str(src), "--target", "nvidia_h200"])
        out = capsys.readouterr().out
        assert code == 1
        payload = json.loads(out)
        assert payload["status"] == "rejected"
        assert [r["code"] for r in payload["rejections"]] == ["E0303"]
        r = payload["rejections"][0]
        assert r["category"] == "state-ctor-arg" and "tis.cast" in r["suggestion"]
        assert set(r.keys()) == {"code", "line", "col", "category", "suggestion"}


class TestCrossTargetConsistency:
    """hal「跨 HAL 行为不变面」Scenario 3：同源语言层拒绝两目标逐条一致；
    HAL 依赖拒绝按目标分化（tasks 6.2）。"""

    _SOURCE = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(d: Tensor[f32, (64, 8), Global], s: Tensor[f32, (64, 8), Register]):\n"
        "    tis.load(d, s, mode=Fast)\n"                              # 语言层 E0406
        "    r = tis.reduce(s, axis=1, op=Max, scope=Warp)\n"          # HAL 依赖面
        "    return\n"
    )

    def test_language_layer_identical_hal_differs(self):
        h200 = pipeline.compile_stages(self._SOURCE, target="nvidia_h200")
        ascend = pipeline.compile_stages(self._SOURCE, target="ascend_910b")
        # 语言层条目（非 HAL 依赖）两目标逐条一致（含建议文本）。
        lang_h = [r.to_dict() for r in h200 if r.category != "scope-unsupported"]
        lang_a = [r.to_dict() for r in ascend if r.category != "scope-unsupported"]
        assert lang_h == lang_a and len(lang_h) == 1
        # h200 支持 Warp：仅语言层一条；ascend_910b 追加 HAL 支持面拒绝。
        assert [r.code for r in h200] == ["E0406"]
        assert [r.code for r in ascend] == ["E0406", "E0408"]
        assert ascend[1].category == "scope-unsupported"
        assert "Block" in ascend[1].suggestion  # 目标支持清单
