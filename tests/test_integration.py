"""集成样例（tasks 6.1）：veps §7 FlashAttention 源码的语法段实例化。

syntax spec Scenario L100（白名单语句组合被接受）引用 veps/design.md §7；
该源码以源文件本身为模块——`module flash_attention:` 行是文档组织伪代码，
按 E0101 拒绝，集成样例须去除（tasks 6.1）。
"""

import json

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
