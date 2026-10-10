"""存取原语检查测试（memory-ops spec：E0406 参数集/值域/形态面）。

3.1 面：R1×5（参数集结构含 delta 补的必选实参缺失、mode 值域）、
R5×6（分配与视图构造值域）、R6×3（cast）、R7 值域（barrier）、
R3 的实参类别面（Pointer/标量 → E0406）。
E0404/E0405/E0407 由 3.2/3.3/3.4 增量接入。

入口走公共面 primitives.check_module（接受例同时是后续段的前瞻合法构造）。
"""

from tilescript.frontend import carrier
from tilescript.primitives import check_module

_PARAMS = (
    "gl2: Tensor[f16, (64,), Global], "
    "gl: Tensor[f16, (64,), Global], "
    "sh: Tensor[f16, (64,), Shared], "
    "sh2: Tensor[f16, (64,), Shared], "
    "reg: Tensor[f16, (64,), Register], "
    "reg2: Tensor[f16, (64,), Register], "
    "f32r: Tensor[f32, (64,), Register], "
    "p: Pointer[f16, Global], "
    "psh: Pointer[f16, Shared], "
    "n: int, "
    "BR: comptime[int]"
)


def _run(body: str, params=_PARAMS):
    source = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        f"def k({params}):\n"
        f"{body}"
    )
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return check_module(tree, "nvidia_h200")


def _stmt(line: str):
    return f"    {line}\n"


class TestCallStructure:
    """R1：参数集结构 + mode 值域（含 delta 补的必选实参缺失）。"""

    def test_default_load_accepted(self):
        """Scenario 1：全默认 load 接受（mode 取 Sync；合法转移格前瞻）。"""
        assert _run(_stmt("tis.load(gl, sh)")) == []

    def test_unknown_keyword_rejected(self):
        """Scenario 2：group= 不在参数集 → E0406 报合法参数集。"""
        rs = _run(_stmt("tis.load(gl, sh, group=reg)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "param-set"
        assert "src" in rs[0].suggestion and "dst" in rs[0].suggestion \
            and "mode" in rs[0].suggestion

    def test_positional_optional_accepted(self):
        """Scenario 3：zeros 第三位置实参传 scope → 等价关键字形式接受。"""
        assert _run(_stmt("tis.zeros((BR, BR), f32, Register)")) == []

    def test_too_many_positional_rejected(self):
        """Scenario 4：cast 三位置实参 → E0406 注明只接受两个。"""
        rs = _run(_stmt("tis.cast(reg, f16, reg)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "param-set" and "位置实参" in rs[0].suggestion \
            and "x/dtype" in rs[0].suggestion

    def test_missing_required_arg_rejected(self):
        """delta Scenario：必选实参缺失（cast 缺 dtype / make_tensor 缺 shape）。"""
        rs = _run(_stmt("tis.cast(reg)"))
        assert [r.code for r in rs] == ["E0406"] and rs[0].category == "param-set"
        assert "dtype" in rs[0].suggestion and "缺少必选" in rs[0].suggestion
        rs = _run(_stmt("tis.make_tensor(p)"))
        assert [r.code for r in rs] == ["E0406"]
        assert "shape" in rs[0].suggestion

    def test_mode_value_domain(self):
        """mode 值域 Sync/Async：Sync 接受、集合外值拒绝。"""
        assert _run(_stmt("tis.load(gl, sh, mode=Sync)")) == []
        rs = _run(_stmt("tis.load(gl, sh, mode=Fast)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "mode-value"
        assert "Sync/Async" in rs[0].suggestion


class TestAllocAndView:
    """R5：make_tensor / alloc_shared / zeros / full 值域。"""

    def test_make_tensor_dynamic_global_view(self):
        """Scenario 1：make_tensor(p, (n, BR)) 接受；结果 Tensor 经 cast 可观察。"""
        assert _run(_stmt("q = tis.make_tensor(p, (n, BR))\n    q2 = tis.cast(q, f32)")) == []

    def test_make_tensor_shared_pointer_rejected(self):
        """Scenario 2：Pointer[f16, Shared] → E0406 只接受 Global scope Pointer。"""
        rs = _run(_stmt("tis.make_tensor(psh, (64,))"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "ptr-scope"
        assert "Global" in rs[0].suggestion and "Pointer" in rs[0].suggestion

    def test_make_tensor_non_pointer_and_shape_component(self):
        """ptr 非 Pointer（Tensor）与 shape 浮点组件 → E0406。"""
        rs = _run(_stmt("tis.make_tensor(gl, (64,))"))
        assert [r.code for r in rs] == ["E0406"] and rs[0].category == "ptr-kind"
        rs = _run(_stmt("tis.make_tensor(p, (2.5,))"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "shape-component"

    def test_alloc_shared_default_and_swizzled(self):
        """Scenario 3：默认 RowMajor 与 swizzled(xor=0b11100) 均接受。"""
        assert _run(_stmt("tis.alloc_shared((BR, 64), f16)")) == []
        assert _run(_stmt(
            "tis.alloc_shared((BR, 64), f16, layout=tis.Layout.swizzled(xor=0b11100))"
        )) == []

    def test_negative_xor_rejected(self):
        """Scenario 4：xor=-1 → E0406 注明非负 comptime[int]。"""
        rs = _run(_stmt(
            "tis.alloc_shared((BR, 64), f16, layout=tis.Layout.swizzled(xor=-1))"
        ))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "layout-value"
        assert "非负" in rs[0].suggestion and "comptime" in rs[0].suggestion

    def test_runtime_int_xor_rejected(self):
        """xor=运行期 int 名（n）→ E0406：值非静态可知，comptime 判定
        区分运行期名与 comptime[int] 名。"""
        rs = _run(_stmt(
            "tis.alloc_shared((BR, 64), f16, layout=tis.Layout.swizzled(xor=n))"
        ))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "layout-value"
        assert "comptime" in rs[0].suggestion

    def test_comptime_name_xor_accepted(self):
        """xor=comptime[int] 名（BR）→ 让渡为接受面（约束域在名上）。"""
        assert _run(_stmt(
            "tis.alloc_shared((BR, 64), f16, layout=tis.Layout.swizzled(xor=BR))"
        )) == []

    def test_alloc_shared_dtype_and_layout_domain(self):
        """dtype 六封闭集外与 layout 集合外值 → E0406。"""
        rs = _run(_stmt("tis.alloc_shared((64,), f64)"))
        assert [r.code for r in rs] == ["E0406"] and rs[0].category == "dtype-value"
        rs = _run(_stmt("tis.alloc_shared((64,), f16, layout=ColumnMajor)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "layout-value"

    def test_zeros_to_global_rejected(self):
        """Scenario 5：zeros scope=Global → E0406 建议 make_tensor。"""
        rs = _run(_stmt("tis.zeros((BR, BR), f32, Global)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "scope-value" and "make_tensor" in rs[0].suggestion

    def test_full_named_constant_accepted(self):
        """Scenario 6：full((BR,), -inf, f32, Register) 具名常量一元负接受。"""
        assert _run(_stmt("tis.full((BR,), -inf, f32, Register)")) == []

    def test_full_tensor_value_rejected(self):
        """full 的 value 为 Tensor → E0406（标量形态约束）。"""
        rs = _run(_stmt("tis.full((BR,), reg, f32, Register)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "value-form" and "标量" in rs[0].suggestion


class TestCast:
    """R6：cast 实参类别与 dtype 值域。"""

    def test_cast_same_shape_new_dtype(self):
        """Scenario 1：cast(f32r, f16) 接受（shape/scope 不变）。"""
        assert _run(_stmt("x = tis.cast(f32r, f16)")) == []

    def test_cast_scalar_rejected(self):
        """Scenario 2：int 标量 x → E0406 只接受 Tensor。"""
        rs = _run(_stmt("tis.cast(n, f32)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "arg-kind" and "Tensor" in rs[0].suggestion

    def test_cast_dtype_domain(self):
        """dtype 六封闭集外（f64）→ E0406。"""
        rs = _run(_stmt("tis.cast(reg, f64)"))
        assert [r.code for r in rs] == ["E0406"] and rs[0].category == "dtype-value"


class TestBarrier:
    """R7 值域：scope Block/WarpGroup。"""

    def test_barrier_default_and_block_accepted(self):
        assert _run(_stmt("tis.barrier()")) == []
        assert _run(_stmt("tis.barrier(scope=Block)")) == []

    def test_barrier_unknown_scope_rejected(self):
        """Scenario 2：scope=Grid → E0406 注明值域 Block/WarpGroup。"""
        rs = _run(_stmt("tis.barrier(scope=Grid)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "scope-value"
        assert "Block/WarpGroup" in rs[0].suggestion


class TestMoveArgKind:
    """R3 实参类别面：load/store 实参非 Tensor → E0406（建议 make_tensor）。"""

    def test_pointer_src_rejected(self):
        """Scenario 4：Pointer[f16, Global] 直接为 src → E0406。"""
        rs = _run(_stmt("tis.load(p, sh)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "arg-kind" and "make_tensor" in rs[0].suggestion

    def test_scalar_dst_rejected(self):
        rs = _run(_stmt("tis.store(reg, n)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "arg-kind"


class TestTransferCell:
    """R2：转移格承载映射（E0404）——load/store 格、类别不匹配、无承载格、
    矩阵非法格让 E0301。"""

    def test_load_global_to_shared_accepted(self):
        """Scenario 1：load 承载 Global→Shared（load 格）通过。"""
        assert _run(_stmt("tis.load(gl, sh)")) == []

    def test_store_register_to_global_accepted(self):
        """Scenario 2：store 承载 Register→Global（store 格）通过。"""
        assert _run(_stmt("tis.store(reg, gl)")) == []

    def test_store_on_load_cell_rejected(self):
        """Scenario 3：store 承载 Global→Shared → E0404 建议 tis.load。"""
        rs = _run(_stmt("tis.store(gl, sh)"))
        assert [r.code for r in rs] == ["E0404"]
        assert rs[0].category == "transfer-cell"
        assert "tis.load" in rs[0].suggestion

    def test_load_on_store_cell_rejected(self):
        """对称面：load 承载 Register→Global → E0404 建议 tis.store。"""
        rs = _run(_stmt("tis.load(reg, gl)"))
        assert [r.code for r in rs] == ["E0404"]
        assert "tis.store" in rs[0].suggestion

    def test_copy_cell_no_carrier_rejected(self):
        """Scenario 4：Shared→Shared（copy 格）无承载原语 → E0404。"""
        for call in ("tis.load(sh, sh2)", "tis.store(sh, sh2)"):
            rs = _run(_stmt(call))
            assert [r.code for r in rs] == ["E0404"]
            assert rs[0].category == "transfer-cell"
            assert "copy" in rs[0].suggestion and "无承载" in rs[0].suggestion

    def test_move_cell_no_carrier_rejected(self):
        """move 格（Register→Register）同拒（显式延期承载）。"""
        rs = _run(_stmt("tis.load(reg, reg2)"))
        assert [r.code for r in rs] == ["E0404"]
        assert "move" in rs[0].suggestion and "赋值" in rs[0].suggestion

    def test_illegal_cell_left_to_type_system(self):
        """Scenario 5：Global→Global 矩阵非法格归 E0301——本段零 E04xx
        （管线下类型段已拒即短路；直调验证不重复）。"""
        assert _run(_stmt("tis.load(gl, gl2)")) == []
        assert _run(_stmt("tis.store(gl, gl2)")) == []


class TestDataDims:
    """R3：load/store 数据维度契约（E0405）——dtype、逐维四支、折叠支 (d)。

    Scenario 1 的 `buf.K`（执行结构对象 M1 未知）按 design D10 以 kernel
    形参 Tensor 的等价切片构造承载同一折叠支语义。
    """

    _FLASH_LOAD = (
        "import tis\n"
        "\n"
        "@tis.kernel\n"
        "def k(j: int, kg: Tensor[f16, (128, D), Global],"
        " kbuf: Tensor[f16, (BC, D), Shared]):\n"
        "    pipe = tis.Pipeline(stages=2)\n"
        "\n"
        "    @pipe.produce\n"
        "    def p(jj: int, buf):\n"
        "        tis.load(kg[jj*BC:(jj+1)*BC, :], kbuf, mode=Async)\n"
        "\n"
        "    return\n"
    )

    def test_folded_slice_dim_compatible(self):
        """Scenario 1：切片维 (jj+1)*BC-jj*BC 折叠为 BC、与 comptime BC
        相容（折叠支 (d)；Async 置于 produce 内合法语境）。"""
        tree, rejection = carrier.parse(self._FLASH_LOAD)
        assert rejection is None
        assert check_module(tree, "nvidia_h200") == []

    def test_dtype_mismatch_rejected(self):
        """Scenario 2：f16 × f32 → E0405 两侧类型 + tis.cast 建议。"""
        rs = _run(_stmt("tis.load(gl, sh32)"),
                  params=_PARAMS + ", sh32: Tensor[f32, (64,), Shared]")
        assert [r.code for r in rs] == ["E0405"]
        assert rs[0].category == "dtype-mismatch"
        assert "tis.cast" in rs[0].suggestion
        assert "f16" in rs[0].suggestion and "f32" in rs[0].suggestion

    def test_incompatible_dim_rejected(self):
        """Scenario 3：常量维 64 × 运行期符号 seq_len → E0405 注明维度。"""
        params = (_PARAMS + ", a2: Tensor[f16, (64, D), Register],"
                  " g2: Tensor[f16, (seq_len, D), Global]")
        rs = _run(_stmt("tis.store(a2, g2)"), params=params)
        assert [r.code for r in rs] == ["E0405"]
        assert rs[0].category == "dim-incompatible"
        assert "维度" in rs[0].suggestion and "seq_len" in rs[0].suggestion
        assert "64" in rs[0].suggestion

    def test_rank_mismatch_rejected(self):
        """维度数不同（(64,) × (64, 64)）→ E0405。"""
        params = _PARAMS + ", w: Tensor[f16, (64, 64), Global]"
        rs = _run(_stmt("tis.store(reg, w)"), params=params)
        assert [r.code for r in rs] == ["E0405"]
        assert rs[0].category == "dim-incompatible"

    def test_same_symbol_and_derived_structure_compatible(self):
        """支 (b) 同运行期符号与支 (c) 派生结构等价：均通过。"""
        params = (_PARAMS + ", dyn: Tensor[f16, (seq_len, D), Shared],"
                  " dst2: Tensor[f16, (seq_len, D), Register],"
                  " d1: Tensor[f16, (seq_len*2, D), Register],"
                  " d2: Tensor[f16, (seq_len+seq_len, D), Shared]")
        # Shared→Register（load 格）：同名符号维 + 同名 D。
        assert _run(_stmt("tis.load(dyn, dst2)"), params=params) == []
        # 派生结构等价（j*2 与 j+j 折叠等价——(c) 的数学等价面）。
        assert _run(_stmt("tis.load(d2, d1)"), params=params) == []


class TestContext:
    """R4：语境检查（E0407）——Async 限嵌套体、无值原语值位置封闭集
    （赋值右侧/调用实参/return 值）；Sync/barrier 运行时语义 2S 以
    「调用被接受」间接承载（test_mode_value_domain 已含 Sync 面）。"""

    def test_async_at_kernel_top_rejected(self):
        """Scenario 2：kernel 顶层 mode=Async → E0407 建议 produce/consume
        或 Sync。"""
        rs = _run(_stmt("tis.load(gl, sh, mode=Async)"))
        assert [r.code for r in rs] == ["E0407"]
        assert rs[0].category == "async-context"
        assert "produce/consume" in rs[0].suggestion and "Sync" in rs[0].suggestion

    def test_async_in_produce_accepted(self):
        """Scenario 3：produce 嵌套体内 Async 接受（完成保证让渡 Pipeline）。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(gl: Tensor[f16, (64,), Global], sh: Tensor[f16, (64,), Shared]):\n"
            "    pipe = tis.Pipeline(stages=2)\n"
            "\n"
            "    @pipe.produce\n"
            "    def p(j: int, buf):\n"
            "        tis.load(gl, sh, mode=Async)\n"
            "\n"
            "    return\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert check_module(tree, "nvidia_h200") == []

    def test_valueless_in_assign_rhs_rejected(self):
        """Scenario 4：`v = tis.load(...)` / `x = tis.barrier()` → E0407。"""
        rs = _run(_stmt("v = tis.load(gl, sh)"))
        assert [r.code for r in rs] == ["E0407"]
        assert rs[0].category == "value-position"
        assert "不产生值" in rs[0].suggestion and "表达式语句" in rs[0].suggestion
        rs = _run(_stmt("x = tis.barrier()"))
        assert [r.code for r in rs] == ["E0407"]
        assert rs[0].category == "value-position"

    def test_valueless_in_call_arg_and_return_rejected(self):
        """值位置封闭集另两支：调用实参与 return 值。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(gl: Tensor[f16, (64,), Global],"
            " sh: Tensor[f16, (64,), Shared]):\n"
            "    y = tis.cast(tis.load(gl, sh), f32)\n"   # load 为 cast 实参
            "    return tis.barrier()\n"                    # barrier 为 return 值
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        rs = check_module(tree, "nvidia_h200")
        assert [r.code for r in rs] == ["E0407", "E0407"]
        assert [r.line for r in rs] == [5, 6]  # 位置升序
        assert all(r.category == "value-position" for r in rs)

    def test_value_position_undefined_locations_not_rejected(self):
        """封闭集外（If/While 条件位）为 spec undefined——M1 不裁
        （design D6 收窄；负例固定当前裁决）。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(gl: Tensor[f16, (64,), Global],"
            " sh: Tensor[f16, (64,), Shared]):\n"
            "    if tis.barrier():\n"
            "        tis.load(gl, sh)\n"
            "    while tis.load(gl, sh):\n"
            "        break\n"
            "    return\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        assert check_module(tree, "nvidia_h200") == []

    def test_nested_call_in_arg_reports_inner(self):
        """store 实参位内嵌 barrier：内层命中（实参位置），外层语句位合法、
        dst 让渡不再产生 E04xx。"""
        source = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def k(gl: Tensor[f16, (64,), Global],"
            " sh: Tensor[f16, (64,), Shared]):\n"
            "    tis.store(gl, tis.barrier())\n"
            "    return\n"
        )
        tree, rejection = carrier.parse(source)
        assert rejection is None
        rs = check_module(tree, "nvidia_h200")
        assert [r.code for r in rs] == ["E0407"]
        assert rs[0].category == "value-position" and rs[0].line == 5

    def test_e0406_beats_e0407_same_call(self):
        """段内序：同调用结构违规（E0406 order 3）优先于值位置（E0407 order 4）。"""
        rs = _run(_stmt("v = tis.load(gl, sh, mode=Fast)"))
        assert [r.code for r in rs] == ["E0406"]
        assert rs[0].category == "mode-value"
