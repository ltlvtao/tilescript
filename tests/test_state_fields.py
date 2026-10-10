"""E0304 状态类字段类型约束与注册表测试（type-system spec R5「状态类」×3 + 注册表面）。

Scenario 映射：
- 纯 Register Tensor 字段类通过（L90）
- Shared scope 字段被拒绝（L96）
- Pointer 字段被拒绝（L102）
- 标量种类字段被拒绝（R5 delta 新增 Scenario）
"""

from tilescript.frontend import carrier
from tilescript.typecheck import annotations, state_fields, symbols


def _check(source, comptime=()):
    tree, rejection = carrier.parse(source)
    assert rejection is None, f"载体意外拒绝：{rejection}"
    return state_fields.check(tree, comptime_syms=frozenset(comptime))


_STATE_CLASS = (
    "import tis\n"
    "\n"
    "@tis.state\n"
    "class S:\n"
    "    O_acc: Tensor[f32, (64, 64), Register]\n"
)


class TestRegisterOnlyAccepted:
    def test_pure_register_fields_accepted_and_registered(self):
        """Scenario L90：纯 Register 字段类通过；注册表按类体声明序记录字段。"""
        registry, rejections = _check(_STATE_CLASS)
        assert rejections == []
        assert set(registry) == {"S"}
        fields = registry["S"].fields
        assert [name for name, _ in fields] == ["O_acc"]
        ftype = fields[0][1]
        assert ftype.dtype == "f32" and ftype.scope == "Register"

    def test_field_shape_symbol_phase_follows_kernel_comptime_set(self):
        """字段 shape 符号 comptime 性随 kernel comptime 形参集（同名同相）。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    O_acc: Tensor[f32, (BR, D), Register]\n"
        )
        registry, _ = _check(src, comptime=("BR",))
        dim = registry["S"].fields[0][1].dims[0]
        assert dim == annotations.SymbolDim("BR", True)
        assert registry["S"].fields[0][1].dims[1] == annotations.SymbolDim("D", False)


class TestE0304Rejections:
    def test_shared_scope_field_rejected(self):
        """Scenario L96：Shared scope 字段 → E0304。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    K: Tensor[f16, (64, 64), Shared]\n"
        )
        registry, rejections = _check(src)
        assert len(rejections) == 1
        r = rejections[0]
        assert r.code == "E0304" and r.category == "state-field-scope"
        assert (r.line, r.col) == (5, 8)  # 字段注解起点（col 1 起始）
        assert "Register" in r.suggestion

    def test_pointer_field_rejected(self):
        """Scenario L102：Pointer 字段 → E0304。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    p: Pointer[f16, Global]\n"
        )
        registry, rejections = _check(src)
        assert len(rejections) == 1
        r = rejections[0]
        assert r.code == "E0304" and r.category == "state-field-pointer"
        assert "kernel 形参" in r.suggestion or "形参" in r.suggestion

    def test_scalar_field_rejected(self):
        """R5 delta 新增 Scenario：`count: int` 标量字段 → E0304。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    count: int\n"
        )
        registry, rejections = _check(src)
        assert len(rejections) == 1
        r = rejections[0]
        assert r.code == "E0304" and r.category == "state-field-scalar"
        assert "Tensor" in r.suggestion

    def test_structurally_broken_annotation_reports_e0302_only(self):
        """同位置 E0302 优先：结构坏的字段不产 E0304（一注解一结果）。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    K: Tensor[f64, (64, 64), Shared]\n"
        )
        registry, rejections = _check(src)
        codes = [r.code for r in rejections]
        assert codes == ["E0302"]  # 未知 dtype 在前，无 E0304

    def test_multiple_fields_all_collected(self):
        """收集全部：合法字段不报，两个违规字段各一条。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    O_acc: Tensor[f32, (64,), Register]\n"
            "    K: Tensor[f16, (64,), Shared]\n"
            "    count: int\n"
        )
        registry, rejections = _check(src)
        assert [r.category for r in rejections] == ["state-field-scope",
                                                    "state-field-scalar"]
        # 合法字段仍入注册表（违规字段类型同样入表——绑定检查用）
        assert [n for n, _ in registry["S"].fields] == ["O_acc", "K", "count"]


class TestRegistry:
    def test_consume_annotation_resolves_state_type(self):
        """R5 Scenario：consume 签名状态类注解被接受（查注册表）。"""
        registry, _ = _check(_STATE_CLASS)
        tree, _ = carrier.parse("import tis\n\n@tis.consume\ndef f(st: S) -> None:\n    ...\n")
        fn = tree.body[1]
        t, r = annotations.parse_annotation(fn.args.args[0].annotation,
                                            registry=registry)
        assert r is None and t.class_name == "S"

    def test_first_registration_wins_for_duplicate_class_names(self):
        """同名状态类按首次注册（design D11 residual 2）。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    a: Tensor[f32, (64,), Register]\n"
            "\n"
            "\n"
            "@tis.state\n"
            "class S:\n"
            "    b: Tensor[f16, (64,), Register]\n"
        )
        registry, _ = _check(src)
        assert [n for n, _ in registry["S"].fields] == ["a"]

    def test_kernel_comptime_names_collected(self):
        """symbols：kernel 形参 comptime[int] 名进集，int 名不进。"""
        src = (
            "import tis\n"
            "\n"
            "@tis.kernel\n"
            "def kernel(x: Tensor[f32, (64,), Global],\n"
            "           seq_len: int,\n"
            "           D: comptime[int],\n"
            "           BR: comptime[int]):\n"
            "    ...\n"
        )
        tree, _ = carrier.parse(src)
        assert symbols.kernel_comptime_names(tree) == frozenset({"D", "BR"})
