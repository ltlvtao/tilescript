"""报告契约测试（syntax spec「语法拒绝报告契约」R8×3 + toolchain/cli R2 序列化面）。

Scenario 映射：
- 多条拒绝被收集并按位置排序（L172）
- 同位置多命中只报最早检查阶段（L177）
- 重复编译拒绝清单一致（L182，经 finalize 的确定性排序与 CLI 逐字节测试共同承载）
"""

from tilescript.frontend import report


def _mk(code, line, col, category, order, suggestion="恢复建议"):
    return report.Rejection(
        code=code, line=line, col=col, category=category,
        suggestion=suggestion, order=order,
    )


class TestRejectionDataclass:
    def test_to_dict_field_order_frozen(self):
        """序列化条目字段集与顺序固定：code/line/col/category/suggestion（R2 v1 冻结）。"""
        r = _mk("E0105", 12, 5, "while-loop", 5)
        assert list(r.to_dict().keys()) == ["code", "line", "col", "category", "suggestion"]
        assert r.to_dict() == {
            "code": "E0105", "line": 12, "col": 5,
            "category": "while-loop", "suggestion": "恢复建议",
        }


class TestFinalizeOrdering:
    def test_multi_rejections_sorted_by_position(self):
        """Scenario L172：多条拒绝按位置升序（12 行在前、30 行在后）。"""
        out = report.finalize([
            _mk("E0106", 30, 9, "list-comprehension", 6),
            _mk("E0105", 12, 5, "while-loop", 5),
        ])
        assert [r.code for r in out] == ["E0105", "E0106"]
        assert [r.line for r in out] == [12, 30]

    def test_same_position_earliest_checker_wins(self):
        """Scenario L177：同 (line, col) 多命中只报检查管线最早（order 最小）的一条。"""
        out = report.finalize([
            _mk("E0106", 12, 5, "lambda", 6),
            _mk("E0105", 12, 5, "while-loop", 5),
        ])
        assert len(out) == 1
        assert out[0].code == "E0105"

    def test_same_position_same_checker_keeps_collection_order(self):
        """并列（位置与 order 均同）时按确定性收集序（design D10）——传入序保持不变。"""
        first_collected = _mk("E0106", 7, 3, "lambda", 6, suggestion="第一条")
        second_collected = _mk("E0106", 7, 3, "starred", 6, suggestion="第二条")
        out = report.finalize([first_collected, second_collected])
        assert [r.suggestion for r in out] == ["第一条", "第二条"]
        # 逆序传入则逆序输出：证明键并列时无隐藏次级键、纯靠稳定排序保持收集序。
        out_reversed = report.finalize([second_collected, first_collected])
        assert [r.suggestion for r in out_reversed] == ["第二条", "第一条"]

    def test_empty_input_empty_output(self):
        assert report.finalize([]) == []


class TestDeterminism:
    def test_repeat_finalize_identical(self):
        """Scenario L182：同一输入重复运行清单逐条一致（稳定排序键）。"""
        rejections = [
            _mk("E0106", 30, 9, "list-comprehension", 6),
            _mk("E0105", 12, 5, "while-loop", 5),
            _mk("E0102", 3, 1, "top-level-statement", 2),
        ]
        once = [r.to_dict() for r in report.finalize(rejections)]
        twice = [r.to_dict() for r in report.finalize(rejections)]
        assert once == twice
