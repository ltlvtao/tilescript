"""HAL 能力描述数据层测试（hal/capability-descriptions「字段集」+「列表顺序即优先序」）。

登记值逐字段对齐 spec v1：nvidia_h200 与 ascend_910b 两 Scenario 直测；
Auto 选择（列表第一项）与小写 warp/block ↔ 语言值映射为 E0402/reduce
支持面判定的数据前提。
"""

from tilescript import hal


class TestRegisteredValues:
    def test_nvidia_h200_values(self):
        """Scenario：nvidia_h200 登记值（mma_shapes 两项、reduce_scopes 两项）。"""
        cap = hal.capability("nvidia_h200")
        assert cap.name == "nvidia_h200"
        assert cap.mma_shapes == ((16, 8, 16), (16, 8, 32))
        assert cap.reduce_scopes == ("warp", "block")

    def test_ascend_910b_values(self):
        """Scenario：ascend_910b 登记值（单形状、reduce_scopes 仅 block）。"""
        cap = hal.capability("ascend_910b")
        assert cap.name == "ascend_910b"
        assert cap.mma_shapes == ((16, 16, 16),)
        assert cap.reduce_scopes == ("block",)

    def test_unregistered_target_raises(self):
        """未登记标识符不构成合法输入（入口拒绝已由 CLI 承载；数据层防御）。"""
        try:
            hal.capability("foo_bar")
        except KeyError:
            pass
        else:
            raise AssertionError("未登记目标应拒绝")

    def test_persistent_kernel_registered_values(self):
        """persistent_kernel 登记值（v1 字段集：h200=true / ascend=false）——
        E0506 入口 HAL 支持面判定的唯一数据源（execution change tasks 1.1）。"""
        assert hal.capability("nvidia_h200").persistent_kernel is True
        assert hal.capability("ascend_910b").persistent_kernel is False


class TestAutoSelection:
    def test_auto_mma_is_first_item(self):
        """Scenario：dot Auto 选择第一项（两目标各自唯一确定）。"""
        assert hal.auto_mma("nvidia_h200") == (16, 8, 16)
        assert hal.auto_mma("ascend_910b") == (16, 16, 16)

    def test_auto_reduce_scope_first_item_mapping(self):
        """Scenario：reduce Auto 选择因目标不同（小写→语言值映射）。"""
        assert hal.auto_reduce_scope("nvidia_h200") == "Warp"
        assert hal.auto_reduce_scope("ascend_910b") == "Block"

    def test_explicit_scope_support(self):
        """显式 scope 支持面判定用同一列表（warp/block ↔ Warp/Block）。"""
        assert hal.language_scope_supported("nvidia_h200", "Warp") is True
        assert hal.language_scope_supported("nvidia_h200", "Block") is True
        assert hal.language_scope_supported("ascend_910b", "Warp") is False
        assert hal.language_scope_supported("ascend_910b", "Block") is True

    def test_supported_scopes_line(self):
        """E0408 reduce 支持面报告：目标支持清单文本（登记顺序、语言值）。"""
        assert hal.supported_scopes_line("nvidia_h200") == "Warp/Block"
        assert hal.supported_scopes_line("ascend_910b") == "Block"

    def test_mma_shapes_line(self):
        """E0402 报告面：全部支持形状按登记顺序。"""
        assert hal.mma_shapes_line("nvidia_h200") == "(16, 8, 16)/(16, 8, 32)"
        assert hal.mma_shapes_line("ascend_910b") == "(16, 16, 16)"
