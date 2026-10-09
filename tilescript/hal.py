"""HAL 登记表标识符集（toolchain/cli R1：--target 合法取值为且仅为登记表成员）。

数据来源：openspec/specs/hal/capability-descriptions「标识符登记表」v1 封闭二成员。
YAML 载体与字段加载延期至首个消费段 change（design D2）；此处仅承载标识符层校验。
排序输出（拒绝说明中的合法取值清单）按字母序，保证逐字节一致（design D10）。
"""

REGISTERED_TARGETS = frozenset({"nvidia_h200", "ascend_910b"})


def registered_targets_line() -> str:
    """拒绝说明中使用的合法取值清单（确定性排序）。"""
    return "/".join(sorted(REGISTERED_TARGETS))
