"""HAL 登记表与能力描述（toolchain/cli R1 标识符集 + hal/capability-descriptions 字段加载）。

登记值来源：openspec/specs/hal/capability-descriptions「能力描述字段集」v1 登记
（静态数据、同目标逐字段一致、无设备在环——冻结常量载体满足契约四要点；
YAML 文件载体仍为延期项，design D2）。列表顺序即优先序：mma_shapes/
reduce_scopes 第一项即 Auto 选择；小写 warp/block ↔ 语言值 Warp/Block。
排序输出（拒绝说明中的清单）按字母序或登记序，保证逐字节一致。
"""

from dataclasses import dataclass

REGISTERED_TARGETS = frozenset({"nvidia_h200", "ascend_910b"})


def registered_targets_line() -> str:
    """拒绝说明中使用的合法取值清单（确定性排序）。"""
    return "/".join(sorted(REGISTERED_TARGETS))


@dataclass(frozen=True)
class Capability:
    """能力描述（消费面：mma_shapes / reduce_scopes / persistent_kernel；
    其余字段随消费段加载）。"""

    name: str
    mma_shapes: tuple          # 三元组元组，登记顺序即 Auto 优先序
    reduce_scopes: tuple       # 小写字面 "warp"/"block"，登记顺序即 Auto 优先序
    persistent_kernel: bool    # 入口 HAL 支持面（E0506 判定数据源）


_CAPABILITIES = {
    "nvidia_h200": Capability(
        name="nvidia_h200",
        mma_shapes=((16, 8, 16), (16, 8, 32)),
        reduce_scopes=("warp", "block"),
        persistent_kernel=True,
    ),
    "ascend_910b": Capability(
        name="ascend_910b",
        mma_shapes=((16, 16, 16),),
        reduce_scopes=("block",),
        persistent_kernel=False,
    ),
}

# 小写登记值 ↔ 语言层 scope 值（hal spec「列表顺序即优先序」映射表）。
_SCOPE_MAP = {"warp": "Warp", "block": "Block"}


def capability(target: str) -> Capability:
    """登记目标的能力描述（未登记目标 KeyError——CLI 入口已挡，此处防御）。"""
    return _CAPABILITIES[target]


def auto_mma(target: str) -> tuple:
    """dot mma=Auto 的确定性选择：列表第一项。"""
    return capability(target).mma_shapes[0]


def auto_reduce_scope(target: str) -> str:
    """reduce scope=Auto 的确定性选择：列表第一项的语言值映射。"""
    return _SCOPE_MAP[capability(target).reduce_scopes[0]]


def language_scope_supported(target: str, language_scope: str) -> bool:
    """显式 reduce scope 支持面判定（语言值 → 小写列表成员）。"""
    lowered = {v: k for k, v in _SCOPE_MAP.items()}[language_scope]
    return lowered in capability(target).reduce_scopes


def supported_scopes_line(target: str) -> str:
    """E0408 reduce 支持面报告的目标支持清单（登记顺序、语言值）。"""
    return "/".join(_SCOPE_MAP[s] for s in capability(target).reduce_scopes)


def mma_shapes_line(target: str) -> str:
    """E0402 报告面：全部支持形状按登记顺序。"""
    return "/".join(f"({m}, {n}, {k})" for m, n, k in capability(target).mma_shapes)
