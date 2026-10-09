"""命令行入口（toolchain/cli：编译入口、拒绝清单 JSON 序列化、诚实退出）。

退出码封闭三值（R1/R2/R3）：
- 2 工具链入口拒绝（不触源文件、非 E0xxx）；
- 1 已实现检查段存在拒绝；
- 0 已实现段零命中（诚实 incomplete——pending 非空期间 MUST NOT 宣称完成）。
"""

import argparse
import json
import pathlib
import sys

from . import frontend, hal


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 工具链入口拒绝（R1）：--target 校验先于源文件读取。
    if args.target is None:
        print(f"拒绝：缺少必选参数 --target；合法取值：{hal.registered_targets_line()}",
              file=sys.stderr)
        return 2
    if args.target not in hal.REGISTERED_TARGETS:
        print(f"拒绝：未登记的目标标识符 {args.target}；"
              f"合法取值：{hal.registered_targets_line()}", file=sys.stderr)
        return 2
    if args.source is None:
        print("拒绝：缺少源文件路径位置参数", file=sys.stderr)
        return 2
    try:
        source_text = pathlib.Path(args.source).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        print(f"拒绝：源文件不可读：{args.source}", file=sys.stderr)
        return 2

    # 语法段（五段管线的第一段；其余段由后续 change 实现，诚实列出）。
    rejections = frontend.check_module(source_text)
    if rejections:
        payload = {
            "status": "rejected",
            "rejections": [r.to_dict() for r in rejections],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    payload = {
        "status": "incomplete",
        "implemented_stages": [frontend.STAGE_NAME],
        "pending_stages": [s for s in frontend.ALL_STAGES if s != frontend.STAGE_NAME],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tilescript",
        description="TileScript 编译器（确定性编译、结构化拒绝清单）",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compile_cmd = sub.add_parser("compile", help="编译 TileScript 源文件")
    # source 用 nargs="?" 自校验：保证「缺少源文件路径」的 stderr 一行说明
    # 与 exit 2 一致（R1 Scenario 4），不混入 argparse 通用 usage 错误。
    compile_cmd.add_argument("source", nargs="?")
    # --target 不用 choices：拒绝说明须含登记表清单且与通用用法错误区分（design D7）。
    compile_cmd.add_argument("--target", default=None)
    return parser
