"""toolchain/cli 行为测试（R1×5S、R2×3S、R3×2S——change specs/toolchain/cli）。

直接调用 main(argv) 断言退出码/stdout/stderr；另以子进程覆盖
`python -m tilescript` 入口形态（Scenario 的字面命令形态）。
"""

import json
import subprocess
import sys

import pytest

from tilescript.cli import main

REPO_ROOT = __file__.rsplit("/tests/", 1)[0]


def _run(argv, capsys):
    capsys.readouterr()
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


VALID_KERNEL = (
    "import tis\n"
    "\n"
    "@tis.kernel\n"
    "def k(src: Pointer[f16, Global], n: int, out: Tensor[f16, (16,), Register]):\n"
    "    acc = tis.zeros((16,), f32, Register)\n"
    "    acc = out\n"
    "    return\n"
)

INVALID_KERNEL = (
    "import tis\n"
    "\n"
    "@tis.kernel\n"
    "def k(src: Pointer[f16], n: int):\n"
    "    while n > 0:\n"          # L5：E0105
    "        pass\n"
    "    x = [i for i in row]\n"  # L7：E0106
)


class TestR1Entrance:
    def test_missing_target_rejected(self, tmp_path, capsys):
        """Scenario：缺 --target → stderr 含清单、exit 2、源文件不读（不报文件不可读）。"""
        src = _write(tmp_path, "kernel.tis", VALID_KERNEL)
        code, out, err = _run(["compile", src], capsys)
        assert code == 2 and out == ""
        assert "nvidia_h200" in err and "ascend_910b" in err
        assert "--target" in err

    def test_missing_target_beats_missing_file(self, tmp_path, capsys):
        """缺 --target 时不触源文件：文件不存在仍只报 target（MUST NOT 读取源文件）。"""
        code, out, err = _run(["compile", "no_such_file.tis"], capsys)
        assert code == 2
        assert "不可读" not in err  # 未读文件，报的是 target 缺失

    def test_unregistered_target_rejected(self, tmp_path, capsys):
        """Scenario：--target foo_bar → exit 2、列出两成员、无 E0xxx。"""
        src = _write(tmp_path, "kernel.tis", VALID_KERNEL)
        code, out, err = _run(["compile", src, "--target", "foo_bar"], capsys)
        assert code == 2 and out == ""
        assert "nvidia_h200" in err and "ascend_910b" in err and "foo_bar" in err
        assert "E0" not in err

    def test_missing_source_path_rejected(self, capsys):
        """Scenario：无位置参数 → stderr 一行说明缺少源文件路径、exit 2。"""
        code, out, err = _run(["compile", "--target", "nvidia_h200"], capsys)
        assert code == 2 and out == ""
        assert "源文件路径" in err

    def test_unreadable_source_rejected(self, tmp_path, capsys):
        """Scenario：文件不存在 → exit 2、stderr 说明不可读。"""
        code, out, err = _run(
            ["compile", str(tmp_path / "no_such_file.tis"), "--target", "nvidia_h200"],
            capsys,
        )
        assert code == 2 and "不可读" in err

    def test_non_tis_extension_not_rejected(self, tmp_path, capsys):
        """Scenario：.py 扩展名合法源 → 不因扩展名拒绝，照常进语法检查。"""
        src = _write(tmp_path, "kernel.py", VALID_KERNEL)
        code, out, err = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        assert code == 0
        assert json.loads(out)["status"] == "incomplete"


class TestR2Serialization:
    def test_multi_rejection_json_sorted_exit1(self, tmp_path, capsys):
        """Scenario：两条拒绝 → 单 JSON 对象、五字段、位置序、exit 1。"""
        src = _write(tmp_path, "kernel.tis", INVALID_KERNEL)
        code, out, err = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        assert code == 1
        payload = json.loads(out)
        assert payload["status"] == "rejected"
        assert len(payload["rejections"]) == 2
        first, second = payload["rejections"]
        assert first["code"] == "E0105" and first["line"] == 5
        assert second["code"] == "E0106" and second["line"] == 7
        for item in payload["rejections"]:
            assert set(item.keys()) == {"code", "line", "col", "category", "suggestion"}
            assert item["line"] >= 1 and item["col"] >= 1

    def test_repeat_run_byte_identical(self, tmp_path, capsys):
        """Scenario：同输入连续两次 stdout 逐字节一致。"""
        src = _write(tmp_path, "kernel.tis", INVALID_KERNEL)
        _, out1, _ = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        _, out2, _ = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        assert out1 == out2

    def test_v1_field_set_frozen(self, tmp_path, capsys):
        """Scenario：字段集恰为 v1 冻结五字段（序列化面逐字段断言）。"""
        src = _write(tmp_path, "kernel.tis", INVALID_KERNEL)
        _, out, _ = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        assert list(json.loads(out).keys()) == ["status", "rejections"]
        assert list(json.loads(out)["rejections"][0].keys()) == [
            "code", "line", "col", "category", "suggestion",
        ]


class TestR3HonestExit:
    def test_syntax_pass_incomplete_exit0(self, tmp_path, capsys):
        """Scenario：已实现段（syntax + type-system + primitive-contract）零命中
        → incomplete + 段清单 + exit 0。"""
        src = _write(tmp_path, "kernel.tis", VALID_KERNEL)
        code, out, err = _run(["compile", src, "--target", "nvidia_h200"], capsys)
        assert code == 0
        payload = json.loads(out)
        assert payload["status"] == "incomplete"
        assert payload["implemented_stages"] == [
            "syntax", "type-system", "primitive-contract"]
        assert set(payload["pending_stages"]) == {
            "execution-structure", "numerics",
        }

    def test_status_values_only_rejected_or_incomplete(self, tmp_path, capsys):
        """Scenario：passed 不可达——本 change 全部输出路径的 status 封闭二值。"""
        src_ok = _write(tmp_path, "ok.tis", VALID_KERNEL)
        src_bad = _write(tmp_path, "bad.tis", INVALID_KERNEL)
        _, out_ok, _ = _run(["compile", src_ok, "--target", "nvidia_h200"], capsys)
        _, out_bad, _ = _run(["compile", src_bad, "--target", "nvidia_h200"], capsys)
        assert json.loads(out_ok)["status"] == "incomplete"
        assert json.loads(out_bad)["status"] == "rejected"


class TestSubprocessEntry:
    @pytest.mark.parametrize("target_flag", [["--target", "nvidia_h200"], []])
    def test_python_dash_m_entry(self, tmp_path, target_flag):
        """`python -m tilescript compile ...` 入口形态（Scenario 的字面命令）。"""
        src = _write(tmp_path, "kernel.tis", VALID_KERNEL)
        proc = subprocess.run(
            [sys.executable, "-m", "tilescript", "compile", src, *target_flag],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        if target_flag:
            assert proc.returncode == 0
            assert json.loads(proc.stdout)["status"] == "incomplete"
        else:
            assert proc.returncode == 2
            assert "nvidia_h200" in proc.stderr
