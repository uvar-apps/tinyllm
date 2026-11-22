from __future__ import annotations

import json
from pathlib import Path

import pytest
from ggufgen import qwen35_kv, write_gguf

from tinyllm import cli


def test_inspect_text_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_gguf(tmp_path / "m.gguf", qwen35_kv(), [("blk.0.ffn_gate.weight", (256, 1024), "Q4_K")])
    assert cli.main(["inspect", str(path), "--ctx", "2048", "--slots", "2"]) == 0
    text = capsys.readouterr().out
    assert "tiny-qwen35  (qwen35)" in text and "Memory plan: 2 slots at 2048 tokens" in text
    assert cli.main(["inspect", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spec"]["blocks"] == 8


def test_inspect_exit_code_when_the_plan_does_not_fit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_gguf(tmp_path / "m.gguf", qwen35_kv(), [])
    # the context is clamped to the model's 4096 tokens, so only an absurd slot count overflows memory
    assert cli.main(["inspect", str(path), "--slots", "100000000"]) == 2
    assert "DOES NOT FIT" in capsys.readouterr().out


def test_missing_model_is_a_clean_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect", "/nonexistent/model.gguf"]) == 1
    assert "not found" in capsys.readouterr().err


def test_device_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["device", "--device", "cpu"]) == 0
    assert capsys.readouterr().out.strip().endswith("DEV=CPU")
    assert cli.main(["device", "--device", "PCI+AMD"]) == 1
    assert "userspace" in capsys.readouterr().err


def test_bench_arguments_parse() -> None:
    args = cli.build_parser().parse_args(["bench", "m.gguf", "--rows", "1,4,8", "--mode", "sample,logits"])
    assert args.rows == [1, 4, 8] and args.mode == ["sample", "logits"] and args.chunk == [32]
    assert args.tolerance == 0.03 and args.device is None
