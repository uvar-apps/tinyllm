from __future__ import annotations

from pathlib import Path

from ggufgen import qwen35_kv, write_gguf

from tinyllm import inspection
from tinyllm.models import memory, spec
from tinyllm.weights import gguf

GIB = 1 << 30


def _qwen38_27b_kv() -> dict[str, object]:
    """The metadata of Qwen3.8-27B that determines its state layout."""
    kv = qwen35_kv(blocks=64, dim=5120)
    a = "qwen35"
    kv.update(
        {
            f"{a}.attention.head_count": 24,
            f"{a}.attention.head_count_kv": 4,
            f"{a}.attention.key_length": 256,
            f"{a}.ssm.state_size": 128,
            f"{a}.ssm.group_count": 16,
            f"{a}.ssm.time_step_rank": 48,
            f"{a}.ssm.inner_size": 6144,
        }
    )
    return kv


def test_qwen38_27b_state_layout(tmp_path: Path) -> None:
    s = spec.from_gguf(gguf.read(write_gguf(tmp_path / "m.gguf", _qwen38_27b_kv(), [])))
    assert (s.blocks, s.mtp_blocks, s.vocab_size) == (64, 1, 100)
    st = s.state
    assert st is not None
    assert (st.attention_layers, st.recurrent_layers) == (16, 48)
    # 16 layers x (keys + values) x 4 KV heads x 256 dims x fp16
    assert st.kv_bytes_per_token == 64 * 1024
    # 48 layers x 48 heads x 128 x 128 x fp32
    assert st.recurrent_bytes == 48 * 48 * 128 * 128 * 4 == 144 << 20
    # 48 layers x 3 past inputs x (6144 + 2*16*128) channels x fp32
    assert st.conv_bytes == 48 * 3 * 10240 * 4
    assert st.slot_bytes(32768) == 2 * GIB + st.checkpoint_bytes


def test_plain_attention_model_has_no_recurrent_state(tmp_path: Path) -> None:
    kv = {
        "general.architecture": "llama",
        "llama.block_count": 16,
        "llama.embedding_length": 2048,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
    }
    st = spec.from_gguf(gguf.read(write_gguf(tmp_path / "m.gguf", kv, []))).state
    assert st is not None and st.recurrent_layers == 0 and st.checkpoint_bytes == 0
    assert st.kv_bytes_per_token == 16 * 2 * 8 * 64 * 2  # head dimension defaults to width / heads


def test_unmodelled_architecture_has_no_layout(tmp_path: Path) -> None:
    kv = {"general.architecture": "qwen4exp", "qwen4exp.block_count": 48}
    assert spec.from_gguf(gguf.read(write_gguf(tmp_path / "m.gguf", kv, []))).state is None


def test_plan_totals_and_fit() -> None:
    layout = spec.StateLayout(16, 48, 64 * 1024, 144 << 20, 6 << 20)
    p = memory.plan(layout, weights=17 * GIB, context=32768, slots=4, host_total=123 * GIB, gpu_mappable=121 * GIB)
    assert p.slot_bytes == 2 * GIB + (150 << 20)
    assert p.state_store == 4 * p.slot_bytes  # default: one parked sequence per slot
    assert p.total == 17 * GIB + 8 * p.slot_bytes + memory.DEFAULT_SCRATCH
    assert p.limit == 121 * GIB and p.fits is True
    assert memory.plan(layout, 17 * GIB, 262144, 8, host_total=123 * GIB).fits is False
    assert memory.plan(layout, 17 * GIB, 4096, 1).fits is None  # unknown machine


def test_inspect_report_flags_generic_path(tmp_path: Path) -> None:
    tensors = [
        ("token_embd.weight", (256, 100), "Q4_K"),
        ("blk.0.ffn_gate.weight", (256, 1024), "Q4_K"),
        ("blk.3.attn_q.weight", (256, 4096), "Q5_1"),
        ("output.weight", (256, 100), "Q6_K"),
    ]
    path = write_gguf(tmp_path / "m.gguf", qwen35_kv(), tensors)
    r = inspection.report(path, context=1024, slots=2)
    assert r["types"]["Q5_1"] == 256 * 4096
    assert list(r["slow_linear_families"]) == ["blk.N.attn_q.weight [Q5_1]"]
    assert 0 < r["fast_kernel_share"] < 1
    assert r["plan"]["slots"] == 2 and r["plan"]["context"] == 1024
    text = inspection.render(r)
    assert "tiny-qwen35" in text and "generic path: blk.N.attn_q.weight [Q5_1]" in text and "Memory plan" in text


def test_inspect_reports_mtp_block_tensors_separately(tmp_path: Path) -> None:
    # qwen35_kv() declares 8 main blocks plus one MTP block, which is blk.8
    tensors = [
        ("blk.0.ffn_gate.weight", (256, 1024), "Q4_K"),
        ("blk.8.ffn_gate.weight", (256, 1024), "Q5_1"),
        ("blk.8.nextn.eh_proj.weight", (512, 256), "Q5_1"),
    ]
    r = inspection.report(write_gguf(tmp_path / "m.gguf", qwen35_kv(), tensors), context=1024, slots=1)
    assert r["fast_kernel_share"] == 1.0 and r["slow_linear_families"] == {}
    assert set(r["mtp_slow_linear_families"]) == {"blk.N.ffn_gate.weight [Q5_1]", "blk.N.nextn.eh_proj.weight [Q5_1]"}
    assert "MTP block, used only for speculative decoding" in inspection.render(r)
