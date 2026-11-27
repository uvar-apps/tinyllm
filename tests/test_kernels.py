"""tinyllm's kernels: detection, layout and routing on the CPU, the kernels themselves on the GPU."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from conftest import REPO

from tinyllm import tg


def _decoded(bits: np.ndarray, ggml_type: int, shape: tuple[int, int]):  # type: ignore[no-untyped-def]
    """A weight wrapped exactly as tinygrad's GGUF loader wraps one read from a file."""
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor

    raw = Tensor(bits.reshape(-1).view(np.uint8)).realize()
    return ggml_data_to_tensor(raw, bits.size, ggml_type).reshape(shape).cast("float16")


def test_find_dense_recognises_bf16_and_f16_and_nothing_else() -> None:
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import find_dense

    bits = np.arange(64 * 32, dtype=np.uint16).reshape(64, 32)
    for ggml_type in (30, 1):
        found = find_dense(_decoded(bits, ggml_type, (64, 32)))
        assert found is not None and found[0] == ggml_type
        words = found[1].numpy()
        assert words.dtype == np.uint32 and words.shape == (64 * 32 // 2,)
        assert words[0] == (1 << 16) | 0  # weights 0 and 1, low half first
    # arithmetic after the decode, a quantized tensor, a tensor not read from a file
    assert find_dense(_decoded(bits, 30, (64, 32)) * 2) is None
    q8 = Tensor(np.zeros(34 * 4, dtype=np.uint8)).realize()
    assert find_dense(ggml_data_to_tensor(q8, 128, 8).reshape(4, 32)) is None
    assert find_dense(Tensor.ones(64, 32).contiguous().realize()) is None


@pytest.mark.parametrize(
    ("out_features", "tokens", "expected"),
    [
        (248320, 1, (1, 8, 4)),
        (248320, 8, (8, 8, 4)),
        (5120, 32, (8, 8, 4)),  # 32 rows run as four tiles of 8
        (48, 3, (3, 8, 2)),
        (96, 6, (6, 8, 4)),
        (100, 12, (6, 4, 1)),  # 100 rows are not a multiple of 8 waves; 12 tokens split into two tiles of 6
    ],
)
def test_layout(out_features: int, tokens: int, expected: tuple[int, int, int]) -> None:
    from tinyllm.tg.kernels.dense import _layout

    tile, waves, rows = _layout(out_features, tokens)
    assert (tile, waves, rows) == expected
    assert tokens % tile == 0 and out_features % (waves * rows) == 0 and 1024 % (waves * 32) == 0


def test_hook_is_idempotent_and_leaves_cpu_layers_alone() -> None:
    from tinygrad.llm.kernels.amd import Linear

    from tinyllm.tg import Tensor, kernels

    kernels.install()
    hooked = Linear.__call__
    kernels.install()
    assert Linear.__call__ is hooked and kernels.installed()

    bits = (np.arange(16 * 64, dtype=np.uint32).astype(np.uint16) | 0x3C00).reshape(16, 64)  # finite bf16 values
    layer = Linear(64, 16, bias=False)
    layer.weight = _decoded(bits, 30, (16, 64))
    x = np.random.default_rng(0).standard_normal((2, 64)).astype(np.float32)
    got = layer(Tensor(x)).numpy()
    # on the CPU the layer is not eligible and keeps tinygrad's path (which rounds bf16 to fp16 first)
    assert layer._tinyllm_dense is None
    weights = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(got, x @ weights.T, rtol=1e-3, atol=1e-3)


def test_sparse_attention_hook_is_idempotent_and_leaves_other_layers_alone() -> None:
    from types import SimpleNamespace

    from tinygrad.llm.model import Indexer, TransformerBlock

    from tinyllm.tg import Tensor, kernels
    from tinyllm.tg.kernels import qsa

    kernels.install()
    hooked = (TransformerBlock._attention, TransformerBlock._indexer_mask, Indexer._init_state)
    kernels.install()
    assert (TransformerBlock._attention, TransformerBlock._indexer_mask, Indexer._init_state) == hooked

    idx = SimpleNamespace(qsa=True, top_k=2048, kpool=4, head_dim=128, rope_dim=64, rope_interleave=False)
    cfg = SimpleNamespace(max_context=32768, n_heads=24, n_kv_heads=2, head_dim=256, sliding_window=0)
    block = object.__new__(TransformerBlock)
    block.config, block.indexer = cfg, SimpleNamespace(idx=idx)
    x = Tensor.zeros(1, 1, 8)
    assert not qsa.serves(block, x, (0,))  # on the CPU tinygrad's path runs
    block.indexer = None
    assert not qsa.serves(block, x, (0,))  # no indexer: not a QSA layer
    block.indexer, cfg.max_context = SimpleNamespace(idx=idx), 2048
    assert not qsa.serves(block, x, (0,))  # the indexer keeps every position: dense attention


def _q4_decoded(ggml_type: int, rows: int, cols: int):  # type: ignore[no-untyped-def]
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor

    block = 18 if ggml_type == 2 else 20
    raw = Tensor(np.arange(rows * cols // 32 * block, dtype=np.uint32).astype(np.uint8)).realize()
    return ggml_data_to_tensor(raw, rows * cols, ggml_type).reshape(rows, cols).cast("float16")


def test_find_q4_recognises_q4_0_and_q4_1_only() -> None:
    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import find_dense, find_q4

    for ggml_type, block in ((2, 18), (3, 20)):
        found = find_q4(_q4_decoded(ggml_type, 8, 64), 64)
        assert found is not None and found[0] == ggml_type
        assert found[1].dtype == tg.dtypes.uint16 and found[1].shape == (8 * 64 // 32 * block // 2,)
        assert find_dense(_q4_decoded(ggml_type, 8, 64)) is None
    assert find_q4(_q4_decoded(2, 8, 64) + 1, 64) is None
    assert find_q4(_q4_decoded(2, 8, 64), 48) is None  # rows must be whole blocks
    q8 = Tensor(np.zeros(34 * 4, dtype=np.uint8)).realize()
    assert find_q4(ggml_data_to_tensor(q8, 128, 8).reshape(4, 32), 32) is None


def test_prefill_finders_recognise_their_own_types_only() -> None:
    from types import SimpleNamespace

    from tinygrad.llm.gguf import ggml_data_to_tensor

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import iq4xs, q6k, q8, q45k

    finders = {"q45k": q45k.find_q45k, "iq4xs": iq4xs.find_iq4xs, "q6k": q6k.find_q6k, "q8": q8.find_q8}
    # ggml type: (bytes per 256 weights, the finder that serves it)
    cases = {12: (144, "q45k"), 13: (176, "q45k"), 23: (136, "iq4xs"), 14: (210, "q6k"), 8: (272, "q8")}
    for ggml_type, (size, owner) in cases.items():
        raw = Tensor(np.zeros(2 * size, dtype=np.uint8)).realize()
        weight = ggml_data_to_tensor(raw, 2 * 256, ggml_type).reshape(2, 256).cast("float16")
        layer = SimpleNamespace(weight=weight, in_features=256)
        for name, find in finders.items():
            found = find(layer)
            assert (found is not None) == (name == owner), (ggml_type, name)
            if found is not None:
                assert found[0] == ggml_type


def test_q2_0_decodes_as_ggml_does() -> None:
    """ggml's dequantize_row_q2_0: d * (q - 1), weight j in bits 2(j % 4) of byte j // 4."""
    from tinygrad.llm import gguf as tg_gguf

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import moe_small

    moe_small.install_decoders()
    moe_small.install_decoders()  # once is enough, twice is harmless
    rng = np.random.default_rng(0)
    blocks = np.zeros(6, dtype=np.dtype([("d", "<f2"), ("qs", "u1", (16,))]))
    blocks["d"] = rng.uniform(-1, 1, 6).astype(np.float16)
    blocks["qs"] = rng.integers(0, 256, (6, 16), dtype=np.uint8)
    raw = Tensor(blocks.view(np.uint8).reshape(-1)).realize()
    got = tg_gguf.ggml_data_to_tensor(raw, 6 * 64, 42).reshape(6, 64).numpy()
    q = (blocks["qs"][:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    expected = blocks["d"].astype(np.float32)[:, None] * (q.reshape(6, 64).astype(np.float32) - 1)
    np.testing.assert_array_equal(got, expected)
    assert tg_gguf._GGML_QUANT[42] == (64, 18)


def test_find_small_recognises_q5_0_q5_1_and_q2_0_only() -> None:
    from tinygrad.llm import gguf as tg_gguf

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import moe_small

    moe_small.install_decoders()
    # ggml type: bytes per 256 weights; Q5_0 has Q5_K's size and Q2_0 is new
    sizes = {6: 176, 7: 192, 42: 72, 13: 176, 12: 144, 8: 272, 20: 144}
    for ggml_type, size in sizes.items():
        raw = Tensor(np.zeros(4 * size, dtype=np.uint8)).realize()
        weight = tg_gguf.ggml_data_to_tensor(raw, 4 * 256, ggml_type).reshape(2, 2, 256).cast("float16")
        found = moe_small.find_small(weight, 256)
        assert (found is not None) == (ggml_type in (6, 7, 42)), ggml_type
        if found is not None:
            word = moe_small.WORD[ggml_type]
            assert found[0] == ggml_type and found[1].dtype == word and found[1].numel() == 4 * size // word.itemsize
    raw = Tensor(np.zeros(4 * 192, dtype=np.uint8)).realize()
    assert moe_small.find_small(tg_gguf.ggml_data_to_tensor(raw, 1024, 7).reshape(2, 2, 256) * 2, 256) is None


@pytest.mark.parametrize(
    ("mapped", "cached"), [(True, True), (False, True), (True, False)], ids=["mapped", "preadv2", "pool"]
)
def test_ngram_rows_read_from_the_file_match_tinygrad_s_own(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch, mapped: bool, cached: bool
) -> None:
    """`tinyllm.tg.ngram.FileRows` writes the same host slots as `GGUFPackedTable.write`, cached rows or not: a
    write of a few positions row by row, one of many positions through the mapping (`mapped`) split over the
    thread pool. Not `cached`: no row is taken from the page cache by the first pass, so the pool reads them all."""
    from pathlib import Path

    from tinygrad.llm.gguf import GGUFPackedTable

    from tinyllm.tg import Tensor, ngram

    monkeypatch.setattr(ngram, "MAPPED", mapped)
    rows, width, row_bytes, header = 1000, 64, 36, 100  # IQ4_NL rows of 64 weights: 2 blocks of 18 bytes
    data = np.random.default_rng(1).integers(0, 256, header + rows * row_bytes, dtype=np.uint8)
    path = tmp_path / "table.bin"
    path.write_bytes(data.tobytes())
    view = Tensor(Path(path))[header : header + rows * row_bytes]
    ours, theirs = (
        ngram.FileRows(GGUFPackedTable((view, (rows, width), 20))),
        GGUFPackedTable((view, (rows, width), 20)),
    )
    ours._nowait = cached
    ids = [[3, 999, 0, 3], [500, 1, 2, 998]]
    for start in (0, 5, 5):  # the second write at 5 finds its rows in the page cache
        ours.write(start, ids, 200)
        theirs.write(start, ids, 200)
        ids = [[r * 7 % rows for r in position] for position in ids]
    many = [[(p * 37 + h * 101) % rows for h in range(4)] for p in range(100)]  # >= MAPPED_SPLIT positions
    ours.write(50, np.asarray(many) if mapped else many, 200)
    theirs.write(50, many, 200)
    np.testing.assert_array_equal(ours.inner.rows.numpy(), theirs.rows.numpy())
    assert ours.cold + ours.warm == 3 * 8 + 100 * 4
    assert cached or ours.cold == 3 * 8  # a few positions never go through the mapping
    got, expected = ours.read(5, 2, "CPU").numpy(), theirs.read(5, 2, "CPU").numpy()
    np.testing.assert_array_equal(got, expected)


def test_ngram_row_ids_match_tinygrad_s_hashing() -> None:
    """`tinyllm.tg.ngram.row_ids` (numpy) gives `PLELayer.write_rows`' rows: EOS tokens, starts near 0, and
    Qwen3.8-Flash-Next's multipliers (products near 2**63)."""
    import random
    from types import SimpleNamespace

    from tinygrad.llm.model import PLELayer

    from tinyllm.tg import ngram

    ple = SimpleNamespace(
        multipliers=[23703573157769, 20109073645365, 8052911324071],
        vocab_sizes=[20000003 + 20 * i for i in range(16)],
        eos=248044,
    )
    rng = random.Random(1)
    tokens = [rng.choice([248044, rng.randrange(248056), rng.randrange(248056)]) for _ in range(600)]
    got: list[list[list[int]]] = []
    fake = SimpleNamespace(ple=ple, max_context=1024, table=SimpleNamespace(write=lambda s, ids, n: got.append(ids)))
    for start, n in ((0, 1), (0, 2), (1, 1), (2, 5), (0, 300), (299, 301)):
        got.clear()
        PLELayer.write_rows(fake, tokens, start, n, 32)
        assert ngram.row_ids(ple, tokens, start, n).tolist() == got[0], (start, n)


def test_dense_mm_serves_multiples_of_16_rows_of_large_layers() -> None:
    from types import SimpleNamespace

    from tinyllm.tg import Tensor
    from tinyllm.tg.kernels import densemm

    layer = SimpleNamespace(out_features=5120)
    for rows, served in ((1, False), (8, False), (12, False), (16, True), (24, False), (32, True), (128, True)):
        assert densemm.serves(layer, Tensor.empty(rows, 64)) == served, rows
    assert densemm.serves(SimpleNamespace(out_features=40), Tensor.empty(32, 64)) is False
    assert densemm.serves(SimpleNamespace(out_features=48), Tensor.empty(32, 64)) is False  # DeltaNet gates stay
    assert densemm.serves(layer, Tensor.empty(2, 16, 64))  # rows are counted over all leading dimensions
    assert densemm._config(32, 5120, 30)[1] == "bf16" and densemm._config(32, 5120, 1)[1] == "f16"
    assert densemm._config(32, 96, 30)[0] == 1  # 3 groups of 32 are not a whole number of 2-group steps


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("dense", "dense_mm", "q4", "q45k", "iq4xs", "q6k", "q8", "attention", "moe", "moe_mm", "qsa", "hc")),
        ("all", ("dense", "dense_mm", "q4", "q45k", "iq4xs", "q6k", "q8", "attention", "moe", "moe_mm", "qsa", "hc")),
        ("q8,attention", ("q8", "attention")),
        ("none", ()),
        ("0", ()),
        ("q4", ("q4",)),
        (" dense , q4 ", ("dense", "q4")),
    ],
)
def test_kernel_selection_from_the_environment(value: str | None, expected: tuple[str, ...]) -> None:
    from tinyllm.tg.kernels import ENV, requested

    assert requested({} if value is None else {ENV: value}) == expected


def test_unknown_kernel_names_are_rejected() -> None:
    from tinyllm.tg.kernels import ENV, install, requested

    with pytest.raises(ValueError, match="unknown kernel"):
        requested({ENV: "q5"})
    with pytest.raises(ValueError, match="unknown kernel"):
        install(["fast"])


@pytest.mark.gpu
@pytest.mark.parametrize(
    "check",
    [
        ["tools/dense_kernel_check.py"],
        ["tools/dense_kernel_check.py", "--f16"],
        ["tools/q4_kernel_check.py", "--rows", "1,3,8,16,32"],
        ["tools/q4_kernel_check.py", "--type", "q4_1", "--rows", "1,3,8,16,32"],
        ["tools/prefill_kernel_check.py", "--type", "q6_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q6_k", "--rows", "16,32,64,128", "--groups", "8", "--no-packed"],
        ["tools/prefill_kernel_check.py", "--type", "q8_0", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q8_0", "--rows", "16,32,64,128", "--groups", "8", "--no-packed"],
        ["tools/prefill_kernel_check.py", "--type", "q4_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q4_k", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "q5_k", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "q5_k", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "iq4_xs", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "iq4_xs", "--rows", "16,128", "--configs", "1u,2p,8p,8u"],
        ["tools/prefill_kernel_check.py", "--type", "bf16", "--rows", "1,8,16,32,64"],
        ["tools/prefill_kernel_check.py", "--type", "bf16", "--rows", "16,128", "--configs", "1b,2h,8b,8h"],
        ["tools/prefill_kernel_check.py", "--type", "f16", "--rows", "1,8,16,32,64"],
        [
            "tools/prefill_kernel_check.py",
            *("--type", "q5_k", "--rows", "64,128", "--configs", "1u,1p"),
            *("--set", "q45k.SHARED=(64,2,2)", "--set", "q45k.PREFETCH=True"),
        ],
        ["tools/sampler_check.py"],
        ["tools/rollback_check.py"],
        ["tools/checkpoint_check.py"],
        ["tools/attention_rows_check.py"],
        ["tools/attention_rows_check.py", "--heads", "8,2", "--rows", "4", "--context", "1024"],
        ["tools/decode_mma_check.py"],
        ["tools/decode_mma_check.py", "--rows", "6"],
        ["tools/decode_mma_check.py", "--heads", "8,2", "--rows", "4", "--context", "4096"],
        ["tools/moe_kernel_check.py", "--type", "q5_0"],
        ["tools/moe_kernel_check.py", "--type", "q5_1"],
        ["tools/moe_kernel_check.py", "--type", "q2_0"],
        ["tools/qsa_check.py"],
        ["tools/moe_mm_check.py"],
        ["tools/moe_mm_check.py", "--fused", "--types", "q5_1,q8_0"],
        ["tools/moe_mm_check.py", "--types", "iq4_xs,iq3_xxs,iq2_s,iq4_nl,q5_0,q2_0"],
        ["tools/moe_mm_check.py", "--fused", "--types", "iq4_xs,iq3_xxs,iq2_s"],
        ["tools/moe_mm_check.py", "--skewed", "--types", "q4_k,q5_1,q8_0", "--tokens", "64,300"],
        ["tools/moe_mm_check.py", "--skewed", "--fused", "--types", "q5_1,q8_0", "--tokens", "64,300"],
        ["tools/moe_mm_check.py", "--quick", "--set", "WIDE_TILES=0"],
    ],
    ids=[
        "bf16",
        "f16",
        "q4_0",
        "q4_1",
        "q6_k",
        "q6_k-unpacked",
        "q8_0",
        "q8_0-unpacked",
        "q4_k",
        "q4_k-configs",
        "q5_k",
        "q5_k-configs",
        "iq4_xs",
        "iq4_xs-configs",
        "bf16-prefill",
        "bf16-prefill-configs",
        "f16-prefill",
        "q5_k-shared-prefetch",
        "gpu-sampler",
        "deltanet-rollback",
        "state-checkpoints",
        "rows-attention",
        "rows-attention-small",
        "decode-mma",
        "decode-mma-6-rows",
        "decode-mma-small",
        "moe-q5_0",
        "moe-q5_1",
        "moe-q2_0",
        "qsa",
        "moe-mm",
        "moe-mm-fused",
        "moe-mm-iq4_xs-types",
        "moe-mm-iq4_xs-types-fused",
        "moe-mm-skewed",
        "moe-mm-skewed-fused",
        "moe-mm-per-wave",
    ],
)
def test_kernel_on_the_gpu(check: list[str]) -> None:
    """Small shapes through every code path of each kernel: decode and matrix-core, ragged chunks, odd row counts."""
    for extra in ([],):
        whole = (
            "sampler_check.py",
            "rollback_check.py",
            "checkpoint_check.py",
            "attention_rows_check.py",
            "decode_mma_check.py",
            "qsa_check.py",
        )
        quick = [] if check[0].endswith(whole) else ["--quick"]
        cmd = [sys.executable, check[0], *quick, *check[1:], *extra]
        if check[0].endswith("dense_kernel_check.py"):
            cmd += ["--rows", "1,3,8,32"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=600, env=_gpu_env())
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
        assert ("tinyllm" in proc.stdout or "draws match" in proc.stdout) and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_resumes_from_checkpoints_past_the_sparse_attention_bound() -> None:
    """Qwen3.8-Flash-Next `UD-IQ4_XS`: resuming from a checkpoint gives the logits of the calls it replaces, bit for
    bit, before the 2,051-position bound (request 2 at 2,048) and past it (request 3, where the indexer selects),
    with the n-gram embedding's state in the checkpoint."""
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    cmd = [sys.executable, "tools/checkpoint_check.py", str(path), "--chunk", "1024"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=1200, env=_gpu_env())
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "all identical" in proc.stdout and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_verify_computes_rows_independently_and_rolls_back() -> None:
    """Qwen3.8-Flash-Next `UD-IQ4_XS` with its MTP block: a speculative verify's rows do not depend on the other rows
    (linear, expert and feed-forward layers bit for bit, mixers and attention to rounding), and a rollback puts back
    the state after the accepted row, the n-gram layer's included, after a verify of 4 drafts and after a copied
    block's of 15 (`tools/verify_rows_check.py`)."""
    from tinyllm.server.serve import find_mtp
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if find_mtp(path) is None:
        pytest.skip(f"no MTP block next to {path.name}")
    cmd = [sys.executable, "tools/verify_rows_check.py", str(path), "--copy", "15"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=1800, env=_gpu_env())
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "0 failure(s)" in proc.stdout and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_padded_prefill_call_ends_in_and_keeps_the_right_state() -> None:
    """Qwen3.8-Flash-Next `UD-IQ4_XS` with its MTP block: a prefill call padded past the prompt's end leaves every
    DeltaNet block and the n-gram layer in the state after its last real row, keeps the state 256 rows before its
    end for the checkpoint, and its logits choose as plain calls' do (`tools/padded_call_check.py`)."""
    from tinyllm.server.serve import find_mtp
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if find_mtp(path) is None:
        pytest.skip(f"no MTP block next to {path.name}")
    env = {**_gpu_env(), "TINYLLM_STATIC_SIZES": "768"}  # one padded program besides the full chunk's
    cmd = [sys.executable, "tools/padded_call_check.py", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=2400, env=env)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "0 failure(s)" in proc.stdout and "FAIL" not in proc.stdout


@pytest.mark.gpu
def test_flash_next_resumes_from_a_checkpoint_taken_inside_a_padded_call() -> None:
    """Qwen3.8-Flash-Next `UD-IQ4_XS` with its MTP block (`tools/checkpoint_check.py --mtp find`): the next request
    resumes at the state the prompt's padded last call kept, bit for bit whatever was decoded in between and up to
    rounding against calls that end there; the checkpoints taken while decoding stay bit for bit."""
    from tinyllm.server.serve import find_mtp
    from tinyllm.weights.resolve import find_model

    try:
        path = find_model("Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf")
    except FileNotFoundError as e:
        pytest.skip(str(e))
    if find_mtp(path) is None:
        pytest.skip(f"no MTP block next to {path.name}")
    cmd = [sys.executable, "tools/checkpoint_check.py", str(path), "--mtp", "find", "--speculative", "4"]
    cmd += ["--chunk", "1024"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=2400, env=_gpu_env())
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "up to rounding" in proc.stdout and "FAIL" not in proc.stdout


def _gpu_env() -> dict[str, str]:
    import os

    return {k: v for k, v in os.environ.items() if k not in ("TINYLLM_DEVICE", "DEV")}
