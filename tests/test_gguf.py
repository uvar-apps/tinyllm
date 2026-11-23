from __future__ import annotations

import struct
from pathlib import Path

import pytest
from ggufgen import qwen35_kv, write_gguf

from tinyllm.weights import gguf, quant


def test_reads_metadata_and_tensors(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf",
        {"general.architecture": "llama", "llama.block_count": 2, "x.flag": True, "x.scale": 0.5, "x.list": [1, 2, 3]},
        [("blk.0.attn_q.weight", (256, 64), "Q4_K"), ("output_norm.weight", (64,), "F32")],
    )
    g = gguf.read(path)
    assert g.version == 3 and g.architecture == "llama"
    assert g.get("block_count") == 2 and g.kv["x.flag"] is True and g.kv["x.list"] == [1, 2, 3]
    assert g.kv["x.scale"] == pytest.approx(0.5)
    q, norm = g.tensors
    # GGUF stores dimensions innermost first; shapes are reported row-major, (out, in) for a linear weight
    assert q.shape == (64, 256) and q.qtype.name == "Q4_K" and q.nbytes == 64 * 144
    assert norm.shape == (64,) and norm.nbytes == 256
    assert g.tensor_bytes == 64 * 144 + 256


def test_long_arrays_are_summarised_not_loaded(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", qwen35_kv(vocab=500), [])
    g = gguf.read(path, max_array_items=64)
    assert isinstance(g.kv["tokenizer.ggml.tokens"], gguf.ArraySummary)
    assert g.count("tokenizer.ggml.tokens") == 500
    assert gguf.read(path, max_array_items=1000).count("tokenizer.ggml.tokens") == 500


def test_rejects_non_gguf(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(gguf.GGUFError, match="not a GGUF"):
        gguf.read(bad)


def test_rejects_string_length_past_end_of_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1 << 40) + b"key")
    with pytest.raises(gguf.GGUFError, match="past the end"):
        gguf.read(bad)


def test_rejects_implausible_counts(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1 << 60, 0))
    with pytest.raises(gguf.GGUFError, match="implausible"):
        gguf.read(bad)


def test_rejects_tensor_data_past_end_of_file(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf", {"general.architecture": "llama"}, [("w", (256, 8), "Q8_0")], truncate_data=64
    )
    with pytest.raises(gguf.GGUFError, match="extends past"):
        gguf.read(path)


def test_rejects_unknown_tensor_type(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama"}, [("w", (32,), "F32")])
    raw = bytearray(path.read_bytes())
    at = raw.index(b"w") + 1 + 4 + 8  # name byte, dimension count, one dimension
    raw[at : at + 4] = struct.pack("<I", 999)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="unknown GGML tensor type 999"):
        gguf.read(path)


def test_split_model_is_merged(tmp_path: Path) -> None:
    kv = {"general.architecture": "llama", "split.count": 2, "split.no": 0}
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", kv, [])
    write_gguf(tmp_path / "m-00002-of-00002.gguf", {"split.no": 1}, [("w", (256, 4), "Q6_K")])
    g = gguf.read(first)
    assert len(g.parts) == 2 and [t.name for t in g.tensors] == ["w"] and g.tensors[0].part == 1


def test_split_model_needs_the_first_part_and_all_parts(tmp_path: Path) -> None:
    kv = {"general.architecture": "llama", "split.count": 2, "split.no": 0}
    with pytest.raises(gguf.GGUFError, match="missing part"):
        gguf.read(write_gguf(tmp_path / "m-00001-of-00002.gguf", kv, []))
    with pytest.raises(gguf.GGUFError, match="first part"):
        gguf.read(write_gguf(tmp_path / "other.gguf", kv, []))


def test_block_geometry_matches_ggml() -> None:
    assert quant.BY_NAME["Q4_K"].bits_per_weight == 4.5
    assert quant.BY_NAME["Q8_0"].bits_per_weight == 8.5
    assert quant.BY_NAME["Q6_K"].nbytes(512) == 420
    assert quant.BY_NAME["BF16"].is_float and not quant.BY_NAME["IQ4_XS"].is_float


@pytest.mark.parametrize(
    ("type_name", "out", "inp", "expected"),
    [
        ("Q4_K", 17408, 5120, quant.PACKED),
        ("Q4_K", 16, 288, quant.GENERIC),  # 256-weight blocks need rows that are a multiple of 256
        ("Q8_0", 16, 288, quant.PACKED),  # 32-weight blocks only need a multiple of 32
        ("Q5_1", 64, 256, quant.GENERIC),  # legacy type without a packed kernel
        ("Q4_0", 17408, 5120, quant.TINYLLM_Q4),
        ("Q4_1", 64, 288, quant.TINYLLM_Q4),
        ("BF16", 1024, 5120, quant.TINYLLM_DENSE),
        ("BF16", 248320, 5120, quant.TINYLLM_DENSE),  # the output head of the Q8_K_XL file
        ("F16", 100, 64, quant.TINYLLM_DENSE),
        ("F16", 100, 48, quant.GENERIC),  # rows must be a multiple of 32 wide
        ("F32", 48, 5120, quant.DENSE_SMALL),
    ],
)
def test_kernel_path(type_name: str, out: int, inp: int, expected: str) -> None:
    assert quant.kernel_path(quant.BY_NAME[type_name], out, inp) == expected


def test_kernel_path_without_tinyllm_kernels() -> None:
    assert quant.kernel_path(quant.BY_NAME["Q4_0"], 17408, 5120, tinyllm=False) == quant.GENERIC
    assert quant.kernel_path(quant.BY_NAME["BF16"], 248320, 5120, tinyllm=False) == quant.GENERIC
    assert quant.kernel_path(quant.BY_NAME["BF16"], 1024, 5120, tinyllm=False) == quant.DENSE_SMALL


def test_header_only_accepts_a_truncated_file(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf", {"general.architecture": "llama"}, [("w", (256, 8), "Q8_0")], truncate_data=10**9
    )
    with pytest.raises(gguf.GGUFError, match="extends past"):
        gguf.read(path)
    g = gguf.read(path, header_only=True)
    assert [t.name for t in g.tensors] == ["w"] and g.tensors[0].nbytes == 8 * 34 * 8


def test_header_only_reads_just_the_first_part(tmp_path: Path) -> None:
    kv = {"general.architecture": "llama", "split.count": 2, "split.no": 0}
    g = gguf.read(write_gguf(tmp_path / "m-00001-of-00002.gguf", kv, [("a", (32,), "F32")]), header_only=True)
    assert len(g.parts) == 1 and [t.name for t in g.tensors] == ["a"]
