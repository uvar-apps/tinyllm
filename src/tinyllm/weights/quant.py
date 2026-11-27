"""GGML tensor storage types: names, block geometry, and which ones have fast kernels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantType:
    id: int
    name: str
    block_elems: int  # weights per block
    block_bytes: int  # bytes per block
    upstream: bool = True  # tinygrad at the pinned commit can decode it
    packed_kernel: bool = False  # the amd_rdna kernels multiply on the packed bytes directly
    small_block: bool = False  # 32-weight blocks: packed kernels accept any row length divisible by 32

    @property
    def bits_per_weight(self) -> float:
        return 8 * self.block_bytes / self.block_elems

    @property
    def is_float(self) -> bool:
        return self.name in ("F32", "F16", "BF16", "F64")

    def nbytes(self, elements: int) -> int:
        return elements // self.block_elems * self.block_bytes


def _q(id: int, name: str, elems: int, nbytes: int, **kw: bool) -> QuantType:
    return QuantType(id, name, elems, nbytes, **kw)


_TYPES = [
    _q(0, "F32", 1, 4),
    _q(1, "F16", 1, 2),
    _q(2, "Q4_0", 32, 18),
    _q(3, "Q4_1", 32, 20),
    _q(6, "Q5_0", 32, 22),
    _q(7, "Q5_1", 32, 24),
    _q(8, "Q8_0", 32, 34, packed_kernel=True, small_block=True),
    _q(9, "Q8_1", 32, 36, upstream=False),
    _q(10, "Q2_K", 256, 84, packed_kernel=True),
    _q(11, "Q3_K", 256, 110, packed_kernel=True),
    _q(12, "Q4_K", 256, 144, packed_kernel=True),
    _q(13, "Q5_K", 256, 176, packed_kernel=True),
    _q(14, "Q6_K", 256, 210, packed_kernel=True),
    _q(15, "Q8_K", 256, 292, upstream=False),
    _q(16, "IQ2_XXS", 256, 66, packed_kernel=True),
    _q(17, "IQ2_XS", 256, 74, packed_kernel=True),
    _q(18, "IQ3_XXS", 256, 98, packed_kernel=True),
    _q(19, "IQ1_S", 256, 50, packed_kernel=True),
    _q(20, "IQ4_NL", 32, 18, packed_kernel=True, small_block=True),
    _q(21, "IQ3_S", 256, 110, packed_kernel=True),
    _q(22, "IQ2_S", 256, 82, packed_kernel=True),
    _q(23, "IQ4_XS", 256, 136, packed_kernel=True),
    _q(24, "I8", 1, 1),
    _q(25, "I16", 1, 2),
    _q(26, "I32", 1, 4),
    _q(27, "I64", 1, 8),
    _q(28, "F64", 1, 8),
    _q(29, "IQ1_M", 256, 56),
    _q(30, "BF16", 1, 2),
    _q(34, "TQ1_0", 256, 54, upstream=False),
    _q(35, "TQ2_0", 256, 66, upstream=False),
    _q(39, "MXFP4", 32, 17, packed_kernel=True, small_block=True),
    _q(40, "NVFP4", 64, 36, upstream=False),
    _q(41, "Q1_0", 128, 18),
    _q(42, "Q2_0", 64, 18),  # d * (q - 1), q in 0..3; tinyllm adds tinygrad's decoder (tinyllm.tg.kernels.moe_small)
    _q(43, "F8_E4M3", 1, 1, upstream=False),
]

BY_ID: dict[int, QuantType] = {t.id: t for t in _TYPES}
BY_NAME: dict[str, QuantType] = {t.name: t for t in _TYPES}


def lookup(type_id: int) -> QuantType:
    try:
        return BY_ID[type_id]
    except KeyError:
        raise ValueError(f"unknown GGML tensor type {type_id}") from None


PACKED, TINYLLM_Q4, TINYLLM_DENSE, DENSE_SMALL, GENERIC = (
    "packed",
    "tinyllm-q4",
    "tinyllm-dense",
    "dense-small",
    "generic",
)
DENSE_SMALL_MAX_OUT = 2048


def kernel_path(qtype: QuantType, out_features: int, in_features: int, tinyllm: bool = True) -> str:
    """Which AMD kernel family a linear layer of this storage type and shape runs on.

    Mirrors `Linear.__call__` and `find_packed` in tinygrad/llm/kernels at the pinned commit, plus the routing
    of tinyllm's own kernels (tinyllm.tg.kernels) unless `tinyllm` is False:
      packed         tinygrad: quantized weights multiplied in place; rows a multiple of 32 wide, and of 256
                     unless the type uses 32-weight blocks
      tinyllm-q4     tinyllm: Q4_0 and Q4_1, rows a multiple of 32 wide
      tinyllm-dense  tinyllm: BF16 and F16, rows a multiple of 32 wide
      dense-small    tinygrad: fp16 matrix-vector kernel for float weights with at most 2048 outputs
      generic        tinygrad's ordinary code generation on decoded weights: the slow path
    """
    if qtype.packed_kernel and in_features % 32 == 0 and (qtype.small_block or in_features % 256 == 0):
        return PACKED
    if tinyllm and in_features % 32 == 0:
        if qtype.name in ("Q4_0", "Q4_1"):
            return TINYLLM_Q4
        if qtype.name in ("F16", "BF16"):
            return TINYLLM_DENSE
    if qtype.name in ("F32", "F16", "BF16") and out_features <= DENSE_SMALL_MAX_OUT and in_features % 128 == 0:
        return DENSE_SMALL
    return GENERIC
