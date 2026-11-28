#!/usr/bin/env python3
"""Activations laid out by tile (tinyllm.tg.kernels.gemm.blocked_shape): the tiled kernel that reads them and the
kernels that write them, against the same kernels on rows.

    DEV=KFD+AMD uv run python tools/tile_layout_check.py

No model: random weights and inputs at small shapes.

- The tiled matrix kernel (Q8_0 and Q6_K, wide and narrow output blocks, 128- and 256-row blocks, with and
  without split-K) must give the same result bit for bit from activations in rows and by tile.
- The gated residual's norm kernel (`hc`) must write the same normalised streams in both layouts, bit for bit,
  with the same gates and 1 / rms, and all three must match NumPy. With the residual update of the sublayer
  before in it, the streams it writes must be tinygrad's sum bit for bit, and the rest the plain kernel's on
  those streams to rounding.
- The gated residual's mix (in the up projection's stores) must match NumPy from the fp32 streams and from the
  norm kernel's fp16 tiles, and the two must agree to fp16's rounding of the streams.
- The DeltaNet block's output kernel (`deltafuse`) must write the same values in both layouts, bit for bit.
- The sparse attention's gated output kernel (`qsa`) must match tinygrad's own expression, bit for bit.

Exits non-zero on any difference.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="accepted for the test suite; the check is small as it is")
    ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm import device

    device.configure("amd")  # before anything imports tinygrad
    import prefill_kernel_check as pkc
    from tinygrad.llm.gguf import ggml_data_to_tensor
    from tinygrad.llm.kernels.amd import Linear

    from tinyllm.tg import Tensor, dtypes, kernels
    from tinyllm.tg.kernels import deltafuse, gemm, hc, qsa

    kernels.install()
    rng, failures = np.random.default_rng(0), 0

    def report(name: str, got: np.ndarray, want: np.ndarray, bound: float = 0.0) -> None:
        nonlocal failures
        diff = float(np.abs(got.astype(np.float64) - want.astype(np.float64)).max())
        ok = bool(np.isfinite(got).all()) and diff <= bound
        failures += not ok
        exact = "the same bits" if bound == 0.0 else f"bound {bound:g}"
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max difference {diff:.3g} ({exact})")

    def rows_of(tiles: Any, tokens: int) -> Any:  # (row blocks, input blocks, rows, 32) -> (tokens, inputs)
        return tiles.permute(0, 2, 1, 3).reshape(tokens, -1)

    # the tiled matrix kernel
    shapes = (("q8_0", 256, 1024, 256), ("q8_0", 256, 1024, 384), ("q8_0", 64, 2048, 512), ("q6_k", 128, 1024, 256))
    for kind, out_f, in_f, tokens in shapes:
        fmt = pkc.FORMATS[kind]
        blocks = fmt.blocks(rng, out_f * in_f)
        raw = Tensor(blocks.view(np.uint8).reshape(-1)).realize()
        layer = Linear(in_f, out_f, bias=False)
        layer.weight = ggml_data_to_tensor(raw, out_f * in_f, fmt.ggml_type).reshape(out_f, in_f).cast("float16")
        x_np = rng.standard_normal((tokens, in_f)).astype(np.float16)
        want = layer(Tensor(x_np).realize()).realize().numpy()  # rows; the hook adopts the packed weights here
        ggml_type = gemm.takes_blocked(layer, tokens, raw.device)
        if ggml_type is None:
            failures += 1
            print(f"FAIL {kind} {out_f}x{in_f}, {tokens} rows: not a layer of the tiled kernel")
            continue
        blocks_m, blocks_k, bm, _ = gemm.blocked_shape(tokens, in_f)
        tiles = np.ascontiguousarray(x_np.reshape(blocks_m, bm, blocks_k, 32).transpose(0, 2, 1, 3))
        for unsplit in (False, True):
            saved = gemm.MIN_WORKGROUPS, gemm.MIN_WORKGROUPS_SMALL
            if unsplit:
                gemm.MIN_WORKGROUPS = gemm.MIN_WORKGROUPS_SMALL = 1
            try:
                got = gemm.blocked_linear(layer, ggml_type, Tensor(tiles).realize(), tokens).realize().numpy()
            finally:
                gemm.MIN_WORKGROUPS, gemm.MIN_WORKGROUPS_SMALL = saved
            name = f"matrix kernel {kind} {out_f}x{in_f}, {tokens} rows in blocks of {bm}"
            if unsplit:  # another number of splits sums in another order
                scale = float(np.abs(want).max())
                report(f"{name}, unsplit", got / scale, want / scale, 1e-5)
            else:
                report(name, got, want)

    # the gated residual's norm kernel
    tokens, count, dim = 512, 4, 2560
    weight = (1 + 0.1 * rng.standard_normal(count * dim)).astype(np.float32)
    inject = (rng.standard_normal((count, count * dim)) * 0.01).astype(np.float32)
    block = SimpleNamespace(
        norm=SimpleNamespace(weight=Tensor(weight).realize(), eps=1e-6),
        inject=SimpleNamespace(weight=Tensor(inject).realize()),
    )
    x = (rng.standard_normal((1, tokens, count, dim)) * (0.5 + rng.random((1, tokens, count, 1)))).astype(np.float32)
    xs = Tensor(x).contiguous().realize()
    inv = 1 / np.sqrt((x.astype(np.float64) ** 2).mean(-1) + 1e-6)
    normed = (x * inv[..., None]).reshape(1, tokens, count * dim) * weight
    gates = 2 / (1 + np.exp(-(normed @ inject.T.astype(np.float64)) / count))
    results = {}
    for blocked in (False, True):
        xn, g, iv, _ = hc.norm_inject(block, xs, blocked)
        xn = rows_of(xn, tokens).reshape(1, tokens, -1) if blocked else xn
        results[blocked] = [t.realize().numpy() for t in (xn, g, iv)]
    labels = ("normalised streams", "gates", "1 / rms")
    for label, by_tile, by_row in zip(labels, results[True], results[False], strict=True):
        report(f"norm kernel, {tokens} rows: {label}, tiles against rows", by_tile, by_row)
    report("norm kernel: normalised streams against NumPy", results[True][0], normed, 4e-3)
    report("norm kernel: gates against NumPy", results[True][1], gates, 1e-6)
    report("norm kernel: 1 / rms against NumPy", results[True][2], inv, 1e-6)

    # the same kernel with the residual update of the sublayer before: the streams it writes must be tinygrad's
    # own sum bit for bit; what it derives from them, the plain kernel's on those streams to rounding (the
    # compiler fuses the multiplies and adds of the two kernels' sums differently)
    out_np = rng.standard_normal((1, tokens, dim)).astype(np.float32)
    out_gates_np = (2 * rng.random((1, tokens, count))).astype(np.float32)
    out_t, out_gates_t = Tensor(out_np).realize(), Tensor(out_gates_np).realize()
    summed = (xs + out_t.unsqueeze(2) * out_gates_t.unsqueeze(-1)).contiguous().realize()
    plain = [t.realize().numpy() for t in hc.norm_inject(block, summed, True)[:3]]
    fused = hc.norm_inject(block, xs, True, (out_t, out_gates_t))
    fused_np = [t.realize().numpy() for t in fused[:3]]
    name = "norm kernel with the update"
    report(f"{name}: the streams against tinygrad's sum", fused[3].realize().numpy(), summed.numpy())
    for label, a, b, bound in zip(labels, fused_np, plain, (4e-3, 1e-6, 1e-6), strict=True):
        report(f"{name}: {label} against the kernel on the summed streams", a, b, bound)

    # the gated residual's mix, in the up projection's stores: from the fp32 streams, and from the norm kernel's
    # fp16 streams by tile
    low = 320
    fmt = pkc.FORMATS["q8_0"]
    raw = Tensor(fmt.blocks(rng, count * dim * low).view(np.uint8).reshape(-1)).realize()
    up = Linear(low, count * dim, bias=False)
    up.weight = ggml_data_to_tensor(raw, count * dim * low, fmt.ggml_type).reshape(count * dim, low).cast("float16")
    w_up = up.weight.numpy().astype(np.float64)
    block.up = up
    h_np = (0.5 * rng.standard_normal((1, tokens, low))).astype(np.float32)
    h = Tensor(h_np).realize()
    xn_tiles, _, iv, _ = hc.norm_inject(block, xs, True)
    xn_tiles, iv = xn_tiles.realize(), iv.realize()
    from_streams = hc.mix(block, h, xs, iv, None).realize().numpy()[0]
    from_tiles = hc.mix(block, h, xs, iv, xn_tiles).realize().numpy()[0]
    gate = 1 / (1 + np.exp(-(h_np[0].astype(np.float16).astype(np.float64) @ w_up.T)))  # (tokens, count * dim)
    iv32 = iv.numpy()[0].astype(np.float32)
    normed32 = (x[0] * iv32[..., None] * weight.reshape(count, dim)).astype(np.float64)  # the kernel's fp32 products
    normed16 = rows_of(xn_tiles, tokens).numpy().astype(np.float64).reshape(tokens, count, dim)
    want32 = (gate.reshape(tokens, count, dim) * normed32).mean(1)
    want16 = (gate.reshape(tokens, count, dim) * normed16).mean(1)
    scale = float(np.abs(want32).max())
    report("mix from the fp32 streams against NumPy", from_streams / scale, want32 / scale, 2e-5)
    report("mix from the fp16 tiles against NumPy", from_tiles / scale, want16 / scale, 2e-5)
    report("mix from the fp16 tiles against the fp32 streams", from_tiles / scale, from_streams / scale, 1e-3)

    # the DeltaNet block's output kernel
    tokens, heads, dim = 512, 6, 128
    core = Tensor(rng.standard_normal((1, heads, tokens, dim)).astype(np.float32)).realize()
    gate = Tensor(rng.standard_normal((tokens, heads * dim)).astype(np.float32)).realize()
    norm_weight = Tensor((1 + 0.1 * rng.standard_normal(dim)).astype(np.float32)).realize()
    tiles_shape = gemm.blocked_shape(tokens, heads * dim)
    outputs = []
    for tile in (0, tiles_shape[2]):
        z = Tensor.empty(*(tiles_shape if tile else (tokens, heads * dim)), dtype=dtypes.half, device=core.device)
        fxn = functools.partial(deltafuse._out_kernel, eps=1e-6, silu=True, tile=tile)
        z = Tensor.custom_kernel(z, core, gate, norm_weight, fxn=fxn)[0]
        outputs.append((rows_of(z, tokens) if tile else z).realize().numpy())
    report(f"DeltaNet output kernel, {tokens} rows: tiles against rows", outputs[1], outputs[0])

    # the sparse attention's gated output kernel
    tokens, heads, dim = 256, 6, 128
    attn_np = rng.standard_normal((1, heads, tokens, dim)).astype(np.float32)
    proj_np = rng.standard_normal((tokens, heads * 2 * dim)).astype(np.float32)
    attn, proj = Tensor(attn_np).realize(), Tensor(proj_np).realize()
    tiles_shape = gemm.blocked_shape(tokens, heads * dim)
    z = Tensor.empty(*tiles_shape, dtype=dtypes.half, device=attn.device)
    fxn = functools.partial(qsa._gated_out_kernel, heads=heads, dim=dim, tile=tiles_shape[2])
    got = rows_of(Tensor.custom_kernel(z, attn, proj, fxn=fxn)[0], tokens).realize().numpy()
    gate_view = proj.reshape(1, tokens, heads, 2, dim)[:, :, :, 1, :].reshape(1, tokens, heads * dim)
    own = (attn.transpose(1, 2).reshape(1, tokens, -1) * gate_view.sigmoid()).cast(dtypes.half).realize().numpy()[0]
    report(f"gated attention output kernel, {tokens} rows: against tinygrad's expression", got, own)

    print(f"tinyllm tile layout check: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
