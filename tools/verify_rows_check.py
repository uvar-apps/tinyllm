#!/usr/bin/env python3
"""The speculative verify of a model with residual streams (Qwen3.8-Flash-Next) against one-token steps.

    DEV=KFD+AMD uv run python tools/verify_rows_check.py <...-00001-of-00004.gguf> [--mtp mtp.gguf] [--text code.txt]

Flash-Next's verify runs its K + 1 rows together, on decoding's kernels, and is not bit-exact against one-token
decoding as the 27B's is (tinyllm.tg.speculative): the mixers and the sparse attention round differently for
several rows (1e-7), and decoding's 8-bit activations turn one flipped rounding into 1e-3 within a block. This
tool checks what must hold all the same, after a prompt long enough for the sparse attention to select:

1. Rows are computed independently. Every linear layer, expert layer and feed-forward of a few blocks gives, bit
   for bit, the same first row for one row and for K + 1; the mixers, and the DeltaNet and attention sublayers
   from the same state, within 1e-5.
2. The rollback restores the right row. From the state after the prompt, the next K + 1 tokens of the text run one
   per call, then as one recorded verify. After `restore_from(j)` every state, the DeltaNet blocks' and the n-gram
   layer's convolution, is far closer to the one-token run's after row j than that run's next row is (a wrong row
   would be as far as the next one; the rounding above leaves 1e-7 to a few percent).

With --copy N the second check also runs for the verify of a copied block, N + 1 rows. Those rows run on the
matrix-core kernels, half-precision activations where decoding's are 8-bit, so the first check does not apply to
them.

It prints the verify's logit differences per row for the record. Exits non-zero when a check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXACT, ROUNDING = 0.0, 1e-5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mtp", default=None, help="the MTP block's GGUF (default: found as `tinyllm serve` does)")
    ap.add_argument("--text", default=None, help="a text file for the prompt (default: this repository's sources)")
    ap.add_argument("-k", type=int, default=3, help="drafts per cycle: the verify has k + 1 rows")
    ap.add_argument("--prompt", type=int, default=2300, help="prompt tokens")
    ap.add_argument("--copy", type=int, default=0, help="also check the rollback of a copied block of this many drafts")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    from tinyllm.server.serve import find_mtp
    from tinyllm.tg import Tensor, dtypes  # chooses the device, before anything imports tinygrad
    from tinyllm.tg import speculative as spec
    from tinyllm.tg.kernels.hook import amd
    from tinyllm.tg.upstream import UpstreamSession

    Linear, ExpertWeights = amd.Linear, amd.ExpertWeights

    model_path = Path(args.model).expanduser()
    mtp = Path(args.mtp).expanduser() if args.mtp else find_mtp(model_path)
    if mtp is None:
        raise SystemExit(f"no MTP block found for {model_path.name}; pass --mtp")
    k, n_prompt = args.k, args.prompt
    most = max(k, args.copy)
    s = UpstreamSession(model_path, 8192, rows=1, chunk=2048, logits=True, speculative=k, mtp_path=mtp, copy=args.copy)
    model, cfg = s.model, s.model.config
    if args.text:
        text = Path(args.text).read_text()
    else:  # code, as the model is used: enough of this repository's own sources
        text = "\n".join(p.read_text() for p in sorted((Path(__file__).parent.parent / "src").rglob("*.py"))[:40])
    ids = s.tokenizer().encode(text)[: n_prompt + most + 1]
    if len(ids) < n_prompt + most + 1:
        raise SystemExit(f"the text has {len(ids)} tokens, {n_prompt + most + 1} are needed")
    s.prefill(ids[:n_prompt])
    rows = ids[n_prompt : n_prompt + k + 1]
    tensors = s._state_tensors()  # per DeltaNet block its recurrent and convolution state, then the n-gram layer's
    kinds = [("recurrent", "convolution")[i % 2] for i in range(len(tensors) - 1)] + ["n-gram convolution"]
    base = [t.numpy().copy() for t in tensors]
    rng = np.random.default_rng(0)
    failures = 0

    def check(name: str, value: float, limit: float) -> None:
        nonlocal failures
        ok = value <= limit
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {value:.3g} (limit {limit:g})")

    def states() -> list[np.ndarray]:
        return [t.numpy().copy() for t in tensors]

    def put(saved: list[np.ndarray]) -> None:
        Tensor.realize(*(t.assign(Tensor(a)) for t, a in zip(tensors, saved, strict=True)))

    def at(position: int) -> None:
        s.pos = position
        del s._history[position:]

    def rel(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(b))), 1e-30)

    def randn(*shape: int) -> Tensor:
        return Tensor(rng.standard_normal(shape).astype(np.float32)).realize()

    # ---- 1. rows are computed independently ----
    def layers(obj: object, prefix: str = "", depth: int = 0):  # type: ignore[no-untyped-def]
        for name, value in vars(obj).items():
            if isinstance(value, (Linear, ExpertWeights)):
                yield prefix + name, value
            elif depth < 2 and hasattr(value, "__dict__") and not isinstance(value, Tensor):
                yield from layers(value, f"{prefix}{name}.", depth + 1)

    pos = s._v_verify.bind(n_prompt)
    blocks = [type(b).__name__ for b in model.blk]
    picked = sorted(
        {
            next(i for i, name in enumerate(blocks) if "DeltaNet" in name),
            next(i for i, b in enumerate(model.blk) if getattr(b, "indexer", None) is not None),
            cfg.ple.layer if cfg.ple is not None else 0,
            len(blocks) // 2,
        }
    )
    for i in picked:
        b = model.blk[i]
        worst = 0.0
        for name, layer in layers(b):
            if isinstance(layer, Linear):
                if getattr(layer, "groups", 1) != 1:
                    continue
                x = randn(1, k + 1, int(layer.in_features))
                many, one = layer(x).float().numpy()[0, 0], layer(x[:, :1].contiguous()).float().numpy()[0, 0]
            else:
                slots, width = cfg.num_experts_per_tok, int(layer.weight.shape[2])
                picks = np.stack([rng.permutation(cfg.num_experts)[:slots] for _ in range(k + 1)])
                sel = Tensor(picks[None].astype(np.int32))
                x = randn(1, k + 1, slots if "down" in name else 1, width)
                many = layer(sel, x).float().numpy()[0, 0]
                one = layer(sel[:, :1].contiguous(), x[:, :1].contiguous()).float().numpy()[0, 0]
            worst = max(worst, rel(many, one))
        check(f"block {i} linear and expert layers, 1 row against {k + 1}", worst, EXACT)
        h = randn(1, k + 1, cfg.dim)
        t = Tensor([rows], dtype=dtypes.int32)
        many = b._feed_forward(h, t).float().numpy()[0, 0]
        one = b._feed_forward(h[:, :1].contiguous(), t[:, :1].contiguous()).float().numpy()[0, 0]
        check(f"block {i} feed-forward", rel(many, one), EXACT)
        xs = randn(1, k + 1, cfg.hc.count, cfg.dim)
        worst = 0.0
        for mixer in (b.hc_attn, b.hc_ffn):
            many = mixer(xs, lambda m, h=h: (h, None))[0].float().numpy()[0, 0]
            one = mixer(xs[:, :1].contiguous(), lambda m, h=h: (h[:, :1], None))[0].float().numpy()[0, 0]
            worst = max(worst, rel(many, one))
        check(f"block {i} mixers", worst, ROUNDING)
        out = []
        for n in (k + 1, 1):
            put(base)
            at(n_prompt)
            out.append(b._attention(h[:, :n].contiguous(), (pos,), None).float().numpy()[0, 0])
        check(f"block {i} {blocks[i]} sublayer from the same state", rel(out[0], out[1]), ROUNDING)

    # ---- 2. the rollback restores the right row ----
    def run(tokens: list[int], position: int, record: bool) -> np.ndarray:
        at(position)
        s._feed_ple(tokens)
        t = Tensor([tokens], dtype=dtypes.int32)
        if record:
            with spec.recording(s._verify_blocks):
                h = s._blocks(t, (s._v_verify.bind(position),))
        else:
            h = s._blocks(t, (s._v_verify.bind(position),))
        return np.asarray(model.output(s._head(h)).float()[0].numpy())

    def worst_of(got: list[np.ndarray], want: list[np.ndarray], kind: str) -> float:
        return max(rel(a, b) for name, a, b in zip(kinds, got, want, strict=True) if name == kind)

    def rollback(tokens: list[int]) -> None:
        count = len(tokens)
        put(base)
        plain_logits, plain_states = [], []
        for j, token in enumerate(tokens):
            plain_logits.append(run([token], n_prompt + j, False)[0])
            plain_states.append(states())
        plain = np.stack(plain_logits)
        put(base)
        verify = run(tokens, n_prompt, True)
        final = states()
        diffs = [float(np.max(np.abs(verify[j] - plain[j]))) for j in range(count)]
        same = sum(int(verify[j].argmax() == plain[j].argmax()) for j in range(count))
        print(
            f"info {count} rows against one-token steps: logits differ by {min(diffs):.3g} to {max(diffs):.3g}, "
            f"{same} of {count} top tokens the same"
        )
        for j in range(count - 1):
            put(final)
            Tensor.realize(*(w for b in s._rollback for w in b.restore_from(j)))
            got = states()
            for kind in ("recurrent", "convolution", "n-gram convolution"):
                near = worst_of(got, plain_states[j], kind)
                far = worst_of(plain_states[j + 1], plain_states[j], kind)  # how far the next row's state lies
                check(
                    f"{count} rows, restore to row {j}: {kind} states, against the distance to row {j + 1}'s",
                    near / far,
                    0.5,
                )

    rollback(rows)
    if args.copy:
        rollback(ids[n_prompt : n_prompt + args.copy + 1])
    print(f"tinyllm verify rows check: {failures} failure(s), {most + 1} rows, {len(tensors)} state tensors")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
