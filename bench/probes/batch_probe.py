#!/usr/bin/env python3
"""Probe: aggregate decode/prefill throughput of tinygrad.llm at batch size B (all rows share one position).

KEPT FOR THE RECORD, DO NOT REUSE: its sync() wait inflates every step longer than 200 ms (see the note in
results-2026-11-22.txt). Use `tinyllm bench`, which supersedes it.

usage: batch_probe.py B model.gguf [prefill_chunk] [decode_steps]"""
import sys, time
from tinygrad import Tensor, UOp, Device, GlobalCounters
from tinygrad.llm.model import Transformer

B, path = int(sys.argv[1]), sys.argv[2]
chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 32
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 24
prompt_len = chunk * 8

st = time.perf_counter()
model, kv = Transformer.from_gguf(path, 4096)
print(f"B={B} load {time.perf_counter()-st:.1f}s", flush=True)

v_sp, v_toks = UOp.variable("start_pos", 0, model.max_context-1), UOp.variable("toks", 1, chunk)
temp = Tensor([0.0])
prompt = [1000 + i % 1000 for i in range(prompt_len)]
t = Tensor([prompt + [0]*(model.max_context-len(prompt))]*B, dtype="int32").realize()

def sync(): Device[Device.DEFAULT].synchronize()

# prefill in chunks; the first chunks include JIT capture + kernel compiles
pos, times = 0, []
while pos < prompt_len:
  sp, nt = v_sp.bind(pos), v_toks.bind(chunk)
  st = time.perf_counter()
  out = model(t[:, sp:sp+nt], (sp,), temp).realize(); sync()
  times.append(time.perf_counter()-st)
  pos += chunk
warm = sorted(times[3:])
med = warm[len(warm)//2]
print(f"B={B} prefill chunk={chunk}: first3={[round(x,2) for x in times[:3]]}s  warm median {med*1e3:.1f} ms/chunk "
      f"-> {B*chunk/med:.1f} tok/s aggregate ({chunk/med:.1f} per stream)", flush=True)

# decode: feed the sampled tokens back, all rows advance together
dts = []
for i in range(steps):
  GlobalCounters.reset()
  st = time.perf_counter()
  out = model(out, (v_sp.bind(pos),), temp).realize(); sync()
  dts.append((time.perf_counter()-st, GlobalCounters.global_mem))
  pos += 1
warm = sorted(dts[4:])
dt, mem = warm[len(warm)//2]
rows = [int(x) for r in out.tolist() for x in (r if isinstance(r, list) else [r])]
print(f"B={B} decode: first4={[round(x[0],2) for x in dts[:4]]}s  warm median {dt*1e3:.2f} ms/step "
      f"-> {B/dt:.2f} tok/s aggregate ({1/dt:.2f} per stream), {mem/dt/1e9:.0f} GB/s, rows_equal={len(set(rows)) == 1} tok={rows[0]}", flush=True)
