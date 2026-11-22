#!/usr/bin/env python3
"""Standalone GGUF header inspector: metadata (minus big arrays), tensor type histogram, a few tensor shapes."""
import struct, sys, collections

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
         13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
         35: "TQ2_0", 39: "MXFP4", 41: "Q1_0"}
BPW = {"F32": 32, "F16": 16, "BF16": 16, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q4_K": 4.5, "Q3_K": 3.4375, "Q2_K": 2.625,
       "IQ4_XS": 4.25, "IQ4_NL": 4.5, "Q4_0": 4.5, "Q5_0": 5.5, "MXFP4": 4.25, "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "IQ2_S": 2.5625,
       "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625, "IQ1_S": 1.5625, "IQ1_M": 1.75}

def rd(f, fmt):
  n = struct.calcsize(fmt)
  return struct.unpack("<" + fmt, f.read(n))[0]
def rstr(f): return f.read(rd(f, "Q")).decode("utf-8", "replace")
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def rval(f, t):
  if t in SCALAR: return rd(f, SCALAR[t])
  if t == 8: return rstr(f)
  if t == 9:
    it, n = rd(f, "I"), rd(f, "Q")
    return [rval(f, it) for _ in range(n)]
  raise ValueError(t)

def main(path, show_tensors):
  with open(path, "rb") as f:
    assert f.read(4) == b"GGUF"
    ver, nt, nkv = rd(f, "I"), rd(f, "Q"), rd(f, "Q")
    print(f"== {path}\n   gguf v{ver}, {nt} tensors, {nkv} kv")
    for _ in range(nkv):
      k = rstr(f); t = rd(f, "I"); v = rval(f, t)
      if isinstance(v, list): v = f"[{len(v)} items] {v[:8] if len(v) > 8 else v}" if len(v) > 24 else v
      elif isinstance(v, str) and len(v) > 160: v = v[:160].replace("\n", "\\n") + f"... ({len(v)} chars)"
      print(f"   {k} = {v}")
    hist, bits, params, tensors = collections.Counter(), 0.0, 0, []
    for _ in range(nt):
      name = rstr(f); nd = rd(f, "I"); dims = [rd(f, "Q") for _ in range(nd)]; typ = rd(f, "I"); rd(f, "Q")
      n = 1
      for d in dims: n *= d
      tn = TYPES.get(typ, f"?{typ}")
      hist[tn] += n; params += n; bits += n * BPW.get(tn, 32)
      tensors.append((name, dims, tn))
    print(f"   params: {params/1e9:.3f} B, avg {bits/max(params,1):.2f} bpw")
    for tn, n in hist.most_common(): print(f"     {tn:8s} {n/1e9:8.3f} B params  {100*n/params:5.1f}%")
    if show_tensors:
      seen = set()
      for name, dims, tn in tensors:
        import re
        key = re.sub(r"blk\.\d+\.", "blk.N.", name)
        if key in seen and not name.startswith(("blk.0.", "blk.1.", "blk.2.", "blk.3.")): continue
        seen.add(key)
        if name.startswith(("blk.0.", "blk.3.")) or not name.startswith("blk."): print(f"     {name:48s} {str(dims):28s} {tn}")
      per = collections.defaultdict(collections.Counter)
      for name, dims, tn in tensors:
        import re
        per[re.sub(r"blk\.\d+\.", "blk.N.", name)][tn] += 1
      print("   type per tensor family:")
      for k, c in per.items(): print(f"     {k:48s} {dict(c)}")

if __name__ == "__main__":
  show = "--tensors" in sys.argv
  for p in [a for a in sys.argv[1:] if not a.startswith("--")]: main(p, show)
