"""Which arithmetic reproduces llm-compressor's NVFP4 block scales and element codes? One matrix, many candidates."""
import glob
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "scripts")
from phase0_crosscheck import unpack_fp4  # noqa: E402

from qi.quant.nvfp4 import round_to_e2m1  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "model.layers.30.mlp.gate_proj"


def load(d, key):
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, "pt") as sf:
            if key in sf.keys():
                return sf.get_tensor(key)


w = load("/data/ckpt/bf16", NAME + ".weight")
packed = load("/data/ckpt/q1_llmc", NAME + ".weight_packed")
their_scale = load("/data/ckpt/q1_llmc", NAME + ".weight_scale")
g = load("/data/ckpt/q1_llmc", NAME + ".weight_global_scale").float()
g16 = g.to(torch.bfloat16)
blocks16 = w.reshape(w.shape[0], -1, 16)
bmax16 = blocks16.abs().amax(-1)  # bf16
bmax32 = bmax16.float()
fp8 = torch.float8_e4m3fn
cands = {
    "fp32: (bmax/6*g)->fp8": ((bmax32 / 6.0) * g).to(fp8),
    "bf16: (bmax/6)*g ->fp8": ((bmax16 / 6.0) * g16).to(fp8),
    "bf16: bmax*(1/6)*g ->fp8": ((bmax16 * (1.0 / torch.tensor(6.0, dtype=torch.bfloat16))) * g16).to(fp8),
    "bf16: (bmax*g)/6 ->fp8": ((bmax16 * g16) / 6.0).to(fp8),
    "fp32 then bf16 then fp8": ((bmax32 / 6.0) * g).to(torch.bfloat16).to(fp8),
    "bf16: bmax/6 ->fp32*g->fp8": ((bmax16 / 6.0).float() * g).to(fp8),
    "fp32: bmax*(g/6)->fp8": (bmax32 * (g / 6.0)).to(fp8),
    "bf16 (bmax/6)*g clamp448 ->fp8": ((bmax16 / 6.0) * g16).clamp(max=448).to(fp8),
}
ts = their_scale.float()
print(NAME, "blocks:", ts.numel())
best_name, best = None, -1
for k, v in cands.items():
    frac = (v.float() == ts).float().mean().item()
    print(f"  {k:36s} scale match {frac:.4f}")
    if frac > best:
        best_name, best = k, frac
print("best:", best_name)
# element check with THEIR scales (so only the element rounding is tested)
scale_dq = ts / g
x = blocks16.float() / torch.where(scale_dq > 0, scale_dq, torch.ones_like(scale_dq)).unsqueeze(-1)
ours_codes = round_to_e2m1(x)
theirs_vals = unpack_fp4(packed, True) * scale_dq.repeat_interleave(16, -1)
theirs_codes = (theirs_vals / torch.where(scale_dq > 0, scale_dq, torch.ones_like(scale_dq)).repeat_interleave(16, -1)).reshape(x.shape)
mism = ours_codes != theirs_codes
print(f"element code mismatch with their scales: {mism.float().mean().item():.6f}")
if mism.any():
    xm = x[mism][:12]
    print("  x at mismatches:", [round(v, 4) for v in xm.tolist()])
    print("  ours:", ours_codes[mism][:12].tolist())
    print("  theirs:", theirs_codes[mism][:12].tolist())
    # candidate: x computed in bf16?
    x16 = (blocks16 / scale_dq.to(torch.bfloat16).unsqueeze(-1)).float()
    m2 = round_to_e2m1(x16) != theirs_codes
    print(f"  mismatch if x computed in bf16: {m2.float().mean().item():.6f}")
    # candidate: round half away from zero
    mag = x.abs().clamp(max=6)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.0]); mids = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code_away = (mag.unsqueeze(-1) >= mids).sum(-1)
    m3 = (torch.sign(x) * grid[code_away]) != theirs_codes
    print(f"  mismatch if ties round away from zero: {m3.float().mean().item():.6f}")
