"""Which arithmetic reproduces llm-compressor's NVFP4 per-tensor global scales? Compares candidate rules against
the exported weight_global_scale of every quantized matrix (fused q/k/v and gate/up share one amax)."""
import glob
import re

import torch
from safetensors import safe_open

SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))


def main(bf16_dir="/data/ckpt/bf16", export_dir="/data/ckpt/q1_llmc"):
    amax = {}
    for f in glob.glob(f"{bf16_dir}/*.safetensors"):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if re.search(r"\.layers\.\d+\.", k) and k.endswith(tuple(s + ".weight" for s in SUFFIXES)):
                    amax[k[: -len(".weight")]] = sf.get_tensor(k).abs().amax()  # bf16 tensor scalar
    theirs = {}
    for f in glob.glob(f"{export_dir}/*.safetensors"):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                if k.endswith("weight_global_scale"):
                    theirs[k[: -len(".weight_global_scale")]] = sf.get_tensor(k).float().item()
    shared = {}
    for name in amax:
        li = re.search(r"\.layers\.(\d+)\.", name).group(1)
        suf = name.split(".")[-1]
        for g in GROUPS:
            if suf in g:
                members = [n for n in amax if f".layers.{li}." in n and n.split(".")[-1] in g]
                shared[name] = max(amax[m] for m in members)

    def candidates(a_bf16: torch.Tensor) -> dict:
        a32 = a_bf16.float()
        fp32 = 2688.0 / a32
        nearest = fp32.to(torch.bfloat16)
        step_up = torch.nextafter(nearest, torch.tensor(float("inf"), dtype=torch.bfloat16))
        ceil = step_up if nearest.float() < fp32 else nearest
        b = a_bf16.to(torch.bfloat16)
        return {
            "fp32": fp32.item(),
            "bf16_nearest": nearest.float().item(),
            "bf16_ceil": ceil.float().item(),
            "bf16_div": (torch.tensor(2688.0, dtype=torch.bfloat16) / b).float().item(),
            "bf16_recip_mul": (torch.tensor(2688.0, dtype=torch.bfloat16) * (1.0 / b)).float().item(),
            "fp32_then_bf16_of_448x6_over_bf16amax": (448.0 * 6.0 / b.float()).to(torch.bfloat16).float().item(),
        }

    hits, unexplained = {}, []
    for name, a in amax.items():
        c = candidates(shared.get(name, a))
        t = theirs[name]
        for k, v in c.items():
            hits[k] = hits.get(k, 0) + int(abs(v - t) <= 1e-6 * max(1.0, abs(t)))
        if all(abs(v - t) > 1e-6 * max(1.0, abs(t)) for v in c.values()) and len(unexplained) < 5:
            unexplained.append((name, t, {k: round(v, 3) for k, v in c.items()}))
    print("matrices:", len(amax))
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {k:45s} {v}")
    for u in unexplained:
        print("unexplained:", u)


if __name__ == "__main__":
    main()
