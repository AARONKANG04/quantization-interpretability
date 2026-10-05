"""Gate G0.2 proxy: our NVFP4 QDQ against llm-compressor's NVFP4A16 export, matrix by matrix.

Runs llm-compressor oneshot (weight-only, no calibration data needed) on the model, saves the compressed
checkpoint (also the checkpoint vLLM will serve natively later), unpacks weight_packed / weight_scale /
weight_global_scale from the safetensors directly (no dependence on internal APIs), dequantizes, and compares
with qi.quant.nvfp4 on the original bf16 weights.

usage: python scripts/phase0_crosscheck.py --out /data/ckpt/q1_llmc [--model google/gemma-3-4b-pt] [--skip-oneshot]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

from _common import load_env, run_meta, write_json

load_env()
from qi.quant.nvfp4 import E2M1_GRID, nvfp4_qdq  # noqa: E402

GRID = torch.tensor(E2M1_GRID)


def unpack_fp4(packed: torch.Tensor, low_first: bool) -> torch.Tensor:
    """uint8 [out, in/2] -> values [out, in]; code bit 3 is sign, bits 0-2 index the E2M1 grid."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi] if low_first else [hi, lo], dim=-1).reshape(packed.shape[0], -1).long()
    mag = GRID[codes & 0x7]
    return torch.where(codes & 0x8 > 0, -mag, mag)


def run_oneshot(model_name: str, out: str):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoTokenizer, Gemma3ForCausalLM

    model = Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(model_name)
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--out", default="/data/ckpt/q1_llmc")
    ap.add_argument("--skip-oneshot", action="store_true")
    ap.add_argument("--max-matrices", type=int, default=0, help="0 = all")
    ap.add_argument("--report", default="results/phase0/crosscheck.json")
    ap.add_argument("--no-fuse", action="store_true", help="per-matrix global scales (not what llm-compressor exports)")
    args = ap.parse_args()
    if not args.skip_oneshot:
        run_oneshot(args.model, args.out)

    from qi.models import load_gemma3_text
    from qi.quant.apply import fused_global_amax
    model, _ = load_gemma3_text(args.model, device_map="cpu")
    shared = fused_global_amax(model) if not args.no_fuse else {}
    orig = {n: p for n, p in model.named_parameters() if n.endswith(".weight") and re.search(r"\.layers\.\d+\.", n)
            and any(n.endswith(f"{s}.weight") for s in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))}
    tensors = {}
    for f in sorted(glob.glob(str(Path(args.out) / "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith(("weight_packed", "weight_scale", "weight_global_scale")):
                    tensors[k] = sf.get_tensor(k)
    rows, n = [], 0
    for name, w in orig.items():
        base = name[: -len(".weight")]
        packed, scale, gscale = tensors.get(f"{base}.weight_packed"), tensors.get(f"{base}.weight_scale"), tensors.get(f"{base}.weight_global_scale")
        if packed is None or scale is None or gscale is None:
            rows.append({"name": base, "status": "missing_in_export"})
            continue
        ours, st = nvfp4_qdq(w.data, global_amax=shared.get(base))
        ours = ours.float()  # bf16 values (what the model holds); theirs is cast to the same dtype below
        best = None
        for low_first in (True, False):
            vals = unpack_fp4(packed, low_first)
            theirs = (vals * (scale.float().repeat_interleave(16, dim=-1) / gscale.float())).to(w.dtype).float()
            diff = (theirs - ours).abs()
            cand = {"low_first": low_first, "max_abs_diff": float(diff.max()), "frac_mismatch": float((diff > 0).float().mean()),
                    "gscale_theirs": float(gscale.float()), "gscale_ours": st.global_scale,
                    "rel_err_theirs": float((theirs - w.float()).norm() / w.float().norm()), "rel_err_ours": st.rel_fro_err}
            if best is None or cand["frac_mismatch"] < best["frac_mismatch"]:
                best = cand
        rows.append({"name": base, "status": "ok", **best})
        n += 1
        if args.max_matrices and n >= args.max_matrices:
            break
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {"n_compared": len(ok), "n_missing": len(rows) - len(ok),
               "max_frac_mismatch": max((r["frac_mismatch"] for r in ok), default=None),
               "mean_frac_mismatch": sum(r["frac_mismatch"] for r in ok) / max(len(ok), 1),
               "max_abs_diff": max((r["max_abs_diff"] for r in ok), default=None),
               "bit_exact": all(r["frac_mismatch"] == 0.0 for r in ok) if ok else None}
    write_json(args.report, {**run_meta(args), "summary": summary, "per_matrix": rows})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
