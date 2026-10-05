"""Gemma Scope 2 loading (sae-lens release names, with a direct Hugging Face fallback) and the validation
checks of plan E2.1."""
from __future__ import annotations

import json

import torch

from ..models import ResidualPatch
from .jumprelu import JumpReLUSAE

SITE_SHORT = {"resid_post": "res", "mlp_out": "mlp", "attn_out": "att", "transcoder": "transcoders"}
HF_REPO = "google/gemma-scope-2-{variant}"


def _from_hf(layer, width, l0, site, variant, device, dtype) -> JumpReLUSAE:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    folder = f"{site}/layer_{layer}_width_{width}_l0_{l0}"
    repo = HF_REPO.format(variant=variant)
    params = load_file(hf_hub_download(repo, f"{folder}/params.safetensors"))
    try:
        cfg = json.loads(open(hf_hub_download(repo, f"{folder}/config.json")).read())
    except Exception:
        cfg = {}
    keys = {k.lower(): k for k in params}

    def get(*names):
        for n in names:
            if n.lower() in keys:
                return params[keys[n.lower()]]
        raise KeyError(f"none of {names} in {list(params)}")

    W_enc = get("W_enc", "w_enc", "encoder.weight")
    W_dec = get("W_dec", "w_dec", "decoder.weight")
    if W_enc.shape[0] != W_dec.shape[1]:  # stored as [m, d]: transpose
        W_enc = W_enc.T
    b_enc = get("b_enc", "encoder.bias")
    b_dec = get("b_dec", "decoder.bias")
    thr = get("threshold", "log_threshold")
    if "log_threshold" in keys and "threshold" not in keys:
        thr = thr.exp()
    apply = bool(cfg.get("apply_b_dec_to_input", False))
    return JumpReLUSAE(W_enc.to(dtype), b_enc.to(dtype), thr.to(dtype), W_dec.to(dtype), b_dec.to(dtype), apply).to(device)


def load_gemma_scope2(layer: int, width: str = "65k", l0: str = "medium", site: str = "resid_post",
                      variant: str = "4b-pt", device: str = "cuda", dtype=torch.float32, all_layers: bool = False) -> JumpReLUSAE:
    """Try sae-lens (release gemma-scope-2-<variant>-<res|mlp|att>[-all]); fall back to the raw HF safetensors."""
    release = f"gemma-scope-2-{variant}-{SITE_SHORT[site]}" + ("-all" if all_layers else "")
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    try:
        from sae_lens import SAE

        out = SAE.from_pretrained(release, sae_id, device=device)
        sae = out[0] if isinstance(out, (tuple, list)) else out
        return JumpReLUSAE.from_sae_lens(sae).to(device=device, dtype=dtype)
    except Exception as e:  # noqa: BLE001
        print(f"sae-lens load failed ({type(e).__name__}: {str(e)[:120]}); loading raw safetensors from the Hub")
        return _from_hf(layer, width, l0, site + ("_all" if all_layers else ""), variant, device, dtype)


@torch.no_grad()
def splice_ce_delta(model, sae: JumpReLUSAE, layer: int, input_ids: torch.Tensor, chunk: int = 1024) -> dict:
    """Next-token cross-entropy with and without replacing resid_post at `layer` by its SAE reconstruction.
    One sequence at a time, logits consumed in token chunks, so memory stays small at a 262k vocabulary."""
    def ce():
        tot, n = 0.0, 0
        for i in range(input_ids.shape[0]):
            logits = model(input_ids=input_ids[i : i + 1]).logits[0, :-1]
            tgt = input_ids[i, 1:]
            for s in range(0, logits.shape[0], chunk):
                tot += torch.nn.functional.cross_entropy(logits[s : s + chunk].float(), tgt[s : s + chunk], reduction="sum").item()
            n += logits.shape[0]
            del logits
        return tot / n

    base = ce()
    with ResidualPatch(model, layer, lambda h: sae.decode(sae.encode(h)).to(h.dtype)):
        spliced = ce()
    return {"ce_base": base, "ce_spliced": spliced, "ce_delta": spliced - base}
