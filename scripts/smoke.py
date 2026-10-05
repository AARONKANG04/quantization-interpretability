"""Smoke test on the box: load the model, generate 20 tokens, count quantizable params, time one QDQ."""
import time

import torch

from _common import load_env

load_env()
from qi.models import count_quantizable_params, global_layer_indices, load_gemma3_text  # noqa: E402
from qi.quant import nvfp4_qdq  # noqa: E402

model, tok = load_gemma3_text()
print("layers:", model.config.num_hidden_layers, "global:", global_layer_indices(model.config))
print("quantizable params:", f"{count_quantizable_params(model)/1e9:.3f}B")
ids = tok("The capital of France is", return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**ids, max_new_tokens=20, do_sample=False)[0]))
w = model.model.layers[12].mlp.down_proj.weight
t0 = time.time(); _, st = nvfp4_qdq(w); torch.cuda.synchronize()
print(f"nvfp4 qdq of {tuple(w.shape)} in {time.time()-t0:.2f}s:", st)
