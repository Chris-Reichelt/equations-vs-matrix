"""Capture REAL layer-10 MLP-input activations from the on-disk Bonsai 1.7B
(unpacked, Qwen3 arch) with a torch CPU forward pass over ~512 tokens of
diverse text, then save them for the symbolic spike.

x = hidden state entering model.layers[10].mlp (post-RMSNorm, pre-gate_proj).
Saved as float32 (N, 2048) + token counts + per-component RMS diagnostics.
"""
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "../models/Bonsai-1.7B-unpacked"
OUT = "./results/acts_real.npz"
LAYER = 10
torch.set_num_threads(8)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.eval()

# Diverse calibration text: real code + prose from this machine
corpus = []
for root in ["/home/will/qwen38-opt-lab/scripts", "/home/will/Documents/Sync/wiki"]:
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        if os.path.isfile(p) and os.path.getsize(p) < 200_000 and f.endswith((".py", ".md")):
            corpus.append(open(p, errors="ignore").read(4000))
text = "\n\n".join(corpus)
ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
print("tokens:", ids.shape, flush=True)

acts = {}
def hook(mod, inp, out):
    acts["x"] = inp[0].detach().to(torch.float32)  # (1, seq, 2048)
h = model.model.layers[LAYER].mlp.register_forward_hook(hook)

t0 = time.time()
with torch.no_grad():
    model(ids)
print(f"forward pass {time.time()-t0:.1f}s", flush=True)
x = acts["x"].reshape(-1, acts["x"].shape[-1]).numpy().astype(np.float32)
h.remove()

rms = np.sqrt((x**2).mean(0))
print(f"acts {x.shape} per-component RMS: min={rms.min():.3f} med={np.median(rms):.3f} max={rms.max():.3f}")
print(f"row RMS (||x||/sqrt(2048)): med={np.median(np.linalg.norm(x,axis=1)/np.sqrt(2048)):.3f}")
np.savez_compressed(OUT, x=x)
json.dump(dict(layer=LAYER, tokens=int(ids.shape[1]), shape=list(x.shape),
               rms_min=float(rms.min()), rms_med=float(np.median(rms)), rms_max=float(rms.max())),
          open(OUT.replace(".npz", ".meta.json"), "w"), indent=2)
print("WROTE", OUT)
