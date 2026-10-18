"""Perplexity harness: swap fitted GP programs into the real forward pass and
measure end-to-end perplexity, full-matrix baseline vs hybrid (programs for
the compressible rows, ternary matrix for the rest).

Programs come from results_scaled.jsonl (checkpoint — works with partial rows).
Row selection: rel_mse < threshold (measured compressibility), applied to
gate/up (2048->6144) and down (6144->2048) projections of layer L.
Programs are re-executed on the fly: gplearn _Program.execute over the
activation matrix (vectorized, no per-token python).
"""
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gplearn.functions import _Function
from gplearn._program import _Program

MODEL = "../models/Bonsai-1.7B-unpacked"
CKPT = "./results/results_scaled.jsonl"
OUT = "./results/perplexity.json"
LAYER = 10
FUNC_SET = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']

def rebuild(prog_str):
    """Parse gplearn's LISP string (parens included) into a flat PREFIX
    program list — function before args, matching _Program.execute semantics."""
    import re
    toks = re.findall(r'X\d+|-?\d+\.\d+|[A-Za-z]+|\(|\)', prog_str)
    pos = [0]
    def parse():
        if pos[0] >= len(toks):
            raise ValueError("program truncated")
        t = toks[pos[0]]; pos[0] += 1
        if t in FUNC_SET:
            if toks[pos[0]] != '(':
                raise ValueError("expected '(' after function name")
            pos[0] += 1
            args = []
            for _ in range(ARITY[t]):
                args.append(parse())   # commas are skipped by the tokenizer
            if toks[pos[0]] != ')':
                raise ValueError("expected ')'")
            pos[0] += 1
            return [t] + [a for arg in args for a in arg]   # PREFIX: function first
        if t.startswith('X'):
            return [int(t[1:])]
        try:
            return [float(t)]
        except ValueError:
            raise ValueError(f"bad token {t!r}")
    program = parse()
    if pos[0] != len(toks):
        raise ValueError("program string truncated mid-tree")
    p = _Program.__new__(_Program)
    p.program = program
    p.low_memory = True
    return p

from gplearn import functions as gf
GF = {'add':gf.add2,'sub':gf.sub2,'mul':gf.mul2,'div':gf.div2,'max':gf.max2,'min':gf.min2,
      'neg':gf.neg1,'sqrt':gf.sqrt1,'abs':gf.abs1,'log':gf.log1,'sin':gf.sin1,'cos':gf.cos1,
      'tan':gf.tan1,'inv':gf.inv1}  # identical guarded semantics to training
ARITY = {'add':2,'sub':2,'mul':2,'div':2,'max':2,'min':2,
         'neg':1,'sqrt':1,'abs':1,'log':1,'sin':1,'cos':1,'tan':1,'inv':1}

def exec_program(p, X):
    """Execute flat PREFIX program (function before args) over (N, F) -> (N,),
    args applied in program order — identical semantics to _Program.execute."""
    def walk(i):
        node = p.program[i]
        if isinstance(node, str):
            arity = ARITY[node]
            args, j = [], i + 1
            for _ in range(arity):
                val, j = walk(j)
                args.append(val)
            return (GF[node](*args) if arity == 2 else GF[node](args[0])), j
        if isinstance(node, int):
            return X[:, node], i + 1
        return np.full(X.shape[0], node, dtype=np.float64), i + 1
    return walk(0)[0]

# --- load programs
progs = {}
for line in open(CKPT):
    d = json.loads(line)
    if d["program"].count("(") == d["program"].count(")"):  # skip 200-char truncations
        progs[d["row"]] = d
print(f"programs loaded: {len(progs)} rows", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.eval()

# calibration text for perplexity (held-out from activation corpus: use prose)
import glob
texts = []
for p in sorted(glob.glob("/home/will/Documents/Sync/wiki/*.md"))[:12]:
    texts.append(open(p, errors="ignore").read(3000))
ids = tok("\n\n".join(texts), return_tensors="pt", truncation=True,
          max_length=1024).input_ids
print("eval tokens:", ids.shape, flush=True)

def run_forward():
    with torch.no_grad():
        logits = model(ids).logits
    lp = torch.nn.functional.log_softmax(logits.to(torch.float32), dim=-1)
    nll = -torch.gather(lp[:, :-1, :], 2, ids[:, 1:, None]).mean()
    return float(nll)

def apply_programs(thresh):
    """Replace gate/up rows (and matching down rows) with programs where rel_mse<thresh."""
    n_used = 0
    with torch.no_grad():
        for r, d in progs.items():
            if d["rel_mse"] < thresh:
                # gate/up rows feed silu(gate*up); down rows consume them.
                # We approximate the full FFN row by replacing gate row r AND
                # down column r (the output unit that weight feeds).
                gp = rebuild(d["program"])
                # reconstruct weight row from program on calibration acts
                pass
    return n_used

# Simpler, exact approach: patch the WEIGHTS directly.
# For each selected row r: recompute gate row r (and up row r, and down col r)
# from the program's implied linear map? Programs are nonlinear in x, so we
# cannot express them as weight rows. Instead hook the mlp output:
#   out = silu(gate(x)) * up(x)  ->  down @ out
# Replace the FULL FFN for selected output units: down[:, r] @ silu(gate_r(x))*up_r(x)
# where gate_r(x) is the program. up_r stays matrix (it is one row too).
rebuilt = {}
for r, d in progs.items():
    if d["rel_mse"] >= 0.5:
        continue
    try:
        rebuilt[r] = rebuild(d["program"])
    except Exception:
        pass
selected = sorted(rebuilt.keys())
print(f"selected rows (rel_mse<0.5, parseable): {len(selected)} of {len(progs)}", flush=True)

def hook_mlp(mod, inp, out):
    x = inp[0].to(torch.float32)          # (1, seq, 2048)
    with torch.no_grad():
        gate_w = model.model.layers[LAYER].mlp.gate_proj.weight.float()
        up_w = model.model.layers[LAYER].mlp.up_proj.weight.float()
        down_w = model.model.layers[LAYER].mlp.down_proj.weight.float()
        g = x @ gate_w.T; u = x @ up_w.T
        # replace selected gate units with programs (programs fit gate rows)
        x2d = x.reshape(-1, x.shape[-1]).numpy()
        for r in selected:
            g2 = exec_program(rebuilt[r], x2d)
            g[0, :, r] = torch.from_numpy(g2.astype(np.float32))
        act = torch.nn.functional.silu(g) * u
        o = act @ down_w.T   # down stays matrix; acts carry the program effect
    return o.to(torch.bfloat16)

results = {}
results["baseline_full_matrix"] = run_forward()
print("baseline NLL:", results["baseline_full_matrix"], flush=True)
h = model.model.layers[LAYER].mlp.register_forward_hook(hook_mlp)
results["hybrid_programs"] = run_forward()
h.remove()
print("hybrid NLL:", results["hybrid_programs"], flush=True)
results["ppl_baseline"] = float(np.exp(results["baseline_full_matrix"]))
results["ppl_hybrid"] = float(np.exp(results["hybrid_programs"]))
results["n_program_rows"] = len(selected)
results["n_programs_total"] = len(progs)
json.dump(results, open(OUT, "w"), indent=2)
print("WROTE", OUT)
