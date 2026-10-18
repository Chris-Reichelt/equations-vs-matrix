"""Multi-layer perplexity harness: baseline vs hybrid across ALL 28 layers.

Loads per-layer GP checkpoints (results_scaled_L{N}.jsonl from
gp_sweep_all_layers.py) and replaces, in every layer, the gate units whose
program rel_mse < THRESH with their fitted programs. Programs execute through
gplearn's guarded functions (identical semantics to training) via the
validated prefix-list interpreter from bonsai_perplexity_harness.py.

Reports baseline NLL, hybrid NLL, per-layer replacement counts, and the
bytes-saved estimate (the speed claim).
"""
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gplearn._program import _Program
from gplearn import functions as gf

MODEL = "../models/Bonsai-1.7B-unpacked"
OUTDIR = "./results"
LAYERS = list(range(28))
THRESH = 0.5
BITS_TERNARY = 1.585

FUNC_SET = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']
GF = {'add':gf.add2,'sub':gf.sub2,'mul':gf.mul2,'div':gf.div2,'max':gf.max2,'min':gf.min2,
      'neg':gf.neg1,'sqrt':gf.sqrt1,'abs':gf.abs1,'log':gf.log1,'sin':gf.sin1,'cos':gf.cos1,
      'tan':gf.tan1,'inv':gf.inv1}
ARITY = {'add':2,'sub':2,'mul':2,'div':2,'max':2,'min':2,
         'neg':1,'sqrt':1,'abs':1,'log':1,'sin':1,'cos':1,'tan':1,'inv':1}

def rebuild(prog_str):
    """Parse gplearn LISP (parens) into flat PREFIX list — validated against
    an independent LISP evaluator (max diff 0, 200 programs)."""
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
                args.append(parse())   # commas skipped by tokenizer
            if toks[pos[0]] != ')':
                raise ValueError("expected ')'")
            pos[0] += 1
            return [t] + [a for arg in args for a in arg]   # PREFIX
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

def exec_program(p, X):
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

# --- load per-layer programs
rebuilt = {}      # L -> {row: program}
counts = {}
for L in LAYERS:
    ck = os.path.join(OUTDIR, f"results_scaled_L{L}.jsonl")
    if not os.path.exists(ck):
        print(f"layer {L}: no checkpoint, skipping", flush=True)
        continue
    sel = {}
    for line in open(ck):
        d = json.loads(line)
        if d["rel_mse"] >= THRESH:
            continue
        try:
            sel[d["row"]] = rebuild(d["program"])
        except Exception:
            pass
    rebuilt[L] = sel
    counts[L] = len(sel)
    print(f"layer {L}: {len(sel)} units replaced", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.eval()

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

def make_hook(L):
    def hook(mod, inp, out):
        x = inp[0].to(torch.float32)
        with torch.no_grad():
            gate_w = model.model.layers[L].mlp.gate_proj.weight.float()
            up_w = model.model.layers[L].mlp.up_proj.weight.float()
            down_w = model.model.layers[L].mlp.down_proj.weight.float()
            g = x @ gate_w.T; u = x @ up_w.T
            x2d = x.reshape(-1, x.shape[-1]).numpy()
            for r, p in rebuilt[L].items():
                g2 = exec_program(p, x2d)
                g[0, :, r] = torch.from_numpy(g2.astype(np.float32))
            act = torch.nn.functional.silu(g) * u
            o = act @ down_w.T
        return o.to(torch.bfloat16)
    return hook

results = {}
results["baseline_full_matrix"] = run_forward()
print("baseline NLL:", results["baseline_full_matrix"], flush=True)
handles = [model.model.layers[L].mlp.register_forward_hook(make_hook(L))
           for L in rebuilt]
results["hybrid_programs"] = run_forward()
for h in handles:
    h.remove()
print("hybrid NLL:", results["hybrid_programs"], flush=True)

n_units = sum(counts.values())
bits_saved = sum(counts[L] * (2048 * BITS_TERNARY - 35) for L in counts)
results.update(dict(
    ppl_baseline=float(np.exp(results["baseline_full_matrix"])),
    ppl_hybrid=float(np.exp(results["hybrid_programs"])),
    per_layer_units=counts, n_units=n_units,
    bits_saved=float(bits_saved),
    matrix_bits_total=float(28 * 6144 * 2048 * BITS_TERNARY),
    frac_gate_units_replaced=float(n_units / (28 * 6144))))
json.dump(results, open(os.path.join(OUTDIR, "perplexity_all_layers.json"), "w"), indent=2)
print("WROTE", os.path.join(OUTDIR, "perplexity_all_layers.json"))
