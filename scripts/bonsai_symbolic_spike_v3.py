"""Spike v3: scaled GP budget to find the TRUE error floor.

All 6,144 gate_proj rows of layer 10, GP budget 500 pop x 60 gens (17x v2),
parallel n_jobs=18, scored on real activations (acts_real.npz).
Writes results_scaled.json with per-row bits/rel_mse + summary.
"""
import json, os, time
import numpy as np
from safetensors import safe_open
from gplearn.genetic import SymbolicRegressor

MODEL = "../models/Bonsai-1.7B-unpacked/model.safetensors"
ACTS = "./results/acts_real.npz"
OUT = "./results/results_scaled.json"
LAYER = 10
BITS_TERNARY = 1.585
POP, GENS, JOBS = 500, 60, 3
CKPT = OUT.replace(".json", ".jsonl")

def mem_available_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 99.0

X = np.load(ACTS)["x"].astype(np.float64)
if X.shape[0] > 4096:
    X = X[np.random.default_rng(7).choice(X.shape[0], 4096, replace=False)]
with safe_open(MODEL, framework="numpy", device="cpu") as f:
    gate = f.get_tensor(f"model.layers.{LAYER}.mlp.gate_proj.weight").astype(np.float64)
print(f"acts {X.shape} gate {gate.shape}", flush=True)

func_set = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']
half = len(X) // 2
Xtr, Xte = X[:half], X[half:]
ytr_all, yte_all = Xtr @ gate.T, Xte @ gate.T   # (n, 6144) -> columns per row
var_te = (yte_all ** 2).mean(0)                  # per-row variance of target

def one_row(r):
    sr = SymbolicRegressor(population_size=POP, generations=GENS,
                           stopping_criteria=0.0, function_set=func_set,
                           parsimony_coefficient=0.002, const_range=(-1.0, 1.0),
                           init_method='half and half', tournament_size=20,
                           low_memory=True, n_jobs=1, random_state=int(r), verbose=0)
    sr.fit(Xtr, ytr_all[:, r])
    prog = sr._program
    n_nodes = len(prog.program)
    pred = sr.predict(Xte)
    mse = float(np.mean((pred - yte_all[:, r]) ** 2))
    rel = mse / float(var_te[r])
    import re as _re
    txt = _re.sub(r'X\d+', ' ', str(prog)).replace('(', ' ').replace(')', ' ').replace(',', ' ')
    n_const = 0
    for t in txt.split():
        try: float(t); n_const += 1
        except ValueError: pass
    return dict(row=int(r), n_nodes=n_nodes, n_const=n_const,
                prog_bits=n_nodes*5 + n_const*16, rel_mse=rel,
                program=str(prog))  # FULL program — the perplexity harness parses these

t0 = time.time()
done = {}
if os.path.exists(CKPT):
    for line in open(CKPT):
        try: d = json.loads(line); done[d["row"]] = d
        except Exception: pass
    print(f"resuming with {len(done)} rows done", flush=True)
from joblib import Parallel, delayed
CHUNK = 600
for start in range(0, gate.shape[0], CHUNK):
    rows = [r for r in range(start, min(start+CHUNK, gate.shape[0])) if r not in done]
    if rows:
        while mem_available_gb() < 4.0:
            print(f"waiting for memory: {mem_available_gb():.1f} GB avail", flush=True)
            time.sleep(30)
        try:
            batch = Parallel(n_jobs=JOBS, verbose=5)(delayed(one_row)(r) for r in rows)
        except Exception as e:
            print("joblib failed:", e, flush=True)
            batch = [one_row(r) for r in rows]
        with open(CKPT, "a") as fh:
            for d in batch:
                fh.write(json.dumps(d) + "\n"); done[d["row"]] = d
    print(f"chunk {start}: {len(done)}/{gate.shape[0]} rows, {time.time()-t0:.0f}s", flush=True)
res = [done[r] for r in range(gate.shape[0])]
print(f"scaled GP: {time.time()-t0:.0f}s", flush=True)

bits = np.array([r["prog_bits"] for r in res]); rel = np.array([r["rel_mse"] for r in res])
row_bits = gate.shape[1] * BITS_TERNARY
summary = dict(
    n_rows=len(res), pop=POP, gens=GENS, jobs=JOBS,
    bits=dict(med=float(np.median(bits)), mean=float(bits.mean()), p10=float(np.percentile(bits,10)),
              p90=float(np.percentile(bits,90)), min=int(bits.min()), max=int(bits.max())),
    rel_mse=dict(med=float(np.median(rel)), mean=float(rel.mean()),
                 p10=float(np.percentile(rel,10)), p90=float(np.percentile(rel,90)),
                 min=float(rel.min()), max=float(rel.max()),
                 frac_lt_0p5=float((rel<0.5).mean()), frac_lt_0p2=float((rel<0.2).mean())),
    layer_bits_gp=float(bits.sum()), layer_bits_ternary=float(row_bits*gate.shape[0]),
    compression=float(row_bits*gate.shape[0]/bits.sum()))
json.dump(dict(summary=summary, rows=res), open(OUT, "w"), indent=2)
print("SUMMARY", json.dumps(summary), flush=True)
print("WROTE", OUT)
