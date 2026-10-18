"""GP sweep across ALL 28 layers' gate_proj rows (cheap budget).

v3 showed the error floor is information-limited (17x budget barely moved
0.56 -> 0.52), so the all-layer sweep uses the cheap 150x12 budget:
~0.7s/row -> ~24 min/layer -> ~11h for 28 layers.

Per-layer checkpoints: results/results_scaled_L{N}.jsonl (resumable).
Memory-hardened for this box (chronically 2-6 GB free): 3 workers,
low_memory=True, MemAvailable guard between chunks.
"""
import json, os, sys, time
import numpy as np
from safetensors import safe_open
from gplearn.genetic import SymbolicRegressor

MODEL = "../models/Bonsai-1.7B-unpacked/model.safetensors"
ACTS = "./results/acts_real.npz"
OUTDIR = "./results"
LAYERS = list(range(28))
BITS_TERNARY = 1.585
POP, GENS, JOBS = 150, 12, 3   # cheap budget — floor is information-limited
CHUNK = 600

def mem_available_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 99.0

X = np.load(ACTS)["x"].astype(np.float64)
if X.shape[0] > 4096:
    X = X[np.random.default_rng(7).choice(X.shape[0], 4096, replace=False)]
print(f"acts {X.shape}", flush=True)

func_set = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']
from joblib import Parallel, delayed

def one_row(gate_r, ytr, yte, var_te, r):
    sr = SymbolicRegressor(population_size=POP, generations=GENS,
                           stopping_criteria=0.0, function_set=func_set,
                           parsimony_coefficient=0.002, const_range=(-1.0, 1.0),
                           init_method='half and half', tournament_size=20,
                           low_memory=True, n_jobs=1, random_state=int(r), verbose=0)
    sr.fit(Xtr, ytr)
    prog = sr._program
    n_nodes = len(prog.program)
    pred = sr.predict(Xte)
    mse = float(np.mean((pred - yte) ** 2))
    rel = mse / float(var_te)
    import re as _re
    txt = _re.sub(r'X\d+', ' ', str(prog)).replace('(', ' ').replace(')', ' ').replace(',', ' ')
    n_const = 0
    for t in txt.split():
        try: float(t); n_const += 1
        except ValueError: pass
    return dict(row=int(r), n_nodes=n_nodes, n_const=n_const,
                prog_bits=n_nodes*5 + n_const*16, rel_mse=rel,
                program=str(prog))  # FULL program — the harness parses these

half = len(X) // 2
Xtr, Xte = X[:half], X[half:]
t0 = time.time()
with safe_open(MODEL, framework="numpy", device="cpu") as f:
    for L in LAYERS:
        CKPT = os.path.join(OUTDIR, f"results_scaled_L{L}.jsonl")
        gate = f.get_tensor(f"model.layers.{L}.mlp.gate_proj.weight").astype(np.float64)
        ytr_all, yte_all = Xtr @ gate.T, Xte @ gate.T
        var_te = (yte_all ** 2).mean(0)
        done = {}
        if os.path.exists(CKPT):
            for line in open(CKPT):
                try: d = json.loads(line); done[d["row"]] = d
                except Exception: pass
        for start in range(0, gate.shape[0], CHUNK):
            rows = [r for r in range(start, min(start+CHUNK, gate.shape[0])) if r not in done]
            if rows:
                while mem_available_gb() < 4.0:
                    print(f"waiting for memory: {mem_available_gb():.1f} GB", flush=True)
                    time.sleep(30)
                try:
                    batch = Parallel(n_jobs=JOBS)(delayed(one_row)(
                        gate[r], ytr_all[:, r], yte_all[:, r], var_te[r], r) for r in rows)
                except Exception as e:
                    print("joblib failed:", e, flush=True)
                    batch = [one_row(gate[r], ytr_all[:, r], yte_all[:, r], var_te[r], r) for r in rows]
                with open(CKPT, "a") as fh:
                    for d in batch:
                        fh.write(json.dumps(d) + "\n"); done[d["row"]] = d
        rel = np.array([done[r]["rel_mse"] for r in done])
        bits = np.array([done[r]["prog_bits"] for r in done])
        print(f"layer {L}: {len(done)} rows, rel_mse med={np.median(rel):.3f}, "
              f"bits med={np.median(bits):.0f}, frac<0.5={(rel<0.5).mean():.2f}, "
              f"{time.time()-t0:.0f}s elapsed", flush=True)
print("WROTE sweep complete")
