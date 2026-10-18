"""Spike v2: refit the symbolic-vs-matrix comparison on REAL layer-10
activations captured from the on-disk Bonsai 1.7B forward pass.

Same protocol as v1 (32 gate_proj rows, GP programs, SVD baselines, ternary
bits) plus the sparse-coordinate baseline that v1 flagged as an artifact of
synthetic activations — now measured on real data, where it is meaningful.

Reads: acts_real.npz (N, 2048) from scripts/bonsai_real_activations.py
Writes: results_real.json
"""
import json, os, time
import numpy as np
from safetensors import safe_open
from gplearn.genetic import SymbolicRegressor

MODEL = "../models/Bonsai-1.7B-unpacked/model.safetensors"
ACTS = "./results/acts_real.npz"
OUT = "./results/results_real.json"
LAYER = 10
N_ROWS = 32
BITS_TERNARY = 1.585
os.makedirs(os.path.dirname(OUT), exist_ok=True)

X = np.load(ACTS)["x"].astype(np.float64)
# subsample to <=4096 rows for GP cost
if X.shape[0] > 4096:
    X = X[np.random.default_rng(7).choice(X.shape[0], 4096, replace=False)]
with safe_open(MODEL, framework="numpy", device="cpu") as f:
    gate = f.get_tensor(f"model.layers.{LAYER}.mlp.gate_proj.weight").astype(np.float64)
print(f"acts {X.shape} gate {gate.shape}", flush=True)

rng = np.random.default_rng(1337)
rows = rng.choice(gate.shape[0], N_ROWS, replace=False)
func_set = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']
results = []
for r in rows:
    w = gate[r]
    y = X @ w
    half = len(y) // 2
    ytr, yte, Xtr, Xte = y[:half], y[half:], X[:half], X[half:]
    sr = SymbolicRegressor(population_size=150, generations=12,
                           stopping_criteria=0.0, function_set=func_set,
                           parsimony_coefficient=0.002, const_range=(-1.0, 1.0),
                           init_method='half and half', tournament_size=10,
                           n_jobs=1, random_state=int(r), verbose=0)
    t1 = time.time(); sr.fit(Xtr, ytr)
    prog = sr._program
    n_nodes = len(prog.program)
    pred = sr.predict(Xte)
    mse = float(np.mean((pred - yte) ** 2))
    rel = mse / float(np.mean(yte ** 2))
    import re as _re
    _txt = _re.sub(r'X\d+', ' ', str(prog)).replace('(', ' ').replace(')', ' ').replace(',', ' ')
    n_const = 0
    for _t in _txt.split():
        try: float(_t); n_const += 1
        except ValueError: pass
    prog_bits = n_nodes * 5 + n_const * 16
    results.append(dict(row=int(r), n_nodes=n_nodes, n_const=n_const,
                        prog_bits=prog_bits, row_bits=gate.shape[1]*BITS_TERNARY,
                        compression=gate.shape[1]*BITS_TERNARY/prog_bits,
                        rel_mse=rel, secs=time.time()-t1, program=str(prog)[:400]))
    print(json.dumps(results[-1]), flush=True)

# SVD baselines (full matrix)
U, S, Vt = np.linalg.svd(gate, full_matrices=False)
svd_rows = []
for rank in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank]
    err = np.mean((approx - gate) ** 2) / np.mean(gate ** 2)
    svd_rows.append(dict(rank=rank, rel_mse=float(err),
                         matrix_bits=rank*(gate.shape[0]+gate.shape[1])*BITS_TERNARY))
    print("SVD", json.dumps(svd_rows[-1]), flush=True)

# Sparse-coordinate baselines on REAL activations:
# keep top-k |w| coordinates per row; rel_mse = 1 - sum(top-k w^2)/sum(w^2)
# (exact for Gaussian inputs; on real activations we also measure held-out MSE)
sparse_rows = []
for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    W = gate[rows]
    order = np.abs(W).argsort(1)[:, ::-1][:, :k]
    Ws = np.zeros_like(W); np.put_along_axis(Ws, order, np.take_along_axis(W, order, axis=1), axis=1)
    half = len(X) // 2
    ytr = X[:half] @ W.T; yte = X[half:] @ W.T
    ptr = X[:half] @ Ws.T; pte = X[half:] @ Ws.T
    rel = float(np.mean((pte - yte)**2) / np.mean(yte**2))
    bits = k * (11 + BITS_TERNARY)  # 11-bit feature index + ternary weight
    sparse_rows.append(dict(k=k, rel_mse=rel, bits=bits,
                            compression=(gate.shape[1]*BITS_TERNARY)/bits))
    print("SPARSE", json.dumps(sparse_rows[-1]), flush=True)

json.dump(dict(n_rows=N_ROWS, acts_shape=list(X.shape), gp=results,
               svd=svd_rows, sparse=sparse_rows), open(OUT, "w"), indent=2)
print("WROTE", OUT)
