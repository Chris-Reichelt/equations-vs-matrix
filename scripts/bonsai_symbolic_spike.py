"""Spike: can one FFN weight matrix of a Bonsai model be replaced by a
symbolic program (equations) at lower description cost than the matrix?

Model on disk: ../models/Bonsai-1.7B-unpacked
(Qwen3 arch, hidden=2048, intermediate=6144, 28 layers). The 4B variant is
not on this machine; 1.7B unpacked is the same family and is the honest
stand-in. Ternary weights are ~1.58 bits/value; we compare programs against
matrices at matched bits.

Calibration activations: post-RMSNorm hidden states have per-component RMS
~1, so x ~ N(0,1)^2048 is a defensible synthetic prior without a torch
forward pass (torch is not installed here). Flagged as an assumption.

Outputs: results.json with per-row GP program size/error + SVD baselines.
"""
import json, os, sys, time
import numpy as np
from safetensors import safe_open
from gplearn.genetic import SymbolicRegressor

MODEL = "../models/Bonsai-1.7B-unpacked/model.safetensors"
OUT = "./results/results.json"
LAYER = 10
N_ROWS = 32
N_CAL = 2048
BITS_TERNARY = 1.585  # log2(3)
BITS_FP16 = 16.0
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def load(key):
    with safe_open(MODEL, framework="numpy", device="cpu") as f:
        return f.get_tensor(key)

t0 = time.time()
gate = load(f"model.layers.{LAYER}.mlp.gate_proj.weight")   # (6144, 2048)
up   = load(f"model.layers.{LAYER}.mlp.up_proj.weight")
down = load(f"model.layers.{LAYER}.mlp.down_proj.weight")
print(f"loaded gate {gate.shape} up {up.shape} down {down.shape} in {time.time()-t0:.1f}s", flush=True)

rng = np.random.default_rng(1337)
X = rng.standard_normal((N_CAL, gate.shape[1]))  # calibration activations ~ N(0,1)

# ternary check: are weights actually ternary?
uniq = np.unique(gate)[:8]
print("gate unique values (head):", uniq, flush=True)

func_set = ['add','sub','mul','div','neg','sqrt','abs','log','sin','cos','tan','inv','max','min']
rows = rng.choice(gate.shape[0], N_ROWS, replace=False)
results = []
for i, r in enumerate(rows):
    w = gate[r].astype(np.float64)
    y = X @ w                      # (N_CAL,)
    ytr, yte = y[:N_CAL//2], y[N_CAL//2:]
    Xtr, Xte = X[:N_CAL//2], X[N_CAL//2:]
    sr = SymbolicRegressor(population_size=150, generations=12,
                           stopping_criteria=0.0,
                           function_set=func_set, parsimony_coefficient=0.002,
                           const_range=(-1.0, 1.0),
                           init_method='half and half',
                           tournament_size=10,
                           n_jobs=1, random_state=int(r), verbose=0)
    t1 = time.time()
    sr.fit(Xtr, ytr)
    prog = sr._program
    n_nodes = len(prog.program)  # gplearn 0.4.x: flat prefix-node list
    pred = sr.predict(Xte)
    mse = float(np.mean((pred - yte) ** 2))
    rel = mse / float(np.mean(yte ** 2))
    # bits: program ~ 5 bits per node (opcode+operand class) + 16 per constant.
    # gplearn renders constants as '%.3f' floats and features as X%d.
    import re as _re
    _txt = _re.sub(r'X\d+', ' ', str(prog)).replace('(', ' ').replace(')', ' ').replace(',', ' ')
    n_const = 0
    for _t in _txt.split():
        try:
            float(_t); n_const += 1
        except ValueError:
            pass
    prog_bits = n_nodes * 5 + n_const * 16
    row_bits = gate.shape[1] * BITS_TERNARY
    results.append(dict(row=int(r), n_nodes=n_nodes, n_const=n_const,
                        prog_bits=prog_bits, row_bits=row_bits,
                        compression=row_bits/prog_bits,
                        rel_mse=rel, mse=mse, secs=time.time()-t1,
                        program=str(prog)[:400]))
    print(json.dumps(results[-1]), flush=True)

# SVD baselines on the full gate matrix at matched bits
sv, s, vt = np.linalg.svd(gate.astype(np.float64), full_matrices=False)[:3] if False else np.linalg.svd(gate.astype(np.float64))
svd_rows = []
for rank in [1, 2, 4, 8, 16]:
    approx = sv[:, :rank] @ np.diag(s[:rank]) @ vt[:rank]
    err = np.mean((approx - gate.astype(np.float64)) ** 2) / np.mean(gate.astype(np.float64) ** 2)
    bits = rank * (gate.shape[0] + gate.shape[1]) * BITS_TERNARY
    svd_rows.append(dict(rank=rank, rel_mse=float(err), matrix_bits=bits,
                         matrix_bits_vs_full=len(gate.ravel())*BITS_TERNARY))
    print("SVD", json.dumps(svd_rows[-1]), flush=True)

json.dump(dict(layer=LAYER, gate_dtype=str(gate.dtype), shapes=dict(gate=list(gate.shape), up=list(up.shape), down=list(down.shape)),
               ternary_values=[float(v) for v in uniq], gp=results, svd=svd_rows,
               total_secs=time.time()-t0), open(OUT, "w"), indent=2)
print("WROTE", OUT)
