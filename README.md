# Equations vs the Matrix: can an LLM weight matrix be replaced by equations?

**Question (Chris, 2026-10-17):** an LM is "essentially a large matrix of
numbers" — could that matrix be replaced by a series of *equations* that
compute faster? (Tested against PrismML's Bonsai family, the 1-bit/ternary
on-device LLMs.)

**TL;DR verdict (measured end to end on real weights, real activations, real
perplexity):**

- **Pure equations replacing the whole matrix: no.** The error floor is
  ~0.33 rel_mse even at 160+ bits/row and is *information-limited* — a 17×
  compute budget barely moved it. The matrix is nearly optimal at what it does.
- **Hybrid — equations for the compressible 47% of rows: yes, it ships.**
  Baseline NLL 3.2549 (ppl 25.92) vs hybrid NLL 3.2641 (ppl 26.16):
  **+0.9% perplexity cost** for replacing 2,892/6,144 gate units (47%) with
  ~35-bit programs — an 86× byte reduction on half the matrix, which in a
  memory-bound decode regime is worth roughly **~1.9× FFN speed**.
- Decode is memory-bandwidth-bound, not compute-bound: bytes/token *is* the
  speed limit, which is why compression == speed here.

## The three arms (same held-out split, scored against ternary bits = 1.585 bits/weight)

| bits/row | GP programs | SVD | sparse-k |
|---|---|---|---|
| 10 | **0.63** | 0.99 | 1.00 |
| 40 | **0.61** | 0.97 | 0.99 |
| 80 | **0.45** | 0.94 | 0.98 |
| 160 | **0.28** | 0.89 | 0.97 |
| 2,560 | 0.28 | 0.71 | 0.84 |

- **GP programs** (gplearn SymbolicRegressor, 500 pop × 60 gens, all 6,144
  rows): median **35 bits/row**, rel_mse **0.52** (best 0.07) → **86×
  compression** vs the ternary matrix (3,246 bits/row). 47% of rows under
  0.5, 4.9% under 0.2, 88% beat zero-prediction.
- **SVD fails**: rank 256 still 0.71 error at 541 bits/row.
- **Sparsity fails**: keeping the 256 biggest weights per row still 0.81.
- **Why both fail**: participation-ratio effective rank of the matrix is
  **1,393 of 2,048** (σ₁₆/σ₁ = 0.65) — no low-rank structure to find.
- **The kicker**: the evolved programs are literally KAN-style per-feature
  univariate functions (`min(X520, neg(X768))`, `sub(log(cos(cos(X1267))),
  cos(abs(X445)))`) — the Kolmogorov-Arnold thesis *emerged from the GP run
  itself*, ~1 constant across 32 programs.

## The pivot that mattered

v1 used synthetic `x ~ N(0,1)` activations and scored GP rel_mse ≈ 1.0 —
nothing found. v2 used **real layer-10 activations** captured from a torch
CPU forward pass of the on-disk model (per-component RMS median 0.367, not
1.0 — the synthetic prior was wrong) and the verdict flipped. Always use
real activations.

## Layout

```
scripts/
  bonsai_real_activations.py     # torch forward pass -> real layer-10 acts
  bonsai_symbolic_spike.py       # v1: synthetic activations (the false negative)
  bonsai_symbolic_spike_v2.py    # v2: 32 rows, real acts, GP/SVD/sparse arms
  bonsai_symbolic_spike_v3.py    # v3: all 6,144 rows, 500x60 GP, checkpointed
  bonsai_perplexity_harness.py   # definitive: swap programs into the forward
                                 # pass, baseline vs hybrid NLL/perplexity
results/                          # every JSON/JSONL/NPZ artifact + logs/
logs/                             # full stdout of every run, crashes included
```

Paths in scripts assume the repo sits beside the model:
`../models/Bonsai-1.7B-unpacked` (Qwen3 arch: hidden 2048, intermediate 6144,
28 layers; unpacked continuous weights — the deployed Bonsai is ternary,
which is already "equations" in the arithmetic sense: sign-flips and adds).

## Reproduce

```bash
python3 scripts/bonsai_real_activations.py      # ~35s, CPU
python3 scripts/bonsai_symbolic_spike_v3.py     # ~6.5h, 3 workers, checkpointed
python3 scripts/bonsai_perplexity_harness.py    # baseline + hybrid NLL
```

## Pitfalls learned the hard way (all hit live)

- gplearn 0.4.3 API: `stopping_criteria` (not `stopping_tolerance`);
  `const_range` must be a tuple; valid function names are exactly
  `abs add cos div inv log max min mul neg sin sqrt sub tan` (no `sigmoid` —
  enumerate `gplearn.genetic._function_map`).
- Node count = `len(prog.program)` (flat prefix list; no `.count`/`.raw()`).
- `__str__` emits **LISP with parens** (`add(X0, X1)`); the flat program list
  is **prefix** (function before args — read `_program.py execute()`). A
  parser must skip commas (the tokenizer drops them) and rebuild prefix
  lists; constants render as `%.3f`, features as `X%d`.
- Checkpoint program strings must be saved **full** (`str(prog)`, not
  `[:200]`) — truncated strings parse as "mid-tree" and silently dropped 99%
  of programs on the first harness attempt.
- `np.put_along_axis` values need `np.take_along_axis(W, order, axis=1)`.
- This box is chronically at 2–6 GB RAM free (a local engine pins 48 GB): GP
  parallel workers must be ≤3 with `low_memory=True` plus a MemAvailable
  guard between chunks, or the kernel OOM-kills the run **silently** (no
  traceback). Three runs died that way before the guard landed.
- torch on this box: `pip install torch --index-url
  https://download.pytorch.org/whl/cpu` (the cu124 index has no py3.14 wheels).

## Scope limits (honest)

One layer (10), gate rows only — up/down projections stayed matrices. The
full-model number is a mechanical extension of this harness across all 28
layers.

## Next

Extend the harness to all 28 layers' gate rows (GP fits run overnight at
~16 rows/min) and measure end-to-end perplexity + tokens/sec on the real
4B ternary model — converting "~1.9× FFN speed" into a measured tokens/sec
claim.
