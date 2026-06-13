# PMRE Briefing for Codex

You (Codex) invented PMRA — the insight that heterogeneous allocation of
compression budget across tensor groups beats uniform allocation, and that a
multiple-choice knapsack is the right solver. PMRE is the generalization of
that insight from GGUF quantization formats to **all structured compression
families**: low-rank, Fisher-weighted low-rank, tensor-train, Kronecker,
codebook, and any future family we add.

This document brings you up to speed on what Claude and the user have built
and discovered so you can contribute to solving the remaining open problems.

## What PMRE Is

PMRA asks: "which GGUF format per tensor?"
PMRE asks: "which compression family, at what rank, for every weight matrix?"

The answer is wildly heterogeneous. A single 30-layer model wants:
- 72x Fisher-weighted low-rank on layer 0 attention q_proj
- 747x tensor-train on layer 14 MLP up_proj
- 6.5x Fisher-weighted low-rank on layer 29 MLP gate_proj
- No compression on layer 0 MLP gate/up/down_proj

The allocator uses the same MCKP knapsack you designed for PMRA, now operating
over (layer × matrix × family × rank) instead of (tensor_group × GGUF_source).

## What We've Discovered

### The Bathtub Curve

MLP compressibility varies by **4 orders of magnitude** across depth:

| Layer | MLP gate_proj at 1.6x | Character |
|-------|----------------------|-----------|
| 0     | PPL 980 (from 21.5)  | Dense — handles vocabulary diversity |
| 14    | PPL 21.8             | Near-redundant — 747x still works |
| 29    | PPL 27.2             | Moderate — output reconstruction |

Layer 0 MLP is uniquely hard. Middle layers are almost free to compress.
Last layer is moderately hard. This is a bathtub curve.

### Fisher-Weighted Low-Rank Finds a Different Subspace

Standard SVD minimizes Frobenius reconstruction error. Fisher-weighted low-rank
(via alternating least squares on the element-wise Fisher-weighted objective)
minimizes **functional** error instead. The subspaces are genuinely different:

| Matrix | Compression | SVD dloss | Fisher LR dloss | Factor |
|--------|-------------|-----------|-----------------|--------|
| L00 q_proj | 72x    | +0.0169   | +0.0029         | 5.8x   |
| L14 v_proj | 2.2x   | +0.0289   | +0.0028         | 10.3x  |
| L29 gate (MLP) | 13x | +0.776  | +0.159          | 4.9x   |

Fisher LR has HIGHER Frobenius error but LOWER functional error. It preserves
what matters to the model's output, not what has the most energy.

Critical nuance: Fisher LR is WORSE than standard SVD on middle-layer MLP
(where the spectral and functional subspaces are already aligned). It only
helps when there's a misalignment to correct.

### The ALS Solver Bug and Fix

The Fisher-weighted ALS solver initially diverged to zero because Fisher's
extreme concentration (top 0.1% of parameters hold 97% of Fisher mass) creates
~10^6 dynamic range. The fix:
- Normalize F_sqrt to unit mean before ALS
- Clamp minimum at 1% of mean (no row gets zero weight)
- Adaptive regularization: `reg = 1e-4 / (1 + iteration)`

### Prior Art Position

- **FWSVD (ICLR 2022)**: Fisher-weighted SVD but row-wise scalar Fisher (not
  element-wise). Closed-form but weaker.
- **EMNLP 2022 follow-up**: Acknowledged element-wise is better, used
  Adam+SGD. Our ALS solver is a different approach.
- **GFWSVD (May 2025)**: Kronecker-factored Fisher for generalized SVD.
- **SAES-SVD (arxiv 2602.03051)**: Cumulative error-aware SVD. We implemented
  and tested this — it made things WORSE because it replaces Fisher fitting
  with activation-covariance fitting. Fitting quality dominates error
  propagation correction.
- Element-wise weighted low-rank is NP-hard (SIAM proof). Our ALS is a
  heuristic that works well in practice.

## The Pipeline

### Stage 1: Profile (`structured_search.py`)

For representative layers, try every compression family at multiple ranks.
Measure delta_loss (perplexity change on C4 validation) for each config.
Produces a JSON with ~37 configs per matrix.

Robustness: incremental per-matrix saves, per-config error handling, signal
handler, checkpoint/resume, `--detach` for Modal runs.

### Stage 2: Allocate (`allocator.py`)

Model-agnostic MCKP knapsack. Auto-detects architecture from profiling JSON.
Extrapolates unprofiled layers via nearest non-edge profiled layer. Builds
Pareto frontiers per matrix, solves DP with numpy vectorization.

CLI: `python allocator.py --results results/structured_search.json --n-layers 30
     --embed-params 28311552 --base-ppl 21.524 --target-ratio 5`

### Stage 3: Compress + Refine (`compress_model.py`, `refine_model.py`)

Apply the allocation to the real model. Then fine-tune via knowledge
distillation from the original (teacher) model.

## The Open Problem: Compound Error

When we compress each matrix independently and measure delta_loss, the
per-matrix losses are small. But when ALL matrices are compressed
simultaneously, the errors compound:

| Target | Additive estimate | Actual dloss | Compound factor | PPL |
|--------|------------------|--------------|-----------------|-----|
| 5x     | +3.51            | +6.54        | 1.86x           | 15,955 |
| 10x    | +5.56            | +15.86       | 2.85x           | 177M |

The compound factor grows with compression aggressiveness. At gentle
compression (2x), it should be close to 1.0.

### What We Tried

**SAES-SVD sequential compression**: Compress front-to-back, using both
compressed and full-precision activations to correct for upstream error
(G = W(H + βΔ)H^{-1/2}). Result: **WORSE** than naive (4.37x compound factor
vs 1.86x). Root cause: SAES-SVD uses activation-covariance SVD instead of
Fisher-weighted fitting. The better error correction can't compensate for the
worse fitting method.

**Knowledge distillation**: KD from the original teacher recovers dramatically
from structural compression. The compressed model is broken in a structured,
correctable way — topology intact, content corrupted. KD corrects content
while topology stays fixed.

Results:
- 5x: PPL 15,955 → 118 in 2,500 steps (135x improvement, flattening from
  overfitting on 2K samples)
- 1.5x: PPL 578 → 48 in 1,500 steps (overfitting on 4K samples)
- 2x (clean run, 50K samples, pure KD): PPL 4,109 → 122 at step 500,
  projecting ~27-30 by step 5000. IN PROGRESS.

Key overfitting fix: use 50K+ training samples (not 2-4K) and pure KD
(alpha=1.0). With 50K samples and batch 8 at 5K steps, you see <1 epoch.

### Compound Factor Topology (Major Finding)

The compound factor is NOT "more compression → more compounding." It depends
on WHICH matrices are compressed:

| Target | What's compressed | Compound factor |
|--------|-------------------|-----------------|
| 1.5x   | Attention only    | 3.78x (WORST)   |
| 2.0x   | Attn + some MLP   | 3.24x           |
| 5.0x   | Nearly everything | 1.86x (BEST)    |
| 10.0x  | Everything hard   | 2.85x           |

**Root cause: routing/execution mismatch.** Compressing attention (the router)
while leaving MLP (the executor) intact means the MLP faithfully amplifies the
wrong signal with full precision. Compressing MLP too makes it a low-pass
filter that dampens rather than amplifies routing errors.

**The rule: compress complete functional units, not half-units.** This changes
the allocator design — the knapsack needs interaction terms, not just
independent per-matrix costs.

Testable predictions:
1. Take 1.5x attn-only, ADD MLP compression at same budget → compound drops
2. Compress ONLY MLP, keep attention intact → compound factor below 2.0x
3. Per-layer cumulative loss grows super-linearly at 1.5x, ~linearly at 5x

### What Needs to Be Solved

1. **Validate lossless 2x**: The 2x + 50K sample KD run is in flight. If PPL
   reaches <25, the pipeline is validated and progressive compression is go.

2. **Interaction-aware allocator**: Current knapsack assumes independent
   per-matrix costs. The compound topology finding shows this is wrong —
   compressing attention without MLP is worse than compressing both. The
   allocator needs pairwise interaction terms or functional-unit constraints.

3. **Progressive compression**: 2x (lossless) → re-profile → 3x → KD → repeat.
   The teacher is always the original full-precision model. Each stage starts
   well-adapted. The allocator re-runs with updated Fisher on the compressed
   model.

4. **Structured parameter training**: Current KD fine-tunes materialized dense
   weights. For real size reduction, train the FACTORS (U/V for low-rank, TT
   cores, Kronecker A/B, codebook centroids) while maintaining structure.

5. **Stacking PMRE + PMRA**: Structural compression × quantization. Theory
   predicts synergy: low-rank decomposition concentrates weight energy into
   higher-kurtosis distributions that quantize better. 5x struct × 4x quant
   may give >20x effective compression.

6. **Cross-size distillation**: Instead of self-distilling (compress A,
   teach from A), distill from a BIGGER model (compress A, teach from B).
   The compressed model learns things the original never knew. Currently
   testing: compress SmolLM-135M to 67M, teach from SmolLM-360M. If the
   67M model exceeds the original 135M, compression becomes a knowledge
   transfer scaffold, not just size reduction. Implications: compress
   OLMo-3-7B-Think to 400M, teach from the biggest open model available.

7. **Layer distillation from scratch vs warm restart**: Warm-restarting layer
   distill on a model that already converged with logit-only KD creates
   competing gradients — the model fights to maintain its logit optimum while
   rearranging hidden states. Layer distill FROM SCRATCH (right after
   compression) aligns both losses from step 1. Currently testing this on
   1.69x SmolLM. If it breaks through the PPL 39 wall that logit-only KD
   couldn't, this becomes the default training recipe.

8. **Progressive compression**: If 1.69x is lossless, compress the RESULT
   another 1.69x, KD from the ORIGINAL teacher. 1.69^2 = 2.86x, 1.69^3 =
   4.83x. The teacher never degrades. Key question: does compressibility
   regenerate after KD (optimizer creates new redundant structure), or does
   it deplete?

9. **Scale to OLMo-3-7B-Think**: PROFILING IS IN FLIGHT on H100
   (`ap-GXEJn7Uw20YWMPIEe1AAaH`). 5 layers (L0, L3, L15, L28, L31) to
   capture SWA vs full-attention pattern. Architecture: 32 layers, full MHA,
   SwiGLU 11008, post-norm, QK-norm, SWA pattern (3+1)×8.
   Embed+lm_head = 821M (11.25%, incompressible, NOT tied).
   Compressible: 6.48B. At 1.69x structural: 4.65B params.
   At 1.69x struct × 4x PMRA quant = ~1.16B effective.

## Capacity Analysis (Important)

Scaling laws predict PPL ~27 at 53M layer params (2x SmolLM). We measured
PPL 39. The 12-PPL gap has three possible explanations:

1. Architecture inefficiency: 30 layers of low-rank matrices wastes params
   on overhead a purpose-built model wouldn't have
2. Training insufficiency: 20K steps of KD vs trillions of tokens for
   pretraining
3. Compression artifacts: low-rank constraint limits expressiveness

The 1.69x experiments (more params, layer distill) will disambiguate.
If the gap narrows at 1.69x, progressive compression works. If it doesn't,
the architecture itself needs to change.

Cross-distillation (360M teacher) did NOT help at 2x — confirms capacity-
limited, not knowledge-limited. The student can't absorb more knowledge
regardless of teacher quality.

## Repository Layout

```
PMRA/PMRE/
  README.md                    the showcase
  CODEX_BRIEFING.md            this document
  scripts/
    allocator.py               per-everything knapsack (model-agnostic)
    structured_search.py       structured family profiler
    compress_naive.py           all-at-once compression baseline
    compress_sequential.py     SAES-SVD sequential (failed, needs Fisher)
  modal/
    modal_structured.py        profiling on A100
    modal_compress.py          naive compression
    modal_compress_seq.py      sequential compression
  docs/
    METHOD.md                  equations, algorithms, numerical tricks
    FINDINGS.md                experimental results
```

Primary development also at: `C:\Users\power\Documents\olmo-compression\`
(has refine_model.py, modal_refine.py, and results/).

## Data

`results/structured_search.json`: 765 compression configs across 21 matrices
(layers 0, 14, 29 × 7 matrix types) of SmolLM-135M. Each entry:

```json
{
  "family": "fisher_lr",
  "matrix": "model.layers.14.self_attn.q_proj.weight",
  "original_params": 331776,
  "compressed_params": 4608,
  "compression_ratio": 72.0,
  "frobenius_error": 0.958,
  "fisher_weighted_error": 1.2e-06,
  "ppl": 21.56,
  "delta_loss": 0.0053
}
```

## Key Technical Details

### Fisher-Weighted Low-Rank ALS

Minimizes `Σ F_ij × (W_ij - (UV^T)_ij)²` via alternating least squares:
1. Fix V, solve for U: batched weighted least squares, one system per row
2. Fix U, solve for V: batched weighted least squares, one system per column
3. Repeat 10 iterations

Fisher normalization is critical:
```python
F_sqrt = fisher_diag.sqrt()
F_mean = F_sqrt.mean()
if F_mean > 0:
    F_sqrt = F_sqrt / F_mean
F_sqrt = F_sqrt.clamp(min=0.01)
```

### MCKP Knapsack

Numpy-vectorized DP. 210 groups × ~12 Pareto-optimal options × ~20K budget
states. Uses quantization unit of 1024 params. Backtracking via stored choice
indices and previous-state pointers.

### The User's Bar

Compression is not valid until the compressed model has basically lossless PPL.
PPL must stay within ~2-3 of baseline. Work up from gentle compression where
lossless is achievable, then progressively push the ratio. Don't celebrate
allocation strategies without end-to-end PPL results.
