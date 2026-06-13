# PMRE Method

## Overview

PMRE is a three-stage pipeline: Profile, Allocate, Compress.

## Stage 1: Structured Family Search

For each weight matrix in representative layers, fit every structured
decomposition family at multiple compression levels and measure the functional
cost (delta_loss = compressed_loss - baseline_loss on C4 validation).

### Families

**Low-rank**: `W ~ U @ V^T` via truncated SVD. Ranks: 4, 8, 16, 32, 64, 128, 256.

**Fisher-weighted low-rank**: Same factorization, but minimizes
`sum F_ij * (W - UV^T)_ij^2` via alternating least squares. The Fisher diagonal
is normalized to unit mean and clamped at 1% floor to prevent ALS divergence
from extreme Fisher concentration. Regularization decays with iteration:
`reg = 1e-4 / (1 + iter)`.

**Low-rank + sparse**: `W ~ UV^T + S` where S retains the top-k residual
elements by magnitude. Configs: rank 16/32/64 + 1-5% sparse.

**Kronecker**: `W ~ A kron B`. Nearest Kronecker product via Van Loan-Pitsianis.

**Kronecker sum**: `W ~ sum_i A_i kron B_i`. Multi-term SVD of the rearranged
block matrix.

**Block diagonal**: For square matrices only.

**Tensor-train**: Reshape W into a higher-order tensor, TT-SVD decomposition.

**Product quantization (codebook)**: Split rows into subvectors, k-means
clustering. Configs: 4-16 codebooks x 64-256 entries.

**Fisher-weighted codebook**: Same as codebook but k-means uses Fisher
importance weights for distance and centroid updates.

### Evaluation

Each config is evaluated by substituting the approximation into the model and
measuring perplexity on 64 C4 validation sequences of length 512. The model is
restored to original weights after each measurement.

## Stage 2: Per-Everything Allocation

Given profiling results and a target compression ratio, solve a multiple-choice
knapsack problem (MCKP):

```
minimize   sum_i delta_loss(config_i)
subject to sum_i compressed_params(config_i) <= budget
           exactly one config per matrix group
```

Each matrix group includes a "no compression" option (cost = original_params,
delta_loss = 0). The Pareto frontier is computed per group to prune dominated
configs.

### Extrapolation

Only a few layers need profiling. Unprofiled layers are mapped to the nearest
non-edge profiled layer. Edge layers (first and last) are mapped to their own
profiled equivalents.

### Solver

Numpy-vectorized DP with backtracking. Parameters quantized to 1024-unit
resolution. 210 groups x ~12 options x ~20K budget states = solved in seconds.

## Stage 3: Sequential SAES-SVD Compression

Applying all compressions independently causes errors to compound across layers
(1.9-2.9x worse than additive estimate). The fix: compress front-to-back using
cumulative error compensation.

### Core formula (from SAES-SVD, arxiv 2602.03051)

```
G = W @ (H + beta * Delta) @ H^{-1/2}
```

- `H = X @ X^T`: input covariance from compressed activations
- `Delta = (X_fp - X) @ X^T`: cross-covariance of activation error
- `X_fp`: full-precision activations (cached from one forward pass)
- `X`: actual activations through the already-compressed prefix
- `beta in [0,1]`: correction strength, auto-tuned via ACES

The optimal rank-r approximation is the truncated SVD of G. Closed-form, no
training required.

### ACES coefficient selection

Grid search over beta in [0, 0.95] to maximize retained energy ratio:
`sum(sigma_i^2 for i < r) / sum(sigma_i^2 for all i)` where sigma are the
singular values of G(beta).

### Numerical stability

The covariance matrix H becomes ill-conditioned after several layers are
compressed. H^{-1/2} is computed via SVD with relative eigenvalue clipping
(threshold = max_eigenvalue * 1e-4) and trace-proportional regularization
(1e-3 * trace(H)/d).
