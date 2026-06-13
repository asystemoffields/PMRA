# Production Mixed-Rate Everything

PMRE extends PMRA from quantization format selection to **any structured
compression family**. Where PMRA asks "which GGUF format per tensor?", PMRE
asks "which compression — low-rank, tensor-train, Kronecker, codebook,
Fisher-weighted, or original — at what ratio, for every weight matrix in the
model?"

The answer turns out to be wildly heterogeneous. A single 30-layer model wants
72x Fisher-weighted low-rank on early attention, 747x tensor-train on
middle-layer MLP, 6.5x Fisher-weighted low-rank on the last layer's MLP, and
no compression at all on layer 0's MLP. The allocator finds this automatically
via the same multiple-choice knapsack that powers PMRA.

## What We Found

Three layers of SmolLM-135M were profiled across 10 structured families (765
configurations). The results surprised us:

- **Middle-layer MLP is nearly redundant.** Layer 14's up_proj compresses 747x
  (TT rank 4) with only +0.064 dloss. The weights barely matter.
- **Layer 0 MLP is uniquely dense.** The same matrix at layer 0 gives PPL 980
  at just 1.6x compression. Every direction matters here.
- **Fisher-weighted low-rank finds a different subspace than SVD.** At rank 4
  (72x), Fisher LR gives 5.8x lower functional error than standard SVD —
  while having *higher* Frobenius error. It optimizes what matters, not what's
  big.
- **The compressibility curve is a bathtub.** First and last layers are hard,
  middle layers are easy. Classic neural network sensitivity but quantified
  precisely per matrix per family.

## How It Works

### 1. Profile: Structured Family Search

For representative layers of a model, try every compression family at every
ratio and measure the actual functional cost (delta_loss via C4 perplexity).
This produces a Pareto frontier per matrix: the tradeoff between fewer
parameters and more quality loss.

```
python scripts/structured_search.py \
  --model HuggingFaceTB/SmolLM-135M \
  --layers 0 14 29 \
  --output results/structured_search
```

Families tested: low-rank SVD, Fisher-weighted low-rank (ALS), low-rank +
sparse correction, Kronecker product, Kronecker sum, block diagonal, product
quantization (codebook), Fisher-weighted codebook, tensor-train.

### 2. Allocate: Per-Everything Knapsack

Given profiling results and a target compression ratio, solve a multiple-choice
knapsack over (layer x matrix x family x rank) to minimize total quality loss
at the parameter budget. Model-agnostic — auto-detects architecture from the
profiling data.

```
python scripts/allocator.py \
  --results results/structured_search.json \
  --n-layers 30 \
  --embed-params 28311552 \
  --base-ppl 21.524 \
  --target-ratio 5 10 17.5
```

The knapsack beats greedy allocation by 39-190% on dloss depending on the
target ratio. Uniform compression isn't even in the conversation.

### 3. Compress: Sequential SAES-SVD

Apply the allocation to the actual model. Compresses front-to-back using
cumulative error compensation from SAES-SVD (arxiv 2602.03051): each layer's
compression sees the actual distorted activations from upstream compressed
layers and corrects toward the full-precision reference.

```
python scripts/compress_sequential.py \
  --allocation results/allocation_5.0x.json \
  --model HuggingFaceTB/SmolLM-135M
```

## Vocabulary

Extending the PMRA vocabulary:

- **family**: a structured compression method (low-rank, TT, Kronecker, etc.).
- **config**: a specific (family, rank/parameter) choice for one matrix.
- **profile**: the set of (config, delta_loss) measurements for a matrix.
- **Pareto frontier**: the non-dominated configs for a matrix — best dloss at
  each compression level.
- **allocation**: one chosen config per matrix that fits the budget.
- **compound factor**: ratio of actual whole-model dloss to the additive
  (per-matrix) estimate. Measures how badly errors interact across layers.
- **Fisher-weighted**: compression that minimizes functional error (Fisher
  Information) instead of reconstruction error (Frobenius norm).
- **bathtub curve**: the pattern where first and last layers are hard to
  compress, middle layers are easy.
- **SAES-SVD**: sequential compression with cumulative error compensation.
  Formula: `G = W(H + beta * Delta) @ H^{-1/2}`, truncated SVD of G.

## Why This Is Exciting

PMRA proved that heterogeneous allocation beats uniform allocation for
quantization formats. PMRE proves the same thing holds — even more strongly —
across the entire space of structured decompositions.

The profiling reveals structure in weight matrices that nobody has systematically
mapped before. The fact that Fisher-weighted low-rank discovers a fundamentally
different (and better) subspace than standard SVD means we're not at the frontier
of what's possible — we're just at the frontier of what SVD can see.

The bathtub curve means most of a model's parameters are computational
scaffolding that can be removed, with the real information concentrated in the
edges. A 30-layer model might need 28 of those layers at 100-700x compression
and only 2 layers near their original size.

There is a lot of room out there.

## Repository Layout

```
PMRE/
  scripts/
    allocator.py             per-everything knapsack solver
    structured_search.py     structured family profiler
    compress_naive.py        all-at-once compression (baseline)
    compress_sequential.py   SAES-SVD sequential compression
  modal/
    modal_structured.py      Modal harness for profiling
    modal_compress.py        Modal harness for naive compression
    modal_compress_seq.py    Modal harness for sequential compression
  docs/
    METHOD.md                technical details and equations
    FINDINGS.md              experimental results and analysis
```

## Prior Art

- **PMRA**: this project's parent — byte-budgeted GGUF allocation via knapsack.
- **FWSVD** (ICLR 2022): Fisher-weighted SVD with row-wise scalar Fisher.
- **SAES-SVD** (arxiv 2602.03051): cumulative error-aware SVD compression.
- **GPTQv2** (arxiv 2504.02692): asymmetric calibration for quantization.
- **SVD-LLM** (ICLR 2025): activation-covariance whitening before SVD.

## Status

Early-stage research. The profiling, allocation, and compression pipeline works
end-to-end. The sequential SAES-SVD compressor is the current frontier — closing
the gap between per-matrix quality estimates and actual whole-model performance.

Next: scale to OLMo-3-7B-Think, add gradient refinement of structured
parameters, and connect to OrbitQuant rotation preprocessing for the hard
matrices.
