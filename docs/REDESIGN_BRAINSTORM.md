# PMRA Redesign Brainstorm — from empirical probing to analytic allocation

_2026-06-10. Produced by a research agent grounded in METHOD.md, the PMRE
README, and the allocation_rows corpus (e.g.
`results/gemma4_e2b_it/selector_result_knapsack.json`, 1209 rows with both
`calib_nll_improvement` and `weight_sse_delta`, plus measured joint
`variants`). Companion to the CPU prober (docs/CPU_PROBER.md), which attacks
probe *cost*; this attacks probe *necessity*._

## 0. Diagnose first: was the qwen35-b2 GRAY an additivity failure or a noise failure?

- **Noise hypothesis.** At 4 calibration prompts, per-group dNLL standard
  error is plausibly comparable to most groups' true effects. A knapsack over
  noisy scores is a selection-bias maximizer (winner's curse): it picks
  groups whose noise was positive, and the realized mix regresses to the mean
  — exactly "ties the random control."
- **Interaction hypothesis.** True per-group effects don't add (cross-layer
  error correlation, cancellation, large-perturbation regime at IQ2_M).

**One-hour test on existing data:** split calibration prompts in half;
per-group dNLL on each half; the cross-half correlation is the reliability
ceiling. Low (< ~0.5) => GRAY is mostly noise and a deterministic proxy can
beat empirical probing outright. Also compute the compound factor on
measured `variants`: actual joint dNLL / summed per-group dNLL. Prior:
noise-dominant, interactions secondary.

## 1+2. One forward+backward scores every candidate: the Hessian-sketch scorer

The Tier-1 imatrix-weighted SSE is the diagonal special case of something
exact and barely more expensive. For y = Wx, the exact expected local error
of dW = W_high - W_low is

    E||dW x||^2 = tr(dW * A * dW^T),   A = E[x x^T]

The imatrix is exactly diag(A). Upgrade: cache the full per-layer input
covariance A_l from ONE forward pass; every candidate's exact local error is
one d x d matmul. (This is the GPTQ/OBS Hessian repurposed for allocation.)

**Loss-gain correction (the missing bathtub):** local error is not loss.
K-FAC-style: dNLL ~= tr(G * dW * A * dW^T), with G = E[delta delta^T] the
layer-output gradient second moment; diag(G) from ONE backward pass. This is
OBD made exact in the input dimension — and the input dimension is the right
place to be exact.

**Signed first-order term (the interaction fix):** the model is not at a
wikitext minimum, so cache per-token output gradients g_{l,t} and compute the
signed sum_t g^T (dW x_t) per candidate. First-order terms are exactly
additive across groups and capture cancellation between promotions — they
predict the NLL of any *mix*, not just any candidate.

Costs: one fp16/Q8_0 forward + one backward with hooks (CPU-feasible at 4B,
tens of minutes, once per model); A_l storage ~13 MB fp16 at d=2560; scoring
all ~1500 candidates is minutes of CPU matmuls. Risks: quadratic trust region
at IQ2_M-sized deltas (mitigate: empirically probe only top-decile-error
groups); CPU backward slow at >4B (fallback: per-(role,depth) gain factors
fitted on the allocation_rows corpus — the bathtub as a multiplier table).

**Free validation, before writing production code:** score ladder on
existing allocation_rows — Spearman vs measured improvements AND knapsack
regret (solve knapsack with predicted scores, evaluate the chosen set under
measured scores, compare to the oracle):
bytes-only < imatrix-SSE (current Tier 1) < tr(dW A dW^T) < G-weighted
< G-weighted + signed term.

## 3. Fix additivity at the decision layer

a) **Token-level KL to fp16 as the objective** (cache fp16 logits once).
   Full-distribution comparison per token instead of one-hot log-prob cuts
   estimator variance enormously — 4 prompts behave like ~100. If GRAY was
   noise, this alone may flip it. (v2 already has --objective kl_fp16; make
   it the default for selection.)
b) **Evaluate mixes, not groups.** Tier 1 fixes confident decisions and
   isolates the ambiguous boundary band (10-25 binary-ish choices); search
   over band subsets with JOINT llama-perplexity evals of real spliced GGUFs
   (lazy greedy swaps, or a light surrogate with the proxy as prior mean).
   10-30 joint CPU evals; the additivity assumption is never invoked where
   it can hurt. Each accepted state is a verified improvement, so GRAY
   verdicts die by construction.
c) **Submodularity** is the theory story for (b): if promotions are
   substitutes, cost-benefit lazy greedy carries a 1-1/e guarantee. Cheap
   check on measured variants: do marginal gains shrink with set size?

## 4. Learned allocator (PMRA-zero)

Pool allocation_rows across the 6+ completed models (~4-7k rows). Features:
depth fraction, role one-hot, bpw_low/high/delta, log payload bytes, weight
stats (std, kurtosis, outlier mass, max|w|/rms), imatrix concentration, and
the Tier-1 proxy itself (learn the role x depth correction to the proxy).
Target: improvement normalized by the model's total quant gap; GBT; honest
leave-one-model-out knapsack-regret CV. If the learned model's edge is
"mostly a role x depth multiplier," skip the ML and ship the multiplier
table. Always pair with the joint-mix verification loop from 3(b).

## 5. Drop the splice framing: water-filling over the GGUF type lattice

Bartowski's quant levels are already internally mixed, so splice sources are
correlated bundles and the candidate lattice is publisher-chosen.
`llama-quantize --tensor-type` removes that: for each tensor x each GGUF
type, quantize->dequantize in memory, compute distortion tr(dW A dW^T) from
the cached covariances, take the per-tensor lower convex hull of (bytes,
distortion), then classic Lagrangian rate-distortion: bisect lambda until
marginal distortion per byte is equalized at budget B. No knapsack needed on
convex hulls (keep it as fallback). Emit ONE llama-quantize run with
overrides. Entire pipeline: one imatrix run + one forward/backward cache +
one final validation perplexity.

Risks: loses the "remix of public artifacts" product identity (keep splice
mode as product, direct mode as frontier); per-type shape constraints shrink
the lattice; quantize x 12 types is real one-time CPU work.

## 6. Wild but steelmanned

- **PMRA-LoRA error-correction sidecars:** keep the low base, ship a rank-r
  A-weighted (Fisher) low-rank correction of the quant residual as a
  standard llama.cpp LoRA adapter GGUF — "a 30 MB sidecar that upgrades any
  public IQ2_M toward Q3 quality." Rank-per-tensor slots into the same
  allocator (bytes vs closed-form distortion drop). PMRE's Fisher-ALS code
  is the seed. Risk: adapter fp16 speed hit; residuals may need large r on
  the hardest tensors. Afternoon-of-CPU feasibility check: D(r) curves on
  one model's residuals.
- **Zero-probe deployment mode:** bartowski publishes imatrix.dat next to
  the quants. Learned allocator + published imatrix + GGUF downloads =
  allocation computed in minutes of linear algebra with the model never run;
  one final validation perplexity. PMRA usable by anyone without a GPU.
- **Intermediate-state KL:** per-layer hidden-state L2/cosine to fp16 from a
  single paired forward = a one-forward empirical bathtub, no backprop;
  sanity cross-check on G.

## Ranked recommendation

1. **Hessian-sketch one-pass scorer** (eliminates the entire
   O(groups x sources) forward cost; deterministic scores attack the likely
   GRAY cause; validation is free on existing rows — run the score ladder
   BEFORE writing production code).
2. **Mix-level Tier 2 with KL-to-fp16** (structural additivity fix +
   variance fix; modest refactor of the CPU prober built 2026-06-10; ships
   verified-better-than-proxy mixes by construction).

They compose: the scorer is the prior mean for the mix search. Water-filling
(5) is the elegant medium-term endpoint but inherits the scorer's distortion
model — prototype it after the scorer validates. Step zero: the
noise-vs-interaction diagnosis in section 0.
