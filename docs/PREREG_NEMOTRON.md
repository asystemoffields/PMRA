# PREREG — Nemotron-3-Nano-4B v2 validation (CPU prober + code guardrail)

Registered 2026-07-02, before any Kaggle launch. **Amended same day, still pre-launch, after an
adversarial design review (verdict: fix-then-launch); amendments below override the original text.**

## Amendments (pre-launch, from design review)

1. **Pinned eval stack:** llama.cpp release `b9859` (prebuilt or source-built at that tag). The
   prober stamps every checkpoint row with a cache signature (model file, ctx, llama.cpp build,
   corpus sha, chunks) and discards mismatching rows — no cross-build or cross-model reuse.
2. **Wall-clock fits the 12h cap by construction:** threads=4 (default left half the vCPUs idle),
   MAX_PROBES 32, probe-se-stop 0.004 (paired SEs below that are unresolvable at 24 chunks),
   tier-2 time budget 390 min (projection-based stop; unprobed candidates fall to proxy backfill,
   which is recorded: `backfill_count`/`backfill_extra_bytes` in result.json).
3. **Verdict legs get paired per-chunk SEs** (`verdict_stats` in result.json). Banking a GO
   additionally requires the target-leg paired delta ≥ 2×SE; a GO inside 2×SE is treated as GRAY
   (park, don't ship). This tightens, never loosens, the original bars.
4. **Random control = median of 3 seeded draws** (seeds 17/18/19), so one lucky draw can't flip
   the leg.
5. **Code corpus interleaved** (MBPP-sanitized + HumanEval proportional round-robin — a
   concatenated corpus would leave HumanEval entirely outside the evaluated prefix); new sha256
   `ae93382eca1809dc…`; code evals run at 48 chunks; the guardrail records a paired per-chunk
   regression SE. Boundary rule: if |regression − ε| < 1×SE, adjudicate GRAY with a noise
   hypothesis rather than banking pass/fail.
6. **Secondary qwen35 arm scoring is split:** the coverage hypothesis is scored on the NLL legs
   alone via `verdict_before_guardrail`; the guardrail scores ship-eligibility separately.
   Within-run comparisons only — the 06-13 numbers (different llama.cpp build) are context, not
   a baseline. Fresh kernel slug, no attached checkpoints (asserted in the harness).
7. **Harness fail-fast:** kernel asserts the cloned repo has the profile-aware prober
   (capability probe + prints the git SHA), asserts prober exit code 0, asserts no stray
   checkpoint attachments on fresh runs, validates the imatrix download (size + GGUF magic).
8. **Label honesty:** stock IQ3_XS is llama.cpp's tuned per-tensor ftype mix, not a uniform
   quant — release notes must say "stock IQ3_XS", not "uniform". This is the pre-registered scale for the PMRA v2
release decision on this model: NO-GO here = KILL for the Nemotron v2 release candidate (not for
the method). Secondary arm (qwen35 re-run) registered below with its own bar.

## Question

Does the v2 CPU-prober pipeline (profile-aware grouping, two-tier probing, code guardrail) produce
a mixed GGUF for Nemotron-3-Nano-4B that beats the same-budget uniform quant on held-out NLL
without regressing on code — i.e. a ship-eligible v2 release artifact?

## Arms

| arm | what | source |
|---|---|---|
| candidate | `c2_calib_knapsack_mixed`: knapsack over empirical probes + proxy backfill, low=iq2_m, byte budget = iq3_xs payload, highs {q3_k_s, q3_k_m, iq4_xs} | cpu_prober.py |
| stock control | uniform IQ3_XS (the budget-defining quant) | bartowski GGUF |
| allocation control | `c2_random_same_budget`: random selection at the same budget, seed 17 | cpu_prober.py |

## Conditions ledger (identical across arms unless stated)

- **Model/quants:** bartowski/nvidia_Nemotron-3-Nano-4B-GGUF, files IQ2_M / IQ3_XS / Q3_K_S /
  Q3_K_M / IQ4_XS (fixed repo revision at launch).
- **Eval path:** llama-perplexity from the same in-kernel llama.cpp build (version recorded in
  kernel log), ctx 512, chunks 24, one kernel = one machine for all final evals.
- **Corpora:** calib = wikitext-2-raw-v1 train slice (256KB), eval = test slice (256KB) — same
  recipe as the qwen35 run; code = MBPP-sanitized + HumanEval interleaved via
  tools/build_code_corpus.py (421 tasks, sha256 ae93382eca1809dc…). Selection uses calib only;
  verdict uses eval + code.
- **Grouping:** --tensor-profile nemotron_h, --group-mode tensor; coverage pre-verified 100%
  (263/263 tensors) against the real IQ2_M header (tools/check_gguf_grouping.py, 2026-07-02).
- **Tier-1:** bartowski's precomputed imatrix (verified loadable by load_imatrix, 92 entries);
  SSE reference = target (iq3_xs). Probe set: max_probes 32, probe_fraction/boundary defaults,
  probe-se-stop 0.004, tier-2 time budget 390 min, seed 17 (see Amendment 2).
- **Fresh checkpoints only** — no mounted checkpoint sources (group membership semantics changed
  with the profile-aware mapping; old rows must not be reused).

### Known deviations from the qwen35 CPU run (documented, apply to all arms equally)

1. tier-1 SSE ref = iq3_xs (qwen35 used q8_0) — disk constraint; affects probe-set choice and
   backfill rank only, never the verdict evals.
2. imatrix = bartowski precomputed (qwen35 computed in-kernel on q8_0).
3. group_mode = tensor (finer; qwen35 used layer_family).

## Pre-registered bars

- **GO (ship-eligible):** candidate eval-NLL < stock eval-NLL, AND candidate eval-NLL ≤ random
  control eval-NLL, AND code guardrail passes: candidate code-NLL ≤ stock code-NLL + 0.02 nats.
  - ε = 0.02 ≈ 5× the paired-eval SE observed at 24 chunks (~0.004) — above noise, well below
    typical inter-format gaps (≥0.03).
- **GRAY (either NLL condition half-met, or guardrail fail):** PARK the release. Record which leg
  failed and the predicted cause; one diagnostic re-run is allowed only with a written hypothesis.
- **NO-GO (loses to both stock and random):** KILL the Nemotron v2 release candidate at this scale.
  Method-level conclusions need the qwen35 arm too.

Verdict is computed by the prober itself (result.json `verdict` + `code_guardrail`); no post-hoc
re-scoring. A GO gets an adversarial pass (artifact re-eval, payload accounting check, corpus
hash check) before the trail banks it.

## Secondary arm — qwen35 full-coverage re-run (GRAY-clearing hypothesis)

The 06-13 qwen35 GRAY (mix 2.287 vs target 2.254, random 2.310) ran with 23.4% of payload bytes
excluded from the promotion space (V2_STATUS.md 2026-07-02). Hypothesis: full DeltaNet coverage
(--tensor-profile qwen35, 99.3% of bytes) closes the 0.033 gap to target.

- Same conditions as the 06-13 run (q8_0 ref, in-kernel imatrix, layer_family, corpora recipe,
  seed 17) EXCEPT the profile — one variable. Fresh output dir, no mounted checkpoints.
- **Bar:** GO/GRAY/NO-GO as computed by the prober + code guardrail (same ε). GO here also
  clears the v1-collection Qwen3.5 mix for a v2 refresh. GRAY again = the coverage gap was not
  the (main) cause; bank the negative and stop re-running this model.
- Launch only if a Kaggle CPU slot is free after the Nemotron kernel is confirmed running
  (shared-slot discipline).

## Abort / hygiene

- Push protocol: poll `kernels status` ≤90s after push; only KernelWorkerStatus.* = exists; never
  re-push the same slug (fresh -2/-3 suffix).
- Two consecutive kernel failures on the same arm → stop launching, reproduce the failing stage
  locally at 135M, only then re-push.
- Disk budget pre-checked: 5 quants ≈ 12.3GB + probe/mix ≈ 2.5GB + bins ≪ 19.5GB.

## Run log

- **2026-07-02 run1 (`pmra-nem4b-run1`, ~8h):** prober verdict **GO** — mix 2.49649 vs stock iq3_xs
  2.50262 (mix payload 2434.6MB, 14.5MB under stock), random median-of-3 2.49989, code guardrail
  pass (+0.00537±0.00154 at 48 chunks vs ε=0.02). **Amendment-3 banking bar NOT met:** target-leg
  paired Δ +0.00613±0.00378 = 1.62σ < 2σ ⇒ treated as GRAY, parked, not shipped.
- **Diagnostic re-run (licensed by the GRAY rule, hypothesis written pre-launch):** the target-leg
  delta is real but under-resolved at 24 eval chunks. Prediction: at 96 eval chunks (4× tokens,
  SE ≈ 0.0019) the paired Δ remains ≥ +0.004 and reaches ≥ 2σ. Falsification: Δ shrinks toward 0
  ⇒ the GO was eval noise; bank GRAY and stop. Design: `pmra-nem4b-run2`, STAGES=finalize,
  CHUNKS=96, checkpoints carried from run1 (same pinned llama.cpp b9859, same corpora — cache
  signatures enforce this; code evals at 48 chunks carry over as valid cache). Selection is
  UNCHANGED — only the verdict evals re-run at higher resolution.
- **2026-07-02 cov1 (`pmra-q354b-cov1`) DIED AT THE 12h CAP (~20:10 UTC), output discarded — no
  rows banked.** Post-mortem: the mounted GGUF dataset didn't resolve (workspace snapshot shows HF
  .cache downloads), so it paid full downloads; plus the 06-13-style overhead (in-kernel q8_0
  imatrix, 7 sources, artifact build) on top of tier-2. **cov2 amendments (within-run scoring
  makes these sound; they also align the arm's tier-1 conditions with the Nemotron arm):** drop
  q8_0, ref = target (iq3_xs); imatrix = bartowski's precomputed (verified to exist, 3.6MB);
  MAX_PROBES 24; tier-2 time budget 240 min; BUILD_ARTIFACT=False (verdict-only — the artifact is
  rebuildable from result.json + checkpoints if the arm passes). The original "same conditions as
  06-13" framing is void (it died with the kernel); the arm's question is unchanged: does full
  DeltaNet coverage beat stock IQ3_XS within-run where 76.6%-coverage GRAYed.
- **2026-07-03 run2 (`pmra-nem4b-run2`) ALSO died at the 12h cap (~04:58 UTC), output discarded.**
  96-chunk evals were too greedy: ~10 finalize evals × ~45–50 min + source re-downloads + the
  artifact tail (BUILD_ARTIFACT flag didn't exist when run2 was staged). **No verdict was produced,
  so the single licensed diagnostic re-run remains unconsumed — run3 re-executes the same
  registered hypothesis, it does not add a new look at data.** run3 design: finalize-only,
  CHUNKS=72 (3× run1's eval tokens; expected paired SE ≈ 0.0022 ⇒ hypothesized Δ ≈ +0.006 → ~2.7σ;
  bar unchanged: bank GO iff Δ ≥ 2×SE), BUILD_ARTIFACT=False (run1's artifact is the identical
  selection), checkpoints carried from run1. Wall-clock budget ≈ 7h ≪ 12h cap.
- **2026-07-03 cov2 (`pmra-q354b-cov2`) died at the 12h cap (~10:10 UTC) — but its CHECKPOINTS
  SURVIVED** (allocation_rows 9 probes, tier1_scores, scalar_evals, full kernel log). Forensics
  from the log: **qwen35 evals cost ~25.9 min/probe on Kaggle CPU** (DeltaNet ≈ 2× Nemotron's
  Mamba2); the 240-min tier-2 budget correctly stopped at 9/24 probes at t≈5.3h; the ~7h finalize
  then hit the cap. **Abort-rule note (decide-and-document):** the rule's local-135M reproduction
  cannot reproduce a 4B wall-clock property; the recovered kernel log is the artifact-based
  reproduction (measured cause: eval speed, not a code fault). **cov3 = finalize-only continuation
  carrying cov2's checkpoints** (~8.5h ≪ cap; BUILD_ARTIFACT=False; same pinned b9859/corpora so
  cache signatures validate the carried rows). **Interpretation rule, declared pre-launch:** with
  only 9 empirical probes + proxy backfill the selection is weaker than the 06-13 run's (64
  probes) — a **GO is decisive** (beats stock within-run despite the handicap); a **GRAY is
  AMBIGUOUS** (underpowered selection, NOT a refutation of the coverage hypothesis) → park the
  arm; a properly-powered re-run goes to the hopper.
- **cov3 VOID (declared at launch+15min, before any result):** Kaggle rejected the cancelled cov2
  as a kernel source, so cov3 is running WITHOUT the carried checkpoints — its selection will be
  pure proxy backfill, which is not the registered test. The CLI cannot cancel kernels; cov3's
  eventual verdict is not binding on any bar (at most an unregistered proxy-only-selection
  ablation datapoint). **cov4 is the registered continuation:** cov2's recovered checkpoints
  pushed as private dataset `pmra-q354b-cov2-ckpt` (8KB) and mounted; otherwise identical to
  cov3's design (finalize-only, 24 chunks, BUILD_ARTIFACT=False, asymmetric interpretation as
  declared above).
