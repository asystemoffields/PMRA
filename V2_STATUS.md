# PMRA v2 — status (as of 2026-07-02)

## Update 2026-07-02 (autonomous run)

- **Merge to main: already done.** `pmra-objective-upgrade` is an ancestor of `main`; main carries
  everything plus the Jun-12/13 Hessian/factorial/redist work. The "merge to main" item below is stale.
- **Redist kernel collected** (sat COMPLETE on Kaggle since 06-13) → `results-live/redist/` in the
  qwen35 run area. Sign convention: dnll = base − probe, positive = better (matches RECOVERY.md).
  Base iq3_xs NLL 2.579. Verdicts:
  - conservative redistribution is ≈free byte savings: `demote_only` (L8/L16 mlp→iq2_m)
    +0.0030±0.0037 at −11.1MB; `redist_cons` +0.0042±0.0035 at −2.8MB;
  - aggressive redistribution weakly hurts: `redist_aggr` −0.0050±0.0042 at −2.8MB;
  - attn-band→q3_k_s from iq2_m base: early helps (+0.0057±0.0027), mid/late hurt
    (−0.0092±0.0030, −0.0094±0.0013) with mid at negative net bytes — stock iq2_m likely already
    bumps those attn tensors higher, so the "promotion" was a demotion there;
  - `attn_plus_knap` +0.0818±0.0087 for +224MB — the MLP-promotion knapsack carries the real ΔNLL.
- **Coverage-gap finding (major): the qwen35 CPU-prober GRAY verdict ran with 23.4% of payload
  bytes excluded from the promotion space.** `cpu_prober.group_for_tensor` only knew llama-family
  tails; all 253 DeltaNet tensors (attn_qkv, attn_gate, ssm_*, post_attention_norm) fell out —
  verified against the real bartowski IQ2_M header (old mapping 76.6% of bytes; new `qwen35`
  profile 99.3%, only the 4 nextn.* MTP tensors stay at base, matching the gate spec). The GRAY
  (mix 2.287 vs target 2.254) had the mix locked to IQ2_M on ~1/4 of the model — a full-coverage
  re-run is a live candidate to clear it.
- **Prober is now profile-aware:** `--tensor-profile qwen35|nemotron_h` maps hybrid tails
  (mirrors `build_tensor_specs`; `tests/test_grouping.py`). nemotron_h verified 100% coverage
  (263/263 tensors) against the real bartowski Nemotron-3-Nano-4B IQ2_M header via
  `tools/check_gguf_grouping.py` (new).
- **A/B code guardrail: DONE on the ship path.** `cpu_prober.py --code-text CODE.TXT
  --code-no-regress EPS`: measures code-domain NLL of low/target/mix at finalize, emits a
  `code_guardrail` block in result.json, downgrades GO→GRAY when the mix's code NLL exceeds the
  stock target's by >EPS. Corpus = MBPP-sanitized + HumanEval via `tools/build_code_corpus.py`
  (421 tasks, deterministic, ungated — same benchmarks as evaluate_pmra_code_likelihood.py).
  Tests: `tests/test_code_guardrail.py`. The torch-path flag stays record-only (warns loudly).
- **Anneal/local-refinement objective wiring: explicitly deferred.** Torch-path search nicety;
  the ship path never calls it; not worth the risk without a GPU bench this run.
- **Remaining to ship:** Nemotron-3-Nano-4B CPU-prober validation on Kaggle (single kernel,
  HF-download fallback — the Kaggle GGUF dataset was never built and isn't needed; bartowski's
  imatrix is reused, verified loadable), then the v2 release. Prereg in docs/PREREG_NEMOTRON.md.

# (previous status, 2026-06-10)

v2 is an **additive** upgrade to `scripts/production_mixed_rate_transcoder_gate.py`. Every new flag defaults to v1 behavior; a default run reproduces v1 **byte-for-byte** (proven: smoke parity Δ = 0.00e+00). v1 is pinned at tag `pmra-v1`. Branch: `pmra-objective-upgrade`.

## Done + integration-validated
- **KL-to-fp16 objective** — `--objective kl_fp16` (`--kl-reference {last_token|full_position}`, `--kl-top-k`). Minimizes `KL(fp16 || mix)`. New: `capture_reference_logits`, `_kl_ref_to_cur`, `evaluate_nll_kl`; row keys `kl_to_fp16`/`kl_improvement`/`kl_score_per_mbyte`; variants `c2_kl_{knapsack,greedy}_mixed`.
- **Held-out selection** — `--heldout-prompts N` (`--heldout-split`, `--select-on {calib|heldout}`). Disjoint calib/heldout/eval; `heldout_*_improvement` rows; selection routes to the held-out fold ("compression that generalizes").
- **Validated** via Kaggle CPU kernel `pmra-v2-smoke` (Qwen2.5-0.5B): default==v1 exact, KL + held-out + KL∧held-out all emit correct rows/variants.

## Landed since 2026-06-01 (this file was stale)
- **Triage** — commit #8 wired `--triage` (weight-SSE pre-rank, budget-boundary band, proxy-zero tail) into the probe loop, with unit tests (`tests/test_triage.py`, 5 passing). Fisher pre-rank falls back to weight_sse with a warning.
- **A/B objective comparison** — commit #9 (`calib_nll` vs `kl_fp16`).

## NEW (2026-06-10): CPU prober — GPU-free probing on the llama.cpp path
- `scripts/cpu_prober.py`: two-tier probing with **no GPU and no torch** —
  tier 1 = imatrix-weighted SSE proxy for every candidate (one llama-imatrix
  run + numpy dequant), tier 2 = real llama-perplexity probes on spliced
  GGUFs for the knapsack boundary band only. Emits gate-compatible
  `allocation_rows.jsonl` + `result.json` (artifact builder runs unchanged).
  Probes the actual inference path, sidestepping the torch-side eval drift
  suspected in the qwen35-b2 GRAY verdict (all quants NLL ~13.5 vs fp16 2.8).
- Sharding: `--stages tier2 --shard k/N` + `scripts/merge_allocation_rows.py`
  (same (group,source) dedup as the b1/b2 layer-split merge).
- Harnesses: `.github/workflows/cpu-prober.yml` (free GHA matrix, ≤1B models)
  and `notebooks/pmra_cpu_prober_kaggle.py` (Kaggle CPU kernels — no GPU
  quota — for 4B+). Docs: `docs/CPU_PROBER.md`.
- Validated end-to-end on SmolLM2-135M (bartowski spread, Q2_K→Q4_K_M budget)
  locally on a 12-core CPU box.

## Remaining
- **A/B code guardrail** — eval final variants vs stock on an ungated code corpus; `--code-no-regress`. Wire anneal/local refinement to the active objective.
- **Then**: merge to `main` → one CPU-prober validation on a real 4B (Kaggle CPU kernels) → **Nemotron-3-Nano-4B v2 release** (llama.cpp implements the hybrid natively, so the mamba-ssm compile risk disappears on the CPU path).

## Releases
- **v1 mixes** (calib-NLL objective) → HF **"PMRA v1" collection** (incl. the in-flight Qwen3.5-4B mix once it lands).
- **v2 mixes** (KL / held-out objective) → future PMRA v2 collection. Ship only what *clearly* beats stock per-domain.
