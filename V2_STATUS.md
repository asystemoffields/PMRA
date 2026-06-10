# PMRA v2 — status (as of 2026-06-10)

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
