# PMRA v2 — status (as of 2026-06-01)

v2 is an **additive** upgrade to `scripts/production_mixed_rate_transcoder_gate.py`. Every new flag defaults to v1 behavior; a default run reproduces v1 **byte-for-byte** (proven: smoke parity Δ = 0.00e+00). v1 is pinned at tag `pmra-v1`. Branch: `pmra-objective-upgrade`.

## Done + integration-validated
- **KL-to-fp16 objective** — `--objective kl_fp16` (`--kl-reference {last_token|full_position}`, `--kl-top-k`). Minimizes `KL(fp16 || mix)`. New: `capture_reference_logits`, `_kl_ref_to_cur`, `evaluate_nll_kl`; row keys `kl_to_fp16`/`kl_improvement`/`kl_score_per_mbyte`; variants `c2_kl_{knapsack,greedy}_mixed`.
- **Held-out selection** — `--heldout-prompts N` (`--heldout-split`, `--select-on {calib|heldout}`). Disjoint calib/heldout/eval; `heldout_*_improvement` rows; selection routes to the held-out fold ("compression that generalizes").
- **Validated** via Kaggle CPU kernel `pmra-v2-smoke` (Qwen2.5-0.5B): default==v1 exact, KL + held-out + KL∧held-out all emit correct rows/variants.

## Remaining (flags exist in argparse; probe-loop/eval logic still TODO)
- **Triage** — `--triage` + `--triage-pre-rank {weight_sse|fisher}` + `--triage-probe-fraction` + `--triage-boundary-band`: cheap pre-rank → probe only top fraction + knapsack boundary → zero/skip the tail. Cuts probe forwards 3–5×.
- **A/B + code guardrail** — `--ab-objectives calib_nll,kl_fp16` + `--ab-decide-on`: build both selections, eval on held-out+eval, emit `result["ab_comparison"].winner`. Code domain = release **guardrail** (prose-primary objective; eval final variants vs stock on an ungated code corpus; `--code-no-regress`). Wire anneal/local refinement to the active objective. Selective upcast = pass Q5_K/Q6_K via `--source` (artifact builder already per-group source-flexible).
- **Then**: extend smoke for triage/A&B → merge to `main` → one Kaggle GPU validation on a real 4B → **Nemotron-3-Nano-4B v2 release** (full recipe; `mamba-ssm` compile is the open risk).

## Releases
- **v1 mixes** (calib-NLL objective) → HF **"PMRA v1" collection** (incl. the in-flight Qwen3.5-4B mix once it lands).
- **v2 mixes** (KL / held-out objective) → future PMRA v2 collection. Ship only what *clearly* beats stock per-domain.
