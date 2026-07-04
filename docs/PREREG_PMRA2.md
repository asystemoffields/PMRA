# PREREG — PMRA2 definition probes, round 1 (Kernel A: qwen35 verdict · Kernel B: KLD instrument)

Registered 2026-07-03, pre-launch. Context: Alex's directive — *PMRA v2 = whatever method
combination demonstrably beats modern stock quants; the winning combo becomes the v2 release
definition.* Designed by a 3-lens generator roundtable + adversarial review (reports archived in
the session scratchpad; verdict summary below the bars). Successor to PREREG_NEMOTRON.md, whose
run log (incl. the 07-03 corrections: backfill phantom-spend bug, qwen35 noise floor 3× nemotron)
is incorporated by reference.

## Standing definition change (from the adversarial review — binds all PMRA claims from now on)

**"Stock" = the best published GGUF of the same base model from any major publisher**
(bartowski, mradermacher, unsloth), not bartowski alone. Census for Qwen3.5-4B verified
2026-07-03 via HF API (file sizes): mradermacher i1 ladder i1-IQ1_S 1.359GB … i1-IQ3_XXS 1.904 /
i1-Q2_K 1.915 / i1-Q3_K_S 2.070 / **i1-IQ3_XS 2.078** / i1-IQ3_S 2.140 / i1-IQ3_M 2.163 /
i1-Q3_K_M 2.262GB; unsloth UD-Q2_K_XL 1.941 / UD-IQ3_XXS 1.949 / Q3_K_S 2.106GB; bartowski
IQ2_M 1.954 / Q2_K 2.19 / IQ3_XXS 2.24 / IQ3_XS 2.46GB. All budget claims are stated in
**file GB**, not payload MB.

## Kernel A — `pmra2-ka-1` (qwen35 merged verdict kernel, one CPU slot, ~9–11h)

Fixed-prober selection (dedup fix 2026-07-03 + no-duplicates assertion) recomputed in-kernel from
the cov2 checkpoint dataset (9 empirical probes + tier1 SSE) with **Hessian-ranked backfill** for
`:mlp` groups (hessian_scores.json, Spearman 0.519 vs June's 64 empirical probes, vs SSE's 0.154;
SSE retained for attn/DeltaNet groups the scorer doesn't cover). Two mixes assembled from
bartowski sources (same provenance as all prior arms):

- **MIX-FULL** at the full iq3_xs byte budget (payload 2379.0MB ≈ 2.40GB file).
- **MIX-GAP** at gap budget: target file ≈ 2.03GB (payload ≈ 2005MB) — undercuts i1-Q3_K_S/
  i1-IQ3_XS (2.070/2.078GB), sits above the 1.90–1.95GB shelf.
- **NAIVE-BLEND control** at the same gap bytes: layers promoted iq2_m→iq3_xs in plain ascending
  layer order until the budget fills — the "any interpolation" strawman, materialized honestly.

Eval plan (llama.cpp b9859 pinned, wikitext eval slice, ctx 512, paired per-chunk):
- 24-chunk screening: MIX-FULL smoke (early-abort if Δ vs stock < −0.024), i1-IQ3_XXS (1.904GB),
  UD-IQ3_XXS (1.949GB), i1-IQ3_XS (2.078GB, context).
- 72-chunk verdict legs (SE expected ≈0.0069): stock bartowski IQ3_XS, MIX-FULL, MIX-GAP,
  BEST-BELOW (whichever of i1-IQ3_XXS / UD-IQ3_XXS screens better at 24ch), NAIVE-BLEND.
- Per-stage checkpointing (partial results json after every eval) + wall-clock guard (drop
  NAIVE-BLEND to 48ch if projection exceeds 11h; never start an eval past t=10.2h).

### Bars (Kernel A)

- **Q1 — allocation at equal bytes (mid-band):** GO iff MIX-FULL beats stock IQ3_XS by paired
  Δ ≥ 2×SE at 72ch. KILL for the qwen35 mid-band iff Δ < 2×SE (matches the banked Nemotron
  result → "allocation alone cannot beat tuned mid-ladder rungs" generalizes; the mid-band closes
  on both models).
- **Q2 — gap product (honest competitor set):** product-GO iff MIX-GAP beats BEST-BELOW by
  Δ ≥ 2×SE at 72ch AND MIX-GAP ≥ (i1-Q3_K_S proxy: interpolation between BEST-BELOW and stock at
  the byte ratio) — recorded, not gating. **Method-attribution clause:** the card may credit PMRA
  allocation only if MIX-GAP beats NAIVE-BLEND by ≥ 2×SE; if the blend ties or wins, the honest
  claim is "a blend product" (goes to hopper as a trivially-cheap service; PMRA-the-method gets no
  credit).
- **Q2 kill:** MIX-GAP < 2×SE over BEST-BELOW → the 2.0GB gap is not exploitable vs the real
  publisher set; kill the gap-product line for this model.
- Code guardrail (48ch) runs for any mix that passes its primary bar, time permitting; a ship
  card requires it (regression ≤ 0.02 vs stock IQ3_XS as before).

## Kernel B — `pmra2-kb-1` (KLD instrument calibration, parallel slot, ~3.5–5h)

Mounts `pmra-nem4b-run1` as a kernel source (COMPLETE — contains the built mix artifact).
Sequence (disk-guarded): download Q8_0 (4.2GB) → `llama-perplexity --kl-divergence-base` on the
wikitext eval slice at 72ch (logits ≈9.7GB) → delete Q8_0 → score run3's mix artifact and stock
IQ3_XS with `--kl-divergence` (parse per-chunk KLD + top-1 agreement; raw logs kept in output).

### Bars (Kernel B) — two-sided, instrument-first

- **Primary (instrument):** measured paired per-chunk KLD SE vs the known NLL SE (0.0023 at 72ch
  on this model). GO iff KLD SE ≤ 0.5× NLL SE → KLD becomes a standard verdict metric on future
  arms (parser lands either way). KILL the "sharper instrument" premise otherwise.
- **Secondary (verdict, symmetric):** mix vs stock paired KLD at equal bytes — registered in BOTH
  directions; a KLD-worse result banks as confirmation of the NLL GRAY. No ship claim can come
  from Kernel B alone (the Nemotron artifact stays parked regardless; "closer-to-base at equal
  bytes and equal NLL" has no user hook).

## Explicitly deferred (with reasons, from the review)

- Low-bit equal-bytes arm: competitor set corrected (mradermacher/unsloth publish 8 rungs below
  1.95GB; "only sub-2GB rung" was false); runs only after Kernel A's verdict, competitors
  downloaded not quantized.
- Day-1 header-diff: redesigned to same-publisher revision diffs + slope-converted nats upper
  bound; hopper, zero kernel-hours; a service cadence, not the v2 definition.
- Derivative-coverage census: zero-cost, queued; ship claims wait on the method verdict.
- Hessian scorer → DeltaNet tensor extension: ~2h GPU, queued for the 07-04 quota refresh; gates
  nothing in this round.
- Domain-gap detector + ctx-damage curve (objective lens P2/P3): live candidates for round 2,
  after the KLD instrument verdict picks the metric they'd be judged on.
- Pair-probing for promotion interactions: KILLED pre-launch (interaction terms sit below the
  probe noise floor); offline factorial refit only.

## Abort / hygiene

As PREREG_NEMOTRON: push→poll ≤90s, fresh slugs, per-stage checkpoints, 12h-cap-aware wall-clock
guards, cache signatures (both kernels pin b9859 and record corpus shas). Kernel A carries only
sig-valid cov2 rows; Kernel B touches no selection at all.

## Run log

- **2026-07-04 Kernel B (`pmra2-kb-1`, ~7.5h, COMPLETE): INSTRUMENT GO; verdict leg confirms the
  NLL GRAY from the sharper side.** Mean KLD to Q8_0 at 72 chunks: stock IQ3_XS **0.033417 ±
  0.0003**, parked mix **0.041280 ± 0.000417** — the mix is decisively KLD-worse (+0.0079, ~20σ);
  top-1 agreement mix 89.341 ± 0.228%. KLD SE ≈ 6–7× sharper than the 0.0023 NLL SE (bar was
  ≤0.5× → GO): **KLD becomes the standard verdict metric on future arms.** Same-slice NLL legs
  reproduced run3 across kernels (stock 2.59745, mix 2.60093; Δ −0.00348 vs run3's −0.00349).
  Consequence for the Nemotron artifact: parked-for-good — stock IQ3_XS is closer to the base
  model at equal bytes; the equivalence/TOST framing dies too (equivalent on NLL, but measurably
  less faithful on KLD — an honest card cannot call that parity).
- **2026-07-04 Kernel A (`pmra2-ka-1`, ~11.7h, COMPLETE): Q1 KILL — the qwen35 mid-band closes;
  Q2 killed on evidence in hand (documented deviation).** Q1: fixed-selection full-spend mix
  (447.3/450.2MB, hessian backfill) vs stock IQ3_XS at 72ch: Δ **−0.00261 ± 0.00515 (−0.51σ)** —
  statistically indistinguishable; with Nemotron's banked result, "allocation alone cannot beat
  tuned mid-ladder rungs" now holds on both models at full coverage AND full spend. Postscript
  confirms the bug post-mortem: cov4's −0.034 → −0.003 after the fix (predicted recovery range
  [−0.002,+0.019]). Q2: the wall-clock guard dropped the 72ch gap legs, but the 24ch screens
  decide it — best-below UD-IQ3_XXS 2.28416 / i1-IQ3_XXS 2.29617 vs bartowski iq2_m 2.38722 at
  ≈equal file size: the below-shelf publishers' quants are ~0.08–0.10 nats better than bartowski's
  own rungs at those bytes, so a bartowski-source interpolation at 2.03GB (predicted ≈2.36) loses
  by ~10–15σ. Confirming that with 5.5h of compute serves no decision → **Q2 KILL by measured
  forecast** (deviation from the 72ch bar, logged here; the naive-blend attribution question is
  moot with the product dead). Root finding: modern imatrix/dynamic ladders (unsloth UD,
  mradermacher i1) are better SOURCES, not just tuned ladders — allocation over inferior sources
  cannot reach them.

## Round 2 (registered 2026-07-04, pre-launch): Kernel C — domain-damage detector (`pmra2-kc-1`)

The one deferred line whose premise survives round 1: stock ladders are tuned/validated on
general text; if quantization damage concentrates in some domain, per-domain allocation could
beat stock THERE. Round 1's KLD instrument (6–7× sharper) makes the detector cheap and decisive.
Design (Nemotron — 2× faster evals than qwen35; one CPU kernel ~6h): 4 corpora (wikitext eval
slice = baseline; code = existing interleaved MBPP+HumanEval sha ae93382e; math = GSM8K test
slice; zh = zh-wikipedia slice; each 256KB-recipe, deterministic); per domain: Q8_0 reference
logits at 36ch then KLD of stock IQ3_XS vs that reference. Damage(domain) = mean KLD(stock‖Q8) on
that domain.
- **GO (domain-conditional line lives):** some domain shows damage ≥ 1.5× wiki-damage with
  non-overlapping ±2SE. Next kernel then measures rank-divergence + a selection arm on the argmax
  domain.
- **KILL (line dies, one kernel):** all domains within 1.3× of wiki damage → quant damage is
  domain-flat at this band; no per-domain allocation headroom. Bank and stop.
- Hygiene: per-stage json checkpointing, wall-clock guard, working-dir cleanup before exit (the
  kb-1 clutter lesson), pinned b9859.
- **2026-07-04 Kernel C (`pmra2-kc-1`, ~3.7h, COMPLETE): KILL — the domain-conditional line dies.**
  Mean KLD(stock IQ3_XS ‖ Q8_0) at 36ch: wiki **0.035243±0.000489**, code **0.014589±0.000461**
  (0.414× wiki), math **0.013701±0.000327** (0.389×), zh **0.037944±0.000509** (1.077×). All
  domains ≤1.3× wiki (kill bar); code/math show LESS damage than general text. Quant damage is
  domain-flat-or-better at this band → no per-domain allocation headroom. Banked.

## Round 2b (registered 2026-07-04, pre-launch): Kernel D — context-length damage curve (`pmra2-kd-1`)

The last distinctive allocation-side mechanism from the roundtable (objective lens P3): hybrid
`ssm_*` state-update quantization noise may compound through the recurrent state across long
contexts — invisible at ctx 512, where all prior evals ran; stock ftype rules were tuned at short
ctx on transformers. Design (Nemotron, ~3h, evals only): same wikitext eval text; per ctx setting,
Q8_0 reference logits then stock IQ3_XS KLD — ctx 512×36ch / 2048×9ch / 4096×4ch (≈equal tokens).
- **GO:** damage(4096) ≥ 1.5× damage(512) → design the long-ctx selection arm (probe/select at
  long ctx; the `ssm_*` groups are already 100% covered by the profile-aware prober).
- **KILL:** damage(4096) ≤ 1.2× damage(512) → no context-concentrated damage at this band; the
  final allocation-side candidate closes. Between = park with a scale note.
