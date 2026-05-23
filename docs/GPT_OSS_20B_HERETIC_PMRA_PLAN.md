# GPT-OSS 20B Heretic PMRA Plan

This is the planning card for mixing
[p-e-w/gpt-oss-20b-heretic](https://huggingface.co/p-e-w/gpt-oss-20b-heretic)
with PMRA. The GGUF controls currently target the
[mradermacher GGUF set](https://huggingface.co/mradermacher/gpt-oss-20b-heretic-GGUF).

The release docs and model card must credit P-E-W and the Heretic project as the
upstream model creators. PMRA should be described as a mixed GGUF derived from
their Heretic checkpoint, not as a replacement for it.

## Profile

The model uses the `gpt_oss` architecture with 24 transformer layers and MoE
expert tensors. The HF checkpoint stores expert gate and up projections in a
combined `gate_up_proj` tensor, while GGUF stores them as separate
`ffn_gate_exps.*` and `ffn_up_exps.*` tensors. The PMRA tensor profile therefore
uses slice-aware specs so the full GGUF surface is selectable.

Local header validation against `gpt-oss-20b-heretic.Q3_K_S.gguf` matched:

- GGUF tensor count: 459
- PMRA spec count: 459
- missing GGUF specs: 0
- extra GGUF tensors outside the profile: 0
- split gate/up specs: 96

This is a heavy run. Selector jobs for this model route through the 80 GB A100
Modal path.

## First Selector Bakeoff

The first pass should compare the standard PMRA selectors with interaction-aware
search on the same public calibration/eval split:

- calibration greedy
- calibration knapsack
- calibration and weight-rank blend
- payload-bpw sweeps near the Q2/Q3/MXFP4 size cluster
- seeded genetic search from the scored selector population
- seeded simulated annealing from knapsack
- same-budget random control
- uniform GGUF controls

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys gpt_oss_20b_heretic `
  --seed 7 `
  --calib-prompts 40 `
  --eval-prompts 192 `
  --calib-max-length 128 `
  --eval-max-length 192 `
  --group-mode tensor `
  --low-source q2_k `
  --target-source q3_k_s `
  --high-sources mxfp4_moe,iq4_xs,q3_k_m,q3_k_l,q4_k_s,q4_k_m `
  --candidate-variant c2_calib_knapsack_anneal_mixed `
  --sweep-payload-bpws 4.62,4.80,5.00,5.20,5.45,5.70 `
  --sweep-selectors calib_knapsack,calib_greedy,blend `
  --genetic-search-from c2_calib_knapsack_mixed `
  --genetic-search-generations 8 `
  --genetic-search-population 12 `
  --genetic-search-elite 3 `
  --genetic-search-mutation-rate 0.18 `
  --genetic-search-validation-prompts 8 `
  --genetic-search-rerank-top-k 6 `
  --anneal-search-from c2_calib_knapsack_mixed `
  --anneal-search-steps 60 `
  --anneal-search-mutation-rate 0.12 `
  --anneal-search-initial-temp 0.03 `
  --anneal-search-final-temp 0.001 `
  --anneal-search-validation-prompts 8 `
  --anneal-search-rerank-top-k 8 `
  --result-bucket run_025_gpt_oss_20b_heretic_selector_bakeoff
```

Useful generated variants:

```text
c2_calib_greedy_mixed
c2_calib_knapsack_mixed
c2_weight_mse_mixed
c2_calib_weight_blend_mixed
c2_calib_knapsack_genetic_mixed
c2_calib_knapsack_anneal_mixed
c2_random_same_budget
```

The seeded GA starts from every scored selector candidate that fits the same
budget, not only the literal knapsack candidate. That is the "selected GA"
condition: start from known good selected mixes, then cross over and mutate.

## Direct Search Control

Run a direct GA plus direct annealing pass as a compute-saving control. This
skips per-tensor calibration scoring and evaluates whole promoted subsets
directly.

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys gpt_oss_20b_heretic `
  --seed 7 `
  --calib-prompts 40 `
  --eval-prompts 192 `
  --calib-max-length 128 `
  --eval-max-length 192 `
  --group-mode tensor `
  --low-source q2_k `
  --target-source q3_k_s `
  --high-sources mxfp4_moe,iq4_xs,q3_k_m,q3_k_l,q4_k_s,q4_k_m `
  --candidate-variant c2_direct_anneal_mixed `
  --genetic-search-direct `
  --genetic-search-generations 10 `
  --genetic-search-population 16 `
  --genetic-search-elite 4 `
  --genetic-search-mutation-rate 0.22 `
  --genetic-search-validation-prompts 8 `
  --genetic-search-rerank-top-k 8 `
  --anneal-search-direct `
  --anneal-search-steps 80 `
  --anneal-search-mutation-rate 0.16 `
  --anneal-search-initial-temp 0.035 `
  --anneal-search-final-temp 0.001 `
  --anneal-search-validation-prompts 8 `
  --anneal-search-rerank-top-k 8 `
  --result-bucket run_026_gpt_oss_20b_heretic_direct_search
```

Useful generated variants:

```text
c2_direct_genetic_mixed
c2_direct_anneal_mixed
c2_random_same_budget
```

## Champion Criteria

A publishable champion should clear all of these checks:

- better held-out NLL than the uniform target/control at comparable payload size
- better than same-budget random
- no validation-only win that disappears on held-out public eval
- loadable mixed GGUF with `pmra.*` metadata and an artifact report
- release docs credit P-E-W, Heretic, and the GGUF quant source

If the direct GA or direct annealing run wins, keep the story explicit: it found
a better interaction-aware subset without spending compute on every individual
promotion score. If the seeded run wins, keep knapsack as the strong prior and
describe GA/annealing as search refinements around it.
