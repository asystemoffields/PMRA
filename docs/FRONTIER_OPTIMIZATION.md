# Frontier Optimization

This note records follow-up PMRA experiments for shrinking artifacts below the
published points while tracking quality loss directly. It is forward-looking:
the current published frontier remains the Gemma 4 E2B-it knapsack artifact.

## Selector Additions

The C2 gate now supports two frontier modes:

- promotion sweeps: start from the low source and select stronger tensor
  payloads under several payload-bpw budgets
- reverse-demotion sweeps: start from a base source and demote selected tensor
  groups to cheaper sources with minimum calibration loss

Both modes write normal `selections` entries in `result.json`. Each mixed
selection also has a `selection_base_sources` entry, so artifact building and
public evaluation know which source fills unselected tensors.

## Qwen3-1.7B Follow-Up Sweep

Run a first single-seed Qwen follow-up sweep:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys qwen3_1p7b `
  --seed 7 `
  --calib-prompts 48 `
  --eval-prompts 512 `
  --calib-max-length 192 `
  --eval-max-length 256 `
  --group-mode tensor `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k_l,q3_k_s,q3_k_m,iq4_xs,q4_k_m `
  --candidate-variant c2_calib_knapsack_mixed `
  --sweep-payload-bpws 3.0,3.2,3.4,3.6,3.8 `
  --sweep-selectors calib_knapsack `
  --demotion-base-source iq3_xs `
  --demotion-sources iq2_m,q2_k_l `
  --demotion-selectors reverse_knapsack `
  --result-bucket run_011_qwen17_pmra_frontier
```

Useful generated variant names:

```text
c2_calib_knapsack_bpw_3p000_mixed
c2_calib_knapsack_bpw_3p200_mixed
c2_calib_knapsack_bpw_3p400_mixed
c2_calib_knapsack_bpw_3p600_mixed
c2_calib_knapsack_bpw_3p800_mixed
c2_reverse_knapsack_bpw_3p000_mixed
c2_reverse_knapsack_bpw_3p200_mixed
c2_reverse_knapsack_bpw_3p400_mixed
c2_reverse_knapsack_bpw_3p600_mixed
c2_reverse_knapsack_bpw_3p800_mixed
```

The result card table reports NLL and payload bytes for every generated
frontier point. Promote the smallest point that clears validation margins to
held-out public evaluation.

## Public Eval Of A Frontier Point

Use the result name printed by Modal, then evaluate selected frontier variants:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_eval `
  --model-key qwen3_1p7b `
  --result-bucket run_011_qwen17_pmra_frontier `
  --result-name <modal-result-name> `
  --variants fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_knapsack_bpw_3p200_mixed,c2_reverse_knapsack_bpw_3p200_mixed `
  --candidate-variant c2_calib_knapsack_bpw_3p200_mixed `
  --random-variant none `
  --dataset wikitext `
  --dataset-config wikitext-2-raw-v1 `
  --split test `
  --prompt-count 512 `
  --eval-max-length 256 `
  --public-bucket run_011_qwen17_pmra_frontier_public_eval
```

Repeat the public eval on TinyStories validation and LAMBADA English test for
the smallest surviving point.

## Artifact Build

Build a GGUF for a selected frontier variant with the existing artifact step:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_artifact `
  --model-key qwen3_1p7b `
  --variant c2_calib_knapsack_bpw_3p200_mixed `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k_l,q3_k_s,q3_k_m,iq4_xs,q4_k_m `
  --result-bucket run_011_qwen17_pmra_frontier `
  --result-name <modal-result-name> `
  --artifact-bucket run_011_qwen17_pmra_frontier_artifact
```
