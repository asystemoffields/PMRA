# Ministral 3 8B Instruct PMRA Release Candidate

This note records the PMRA artifacts for Ministral 3 8B Instruct.

## Sources

- HF model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- PMRA model key: `ministral3_8b_instruct`
- tensor profile: `mistral3`

The `mistral3` profile covers the Ministral 3 text stack and the GGUF tensor
names used by llama.cpp for this family.

## Artifacts

Primary quality artifact:

```text
ministral3_8b_pmra_knapsack_iq3xs_budget.gguf
```

Compact artifact for tighter local machines:

```text
ministral3_8b_pmra_knapsack_3p2.gguf
```

The primary artifact stays just below the `IQ3_XS` tensor payload budget. The
compact artifact is a 3.2 bpw sweep point that preserves most of the quality
gain while saving about 310 MB of tensor payload versus `IQ3_XS`.

## Selector Recipe

- group mode: `tensor`
- selector: `c2_calib_knapsack_mixed`
- compact selector: `c2_calib_knapsack_bpw_3p200_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K`, `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- calibration: Wikitext-2 raw train, 12 prompts
- selector evaluation: Wikitext-2 raw validation, 128 prompts
- held-out public evaluation: Wikitext-2 raw test, 512 prompts
- prompt audit overlap count: `0`

The larger 48/512 selector shape was attempted but the tensor/source scoring
workload projected past the Modal job window. These artifacts are therefore
scout-selected and then validated on a larger held-out public test split.

## Selector Result

Lower NLL is better.

| Variant | Validation NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `2.546905` | `16.000000` | `16,979,107,840` |
| `IQ2_M` | `4.874847` | `2.920126` | `3,098,820,608` |
| `IQ3_XS` target | `4.649152` | `3.492735` | `3,706,470,400` |
| `Q3_K_S` | `4.686507` | `3.636073` | `3,858,579,456` |
| `Q3_K_M` | `4.651231` | `3.990001` | `4,234,166,272` |
| `IQ4_XS` | `4.611839` | `4.418161` | `4,688,527,360` |
| PMRA knapsack | `4.456880` | `3.492210` | `3,705,913,344` |
| PMRA knapsack 3.2 bpw | `4.510145` | `3.199730` | `3,395,534,848` |
| same-budget random | `4.825388` | `3.492210` | `3,705,913,344` |

Primary key markers:

- PMRA NLL improvement vs `IQ3_XS`: `0.192272`
- PMRA payload bytes vs `IQ3_XS`: `-557,056`
- PMRA NLL improvement vs same-budget random: `0.368508`
- PMRA NLL improvement vs `Q3_K_S`: `0.229628`
- PMRA payload bytes vs `Q3_K_S`: `-152,666,112`

Compact key markers:

- compact PMRA NLL improvement vs `IQ3_XS`: `0.139007`
- compact PMRA payload bytes vs `IQ3_XS`: `-310,935,552`

## Held-Out Wikitext Test

The frozen selector was evaluated on Wikitext-2 raw test with 512 prompts and
256-token truncation.

| Variant | Test NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `2.393904` | `16.000000` | `16,979,107,840` |
| `IQ2_M` | `4.963936` | `2.920126` | `3,098,820,608` |
| `IQ3_XS` target | `4.722369` | `3.492735` | `3,706,470,400` |
| `Q3_K_S` | `4.757542` | `3.636073` | `3,858,579,456` |
| PMRA knapsack | `4.537475` | `3.492210` | `3,705,913,344` |
| PMRA knapsack 3.2 bpw | `4.600533` | `3.199730` | `3,395,534,848` |
| same-budget random | `4.912780` | `3.492210` | `3,705,913,344` |

Held-out primary markers:

- NLL improvement vs `IQ3_XS`: `0.184894`
- payload bytes vs `IQ3_XS`: `-557,056`
- NLL improvement vs `Q3_K_S`: `0.220067`
- payload bytes vs `Q3_K_S`: `-152,666,112`
- NLL improvement vs same-budget random: `0.375305`
- public eval decision: `GO`

Held-out compact markers:

- NLL improvement vs `IQ3_XS`: `0.121836`
- payload bytes vs `IQ3_XS`: `-310,935,552`
- NLL improvement vs `Q3_K_S`: `0.157010`

## Artifact Markers

Primary quality artifact:

- file size: `3,713,801,312` bytes
- tensor payload bytes: `3,705,913,344`
- file bpw: `3.499643`
- payload bpw: `3.492210`
- SHA-256: `7f88294593cf419a5b39b4da2c7df356fee9528de947d6547b9d11d60a84ac5d`
- tensor reload mismatches: `0`
- status: `GO`

Compact artifact:

- file size: `3,403,422,816` bytes
- tensor payload bytes: `3,395,534,848`
- file bpw: `3.207163`
- payload bpw: `3.199730`
- SHA-256: `ff95384e68f211b238767e1783d20ce0b4a8be8a56ac8b906756c481831421a3`
- tensor reload mismatches: `0`
- status: `GO`

## Source Mix

Primary quality artifact:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `147` | `1,370,210,304` |
| `IQ4_XS` | `67` | `1,045,954,560` |
| `Q2_K_L` | `42` | `411,500,544` |
| `Q3_K_M` | `14` | `189,792,256` |
| `Q3_K_S` | `39` | `688,455,680` |

Compact artifact:

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `172` | `1,733,935,104` |
| `IQ4_XS` | `41` | `425,852,928` |
| `Q2_K_L` | `54` | `611,057,664` |
| `Q3_K_M` | `8` | `62,390,272` |
| `Q3_K_S` | `34` | `562,298,880` |

## Local Fit Note

On an 8 GB Windows machine with integrated graphics, prefer the compact 3.2 bpw
GGUF and a small CPU-only context. This workspace machine had about 7.28 GiB RAM
total, 512 MB integrated AMD VRAM, and enough disk space, so storage is fine but
runtime RAM is tight.

## Reproduce

Run the selector shape that produced both artifacts:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys ministral3_8b_instruct `
  --seed 7 `
  --calib-prompts 12 `
  --eval-prompts 128 `
  --calib-max-length 128 `
  --eval-max-length 192 `
  --group-mode tensor `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k,q2_k_l,q3_k_s,q3_k_m,iq4_xs `
  --candidate-variant c2_calib_knapsack_mixed `
  --sweep-payload-bpws 3.2,3.4,3.6 `
  --sweep-selectors calib_knapsack `
  --result-bucket run_011_ministral3_8b_pmra_frontier
```

Selector result name:

```text
ministral3_8b_instruct_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_q2_k_l_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_tensor_len_192_candidate_calib_knapsack_sweep_3p2_3p4_3p6
```

Build the primary artifact:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_artifact `
  --model-key ministral3_8b_instruct `
  --variant c2_calib_knapsack_mixed `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k,q2_k_l,q3_k_s,q3_k_m,iq4_xs `
  --result-bucket run_011_ministral3_8b_pmra_frontier `
  --result-name ministral3_8b_instruct_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_q2_k_l_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_tensor_len_192_candidate_calib_knapsack_sweep_3p2_3p4_3p6 `
  --artifact-bucket run_013_ministral3_8b_artifact `
  --artifact-name ministral3_8b_pmra_knapsack_iq3xs_budget_artifact `
  --output-gguf ministral3_8b_pmra_knapsack_iq3xs_budget.gguf
```

Evaluate the frozen selector on Wikitext-2 raw test:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_eval `
  --model-key ministral3_8b_instruct `
  --seed 7 `
  --eval-prompts 128 `
  --calib-prompts 12 `
  --group-mode tensor `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k,q2_k_l,q3_k_s,q3_k_m,iq4_xs `
  --variants fp16,iq2_m,iq3_xs,q3_k_s,c2_calib_knapsack_mixed,c2_calib_knapsack_bpw_3p200_mixed,c2_random_same_budget `
  --candidate-variant c2_calib_knapsack_mixed `
  --random-variant c2_random_same_budget `
  --prompt-count 512 `
  --eval-max-length 256 `
  --dataset wikitext `
  --dataset-config wikitext-2-raw-v1 `
  --split test `
  --result-bucket run_011_ministral3_8b_pmra_frontier `
  --result-name ministral3_8b_instruct_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_q2_k_l_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_tensor_len_192_candidate_calib_knapsack_sweep_3p2_3p4_3p6 `
  --public-bucket run_013_ministral3_8b_public_eval_wikitext_test
```
