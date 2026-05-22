# Result Card - C2 Mixed GGUF Artifact

## Status

GO

## Artifact

- file: `<pmra-output-root>/run_010_gemma4_knapsack_artifact/gemma4_e2b_it_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_q2_k_target_q3_k_s_high_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_256_calib_24_tensor_len_256_candidate_calib_knapsack_calib_knapsack_artifact/gemma4_e2b_it_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_q2_k_target_q3_k_s_high_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_256_calib_24_tensor_len_256_candidate_calib_knapsack_calib_knapsack.gguf`
- file size: `3110215968` bytes
- payload bytes: `3094397068`
- metadata + alignment overhead: `15818900` bytes
- payload bpw: `5.326615`
- file bpw: `5.353845`
- PMRA metadata fields: `11`

## C2 Quality Marker

- candidate NLL: `12.878809`
- target `q3_k_s` NLL: `17.993582`
- NLL improvement vs target: `5.114774`
- candidate payload bytes vs target: `1024`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| iq4_xs | 40 | 83140608 |
| q2_k | 397 | 2637615244 |
| q3_k_l | 24 | 21356544 |
| q3_k_m | 84 | 233001984 |
| q4_k_m | 56 | 119282688 |

## Load Check

- tensor count: `601`
- kv count: `63`
- mismatched tensors: `0`

## Decision

GO: selected production-format tensor payloads were materialized into one loadable mixed GGUF artifact.
