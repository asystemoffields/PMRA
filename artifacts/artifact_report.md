# Result Card - C2 Mixed GGUF Artifact

## Status

GO

## Artifact

- file: `<pmra-output-root>/run_008_c2_publiccal_qwen17_artifact/qwen3_1p7b_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_l_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_512_calib_48_tensor_len_256_calib_greedy_artifact/qwen3_1p7b_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_l_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_512_calib_48_tensor_len_256_calib_greedy.gguf`
- file size: `961694976` bytes
- payload bytes: `955742208`
- metadata + alignment overhead: `5952768` bytes
- payload bpw: `3.763246`
- file bpw: `3.786685`
- PMRA metadata fields: `11`

## C2 Quality Marker

- candidate NLL: `3.279698`
- target `iq3_xs` NLL: `3.435756`
- NLL improvement vs target: `0.156058`
- candidate payload bytes vs target: `-6232064`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| iq2_m | 158 | 431355904 |
| iq4_xs | 62 | 236191744 |
| q2_k_l | 60 | 174391296 |
| q3_k_m | 15 | 50724864 |
| q3_k_s | 16 | 63078400 |

## Load Check

- tensor count: `311`
- kv count: `42`
- mismatched tensors: `0`

## Decision

GO: selected production-format tensor payloads were materialized into one loadable mixed GGUF artifact.
