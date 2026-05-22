# Result Card - C2 Mixed GGUF Artifact

## Status

GO

## Artifact

- file: `<pmra-output-root>/run_009_gemma4_pmra_gguf/gemma4_e2b_it_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_q2_k_target_q3_k_s_high_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_256_calib_24_tensor_len_256_calib_greedy_artifact/gemma4_e2b_it_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_q2_k_target_q3_k_s_high_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_256_calib_24_tensor_len_256_calib_greedy.gguf`
- file size: `3110028576` bytes
- payload bytes: `3094209676`
- metadata + alignment overhead: `15818900` bytes
- payload bpw: `5.326292`
- file bpw: `5.353523`
- PMRA metadata fields: `11`

## C2 Quality Marker

- candidate NLL: `13.281400`
- target `q3_k_s` NLL: `17.993582`
- NLL improvement vs target: `4.712182`
- candidate payload bytes vs target: `-186368`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| iq4_xs | 36 | 118861824 |
| q2_k | 385 | 2590785676 |
| q3_k_l | 22 | 11354112 |
| q3_k_m | 146 | 359955456 |
| q4_k_m | 12 | 13252608 |

## Load Check

- tensor count: `601`
- kv count: `63`
- mismatched tensors: `0`

## Decision

GO: selected production-format tensor payloads were materialized into one loadable mixed GGUF artifact.
