# Result Card - C2 Production Mixed-Rate Transcoder Gate

## Status

GO

## Decisive Measurement

The Gemma 4 E2B-it model was forward-evaluated after patching real production GGUF tensor payloads. The candidate starts from the low source and promotes selected tensor groups to stronger source formats under the target tensor-payload byte budget.

## Variants

| Variant | NLL | Delta vs FP16 | Payload bpw | Payload bytes | Last-logit MSE | Top-10 overlap |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 14.381222 | 0.000000 | 16.000000 | 9294899782 | n/a | n/a |
| q2_k | 20.376913 | 5.995691 | 5.118105 | 2973267084 | 63.9394 | 0.030 |
| q3_k_s | 17.993582 | 3.612360 | 5.326613 | 3094396044 | 42.6646 | 0.054 |
| q3_k_m | 15.619944 | 1.238721 | 5.483489 | 3185529996 | 38.301 | 0.098 |
| q3_k_l | 15.756687 | 1.375465 | 5.622925 | 3266532492 | 24.91 | 0.110 |
| iq4_xs | 16.043206 | 1.661984 | 5.670221 | 3294008460 | 14.1852 | 0.214 |
| q4_k_m | 13.549753 | -0.831470 | 5.873431 | 3412059276 | 16.6347 | 0.249 |
| c2_calib_greedy_mixed | 13.281400 | -1.099822 | 5.326291 | 3094208652 | 52.7316 | 0.126 |
| c2_weight_mse_mixed | 19.224024 | 4.842802 | 5.326608 | 3094392972 | 43.3391 | 0.053 |
| c2_calib_weight_blend_mixed | 13.567505 | -0.813717 | 5.326550 | 3094359180 | 39.3155 | 0.119 |
| c2_random_same_budget | 18.629201 | 4.247979 | 5.326291 | 3094208652 | 41.9204 | 0.042 |

## Selector Summary

- low source: `q2_k`
- target source: `q3_k_s`
- high sources: `q3_k_m, q3_k_l, iq4_xs, q4_k_m`
- group mode: `tensor`
- calibration groups tested: `1209`
- selected groups: `216`
- selected extra bytes: `120941568`

## GO / NO-GO

GO: mixed production-format allocation beats the target uniform production baseline at or below its tensor payload bytes.

## Next Step

Escalate to three seeds and page/block-level artifact accounting.
