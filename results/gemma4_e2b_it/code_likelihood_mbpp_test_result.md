# Result Card - PMRA Code Likelihood Benchmark

## Status

GO

## Benchmark

- benchmark: `mbpp_sanitized`
- dataset: `google-research-datasets/mbpp`
- config: `sanitized`
- split: `test`
- tasks: `257`
- task hash: `f4d75a5f3d5ff4c2942fed671b14e93e2b80d1e0dbeee66bb9d5b04d7c47f591`

## Variants

| Variant | Code NLL | Code PPL | Solution tokens | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|---:|
| fp16 | 12.573706 | 288862.406481 | 15821 | 16.000000 | 9294899782 |
| q2_k | 17.129356 | 27490626.514121 | 15821 | 5.118105 | 2973267084 |
| q3_k_s | 15.381973 | 4789672.142053 | 15821 | 5.326613 | 3094396044 |
| q4_k_m | 12.129562 | 185268.535122 | 15821 | 5.873431 | 3412059276 |
| c2_calib_greedy_mixed | 11.762667 | 128369.316163 | 15821 | 5.326291 | 3094208652 |
| c2_random_same_budget | 17.828374 | 55305019.182074 | 15821 | 5.326291 | 3094208652 |

## Key Comparisons

- candidate NLL delta vs target: `-3.619306`
- candidate payload bytes vs target: `-187392`
- candidate NLL delta vs same-budget random: `-6.065708`

## Decision

Candidate preserves or improves canonical-solution code likelihood at the target tensor-payload budget.

## Next Step

Replicate on another seed or code corpus before method-level generalization.
