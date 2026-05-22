# Result Card - PMRA Code Likelihood Benchmark

## Status

GO

## Benchmark

- benchmark: `humaneval`
- dataset: `openai/openai_humaneval`
- config: `None`
- split: `test`
- tasks: `164`
- task hash: `a0eaf2feb359df052fa10c3a1a4c2820eed985ec2c70ffd7028430c104b4cbc8`

## Variants

| Variant | Code NLL | Code PPL | Solution tokens | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|---:|
| fp16 | 12.421707 | 248129.620990 | 10451 | 16.000000 | 9294899782 |
| q2_k | 18.091321 | 71938440.184070 | 10451 | 5.118105 | 2973267084 |
| q3_k_s | 14.657358 | 2320650.853841 | 10451 | 5.326613 | 3094396044 |
| q4_k_m | 11.835313 | 138042.028125 | 10451 | 5.873431 | 3412059276 |
| c2_calib_greedy_mixed | 11.032705 | 61864.676274 | 10451 | 5.326291 | 3094208652 |
| c2_random_same_budget | 17.707267 | 48996891.451171 | 10451 | 5.326291 | 3094208652 |

## Key Comparisons

- candidate NLL delta vs target: `-3.624654`
- candidate payload bytes vs target: `-187392`
- candidate NLL delta vs same-budget random: `-6.674563`

## Decision

Candidate preserves or improves canonical-solution code likelihood at the target tensor-payload budget.

## Next Step

Replicate on another seed or code corpus before method-level generalization.
