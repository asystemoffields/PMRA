# Result Card - Stage 12 C2 Sub-Q3 Public Wikitext Eval

## Status

NO-GO for broad release claim.

## Test

The seed `7` PMRA selection from the clean sub-q3 `calib48/eval1024` gate was
frozen and evaluated on Wikitext-2 raw test chunks. Allocation was not rerun on
the public data.

- public dataset: `wikitext`, config `wikitext-2-raw-v1`, split `test`
- prompts: `512`
- eval max length: `256`
- tokens: `128,765`
- selector source: project-local clean disjoint prompt gate

## Qwen3-1.7B

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| `fp16` | `3.262141` | `16.000000` | `4,063,479,808` |
| `IQ2_M` | `3.661225` | `3.240310` | `822,933,504` |
| `IQ3_XS` | `3.373696` | `3.787785` | `961,974,272` |
| `Q3_K_S` | `3.442231` | `3.917842` | `995,004,416` |
| `PMRA` | `3.393122` | `3.506939` | `890,648,576` |
| `random same-budget` | `3.487743` | `3.506939` | `890,648,576` |

PMRA deltas:

- NLL improvement vs `IQ3_XS`: `-0.019426`
- NLL improvement vs `Q3_K_S`: `0.049109`
- NLL improvement vs random same-budget: `0.094621`
- payload bytes vs `IQ3_XS`: `-71,325,696`
- payload bytes vs `Q3_K_S`: `-104,355,840`

Interpretation: the 1.7B artifact remains materially smaller and beats `Q3_K_S`
and random on public Wikitext, but it misses the target `IQ3_XS` quality line.

## Qwen3-0.6B-Base

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| `fp16` | `3.001541` | `16.000000` | `1,503,264,768` |
| `IQ2_M` | `4.806989` | `3.467750` | `325,809,152` |
| `IQ3_XS` | `3.959172` | `3.976981` | `373,653,504` |
| `Q3_K_S` | `3.875285` | `4.086843` | `383,975,424` |
| `PMRA` | `4.077658` | `3.886782` | `365,178,880` |
| `random same-budget` | `4.359774` | `3.886782` | `365,178,880` |

PMRA deltas:

- NLL improvement vs `IQ3_XS`: `-0.118486`
- NLL improvement vs `Q3_K_S`: `-0.202373`
- NLL improvement vs random same-budget: `0.282116`
- payload bytes vs `IQ3_XS`: `-8,474,624`
- payload bytes vs `Q3_K_S`: `-18,796,544`

Interpretation: the 0.6B project-local replication does not transfer to public
Wikitext with the frozen selector.

## Decision

The PMRA mechanism is not killed: both public tests still beat same-budget random
allocation, and the 1.7B result is smaller than `IQ3_XS` while beating `Q3_K_S`.
However, the frozen project-local selector is not release-grade. The next real
test is public-calibrated allocation: score tensor promotions on Wikitext train
and evaluate on held-out Wikitext validation without prompt overlap.

## Local Artifacts

```text
run008_pmra_public_wikitext_seed7_512x256/qwen17_result.json
run008_pmra_public_wikitext_qwen06_seed7_512x256/qwen06_result.json
```
