---
license: apache-2.0
base_model:
  - mistralai/Ministral-3-8B-Instruct-2512-BF16
tags:
  - gguf
  - mistral
  - ministral
  - pmra
  - mixed-quantization
  - text-generation
language:
  - en
---

# Ministral 3 8B Instruct PMRA GGUF

This repository contains PMRA GGUF artifacts for
`mistralai/Ministral-3-8B-Instruct-2512-BF16`.

Production Mixed-Rate Allocation (PMRA) builds one standard GGUF by selecting
tensor payloads from existing production GGUF quantizations of the same
checkpoint. These artifacts use tensor-level mixed allocation over the
Ministral 3 8B text stack.

## Files

Primary quality artifact:

```text
ministral3_8b_pmra_knapsack_iq3xs_budget.gguf
```

Compact artifact:

```text
ministral3_8b_pmra_knapsack_3p2.gguf
```

The primary artifact targets the `IQ3_XS` payload budget. The compact artifact
is the 3.2 bpw sweep point and is the better first choice for tight 8 GB RAM
machines.

## Artifact Summary

| File | Selector | File size | Payload bpw | SHA-256 |
|---|---|---:|---:|---|
| `ministral3_8b_pmra_knapsack_iq3xs_budget.gguf` | `c2_calib_knapsack_mixed` | `3,713,801,312` | `3.492210` | `7f88294593cf419a5b39b4da2c7df356fee9528de947d6547b9d11d60a84ac5d` |
| `ministral3_8b_pmra_knapsack_3p2.gguf` | `c2_calib_knapsack_bpw_3p200_mixed` | `3,403,422,816` | `3.199730` | `ff95384e68f211b238767e1783d20ce0b4a8be8a56ac8b906756c481831421a3` |

Both GGUFs were materialized and reloaded by the artifact builder with `0`
tensor mismatches.

## Method Settings

- base model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- tensor profile: `mistral3`
- group mode: `tensor`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K`, `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- selector calibration: Wikitext-2 raw train, 12 prompts
- selector evaluation: Wikitext-2 raw validation, 128 prompts
- held-out public evaluation: Wikitext-2 raw test, 512 prompts
- prompt audit overlap count: `0`

The larger 48/512 selector shape was attempted but projected past the Modal job
window for this 8B tensor/source sweep. The artifacts here are scout-selected
and then validated on a larger held-out public test split.

## Validation

Lower NLL is better.

Selector validation, Wikitext-2 raw validation:

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `2.546905` | `16.000000` | `16,979,107,840` |
| `IQ2_M` | `4.874847` | `2.920126` | `3,098,820,608` |
| `IQ3_XS` target | `4.649152` | `3.492735` | `3,706,470,400` |
| `Q3_K_S` | `4.686507` | `3.636073` | `3,858,579,456` |
| PMRA knapsack | `4.456880` | `3.492210` | `3,705,913,344` |
| PMRA knapsack 3.2 bpw | `4.510145` | `3.199730` | `3,395,534,848` |
| same-budget random | `4.825388` | `3.492210` | `3,705,913,344` |

Held-out Wikitext-2 raw test:

| Variant | NLL | Payload bpw | Payload bytes |
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

## Use

Use with a recent llama.cpp build that supports Ministral 3 GGUF models.

```bash
llama-cli -m ministral3_8b_pmra_knapsack_3p2.gguf -p "Write a short hello from PMRA." -n 80 --ctx-size 2048
```

For 8 GB RAM Windows machines, start with the 3.2 bpw file, close memory-heavy
apps, and keep context small. The larger IQ3_XS-budget file is a better fit for
machines with more RAM.

## Included Reports

- `artifact_report.json`
- `artifact_report.md`
- `artifact_3p2_report.json`
- `artifact_3p2_report.md`
- `selector_result.json`
- `selector_result.md`
- `public_eval_wikitext_test_result.json`
- `public_eval_wikitext_test_result.md`
- `MINISTRAL3_8B_INSTRUCT_PMRA.md`

## Attribution

This artifact derives from:

- `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF quantizations from `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- llama.cpp GGUF tooling

Preserve upstream model, license, and quantization attribution when
redistributing derived artifacts.
