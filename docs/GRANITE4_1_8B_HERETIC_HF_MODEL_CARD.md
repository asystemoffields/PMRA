---
license: apache-2.0
base_model:
  - heretic-org/IBM-granite-4.1-8b-heretic
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - granite
  - granite-4.1
  - pmra
  - mixed-quantization
  - heretic
  - heretic-org
  - abliterated
  - uncensored
  - conversational
language:
  - en
quantized_by: Asystemoffields
---

# Granite 4.1 8B Heretic PMRA GGUF

This repository contains a PMRA GGUF artifact for
[`heretic-org/IBM-granite-4.1-8b-heretic`](https://huggingface.co/heretic-org/IBM-granite-4.1-8b-heretic).
Credit for the upstream Heretic checkpoint and abliterated/decensored release
goes to [`heretic-org`](https://huggingface.co/heretic-org). This repository
only remixes existing GGUF tensor payloads into a standard mixed GGUF artifact.

Production Mixed-Rate Allocation (PMRA) builds one standard GGUF by selecting
tensor-group payloads from existing production GGUF quantizations of the same
checkpoint. This artifact uses layer-family mixed allocation over the Granite
4.1 8B text stack.

## Artifact

```text
granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf
```

- base model: `heretic-org/IBM-granite-4.1-8b-heretic`
- original base listed by upstream: `ibm-granite/granite-4.1-8b`
- GGUF source repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- tensor profile: `granite`
- group mode: `layer_family`
- selector: `c2_calib_knapsack_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K_S`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- calibration: Wikitext-2 raw train, 12 prompts
- selector evaluation: Wikitext-2 raw validation, 128 prompts
- held-out public evaluation: Wikitext-2 raw test, 512 prompts
- file size: `3,600,448,224` bytes
- tensor payload bytes: `3,596,877,824`
- file bpw: `3.436956`
- payload bpw: `3.433548`
- SHA-256: `29d3d2b33583127789ee26b0b5e1d7204cb5330af2c265bef6b42c7a4a4a291a`
- tensor reload mismatches: `0`

## Validation

Lower NLL is better.

Selector validation, Wikitext-2 raw validation:

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `3.038160` | `16.000000` | `16,761,102,336` |
| `IQ2_M` | `5.028465` | `2.843966` | `2,979,250,176` |
| `IQ3_XS` target | `4.845994` | `3.434877` | `3,598,270,464` |
| `Q2_K` | `4.707305` | `3.125205` | `3,273,867,264` |
| `Q3_K_S` | `4.823249` | `3.591903` | `3,762,765,824` |
| `IQ4_XS` | `4.579525` | `4.389544` | `4,598,349,824` |
| PMRA knapsack | `4.469497` | `3.433548` | `3,596,877,824` |
| same-budget random | `4.840297` | `3.433548` | `3,596,877,824` |

Held-out Wikitext-2 raw test:

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `2.742451` | `16.000000` | `17,583,185,920` |
| `IQ2_M` | `5.150425` | `2.710999` | `2,979,250,176` |
| `IQ3_XS` target | `4.960251` | `3.274283` | `3,598,270,464` |
| `Q2_K` | `4.754195` | `2.979089` | `3,273,867,264` |
| `Q3_K_S` | `4.933018` | `3.423967` | `3,762,765,824` |
| `IQ4_XS` | `4.672932` | `4.184315` | `4,598,349,824` |
| PMRA knapsack | `4.539084` | `3.273016` | `3,596,877,824` |
| same-budget random | `4.939853` | `3.273016` | `3,596,877,824` |

Held-out markers:

- public eval decision: `GO`
- NLL improvement vs `IQ3_XS`: `0.421167`
- payload bytes vs `IQ3_XS`: `-1,392,640`
- NLL improvement vs `Q3_K_S`: `0.393934`
- payload bytes vs `Q3_K_S`: `-165,888,000`
- NLL improvement vs same-budget random: `0.400769`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `110` | `585,269,248` |
| `Q2_K_S` | `40` | `516,259,840` |
| `Q2_K` | `56` | `359,530,496` |
| `Q3_K_S` | `62` | `1,035,780,096` |
| `Q3_K_M` | `46` | `423,198,720` |
| `IQ4_XS` | `48` | `676,839,424` |

## Use

Use with a recent llama.cpp build that supports Granite 4.1 GGUF models.

```bash
llama-cli -m granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf -p "Write a short hello from PMRA." -n 80 --ctx-size 2048
```

On 8 GB-class Windows machines, start CPU-only with a small context and close
memory-heavy apps. The file is about `3.36` GiB on disk, but runtime memory can
still be tight.

## Included Reports

- `artifact_report.json`
- `artifact_report.md`
- `selector_result.json`
- `selector_result.md`
- `public_eval_wikitext_test_result.json`
- `public_eval_wikitext_test_result.md`
- `GRANITE4_1_8B_HERETIC_PMRA.md`

## Attribution

This artifact derives from:

- [`heretic-org/IBM-granite-4.1-8b-heretic`](https://huggingface.co/heretic-org/IBM-granite-4.1-8b-heretic)
- [`heretic-org`](https://huggingface.co/heretic-org), credited for the Heretic release
- `ibm-granite/granite-4.1-8b`, as listed by the upstream model card
- GGUF quantizations from `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- llama.cpp GGUF tooling

Preserve upstream model, Heretic release, license, and quantization attribution
when redistributing derived artifacts. The upstream checkpoint is an
abliterated/decensored conversational model. Review outputs against your
intended policy and deployment setting.
