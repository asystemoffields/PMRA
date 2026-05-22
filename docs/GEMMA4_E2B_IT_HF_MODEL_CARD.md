---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
  - google/gemma-4-E2B-it
library_name: llama.cpp
tags:
  - gguf
  - gemma4
  - quantization
  - compression
  - mixed-quantization
  - pmra
---

# Gemma 4 E2B-it PMRA Mix GGUF

This is the current PMRA frontier mix from the companion repo.

Production Mixed-Rate Allocation (PMRA) builds one standard GGUF by selecting
tensor payloads from existing production GGUF quantizations under a byte budget.
In plain terms: start with a small GGUF, spend stronger formats on the tensors
where calibration says the bytes help most, and keep everything loadable by
normal llama.cpp tooling.

Recommended file:

```text
gemma4_e2b_it_pmra_calib_knapsack.gguf
```

The previous greedy-selector mix remains available as:

```text
gemma4_e2b_it_pmra_calib_greedy.gguf
```

## Recommended Mix

- file: `gemma4_e2b_it_pmra_calib_knapsack.gguf`
- base model: `google/gemma-4-E2B-it`
- public GGUF source repo: `mradermacher/gemma-4-E2B-it-GGUF`
- low source: `Q2_K`
- target/control budget: `Q3_K_S`
- stronger sources: `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- selector: `c2_calib_knapsack_mixed`
- tensor profile: `gemma4`
- tensor count: `601`
- file size: `3,110,215,968` bytes
- tensor payload bytes: `3,094,397,068`
- payload bpw: `5.326615`
- file bpw: `5.353845`
- SHA-256: `a5a80f2628e236a228f2016bcc3ac660a268f2c8757d21d901095c74b60e3d97`
- tensor reload mismatches: `0`

`general.file_type` is inherited from the metadata source because GGUF has no
single enum for this mixed tensor allocation. Use the embedded `pmra.*`
metadata and `artifact_report_knapsack.json` for PMRA payload accounting.

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `Q2_K` | 397 | 2,637,615,244 |
| `Q3_K_M` | 84 | 233,001,984 |
| `Q4_K_M` | 56 | 119,282,688 |
| `IQ4_XS` | 40 | 83,140,608 |
| `Q3_K_L` | 24 | 21,356,544 |

## Wikitext Selector Result

The selector used Wikitext-2 raw train prompts for calibration and Wikitext-2
raw validation prompts for evaluation. Lower NLL is better.

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | 14.381222 | 16.000000 | 9,294,899,782 |
| `Q2_K` | 20.376913 | 5.118105 | 2,973,267,084 |
| `Q3_K_S` target | 17.993582 | 5.326613 | 3,094,396,044 |
| `Q3_K_M` | 15.619944 | 5.483489 | 3,185,529,996 |
| `Q3_K_L` | 15.756687 | 5.622925 | 3,266,532,492 |
| `IQ4_XS` | 16.043206 | 5.670221 | 3,294,008,460 |
| `Q4_K_M` | 13.549753 | 5.873431 | 3,412,059,276 |
| same-budget random | 20.488594 | 5.326613 | 3,094,396,044 |
| PMRA `c2_calib_greedy_mixed` | 13.281400 | 5.326291 | 3,094,208,652 |
| PMRA `c2_calib_knapsack_mixed` | 12.878809 | 5.326613 | 3,094,396,044 |

Key comparisons:

- knapsack PMRA vs `Q3_K_S` target: `+5.114774` NLL improvement
- knapsack PMRA vs same-budget random: `+7.609785` NLL improvement
- knapsack PMRA vs greedy PMRA: `+0.402591` NLL improvement
- selector-reported payload bytes vs `Q3_K_S` target: `0`
- materialized artifact payload bytes vs `Q3_K_S` target: `+1,024`
- selected tensor groups: `204`

## Runtime Smoke

Local llama.cpp build: `a8fd165`.

- `llama-cli` loaded the knapsack GGUF and generated text from a single-turn
  prompt.
- smoke prompt speed: `30.5` prompt tok/s.
- smoke generation speed: `10.6` tok/s.

The llama.cpp display label inherits the `general.file_type` metadata source.
The PMRA accounting is available in the embedded `pmra.*` metadata and artifact
report.

## Files Included

- `gemma4_e2b_it_pmra_calib_knapsack.gguf`
- `artifact_report_knapsack.json`
- `artifact_report_knapsack.md`
- `selector_result_knapsack.json`
- `selector_result_knapsack.md`
- `llama_cli_smoke_knapsack.log`
- `GEMMA4_E2B_IT_RELEASE.md`
- `gemma4_e2b_it_pmra_calib_greedy.gguf`
- greedy-selector reports and code-likelihood cards from the previous release

## Method And Reproduction

The method, release notes, result cards, and upload helpers are maintained in:

```text
https://github.com/asystemoffields/PMRA
```

## Attribution

This mix is derived from:

- `google/gemma-4-E2B-it`
- public Gemma 4 E2B-it GGUF quantizations from `mradermacher/gemma-4-E2B-it-GGUF`
- llama.cpp GGUF tooling

Preserve upstream model, license, and quantization attribution when
redistributing derived artifacts.
