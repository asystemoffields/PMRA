# PMRA Release Overview

## Scope

Production Mixed-Rate Allocation (PMRA) is a method for mixing one standard
GGUF from several existing production GGUF quantizations of the same base model.

It is not a new quantizer, not a custom runtime, and not a universal benchmark
claim. PMRA selects stronger tensor formats where calibration says the bytes
matter, then materializes the result as a normal GGUF with `pmra.*` metadata.

## Current Frontier Mix

The current recommended mix is the Gemma 4 E2B-it knapsack PMRA GGUF:

```text
gemma4_e2b_it_pmra_calib_knapsack.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/gemma-4-E2B-it-PMRA-GGUF
```

Facts:

- base model: `google/gemma-4-E2B-it`
- source GGUF repo: `mradermacher/gemma-4-E2B-it-GGUF`
- selector: `c2_calib_knapsack_mixed`
- low source: `Q2_K`
- target/control budget: `Q3_K_S`
- stronger sources: `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- calibration: Wikitext-2 raw train, 24 prompts
- evaluation: Wikitext-2 raw validation, 256 prompts
- prompt overlap: `0`
- file size: `3,110,215,968` bytes
- payload bytes: `3,094,397,068`
- payload bpw: `5.326615`
- file bpw: `5.353845`
- GGUF SHA-256: `a5a80f2628e236a228f2016bcc3ac660a268f2c8757d21d901095c74b60e3d97`
- tensor reload mismatches: `0`
- llama.cpp smoke: passed

## Why This Is The Frontier

Lower NLL is better.

| Variant | NLL | Selector payload bytes |
|---|---:|---:|
| `Q2_K` low source | `20.376913` | `2,973,267,084` |
| `Q3_K_S` target/control | `17.993582` | `3,094,396,044` |
| same-budget random | `20.488594` | `3,094,396,044` |
| PMRA greedy | `13.281400` | `3,094,208,652` |
| PMRA knapsack | `12.878809` | `3,094,396,044` |

Key comparisons:

- knapsack PMRA vs `Q3_K_S` target: `+5.114774` NLL improvement
- knapsack PMRA vs same-budget random: `+7.609785` NLL improvement
- knapsack PMRA vs greedy PMRA: `+0.402591` NLL improvement
- selector-reported payload bytes vs `Q3_K_S` target: `0`
- materialized artifact payload bytes vs `Q3_K_S` target: `+1,024`

The mix is the current frontier because it keeps the same practical byte
budget as `Q3_K_S`, beats the uniform target, beats random selection at the same
budget, and improves on the previous greedy PMRA selector. The materialized
GGUF payload is `+1,024` bytes versus the selector-reported `Q3_K_S` budget.

## Additional 8B Release Mixes

Ministral 3 8B Instruct and Granite 4.1 8B Heretic are completed 8B-scale
release artifacts using the scout-selector plus held-out Wikitext-test pattern.

Granite 4.1 8B Heretic:

```text
granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/IBM-granite-4.1-8b-heretic-PMRA-GGUF
```

Facts:

- base model: `heretic-org/IBM-granite-4.1-8b-heretic`
- upstream Heretic organization: `heretic-org`
- source GGUF repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- selector: `c2_calib_knapsack_mixed`
- group mode: `layer_family`
- target/control budget: `IQ3_XS`
- payload bytes: `3,596,877,824`
- payload bpw: `3.433548`
- file size: `3,600,448,224` bytes
- GGUF SHA-256:
  `29d3d2b33583127789ee26b0b5e1d7204cb5330af2c265bef6b42c7a4a4a291a`
- held-out Wikitext test NLL improvement vs `IQ3_XS`: `0.421167`
- held-out Wikitext test NLL improvement vs same-budget random: `0.400769`
- tensor reload mismatches: `0`

The Granite artifact beat `IQ3_XS`, `Q3_K_S`, and same-budget random on the
held-out Wikitext test while staying slightly below the `IQ3_XS` tensor payload
budget. The release docs and HF model card explicitly credit `heretic-org`.

## Earlier Qwen Mixes

The Huihui Qwen3.5 4B abliterated PMRA GGUF is a released follow-up artifact
using the `qwen35` profile:

```text
huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF
```

Facts:

- base model: `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- selector: `c2_calib_weight_blend_mixed`
- group mode: `layer_family`
- target/control budget: `IQ3_XS`
- payload bytes: `1,999,682,304`
- payload bpw: `3.803710`
- file size: `2,010,651,904` bytes
- GGUF SHA-256: `0d7fff15074b8146c37ce3d74adb7d377bb6c686b543840da468c1b683baeb03`
- tensor reload mismatches: `0`

The released Qwen3.5 artifact beat `IQ3_XS`, `Q3_K_S`, greedy PMRA, knapsack
PMRA, and same-budget random on Wikitext validation while staying slightly below
the `IQ3_XS` tensor payload budget.

The Qwen3-1.7B mix remains an important public-calibrated PMRA result, but it
is now prior evidence rather than the lead mix.

```text
qwen17_publiccal_pmra_calib_greedy.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/Qwen3-1.7B-PMRA-IQ3XS-budget-GGUF
```

Facts:

- base model: `Qwen/Qwen3-1.7B`
- selector: `c2_calib_greedy_mixed`
- target/control budget: `IQ3_XS`
- stronger control: `Q3_K_S`
- payload bytes: `955,742,208`
- payload bpw: `3.763246`
- file size: `961,694,976` bytes
- GGUF SHA-256: `cc405feb01fe8f79e44fc27f48fe15e5f591f9860dc304be6477886bf7548420`
- tensor reload mismatches: `0`
- local llama-bench prompt/decode: `37.6608` / `10.5323` tok/s

Qwen passed Wikitext held-out evaluation plus TinyStories and LAMBADA
cross-corpus checks while staying smaller than `IQ3_XS` and `Q3_K_S`.

## Publishable Claim

PMRA mixes standard GGUF artifacts by combining tensor payloads from existing
production GGUF quantizations. In this repo, the current strongest mix is Gemma
4 E2B-it knapsack PMRA at a `Q3_K_S` payload budget, with Qwen3.5, Ministral 3
8B, Granite 4.1 8B Heretic, and Qwen3 public-calibrated results preserved as
supporting evidence.

## Non-Claims

- No claim that PMRA is the best possible quantization method.
- No claim that PMRA beats every production GGUF format on every model family.
- No claim that the tensor-level selector is globally optimal.
- No claim that GGUF `general.file_type` describes the mixed payload. Use the
  embedded `pmra.*` metadata and artifact reports for PMRA accounting.

## Next Work

- broaden Gemma held-out and cross-corpus evaluation
- continue frontier sweeps below the current operating point
- repeat the strongest selector family across more model families and seeds
- investigate finer-than-tensor allocation only with explicit metadata and
  runtime accounting
