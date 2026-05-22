---
license: apache-2.0
base_model:
  - Qwen/Qwen3-1.7B
library_name: llama.cpp
tags:
  - gguf
  - qwen3
  - quantization
  - compression
  - mixed-quantization
  - pmra
---

# Qwen3-1.7B PMRA Mix GGUF

This is a GGUF mix produced with Production Mixed-Rate Allocation (PMRA):
tensor-level mixing over existing production GGUF quantized payloads.

It is a standard GGUF file. It does not require a custom runtime.

## What This Thing Is

PMRA starts from a low-bit GGUF source and promotes selected tensors to stronger
existing GGUF formats when calibration says the bytes are worth spending.

For this mix:

- base model: `Qwen/Qwen3-1.7B`
- low source: `IQ2_M`
- target/control budget: `IQ3_XS`
- stronger control: `Q3_K_S`
- selector: `c2_calib_greedy_mixed`
- calibration corpus: Wikitext-2 raw train
- evaluation corpora: Wikitext-2 raw validation/test, TinyStories validation,
  LAMBADA English test
- output format: standard GGUF with per-tensor quantization types

## Mix Facts

- file size: `961,694,976` bytes
- payload bytes: `955,742,208`
- payload bpw: `3.763246`
- file bpw: `3.786685`
- GGUF SHA-256: `cc405feb01fe8f79e44fc27f48fe15e5f591f9860dc304be6477886bf7548420`
- tensor reload mismatches: `0`
- local llama-bench prompt/decode: `37.6608` / `10.5323` tok/s

`general.file_type` is inherited from a source GGUF because GGUF does not have a
single enum for mixed tensor allocations. Use the embedded `pmra.*` metadata and
the artifact report for the actual PMRA payload accounting.

## Evaluation Summary

Lower NLL is better. Positive improvement means PMRA was better than the
control.

| Eval | PMRA NLL | vs IQ3_XS | vs Q3_K_S | vs same-budget random |
|---|---:|---:|---:|---:|
| Wikitext validation | `3.279698` | `+0.156058` | `+0.223327` | `+0.229276` |
| Wikitext test | `3.227878` | `+0.145818` | `+0.214353` | `+0.217721` |
| TinyStories validation | `2.060892` | `+0.086969` | `+0.239027` | `+0.180680` |
| LAMBADA English test | `3.790112` | `+0.110495` | `+0.136635` | `+0.183961` |

Payload comparison:

- PMRA vs `IQ3_XS`: `-6,232,064` bytes
- PMRA vs `Q3_K_S`: `-39,262,208` bytes

## Side-By-Side Baselines

All rows use the same `512` prompt evaluation setup per dataset. Lower NLL/PPL
is better.

### Wikitext-2 Raw Validation

| Variant | NLL | PPL | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|
| FP16 reference | `3.325596` | `27.815572` | `16.000000` | `4,063,479,808` |
| `IQ2_M` | `3.714917` | `41.055163` | `3.240310` | `822,933,504` |
| `IQ3_XS` | `3.435756` | `31.054871` | `3.787785` | `961,974,272` |
| `Q3_K_S` | `3.503025` | `33.215762` | `3.917842` | `995,004,416` |
| same-budget random | `3.508974` | `33.413963` | `3.763246` | `955,742,208` |
| PMRA | `3.279698` | `26.567736` | `3.763246` | `955,742,208` |

### Wikitext-2 Raw Test

| Variant | NLL | PPL | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|
| FP16 reference | `3.262141` | `26.105378` | `16.000000` | `4,063,479,808` |
| `IQ2_M` | `3.661225` | `38.908961` | `3.240310` | `822,933,504` |
| `IQ3_XS` | `3.373696` | `29.186208` | `3.787785` | `961,974,272` |
| `Q3_K_S` | `3.442231` | `31.256622` | `3.917842` | `995,004,416` |
| same-budget random | `3.445599` | `31.362067` | `3.763246` | `955,742,208` |
| PMRA | `3.227878` | `25.226066` | `3.763246` | `955,742,208` |

### TinyStories Validation

| Variant | NLL | PPL | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|
| FP16 reference | `2.090523` | `8.089144` | `16.000000` | `4,063,479,808` |
| `IQ2_M` | `2.343963` | `10.422455` | `3.240310` | `822,933,504` |
| `IQ3_XS` | `2.147862` | `8.566519` | `3.787785` | `961,974,272` |
| `Q3_K_S` | `2.299919` | `9.973373` | `3.917842` | `995,004,416` |
| same-budget random | `2.241572` | `9.408108` | `3.763246` | `955,742,208` |
| PMRA | `2.060892` | `7.852973` | `3.763246` | `955,742,208` |

### LAMBADA English Test

| Variant | NLL | PPL | Payload bpw | Payload bytes |
|---|---:|---:|---:|---:|
| FP16 reference | `3.825314` | `45.847181` | `16.000000` | `4,063,479,808` |
| `IQ2_M` | `4.105106` | `60.649195` | `3.240310` | `822,933,504` |
| `IQ3_XS` | `3.900607` | `49.432431` | `3.787785` | `961,974,272` |
| `Q3_K_S` | `3.926747` | `50.741661` | `3.917842` | `995,004,416` |
| same-budget random | `3.974073` | `53.200798` | `3.763246` | `955,742,208` |
| PMRA | `3.790112` | `44.261355` | `3.763246` | `955,742,208` |

### Local llama.cpp Bench

CPU `llama-bench`, `-p 128 -n 64 -r 3`:

| Variant | Prompt tok/s | Decode tok/s | Payload bytes |
|---|---:|---:|---:|
| PMRA | `37.6608` | `10.5323` | `955,742,208` |
| `IQ3_XS` | `11.8432` | `8.0709` | `961,974,272` |
| `Q3_K_S` | `27.3281` | `6.9624` | `995,004,416` |

## Reproduction

The method and reproduction commands are maintained in the companion GitHub
repository:

```text
https://github.com/asystemoffields/PMRA
```

The minimal reproduction path is:

1. Run public-calibrated allocation on Wikitext-2 raw train/validation.
2. Evaluate the frozen selection on Wikitext-2 raw test and TinyStories
   validation.
3. Mix the selected tensor payloads into one GGUF.
4. Verify the mix with llama.cpp.

The release docs include exact commands, source controls, hashes, and result
cards.

## Source And Attribution

This mix is derived from:

- `Qwen/Qwen3-1.7B`
- public Qwen3 GGUF quantizations from `bartowski/Qwen_Qwen3-1.7B-GGUF`
- llama.cpp GGUF tooling

Please preserve upstream attribution when redistributing.

## Known Limitations

- Broader public benchmark coverage beyond Wikitext, TinyStories, and LAMBADA
  is still future work.
- The current selector is tensor-level and coarse; finer block/page allocation
  may improve the frontier but would require careful metadata/runtime accounting.
- Results should not be generalized to non-Qwen model families without
  replication.
