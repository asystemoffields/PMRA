---
license: apache-2.0
base_model:
  - huihui-ai/Huihui-Qwen3.5-4B-abliterated
tags:
  - gguf
  - qwen3_5
  - pmra
  - mixed-quantization
  - abliterated
  - conversational
language:
  - en
---

# Huihui Qwen3.5 4B Abliterated PMRA GGUF

This repository contains a PMRA GGUF artifact for
`huihui-ai/Huihui-Qwen3.5-4B-abliterated`.

Production Mixed-Rate Allocation (PMRA) builds one standard GGUF by selecting
tensor-group payloads from existing production GGUF quantizations of the same
checkpoint. This artifact uses layer-family mixed allocation over the
Qwen3.5 hybrid text stack.

## Artifact

```text
huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf
```

- base model: `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- GGUF source repo: `mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF`
- tensor profile: `qwen35`
- group mode: `layer_family`
- selector: `c2_calib_weight_blend_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- calibration: Wikitext-2 raw train, 48 prompts
- evaluation: Wikitext-2 raw validation, 512 prompts
- file size: `2,010,651,904` bytes
- tensor payload bytes: `1,999,682,304`
- file bpw: `3.824576`
- payload bpw: `3.803710`
- SHA-256: `0d7fff15074b8146c37ce3d74adb7d377bb6c686b543840da468c1b683baeb03`
- tensor reload mismatches: `0`

## Validation

Lower NLL is better.

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `3.171504` | `16.000000` | `8,411,502,592` |
| `IQ2_M` | `14.179427` | `3.059981` | `1,608,689,664` |
| `IQ3_XS` target | `14.073741` | `3.803868` | `1,999,765,504` |
| `Q3_K_S` | `13.977966` | `3.916374` | `2,058,911,744` |
| `Q3_K_M` | `13.865006` | `4.273212` | `2,246,508,544` |
| `Q3_K_L` | `13.911635` | `4.465188` | `2,347,433,984` |
| `IQ4_XS` | `13.814762` | `4.612112` | `2,424,674,304` |
| `Q4_K_M` | `13.877977` | `5.129255` | `2,696,546,304` |
| PMRA blend | `13.471562` | `3.803710` | `1,999,682,304` |
| same-budget random | `13.995436` | `3.802938` | `1,999,276,544` |

Key markers:

- PMRA NLL improvement vs `IQ3_XS`: `0.602179`
- PMRA payload bytes vs `IQ3_XS`: `-83,200`
- PMRA NLL improvement vs same-budget random: `0.523874`
- PMRA NLL improvement vs `Q3_K_S`: `0.506404`
- PMRA payload bytes vs `Q3_K_S`: `-59,229,440`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `67` | `650,262,528` |
| `Q3_K_S` | `212` | `785,808,896` |
| `Q3_K_M` | `19` | `118,192,128` |
| `Q3_K_L` | `37` | `82,221,568` |
| `IQ4_XS` | `77` | `320,533,248` |
| `Q4_K_M` | `14` | `42,663,936` |

## Files

- `huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf`
- `artifact_report.json`
- `artifact_report.md`
- `selector_result.json`
- `selector_result.md`

## Use

Use with a recent llama.cpp build that supports Qwen3.5 GGUF models.

```bash
llama-cli -m huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf -p "Write a short hello from PMRA." -n 80
```

## Attribution

This artifact derives from:

- `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- `Qwen/Qwen3.5-4B`
- GGUF quantizations from `mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF`
- llama.cpp GGUF tooling

The upstream checkpoint is an abliterated conversational model. Review outputs
against your intended policy and deployment setting.
