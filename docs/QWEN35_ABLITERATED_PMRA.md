# Qwen3.5 4B Abliterated PMRA Release

This note records the released PMRA artifact for the Huihui Qwen3.5 4B
abliterated checkpoint.

## Sources

- HF model: `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- GGUF source repo: `mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF`
- PMRA release repo:
  `https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF`
- PMRA model key: `huihui_qwen35_4b_abliterated`
- tensor profile: `qwen35`

The `qwen35` profile covers the hybrid Qwen3.5 text stack: linear-attention
layers, regular self-attention layers, MLP tensors, normalization tensors, tied
embedding/output handling, and the GGUF `qwen35` tensor names used by
llama.cpp.

## Released Artifact

```text
huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf
```

Release recipe:

- group mode: `layer_family`
- selector: `c2_calib_weight_blend_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- calibration: Wikitext-2 raw train, 48 prompts
- evaluation: Wikitext-2 raw validation, 512 prompts

Artifact markers:

- file size: `2,010,651,904` bytes
- tensor payload bytes: `1,999,682,304`
- file bpw: `3.824576`
- payload bpw: `3.803710`
- SHA-256: `0d7fff15074b8146c37ce3d74adb7d377bb6c686b543840da468c1b683baeb03`
- tensor reload mismatches: `0`
- status: `GO`

## Selector Result

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
| PMRA greedy | `13.475620` | `3.799710` | `1,997,579,264` |
| PMRA knapsack | `13.530774` | `3.802948` | `1,999,281,664` |
| PMRA weight blend | `13.471562` | `3.803710` | `1,999,682,304` |
| same-budget random | `13.995436` | `3.802938` | `1,999,276,544` |

The release used weight blend because it was the best validation candidate in
the selector result while staying below the `IQ3_XS` target payload.

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

## Reproduce

Run the public-calibrated selector shape that produced the release comparison
table. The run's candidate flag was knapsack, but the result also recorded
greedy, weight-MSE, weight-blend, and random controls. The released GGUF was
then built from the best validation variant, `c2_calib_weight_blend_mixed`.

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys huihui_qwen35_4b_abliterated `
  --seed 7 `
  --calib-prompts 48 `
  --eval-prompts 512 `
  --calib-max-length 192 `
  --eval-max-length 256 `
  --group-mode layer_family `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m `
  --candidate-variant c2_calib_knapsack_mixed `
  --result-bucket run_012_huihui_qwen35_4b_abliterated_pmra
```

Released selector result name:

```text
huihui_qwen35_4b_abliterated_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q3_k_s_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_512_calib_48_layer_family_len_256_candidate_calib_knapsack
```

Build the released GGUF from the completed selector result:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_artifact `
  --model-key huihui_qwen35_4b_abliterated `
  --variant c2_calib_weight_blend_mixed `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q3_k_s,q3_k_m,q3_k_l,iq4_xs,q4_k_m `
  --result-bucket run_012_huihui_qwen35_4b_abliterated_pmra `
  --result-name huihui_qwen35_4b_abliterated_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q3_k_s_q3_k_m_q3_k_l_iq4_xs_q4_k_m_seed_7_eval_512_calib_48_layer_family_len_256_candidate_calib_knapsack `
  --artifact-bucket run_012_huihui_qwen35_4b_abliterated_artifact `
  --artifact-name huihui_qwen35_4b_abliterated_pmra_calib_weight_blend_artifact `
  --output-gguf huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf
```
