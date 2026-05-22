# Granite 4.1 8B Heretic PMRA Release Candidate

This note records the PMRA artifact for
`heretic-org/IBM-granite-4.1-8b-heretic`.

## Sources

- HF model: `heretic-org/IBM-granite-4.1-8b-heretic`
- original base listed by upstream: `ibm-granite/granite-4.1-8b`
- upstream Heretic organization: https://huggingface.co/heretic-org
- GGUF source repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- PMRA model key: `granite4_1_8b_heretic`
- tensor profile: `granite`

The `granite` profile covers the Granite 4.1 text stack using llama-style GGUF
tensor names and Granite's layer norm layout. The upstream Heretic checkpoint is
credited as the source model; this repository only remixes existing production
GGUF tensor payloads into one PMRA GGUF artifact.

## Artifact

```text
granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf
```

The artifact stays slightly below the `IQ3_XS` tensor payload budget while
beating `IQ3_XS`, `Q3_K_S`, and a same-budget random mixed allocation on
held-out Wikitext-2 raw test.

## Selector Recipe

- group mode: `layer_family`
- selector: `c2_calib_knapsack_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K_S`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- calibration: Wikitext-2 raw train, 12 prompts
- selector evaluation: Wikitext-2 raw validation, 128 prompts
- held-out public evaluation: Wikitext-2 raw test, 512 prompts
- prompt audit overlap count: `0`

The selector is a scout shape sized for an 8B multi-source run. It was then
validated on a larger held-out public test split before release packaging.

## Selector Result

Lower NLL is better.

| Variant | Validation NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `3.038160` | `16.000000` | `16,761,102,336` |
| `IQ2_M` | `5.028465` | `2.843966` | `2,979,250,176` |
| `IQ3_XS` target | `4.845994` | `3.434877` | `3,598,270,464` |
| `Q2_K_S` | `4.878147` | `2.915472` | `3,054,157,824` |
| `Q2_K` | `4.707305` | `3.125205` | `3,273,867,264` |
| `Q3_K_S` | `4.823249` | `3.591903` | `3,762,765,824` |
| `Q3_K_M` | `4.720242` | `3.977648` | `4,166,860,800` |
| `IQ4_XS` | `4.579525` | `4.389544` | `4,598,349,824` |
| PMRA knapsack | `4.469497` | `3.433548` | `3,596,877,824` |
| same-budget random | `4.840297` | `3.433548` | `3,596,877,824` |

Selector markers:

- NLL improvement vs `IQ3_XS`: `0.376498`
- payload bytes vs `IQ3_XS`: `-1,392,640`
- NLL improvement vs `Q3_K_S`: `0.353752`
- payload bytes vs `Q3_K_S`: `-165,888,000`
- NLL improvement vs same-budget random: `0.370800`

## Held-Out Wikitext Test

The frozen selector was evaluated on Wikitext-2 raw test with 512 prompts and
256-token truncation.

| Variant | Test NLL | Payload bpw | Payload bytes |
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
- public eval prompt hash:
  `7b03faaa342e5554d39f41fe5d2c1da66eb55f7c579543c2471ff3b08e9123d1`

## Artifact Markers

- file size: `3,600,448,224` bytes
- tensor payload bytes: `3,596,877,824`
- metadata and alignment overhead: `3,570,400` bytes
- file bpw: `3.436956`
- payload bpw: `3.433548`
- SHA-256:
  `29d3d2b33583127789ee26b0b5e1d7204cb5330af2c265bef6b42c7a4a4a291a`
- tensor reload mismatches: `0`
- status: `GO`

## Source Mix

| Source | Tensors | Payload bytes |
|---|---:|---:|
| `IQ2_M` | `110` | `585,269,248` |
| `Q2_K_S` | `40` | `516,259,840` |
| `Q2_K` | `56` | `359,530,496` |
| `Q3_K_S` | `62` | `1,035,780,096` |
| `Q3_K_M` | `46` | `423,198,720` |
| `IQ4_XS` | `48` | `676,839,424` |

## Local Fit Note

The GGUF is about `3.36` GiB on disk. On this workspace's 8 GB-class Windows
machine, storage is fine and the file is slightly smaller than the primary
Ministral 3 8B PMRA artifact, but runtime RAM is still tight. Start CPU-only
with a small context and close memory-heavy apps before trying larger contexts.

## Reproduce

Run the selector:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys granite4_1_8b_heretic `
  --seed 7 `
  --calib-prompts 12 `
  --eval-prompts 128 `
  --calib-max-length 128 `
  --eval-max-length 128 `
  --group-mode layer_family `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k_s,q2_k,q3_k_s,q3_k_m,iq4_xs `
  --candidate-variant c2_calib_knapsack_mixed `
  --knapsack-max-states 50000 `
  --result-bucket run_014_granite4_1_8b_heretic_pmra
```

Selector result name:

```text
granite4_1_8b_heretic_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_s_q2_k_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_layer_family_len_128_candidate_calib_knapsack
```

Build the artifact:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_artifact `
  --model-key granite4_1_8b_heretic `
  --seed 7 `
  --eval-prompts 128 `
  --calib-prompts 12 `
  --group-mode layer_family `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k_s,q2_k,q3_k_s,q3_k_m,iq4_xs `
  --variant c2_calib_knapsack_mixed `
  --result-bucket run_014_granite4_1_8b_heretic_pmra `
  --result-name granite4_1_8b_heretic_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_s_q2_k_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_layer_family_len_128_candidate_calib_knapsack `
  --artifact-bucket run_014_granite4_1_8b_heretic_artifact `
  --artifact-name granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget_artifact `
  --output-gguf granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf
```

Evaluate the frozen selector on Wikitext-2 raw test:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_eval `
  --model-key granite4_1_8b_heretic `
  --seed 7 `
  --eval-prompts 128 `
  --calib-prompts 12 `
  --group-mode layer_family `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k_s,q2_k,q3_k_s,q3_k_m,iq4_xs `
  --variants fp16,iq2_m,iq3_xs,q2_k,q3_k_s,iq4_xs,c2_calib_knapsack_mixed,c2_random_same_budget `
  --candidate-variant c2_calib_knapsack_mixed `
  --random-variant c2_random_same_budget `
  --prompt-count 512 `
  --eval-max-length 256 `
  --dataset wikitext `
  --dataset-config wikitext-2-raw-v1 `
  --split test `
  --result-bucket run_014_granite4_1_8b_heretic_pmra `
  --result-name granite4_1_8b_heretic_c2_publiccal_wikitext_wikitext-2-raw-v1_train_to_validation_low_iq2_m_target_iq3_xs_high_q2_k_s_q2_k_q3_k_s_q3_k_m_iq4_xs_seed_7_eval_128_calib_12_layer_family_len_128_candidate_calib_knapsack `
  --public-bucket run_014_granite4_1_8b_heretic_public_eval_wikitext_test
```

## Attribution

This artifact derives from:

- `heretic-org/IBM-granite-4.1-8b-heretic`
- upstream Heretic organization page: https://huggingface.co/heretic-org
- `ibm-granite/granite-4.1-8b`, as listed by the Heretic model card
- GGUF quantizations from `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- llama.cpp GGUF tooling

Preserve upstream model, Heretic release, license, and GGUF quantization
attribution when redistributing derived artifacts. The upstream checkpoint is
an abliterated/decensored conversational model, so review outputs against your
intended policy and deployment setting.
