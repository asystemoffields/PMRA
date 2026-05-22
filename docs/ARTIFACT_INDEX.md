# Artifact Index - Production Mixed-Rate Allocation

This is the quick map for people trying to decide which PMRA mix to inspect
first. The Gemma 4 E2B-it knapsack mix is the current frontier. The Huihui
Qwen3.5 4B abliterated mix is a released `qwen35` profile example, Ministral 3
8B and Granite 4.1 8B Heretic are completed 8B-scale releases, and the
Qwen3-1.7B mix remains prior public-calibrated evidence.

## Current Frontier - Gemma 4 E2B-it

Gemma 4 E2B-it public-calibrated PMRA, seed `7`,
`c2_calib_knapsack_mixed`.

Recommended GGUF:

```text
gemma4_e2b_it_pmra_calib_knapsack.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/gemma-4-E2B-it-PMRA-GGUF
```

Report:

```text
results/gemma4_e2b_it/artifact_report.json
results/gemma4_e2b_it/artifact_report.md
results/gemma4_e2b_it/artifact_report_knapsack.json
results/gemma4_e2b_it/artifact_report_knapsack.md
results/gemma4_e2b_it/selector_result_knapsack.json
results/gemma4_e2b_it/selector_result_knapsack.md
results/gemma4_e2b_it/llama_cli_smoke_knapsack.log
```

Metrics:

- file size: `3,110,215,968` bytes
- payload bytes: `3,094,397,068`
- file bpw: `5.353845`
- payload bpw: `5.326615`
- GGUF SHA-256: `a5a80f2628e236a228f2016bcc3ac660a268f2c8757d21d901095c74b60e3d97`
- tensor reload mismatches: `0`
- Wikitext validation NLL improvement vs `Q3_K_S`: `5.114774`
- Wikitext validation NLL improvement vs same-budget random: `7.609785`
- Wikitext validation NLL improvement vs greedy PMRA: `0.402591`
- selector-reported payload bytes vs `Q3_K_S`: `0`
- materialized artifact payload bytes vs `Q3_K_S`: `+1,024`
- llama.cpp smoke: load and single-turn generation passed
- smoke prompt/generation speed: `30.5` / `10.6` tok/s

Earlier Gemma greedy artifact, kept for comparison:

```text
gemma4_e2b_it_pmra_calib_greedy.gguf
results/gemma4_e2b_it/artifact_report_greedy.json
results/gemma4_e2b_it/artifact_report_greedy.md
results/gemma4_e2b_it/selector_result_greedy.json
results/gemma4_e2b_it/selector_result_greedy.md
```

## Released Qwen3.5 Abliterated Mix

Huihui Qwen3.5 4B abliterated PMRA, seed `7`,
`c2_calib_weight_blend_mixed`.

Release GGUF:

```text
huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF
```

Report:

```text
artifact_report.json
artifact_report.md
selector_result.json
selector_result.md
docs/HUIHUI_QWEN35_4B_ABLITERATED_HF_MODEL_CARD.md
docs/QWEN35_ABLITERATED_PMRA.md
```

Metrics:

- base model: `huihui-ai/Huihui-Qwen3.5-4B-abliterated`
- GGUF source repo: `mradermacher/Huihui-Qwen3.5-4B-abliterated-i1-GGUF`
- tensor profile: `qwen35`
- group mode: `layer_family`
- selector: `c2_calib_weight_blend_mixed`
- low source: `IQ2_M`
- target/control budget: `IQ3_XS`
- stronger sources: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- file size: `2,010,651,904` bytes
- payload bytes: `1,999,682,304`
- file bpw: `3.824576`
- payload bpw: `3.803710`
- GGUF SHA-256: `0d7fff15074b8146c37ce3d74adb7d377bb6c686b543840da468c1b683baeb03`
- tensor reload mismatches: `0`
- Wikitext validation NLL improvement vs `IQ3_XS`: `0.602179`
- Wikitext validation NLL improvement vs `Q3_K_S`: `0.506404`
- Wikitext validation NLL improvement vs same-budget random: `0.523874`
- payload bytes vs `IQ3_XS`: `-83,200`
- payload bytes vs `Q3_K_S`: `-59,229,440`

## Ministral 3 8B Instruct Mix

Ministral 3 8B Instruct PMRA, seed `7`, `c2_calib_knapsack_mixed`.

Primary quality GGUF:

```text
ministral3_8b_pmra_knapsack_iq3xs_budget.gguf
```

Compact GGUF:

```text
ministral3_8b_pmra_knapsack_3p2.gguf
```

Local reports:

```text
tmp/ministral3_8b_release/selector_result.json
tmp/ministral3_8b_release/selector_result.md
tmp/ministral3_8b_release/artifact_iq3xs_budget_report.json
tmp/ministral3_8b_release/artifact_iq3xs_budget_report.md
tmp/ministral3_8b_release/artifact_3p2_report.json
tmp/ministral3_8b_release/artifact_3p2_report.md
tmp/ministral3_8b_release/public_eval_wikitext_test_result.json
tmp/ministral3_8b_release/public_eval_wikitext_test_result.md
docs/MINISTRAL3_8B_INSTRUCT_PMRA.md
docs/MINISTRAL3_8B_INSTRUCT_HF_MODEL_CARD.md
```

Metrics:

- base model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- tensor profile: `mistral3`
- group mode: `tensor`
- selector: `c2_calib_knapsack_mixed`
- low source: `IQ2_M`
- target/control budget: `IQ3_XS`
- stronger sources: `Q2_K`, `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- primary file size: `3,713,801,312` bytes
- primary payload bytes: `3,705,913,344`
- primary file bpw: `3.499643`
- primary payload bpw: `3.492210`
- primary GGUF SHA-256:
  `7f88294593cf419a5b39b4da2c7df356fee9528de947d6547b9d11d60a84ac5d`
- compact file size: `3,403,422,816` bytes
- compact payload bytes: `3,395,534,848`
- compact file bpw: `3.207163`
- compact payload bpw: `3.199730`
- compact GGUF SHA-256:
  `ff95384e68f211b238767e1783d20ce0b4a8be8a56ac8b906756c481831421a3`
- tensor reload mismatches: `0`
- held-out Wikitext test NLL improvement vs `IQ3_XS`: `0.184894`
- held-out Wikitext test NLL improvement vs `Q3_K_S`: `0.220067`
- held-out Wikitext test NLL improvement vs same-budget random: `0.375305`
- held-out Wikitext test payload bytes vs `IQ3_XS`: `-557,056`
- held-out Wikitext test payload bytes vs `Q3_K_S`: `-152,666,112`
- compact held-out Wikitext test NLL improvement vs `IQ3_XS`: `0.121836`
- compact held-out Wikitext test payload bytes vs `IQ3_XS`: `-310,935,552`

## Released Granite 4.1 8B Heretic Mix

Granite 4.1 8B Heretic PMRA, seed `7`, `c2_calib_knapsack_mixed`.

Release GGUF:

```text
granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/IBM-granite-4.1-8b-heretic-PMRA-GGUF
```

Local reports:

```text
tmp/granite4_1_8b_heretic/release/selector_result.json
tmp/granite4_1_8b_heretic/release/selector_result.md
tmp/granite4_1_8b_heretic/release/artifact_report.json
tmp/granite4_1_8b_heretic/release/artifact_report.md
tmp/granite4_1_8b_heretic/release/public_eval_wikitext_test_result.json
tmp/granite4_1_8b_heretic/release/public_eval_wikitext_test_result.md
docs/GRANITE4_1_8B_HERETIC_PMRA.md
docs/GRANITE4_1_8B_HERETIC_HF_MODEL_CARD.md
```

Metrics:

- base model: `heretic-org/IBM-granite-4.1-8b-heretic`
- upstream Heretic organization: `heretic-org`
- GGUF source repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- tensor profile: `granite`
- group mode: `layer_family`
- selector: `c2_calib_knapsack_mixed`
- low source: `IQ2_M`
- target/control budget: `IQ3_XS`
- stronger sources: `Q2_K_S`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- file size: `3,600,448,224` bytes
- payload bytes: `3,596,877,824`
- file bpw: `3.436956`
- payload bpw: `3.433548`
- GGUF SHA-256:
  `29d3d2b33583127789ee26b0b5e1d7204cb5330af2c265bef6b42c7a4a4a291a`
- tensor reload mismatches: `0`
- held-out Wikitext test NLL improvement vs `IQ3_XS`: `0.421167`
- held-out Wikitext test NLL improvement vs `Q3_K_S`: `0.393934`
- held-out Wikitext test NLL improvement vs same-budget random: `0.400769`
- held-out Wikitext test payload bytes vs `IQ3_XS`: `-1,392,640`
- held-out Wikitext test payload bytes vs `Q3_K_S`: `-165,888,000`

## Prior Qwen Mix

Qwen3-1.7B public-calibrated PMRA, seed `7`, `c2_calib_greedy_mixed`.

Release GGUF:

```text
qwen17_publiccal_pmra_calib_greedy.gguf
```

Report:

```text
artifacts/artifact_report.json
artifacts/artifact_report.md
```

Metrics:

- file size: `961,694,976` bytes
- payload bytes: `955,742,208`
- file bpw: `3.786685`
- payload bpw: `3.763246`
- GGUF SHA-256: `cc405feb01fe8f79e44fc27f48fe15e5f591f9860dc304be6477886bf7548420`
- artifact report SHA-256: `7e1849a3c214c79f864190cc00eaf9c205ddda5faebddf7a6714815fb815f296`
- tensor reload mismatches: `0`
- Wikitext test NLL improvement vs `IQ3_XS`: `0.145818`
- Wikitext test NLL improvement vs `Q3_K_S`: `0.214353`
- Wikitext test NLL improvement vs same-budget random: `0.217721`
- TinyStories NLL improvement vs `IQ3_XS`: `0.086969`
- TinyStories NLL improvement vs `Q3_K_S`: `0.239027`
- TinyStories NLL improvement vs same-budget random: `0.180680`
- LAMBADA NLL improvement vs `IQ3_XS`: `0.110495`
- LAMBADA NLL improvement vs `Q3_K_S`: `0.136635`
- LAMBADA NLL improvement vs same-budget random: `0.183961`
- local llama-bench prompt/decode: `37.6608` / `10.5323` tok/s

## Qwen Controls

Controls used for size and runtime comparison:

```text
Qwen_Qwen3-1.7B-IQ3_XS.gguf
Qwen_Qwen3-1.7B-Q3_K_S.gguf
```

Hashes:

- `IQ3_XS`: `1b166b349d5c2dc2f717a688018d1878a283c630b7729df3fdaecf76271803f0`
- `Q3_K_S`: `6b7ecc78b5941d658c5fb055c7987d4f7fe5289da64bb92e99d8022c99ce81c3`
