# Evidence - Production Mixed-Rate Allocation

## Leak-Audit Status

The initial C2 evaluator reused a fixed prompt pool for both calibration and eval. With `256` eval prompts, the `12` calibration prompts were included in the eval set for seeds `6`, `7`, and `8`.

Impact:

- calibration-greedy selector claims are contaminated until rerun
- weight-MSE selector claims are less directly exposed because the selector is weight-only, but the headline still needs a clean disjoint eval rerun
- clean reruns now write a `prompt_audit` block with prompt hashes and `overlap_count: 0`

Current authority: public-calibrated runs with prompt-audit blocks. The current
frontier is PMRA-021, the Gemma 4 E2B-it knapsack mix. PMRA-022 records the
released Huihui Qwen3.5 4B abliterated PMRA mix. The Qwen3 public-calibrated
suite in PMRA-018 through PMRA-020 remains the broader cross-corpus support set.
PMRA-023 records the completed Ministral 3 8B Instruct mixes and held-out
Wikitext test confirmation. PMRA-024 records the Granite 4.1 8B Heretic PMRA
mix with explicit Heretic upstream credit and held-out Wikitext test
confirmation.

## Confirmed Findings

### PMRA-001 - Three-Seed 64-Prompt Model-Forward Marker

Qwen3-1.7B mixed production allocation beat uniform `Q3_K_M` across seeds `6`, `7`, and `8`.

- mean NLL improvement vs `Q3_K_M`: `0.152334`
- mean tensor-payload saving vs `Q3_K_M`: `14,084,779` bytes
- mean improvement vs same-budget random control: `0.201847` NLL
- mean improvement vs same-budget weight-MSE control: `0.035187` NLL

Source:

```text
codex_ladder/runs/run_007/stage5_c2_production_mixed_rate_robustness_result.md
```

### PMRA-002 - Real Mixed GGUF Artifact

Seed `8` mixed allocation was written as a single GGUF and reloaded with zero tensor mismatches.

- file size: `1,071,604,128` bytes
- payload bytes: `1,065,652,224`
- metadata/alignment overhead: `5,951,904` bytes
- file bpw: `4.219454`

Source:

```text
codex_ladder/runs/run_007/stage5_c2_mixed_gguf_artifact_result.md
```

### PMRA-003 - llama.cpp Runtime Load

`llama-cli.exe` loaded `mixed_seed8.gguf` and generated text with exit code `0`.

This shows the artifact is not merely readable by Python GGUF tooling.

### PMRA-004 - 256-Prompt Robustness Held

The larger three-seed eval held the production mixed-rate signal.

Calibration-greedy mean result:

- mean NLL improvement vs `Q3_K_M`: `0.083145`
- mean NLL improvement vs `IQ3_M`: `0.287746`
- mean NLL improvement vs `IQ4_XS`: `0.099260`
- mean payload bytes vs `Q3_K_M`: `-14,084,779`

Weight-MSE same-budget control was stronger:

- seed `6`: `0.125789` NLL better than `Q3_K_M`
- seed `7`: `0.122095` NLL better than `Q3_K_M`
- seed `8`: `0.122303` NLL better than `Q3_K_M`
- payload bytes vs `Q3_K_M`: `-215,040`

Source:

```text
codex_ladder/runs/run_007/stage5_c2_eval256_result.md
```

### PMRA-005 - Weight-MSE Mixed Artifact

The stronger 256-prompt weight-MSE selector was materialized as a GGUF and loaded by llama.cpp.

- file size: `1,073,027,488` bytes
- file bytes vs `Q3_K_M`: `-215,040`
- payload bpw: `4.201623`
- llama.cpp load smoke: passed

Caveat: CPU llama.cpp runtime is slower than uniform `Q3_K_M`.

Source:

```text
codex_ladder/runs/run_007/stage5_c2_weight_mse_artifact_runtime_result.md
```

## Killed Adjacent Branches

### C1 Production-Base Residual Overlay

Killed. Side payloads over production baselines worsened both `Q3_K_M` and `IQ3_XS` at the tested operating point.

Source:

```text
codex_ladder/runs/run_007/stage5_c1_production_base_residual_overlay_result.md
```

### C5 Simple IQ4 Semantic Erasure

Killed. Simple IQ4 bitplane/subfield erasure caused severe quality collapse and missed the saving target.

Source:

```text
codex_ladder/runs/run_007/stage5_c5_iq4_semantic_erasure_result.md
```

## Open Evidence

### PMRA-006 - IQ3_M-Budget Compression Gate

Passed on seeds `6`, `7`, and `8`.

Mean weight-MSE selector result:

- NLL improvement vs `Q3_K_M`: `0.117052`
- NLL improvement vs `IQ3_M`: `0.321653`
- NLL improvement vs random same-budget: `0.213092`
- payload bytes vs `Q3_K_M`: `-43,958,272`
- payload bytes vs `IQ3_M`: `-81,920`

Source:

```text
codex_ladder/runs/run_007/stage5_c2_iq3m_budget_result.md
```

### PMRA-007 - IQ3_M-Budget Artifact And Runtime

The lower-budget C2 artifact materialized as a GGUF and loaded in llama.cpp.

- file size: `1,029,284,256` bytes
- file bytes vs `Q3_K_M`: `-43,958,272`
- file bytes vs `IQ3_M`: `-81,920`
- decode benchmark: `9.36 tok/s` vs `9.59 tok/s` for `Q3_K_M`
- prompt benchmark: `36.70 tok/s` vs `59.22 tok/s` for `Q3_K_M`

Source:

```text
codex_ladder/runs/run_007/stage5_c2_iq3m_budget_artifact_runtime_result.md
```

## Open Evidence

### PMRA-009 - Clean 1.7B Leak-Audit Rerun

Passed against the original `IQ3_M` and `Q3_K_M` controls.

- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL improvement vs `IQ3_M`: `0.332992`
- mean NLL improvement vs `Q3_K_M`: `0.119953`
- mean NLL improvement vs `IQ4_XS`: `0.145736`
- mean NLL improvement vs random same-budget: `0.229526`
- payload bytes vs `Q3_K_M`: `-43,958,272`

Source:

```text
codex_ladder/runs/run_008/stage5_c2_17b_clean_iq3m_budget_result.md
```

### PMRA-010 - Clean Artifact And Runtime

Clean seed `8` was materialized as a GGUF and loaded in llama.cpp.

- file size: `1,029,284,256` bytes
- payload bytes: `1,023,332,352`
- tensor reload mismatches: `0`
- prompt speed: `29.73 tok/s`
- decode speed: `7.89 tok/s`

Caveat: `Q3_K_S` is a stronger production control than expected. It is smaller and faster than the clean PMRA artifact in local llama.cpp benchmarks.

Source:

```text
codex_ladder/runs/run_008/stage5_c2_clean_artifact_runtime_result.md
```

### PMRA-011 - Q3_K_S-Target Stronger-Control Gate

The tensor-level PMRA effect held against `Q3_K_M`, `IQ4_XS`, and random
same-budget allocation, but it did not beat uniform `Q3_K_S` at the same
`Q3_K_S` payload budget.

- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL delta vs `Q3_K_S`: `-0.190351`
- mean NLL improvement vs `Q3_K_M`: `0.041247`
- mean NLL improvement vs `IQ4_XS`: `0.067031`
- mean NLL improvement vs random same-budget: `0.170860`
- payload bpw: `3.917842`, equal to uniform `Q3_K_S`

The matching-HF guard using `Qwen/Qwen3-1.7B` sharded safetensors reproduced the
same numbers, so the miss is not explained by the earlier HF-reference mismatch.

Decision: current tensor-level PMRA is real but blocked as a deployable
headline by the stronger public `Q3_K_S` baseline unless sub-q3 stacking changes
the frontier.

Source:

```text
codex_ladder/runs/run_008/stage6_c2_q3ks_target_result.md
```

### PMRA-012 - Sub-Q3 IQ2_M to IQ3_XS Gate

Sub-q3 calibration-greedy PMRA beat uniform `IQ3_XS` while staying below
`IQ3_XS` payload bytes on all three clean seeds.

- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL improvement vs `IQ3_XS`: `0.412315`
- mean NLL improvement vs `Q3_K_S`: `0.018464`
- mean NLL improvement vs random same-budget: `0.231937`
- mean payload bytes vs `IQ3_XS`: `-11,782,827`
- mean payload bytes vs `Q3_K_S`: `-44,812,971`
- mean payload bpw: `3.741390`

Decision: promote seed `8` calibration-greedy PMRA to artifact/runtime
validation. Caveat: seed `6` is `GRAY` because random same-budget allocation was
too close/better, so this is not publication-ready until artifact and broader
validation pass.

Source:

```text
codex_ladder/runs/run_008/stage7_c2_subq3_iq2m_to_iq3xs_result.md
```

### PMRA-013 - Sub-Q3 Artifact And Runtime

Seed `8` calibration-greedy sub-q3 PMRA was materialized as a GGUF and loaded in
llama.cpp.

- file size: `959,521,184` bytes
- payload bytes: `953,569,280`
- tensor reload mismatches: `0`
- seed `8` NLL improvement vs `IQ3_XS`: `0.439747`
- seed `8` NLL improvement vs `Q3_K_S`: `0.057056`
- payload bytes vs `IQ3_XS`: `-8,404,992`
- payload bytes vs `Q3_K_S`: `-41,435,136`
- prompt speed: `37.79 tok/s` vs `19.45 tok/s` for `IQ3_XS`
- decode speed: `9.36 tok/s` vs `7.68 tok/s` for `IQ3_XS`

Decision: keep going. A larger 1024-prompt confirmation run is in progress
before any public claim.

Source:

```text
codex_ladder/runs/run_008/stage8_c2_subq3_artifact_runtime_result.md
```

### PMRA-014 - Sub-Q3 1024-Prompt Confirmation

The sub-q3 signal survived a larger held-out eval.

- eval prompts: `1024`
- calibration prompts: `12`
- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL improvement vs `IQ3_XS`: `0.420092`
- mean NLL improvement vs `Q3_K_S`: `0.022041`
- mean NLL improvement vs random same-budget: `0.236058`
- mean payload bytes vs `IQ3_XS`: `-11,782,827`
- mean payload bytes vs `Q3_K_S`: `-44,812,971`

Decision: quality/size survived, selector did not fully clear. Seed `6` remains
`GRAY` because random same-budget allocation beat calibration-greedy on that
seed. Next step is selector hardening, starting with a larger calibration budget
on the failure seed.

Source:

```text
codex_ladder/runs/run_008/stage9_c2_subq3_eval1024_confirmation.md
```

### PMRA-015 - Sub-Q3 Calib48 Production-Shaped Pass

The larger calibration selector fixed the seed `6` random-control failure and
produced a stronger artifact candidate.

- model: `Qwen/Qwen3-1.7B`
- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL improvement vs `IQ3_XS`: `0.869488`
- mean NLL improvement vs `Q3_K_S`: `0.467529`
- mean NLL improvement vs random same-budget: `0.502104`
- mean payload bytes vs `IQ3_XS`: `-64,160,427`
- mean payload bytes vs `Q3_K_S`: `-97,190,571`
- mean payload bpw: `3.535153`

Seed `7` was materialized as a GGUF:

- file size: `896,601,344` bytes
- payload bytes: `890,648,576`
- payload bpw: `3.506939`
- tensor reload mismatches: `0`
- PMRA metadata fields: `11`
- prompt speed: `33.58 tok/s`
- decode speed: `10.85 tok/s`

Same local runtime comparison:

- `Q3_K_S`: `26.97` prompt tok/s, `7.27` decode tok/s
- `IQ3_XS`: `12.27` prompt tok/s, `7.99` decode tok/s

Source:

```text
codex_ladder/runs/run_008/stage10_c2_subq3_calib48_result.md
```

### PMRA-016 - Sub-Q3 Calib48 Second-Size Replication

The same sub-q3 PMRA gate passed on `Qwen/Qwen3-0.6B-Base`.

- prompt overlap count: `0` on seeds `6`, `7`, and `8`
- mean NLL improvement vs `IQ3_XS`: `1.127286`
- mean NLL improvement vs `Q3_K_S`: `0.507815`
- mean NLL improvement vs random same-budget: `1.121200`
- mean payload bytes vs `IQ3_XS`: `-13,607,936`
- mean payload bytes vs `Q3_K_S`: `-23,929,856`
- mean payload bpw: `3.832145`

Decision: replication supported promoting PMRA to a production-shaped method.
It did not yet establish cross-family generality.

Source:

```text
codex_ladder/runs/run_008/stage11_c2_subq3_qwen06_calib48_replication_result.md
```

### PMRA-017 - Frozen Selector Public Wikitext Eval

The seed `7` sub-q3 PMRA selection was frozen and evaluated on Wikitext-2 raw
test chunks without rerunning allocation on public data.

Qwen3-1.7B:

- status: `NO-GO`
- NLL improvement vs `IQ3_XS`: `-0.019426`
- NLL improvement vs `Q3_K_S`: `0.049109`
- NLL improvement vs random same-budget: `0.094621`
- payload bytes vs `IQ3_XS`: `-71,325,696`
- payload bytes vs `Q3_K_S`: `-104,355,840`

Qwen3-0.6B-Base:

- status: `NO-GO`
- NLL improvement vs `IQ3_XS`: `-0.118486`
- NLL improvement vs `Q3_K_S`: `-0.202373`
- NLL improvement vs random same-budget: `0.282116`
- payload bytes vs `IQ3_XS`: `-8,474,624`
- payload bytes vs `Q3_K_S`: `-18,796,544`

Decision: do not publish the frozen project-local selector as a broad quality
claim. The mechanism remains live because both public runs beat same-budget
random allocation and the 1.7B result still beats `Q3_K_S` while materially
smaller, but release now depends on a public-calibrated held-out selector.

Source:

```text
results/run_008/stage12_c2_subq3_public_wikitext_eval.md
```

### PMRA-018 - Public-Calibrated PMRA Release Candidate

Public calibration resolved the frozen-selector Wikitext transfer failure.

Qwen3-1.7B, `c2_calib_greedy_mixed`, Wikitext train -> validation -> frozen
test:

- validation NLL improvement vs `IQ3_XS`: `0.156058`
- validation NLL improvement vs `Q3_K_S`: `0.223327`
- validation NLL improvement vs same-budget random: `0.229276`
- test NLL improvement vs `IQ3_XS`: `0.145818`
- test NLL improvement vs `Q3_K_S`: `0.214353`
- test NLL improvement vs same-budget random: `0.217721`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`

Qwen3-0.6B-Base, `c2_calib_weight_blend_mixed`:

- validation NLL improvement vs `IQ3_XS`: `0.218488`
- validation NLL improvement vs `Q3_K_S`: `0.153261`
- frozen test NLL improvement vs `IQ3_XS`: `0.225181`
- frozen test NLL improvement vs `Q3_K_S`: `0.141294`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`

The 1.7B selector was materialized as one GGUF:

- file size: `961,694,976` bytes
- payload bytes: `955,742,208`
- payload bpw: `3.763246`
- tensor mismatches: `0`
- local llama-bench: `37.6608` prompt tok/s, `10.5323` decode tok/s

Decision: PMRA is live as a production-shaped method. Cross-corpus TinyStories
eval is recorded in PMRA-019.

Source:

```text
results/run_008/stage13_c2_public_calibrated_pmra_result.md
```

### PMRA-019 - Public-Calibrated Cross-Corpus TinyStories Eval

The public-calibrated PMRA selections survived a distribution shift to
`roneneldan/TinyStories` validation.

Qwen3-1.7B, `c2_calib_greedy_mixed`:

- NLL improvement vs `IQ3_XS`: `0.086969`
- NLL improvement vs `Q3_K_S`: `0.239027`
- NLL improvement vs same-budget random: `0.180680`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`

Qwen3-0.6B-Base, `c2_calib_weight_blend_mixed`:

- NLL improvement vs `IQ3_XS`: `0.140297`
- NLL improvement vs `Q3_K_S`: `0.080590`
- NLL improvement vs same-budget random: `0.359056`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`

Decision: Gate 6C passes. PMRA can move into release packaging, with claims
scoped to public-calibrated Qwen3 evidence and no cross-family benchmark claim.

Source:

```text
results/run_008/stage14_c2_public_calibrated_cross_corpus_tinystories.md
```

### PMRA-020 - Public-Calibrated LAMBADA Eval

The public-calibrated PMRA selections also passed `EleutherAI/lambada_openai`
English test.

Qwen3-1.7B, `c2_calib_greedy_mixed`:

- NLL improvement vs `IQ3_XS`: `0.110495`
- NLL improvement vs `Q3_K_S`: `0.136635`
- NLL improvement vs same-budget random: `0.183961`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`

Qwen3-0.6B-Base, `c2_calib_weight_blend_mixed`:

- NLL improvement vs `IQ3_XS`: `0.177553`
- NLL improvement vs `Q3_K_S`: `0.091113`
- NLL improvement vs same-budget random: `0.352031`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`

Decision: the initial broader public benchmark mix passes. PMRA is ready for
method/model release with scoped claims.

Source:

```text
results/run_008/stage15_c2_public_calibrated_lambada_result.md
```

### PMRA-021 - Gemma Knapsack Selector Artifact

The Gemma 4 E2B-it public-calibrated run was rerun with
`c2_calib_knapsack_mixed`, a multiple-choice knapsack selector over the same
candidate tensor/source promotions used by the greedy selector.

Wikitext-2 raw train -> validation, seed `7`:

- prompt audit overlap count: `0`
- knapsack PMRA NLL: `12.878809`
- greedy PMRA NLL: `13.281400`
- `Q3_K_S` target NLL: `17.993582`
- same-budget random NLL: `20.488594`
- NLL improvement vs `Q3_K_S`: `5.114774`
- NLL improvement vs same-budget random: `7.609785`
- NLL improvement vs greedy PMRA: `0.402591`
- selector-reported payload bytes: `3,094,396,044`
- selector-reported payload bytes vs `Q3_K_S`: `0`
- materialized artifact payload bytes: `3,094,397,068`
- materialized artifact payload bytes vs `Q3_K_S`: `+1,024`

The knapsack selection was materialized as one GGUF:

- file size: `3,110,215,968` bytes
- payload bpw: `5.326615`
- file bpw: `5.353845`
- GGUF SHA-256:
  `a5a80f2628e236a228f2016bcc3ac660a268f2c8757d21d901095c74b60e3d97`
- tensor reload mismatches: `0`
- llama.cpp smoke prompt/generation speed: `30.5` / `10.6` tok/s

Source:

```text
results/gemma4_e2b_it/selector_result_knapsack.md
results/gemma4_e2b_it/artifact_report_knapsack.md
results/gemma4_e2b_it/llama_cli_smoke_knapsack.log
```

### PMRA-022 - Huihui Qwen3.5 Abliterated Weight-Blend Release

The Huihui Qwen3.5 4B abliterated PMRA release used the `qwen35` tensor profile
and layer-family allocation over the hybrid Qwen3.5 text stack. The selected
artifact was `c2_calib_weight_blend_mixed`, not the knapsack candidate, because
weight blend had the best Wikitext validation NLL in the selector result while
remaining slightly below the `IQ3_XS` target payload budget.

Wikitext-2 raw train -> validation, seed `7`:

- released artifact: `huihui_qwen35_4b_abliterated_pmra_calib_weight_blend.gguf`
- HF release repo:
  `https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF`
- group mode: `layer_family`
- selector: `c2_calib_weight_blend_mixed`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- fp16 reference NLL: `3.171504`
- `IQ2_M` NLL: `14.179427`
- `IQ3_XS` target NLL: `14.073741`
- `Q3_K_S` NLL: `13.977966`
- greedy PMRA NLL: `13.475620`
- knapsack PMRA NLL: `13.530774`
- weight-blend PMRA NLL: `13.471562`
- same-budget random NLL: `13.995436`
- NLL improvement vs `IQ3_XS`: `0.602179`
- NLL improvement vs `Q3_K_S`: `0.506404`
- NLL improvement vs same-budget random: `0.523874`
- payload bytes: `1,999,682,304`
- payload bytes vs `IQ3_XS`: `-83,200`
- payload bytes vs `Q3_K_S`: `-59,229,440`

The weight-blend selection was materialized as one GGUF:

- file size: `2,010,651,904` bytes
- payload bpw: `3.803710`
- file bpw: `3.824576`
- GGUF SHA-256:
  `0d7fff15074b8146c37ce3d74adb7d377bb6c686b543840da468c1b683baeb03`
- tensor reload mismatches: `0`

Source:

```text
docs/QWEN35_ABLITERATED_PMRA.md
docs/HUIHUI_QWEN35_4B_ABLITERATED_HF_MODEL_CARD.md
https://huggingface.co/Asystemoffields/Huihui-Qwen3.5-4B-Abliterated-PMRA-GGUF
```

### PMRA-023 - Ministral 3 8B Instruct Knapsack Release Candidate

Ministral 3 8B Instruct was completed with the `mistral3` tensor profile and
tensor-level `c2_calib_knapsack_mixed` allocation. The 48/512 selector shape was
attempted but projected beyond the Modal job window, so this record uses the
completed 12/128 scout selector and a separate 512-prompt held-out Wikitext test
confirmation.

Wikitext-2 raw train -> validation, seed `7`:

- primary artifact: `ministral3_8b_pmra_knapsack_iq3xs_budget.gguf`
- compact artifact: `ministral3_8b_pmra_knapsack_3p2.gguf`
- group mode: `tensor`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K`, `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- prompt audit overlap count: `0`
- `IQ2_M` NLL: `4.874847`
- `IQ3_XS` target NLL: `4.649152`
- `Q3_K_S` NLL: `4.686507`
- PMRA knapsack NLL: `4.456880`
- PMRA 3.2 bpw NLL: `4.510145`
- same-budget random NLL: `4.825388`
- PMRA NLL improvement vs `IQ3_XS`: `0.192272`
- PMRA NLL improvement vs `Q3_K_S`: `0.229628`
- PMRA NLL improvement vs same-budget random: `0.368508`
- PMRA payload bytes vs `IQ3_XS`: `-557,056`
- compact PMRA payload bytes vs `IQ3_XS`: `-310,935,552`

Held-out Wikitext-2 raw test, 512 prompts:

- public eval decision: `GO`
- `IQ2_M` NLL: `4.963936`
- `IQ3_XS` target NLL: `4.722369`
- `Q3_K_S` NLL: `4.757542`
- PMRA knapsack NLL: `4.537475`
- PMRA 3.2 bpw NLL: `4.600533`
- same-budget random NLL: `4.912780`
- NLL improvement vs `IQ3_XS`: `0.184894`
- NLL improvement vs `Q3_K_S`: `0.220067`
- NLL improvement vs same-budget random: `0.375305`
- payload bytes vs `IQ3_XS`: `-557,056`
- payload bytes vs `Q3_K_S`: `-152,666,112`
- compact NLL improvement vs `IQ3_XS`: `0.121836`
- compact payload bytes vs `IQ3_XS`: `-310,935,552`

The primary and compact selections were materialized as loadable GGUFs:

- primary file size: `3,713,801,312` bytes
- primary payload bpw: `3.492210`
- primary file bpw: `3.499643`
- primary GGUF SHA-256:
  `7f88294593cf419a5b39b4da2c7df356fee9528de947d6547b9d11d60a84ac5d`
- compact file size: `3,403,422,816` bytes
- compact payload bpw: `3.199730`
- compact file bpw: `3.207163`
- compact GGUF SHA-256:
  `ff95384e68f211b238767e1783d20ce0b4a8be8a56ac8b906756c481831421a3`
- tensor reload mismatches: `0` for both artifacts

Source:

```text
docs/MINISTRAL3_8B_INSTRUCT_PMRA.md
docs/MINISTRAL3_8B_INSTRUCT_HF_MODEL_CARD.md
tmp/ministral3_8b_release/public_eval_wikitext_test_result.md
tmp/ministral3_8b_release/artifact_iq3xs_budget_report.md
tmp/ministral3_8b_release/artifact_3p2_report.md
```

### PMRA-024 - Granite 4.1 8B Heretic Knapsack Release

Granite 4.1 8B Heretic was completed with a new `granite` tensor profile and
layer-family `c2_calib_knapsack_mixed` allocation. The release docs and HF model
card credit `heretic-org` as the upstream Heretic checkpoint provider.

Wikitext-2 raw train -> validation, seed `7`:

- artifact: `granite4_1_8b_heretic_pmra_layer_family_iq3xs_budget.gguf`
- HF release repo:
  `https://huggingface.co/Asystemoffields/IBM-granite-4.1-8b-heretic-PMRA-GGUF`
- base model: `heretic-org/IBM-granite-4.1-8b-heretic`
- GGUF source repo: `mradermacher/IBM-granite-4.1-8b-heretic-i1-GGUF`
- group mode: `layer_family`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K_S`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- prompt audit overlap count: `0`
- `IQ2_M` NLL: `5.028465`
- `IQ3_XS` target NLL: `4.845994`
- `Q2_K` NLL: `4.707305`
- `Q3_K_S` NLL: `4.823249`
- `IQ4_XS` NLL: `4.579525`
- PMRA knapsack NLL: `4.469497`
- same-budget random NLL: `4.840297`
- PMRA NLL improvement vs `IQ3_XS`: `0.376498`
- PMRA NLL improvement vs `Q3_K_S`: `0.353752`
- PMRA NLL improvement vs same-budget random: `0.370800`
- PMRA payload bytes vs `IQ3_XS`: `-1,392,640`
- PMRA payload bytes vs `Q3_K_S`: `-165,888,000`

Held-out Wikitext-2 raw test, 512 prompts:

- public eval decision: `GO`
- `IQ2_M` NLL: `5.150425`
- `IQ3_XS` target NLL: `4.960251`
- `Q2_K` NLL: `4.754195`
- `Q3_K_S` NLL: `4.933018`
- `IQ4_XS` NLL: `4.672932`
- PMRA knapsack NLL: `4.539084`
- same-budget random NLL: `4.939853`
- NLL improvement vs `IQ3_XS`: `0.421167`
- NLL improvement vs `Q3_K_S`: `0.393934`
- NLL improvement vs same-budget random: `0.400769`
- payload bytes vs `IQ3_XS`: `-1,392,640`
- payload bytes vs `Q3_K_S`: `-165,888,000`

The selected tensor payloads were materialized as one loadable GGUF:

- file size: `3,600,448,224` bytes
- payload bytes: `3,596,877,824`
- payload bpw: `3.433548`
- file bpw: `3.436956`
- GGUF SHA-256:
  `29d3d2b33583127789ee26b0b5e1d7204cb5330af2c265bef6b42c7a4a4a291a`
- tensor reload mismatches: `0`

Source:

```text
docs/GRANITE4_1_8B_HERETIC_PMRA.md
docs/GRANITE4_1_8B_HERETIC_HF_MODEL_CARD.md
tmp/granite4_1_8b_heretic/release/public_eval_wikitext_test_result.md
tmp/granite4_1_8b_heretic/release/artifact_report.md
```

### PMRA-008 - Replication

Partial clean replication on Qwen3-0.6B.

With `prompt_audit.overlap_count = 0` on seeds `6`, `7`, and `8`, the weight-MSE mixed selector beat `IQ3_M`, `IQ4_XS`, and random same-budget allocation while staying slightly below `IQ3_M` payload bytes.

- mean NLL improvement vs `IQ3_M`: `0.249980`
- mean NLL improvement vs `IQ4_XS`: `0.057421`
- mean NLL improvement vs random same-budget: `0.244507`
- mean NLL delta vs `Q3_K_M`: `-0.127205`
- payload bytes vs `IQ3_M`: `-65,536`

Source:

```text
codex_ladder/runs/run_008/stage5_c2_qwen06_replication_clean_result.md
```
