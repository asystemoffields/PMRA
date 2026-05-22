# Next PMRA Model Targets

This note records the next models worth treating with PMRA after the Gemma 4
E2B-it, Huihui Qwen3.5, Ministral 3 8B, and Granite 4.1 8B Heretic releases.

## Completed Since This Note Started

- `ministral3_8b_instruct` is now released as a primary IQ3_XS-budget PMRA GGUF
  plus a compact 3.2 bpw GGUF.
- `granite4_1_8b_heretic` is now released as a `granite` profile,
  layer-family PMRA GGUF. The release docs and HF card credit
  `heretic-org/IBM-granite-4.1-8b-heretic` and link the Heretic organization.

## First Finish - Ministral 3 8B Instruct 2512

Status: completed. See:

- [Ministral 3 8B Instruct PMRA](MINISTRAL3_8B_INSTRUCT_PMRA.md)
- [Granite 4.1 8B Heretic PMRA](GRANITE4_1_8B_HERETIC_PMRA.md)

The repo already has a local `mistral3` profile result for
`mistralai/Ministral-3-8B-Instruct-2512-BF16`.

Existing local result:

```text
tmp/ministral3_8b_pmra_3p2/selector_result.json
tmp/ministral3_8b_pmra_3p2/artifact_report.json
```

Recipe:

- model key: `ministral3_8b_instruct`
- HF model: `mistralai/Ministral-3-8B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF`
- tensor profile: `mistral3`
- group mode: `tensor`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger sources: `Q2_K`, `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `IQ4_XS`
- selector comparison run: `c2_calib_knapsack_mixed` plus greedy, blend,
  random, and bpw sweep variants

Validation table from the existing local result:

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | `2.546905` | `16.000000` | `16,979,107,840` |
| `IQ2_M` | `4.874847` | `2.920126` | `3,098,820,608` |
| `IQ3_XS` target | `4.649152` | `3.492735` | `3,706,470,400` |
| `Q3_K_S` | `4.686507` | `3.636073` | `3,858,579,456` |
| `Q3_K_M` | `4.651231` | `3.990001` | `4,234,166,272` |
| `IQ4_XS` | `4.611839` | `4.418161` | `4,688,527,360` |
| PMRA greedy | `4.500706` | `3.457719` | `3,669,311,488` |
| PMRA knapsack | `4.456880` | `3.492210` | `3,705,913,344` |
| PMRA weight blend | `4.512960` | `3.492735` | `3,706,470,400` |
| PMRA knapsack 3.2 bpw | `4.510145` | `3.199730` | `3,395,534,848` |
| PMRA knapsack 3.4 bpw | `4.463643` | `3.399729` | `3,607,773,184` |
| PMRA knapsack 3.6 bpw | `4.440595` | `3.598092` | `3,818,274,816` |
| same-budget random | `4.825388` | `3.492210` | `3,705,913,344` |

Read:

- quality winner under the `IQ3_XS` target budget:
  `c2_calib_knapsack_mixed`
- smallest already-materialized strong point:
  `c2_calib_knapsack_bpw_3p200_mixed`
- best validation NLL in the sweep:
  `c2_calib_knapsack_bpw_3p600_mixed`, but it exceeds the `IQ3_XS` payload
  budget

The existing 3.2 bpw artifact is already loadable:

- file: `ministral3_8b_pmra_3p2.gguf`
- file size: `3,403,422,816` bytes
- payload bytes: `3,395,534,848`
- payload bpw: `3.199730`
- tensor reload mismatches: `0`
- NLL improvement vs `IQ3_XS`: `0.139007`
- payload bytes vs `IQ3_XS`: `-310,935,552`

Recommended finish path:

1. Rerun or verify the seed `7` selector result in Modal so the source result is
   in the canonical volume, not only `tmp/`.
2. Evaluate these variants on Wikitext test, TinyStories validation, and
   LAMBADA English test:
   - `fp16`
   - `iq2_m`
   - `iq3_xs`
   - `q3_k_s`
   - `c2_calib_knapsack_mixed`
   - `c2_calib_knapsack_bpw_3p200_mixed`
   - `c2_random_same_budget`
3. If the target-budget knapsack survives held-out eval, release it as the main
   Ministral 3 8B PMRA artifact.
4. If the 3.2 bpw point survives too, publish it as a smaller frontier variant
   or include it as a second GGUF.
5. Repeat seeds `6` and `8` before making a broad family claim.

Selector command:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_public_calibrated `
  --model-keys ministral3_8b_instruct `
  --seed 7 `
  --calib-prompts 48 `
  --eval-prompts 512 `
  --calib-max-length 192 `
  --eval-max-length 256 `
  --group-mode tensor `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k,q2_k_l,q3_k_s,q3_k_m,iq4_xs `
  --candidate-variant c2_calib_knapsack_mixed `
  --sweep-payload-bpws 3.2,3.4,3.6 `
  --sweep-selectors calib_knapsack `
  --result-bucket run_013_ministral3_8b_pmra
```

Build the likely quality-winner artifact:

```powershell
modal run .\modal\modal_sprint.py::phase_c2_artifact `
  --model-key ministral3_8b_instruct `
  --variant c2_calib_knapsack_mixed `
  --low-source iq2_m `
  --target-source iq3_xs `
  --high-sources q2_k,q2_k_l,q3_k_s,q3_k_m,iq4_xs `
  --result-bucket run_013_ministral3_8b_pmra `
  --result-name <modal-result-name> `
  --artifact-bucket run_013_ministral3_8b_artifact `
  --artifact-name ministral3_8b_pmra_calib_knapsack_artifact `
  --output-gguf ministral3_8b_pmra_calib_knapsack.gguf
```

## Same-Family Replication Targets

### Ministral 3 3B Instruct 2512

Use this as the fast replication point. It is already wired as
`ministral3_3b_instruct`.

- HF model: `mistralai/Ministral-3-3B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF`
- tensor profile: `mistral3`
- layers: `26`
- available GGUF controls include `IQ2_M`, `IQ3_XS`, `Q2_K`, `Q2_K_L`,
  `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, and `Q4_K_M`

### Ministral 3 14B Instruct 2512

Use this after 8B/3B if the family signal holds. It is now wired as
`ministral3_14b_instruct`.

- HF model: `mistralai/Ministral-3-14B-Instruct-2512-BF16`
- GGUF source repo: `bartowski/mistralai_Ministral-3-14B-Instruct-2512-GGUF`
- tensor profile: `mistral3`
- layers: `40`
- available GGUF controls include `IQ2_M`, `IQ2_S`, `IQ3_XS`, `Q2_K`,
  `Q2_K_L`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, and `Q4_K_M`

## Other Good Next Targets

### AllenAI OLMo Plain PMRA Queue

These are plain PMRA targets only. Do not run Heretic, abliterating, or other
behavioral editing passes on AllenAI checkpoints.

1. `olmo2_0425_1b_instruct`
   - HF model: `allenai/OLMo-2-0425-1B-Instruct`
   - GGUF source repo: `mradermacher/OLMo-2-0425-1B-Instruct-i1-GGUF`
   - tensor profile: `olmo2`
   - layers: `16`
   - role: fast preservation/sanity target before spending 7B compute
2. `olmo2_1124_7b_instruct`
   - HF model: `allenai/OLMo-2-1124-7B-Instruct`
   - GGUF source repo: `mradermacher/OLMo-2-1124-7B-Instruct-i1-GGUF`
   - tensor profile: `olmo2`
   - layers: `32`
   - role: main OLMo2 shrink target
3. `olmo3_7b_think`
   - HF model: `allenai/Olmo-3-7B-Think`
   - GGUF source repo: `mradermacher/Olmo-3-7B-Think-i1-GGUF`
   - tensor profile: `olmo3`
   - layers: `32`
   - role: follow-on reasoning target after both OLMo2 runs are done

OLMo 3 Think is Apache-2.0 and has a full imatrix GGUF spread. Its HF tensor
names match the OLMo2 layer/norm pattern, so the repo wires `olmo3` as an
explicit profile alias rather than folding it silently into `olmo2`.

### Heretic Granite Follow-Ups

The first Heretic target passed on Granite 4.1 8B. A sensible next step is a
same-source replication on another Heretic checkpoint only if it has:

- a permissive or clearly redistributable license
- an upstream model card that identifies the base model
- multiple low-bit GGUF controls from the same checkpoint
- enough architectural overlap with an existing tensor profile, or a small new
  profile surface

Keep `heretic-org` attribution explicit in release docs and HF metadata for any
follow-up Heretic artifact.

### Qwen3 8B

This is the cleanest reuse of the existing `qwen` profile at a larger scale.
The `bartowski/Qwen_Qwen3-8B-GGUF` repo has the full low-bit spread needed for
PMRA. It is less novel than the Ministral family because this repo already has
Qwen3-1.7B, Qwen3-0.6B, and Qwen3.5 evidence.

### Qwen3 4B Instruct 2507

This is a good mid-size Qwen target if we want another practical release point.
Use `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, not the official
`Qwen/Qwen3-4B-GGUF`, because the official GGUF repo has too few low-bit
formats for a useful PMRA selector sweep.

### SmolLM3 3B

This is attractive because it is Apache-2.0, small, popular, and has a full
set of Bartowski GGUF controls. It needs a new `smollm3` or llama-like tensor
profile before running PMRA.

### Gemma 3 4B-it

This has plenty of GGUF controls and is a useful comparison to the current
Gemma frontier, but it likely needs a distinct `gemma3` profile and the license
story is less clean than Apache-2.0 targets.

## Suggested Queue

1. Run plain PMRA on `olmo2_0425_1b_instruct`.
2. Run plain PMRA on `olmo2_1124_7b_instruct`.
3. Follow with plain PMRA on `olmo3_7b_think`.
4. Replicate Ministral on `ministral3_3b_instruct`.
5. If both pass, run `ministral3_14b_instruct`.
6. Add a Qwen3 8B or Qwen3 4B Instruct 2507 release for same-profile scale.
7. Evaluate one more Heretic-org target only if the license and GGUF coverage
   are as clean as Granite 4.1 8B Heretic.
8. Add a new profile for SmolLM3 3B if we want a small Apache architecture
   outside Qwen/Mistral/Gemma.
