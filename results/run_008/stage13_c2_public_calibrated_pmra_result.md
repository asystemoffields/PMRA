# Result Card - Stage 13 Public-Calibrated PMRA

## Status

GO for a production-shaped experimental release candidate.

This does not make a cross-family or broad benchmark claim. It does establish a
real PMRA artifact that beats the relevant public GGUF controls on Wikitext
held-out validation and test while staying under the `IQ3_XS` byte budget.

## Qwen3-1.7B Public-Calibrated Selector

Setup:

- calibration corpus: Wikitext-2 raw train
- validation corpus: Wikitext-2 raw validation
- frozen test corpus: Wikitext-2 raw test
- calibration prompts: `48`
- validation/test prompts: `512`
- eval max length: `256`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger control: `Q3_K_S`
- selector: `c2_calib_greedy_mixed`

Validation:

- PMRA NLL: `3.279698`
- NLL improvement vs `IQ3_XS`: `0.156058`
- NLL improvement vs `Q3_K_S`: `0.223327`
- NLL improvement vs same-budget random: `0.229276`
- payload bytes: `955,742,208`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`
- payload bpw: `3.763246`

Frozen Wikitext test:

- PMRA NLL: `3.227878`
- NLL improvement vs `IQ3_XS`: `0.145818`
- NLL improvement vs `Q3_K_S`: `0.214353`
- NLL improvement vs same-budget random: `0.217721`
- payload bytes: `955,742,208`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`
- payload bpw: `3.763246`

## Qwen3-0.6B-Base Public-Calibrated Blend

The `c2_calib_weight_blend_mixed` selector was the stronger 0.6B operating
point. Calibration-greedy beat `IQ3_XS` but missed `Q3_K_S`; blend cleared both.

Validation:

- PMRA NLL: `3.776832`
- NLL improvement vs `IQ3_XS`: `0.218488`
- NLL improvement vs `Q3_K_S`: `0.153261`
- NLL improvement vs random control: `0.375228`
- payload bytes: `373,584,896`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`
- payload bpw: `3.976251`

Frozen Wikitext test:

- PMRA NLL: `3.733991`
- NLL improvement vs `IQ3_XS`: `0.225181`
- NLL improvement vs `Q3_K_S`: `0.141294`
- NLL improvement vs random control: `0.388764`
- payload bytes: `373,584,896`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`
- payload bpw: `3.976251`

## Qwen3-1.7B Artifact

The 1.7B public-calibrated selector was materialized as one GGUF.

- artifact status: `GO`
- file size: `961,694,976` bytes
- payload bytes: `955,742,208`
- file bpw: `3.786685`
- payload bpw: `3.763246`
- tensor count: `311`
- mismatched tensors: `0`
- PMRA metadata fields: present

Local artifact:

```text
qwen17_publiccal_pmra_calib_greedy.gguf
```

## Local llama.cpp Bench

`llama-bench`, CPU, `-p 128 -n 64 -r 3`:

| Artifact | Prompt tok/s | Decode tok/s | Payload bytes |
|---|---:|---:|---:|
| PMRA public-calibrated | `37.6608` | `10.5323` | `955,742,208` |
| `IQ3_XS` | `11.8432` | `8.0709` | `961,974,272` |
| `Q3_K_S` | `27.3281` | `6.9624` | `995,004,416` |

`llama-cli` also loaded the artifact and generated text. This local build of
`llama-cli` remains in chat mode after generation and was terminated with exit
code `137`; `llama-bench` provides the clean load/runtime check.

## Decision

Proceed to packaging and one additional cross-corpus public eval. Do not claim
general benchmark dominance yet; do claim a public Wikitext-backed PMRA method
candidate with a materialized GGUF artifact.
