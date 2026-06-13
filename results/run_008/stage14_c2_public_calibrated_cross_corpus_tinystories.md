# Result Card - Stage 14 Public-Calibrated PMRA Cross-Corpus Eval

## Status

GO.

The public-calibrated PMRA selections survived a cross-corpus public evaluation
on `roneneldan/TinyStories` validation. This does not establish broad benchmark
dominance, but it does move the result past a Wikitext-only selector concern.

## Setup

- calibration corpus: Wikitext-2 raw train
- selection source: Stage 13 public-calibrated PMRA result
- eval corpus: `roneneldan/TinyStories`
- eval split: validation
- eval prompts: `512`
- eval max length: `256`
- prompt hash: `72f55f62922d0ff02119b44bc4c66bd6eae8e5177ef8732231a8754816949c75`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger control: `Q3_K_S`

## Qwen3-1.7B Public-Calibrated Selector

Selector: `c2_calib_greedy_mixed`.

- PMRA NLL: `2.060892`
- `IQ3_XS` NLL: `2.147862`
- `Q3_K_S` NLL: `2.299919`
- same-budget random NLL: `2.241572`
- NLL improvement vs `IQ3_XS`: `0.086969`
- NLL improvement vs `Q3_K_S`: `0.239027`
- NLL improvement vs same-budget random: `0.180680`
- payload bytes: `955,742,208`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`
- payload bpw: `3.763246`

The blend variant was also stronger on quality but is nearly exactly at the
`IQ3_XS` budget and lacks the same-budget random framing used for the release
candidate:

- blend NLL: `2.039977`
- blend payload bytes: `961,968,128`

## Qwen3-0.6B-Base Public-Calibrated Blend

Selector: `c2_calib_weight_blend_mixed`.

- PMRA NLL: `2.457967`
- `IQ3_XS` NLL: `2.598263`
- `Q3_K_S` NLL: `2.538557`
- same-budget random NLL: `2.817023`
- NLL improvement vs `IQ3_XS`: `0.140297`
- NLL improvement vs `Q3_K_S`: `0.080590`
- NLL improvement vs same-budget random: `0.359056`
- payload bytes: `373,584,896`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`
- payload bpw: `3.976251`

## Decision

Gate 6C passes. PMRA is now a public-calibrated experimental release candidate
with:

- Wikitext train -> validation pass
- frozen Wikitext test pass
- TinyStories validation cross-corpus pass
- second-size Qwen3 replication
- materialized Qwen3-1.7B GGUF artifact
- local llama.cpp load and bench evidence

Proceed to release-package cleanup and a larger public benchmark mix. Claims
should still be scoped as tensor-level mixed allocation over existing GGUF
payloads.
