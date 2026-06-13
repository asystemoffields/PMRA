# Result Card - Stage 15 Public-Calibrated PMRA LAMBADA Eval

## Status

GO.

The public-calibrated PMRA selections survived `EleutherAI/lambada_openai`
English test evaluation on both Qwen3 sizes.

## Setup

- calibration corpus: Wikitext-2 raw train
- selection source: Stage 13 public-calibrated PMRA result
- eval corpus: `EleutherAI/lambada_openai`
- eval config: `en`
- eval split: test
- eval prompts: `512`
- eval max length: `256`
- low source: `IQ2_M`
- target/control: `IQ3_XS`
- stronger control: `Q3_K_S`

## Qwen3-1.7B Public-Calibrated Selector

Selector: `c2_calib_greedy_mixed`.

- PMRA NLL: `3.790112`
- `IQ3_XS` NLL: `3.900607`
- `Q3_K_S` NLL: `3.926747`
- same-budget random NLL: `3.974073`
- NLL improvement vs `IQ3_XS`: `0.110495`
- NLL improvement vs `Q3_K_S`: `0.136635`
- NLL improvement vs same-budget random: `0.183961`
- payload bytes: `955,742,208`
- payload bytes vs `IQ3_XS`: `-6,232,064`
- payload bytes vs `Q3_K_S`: `-39,262,208`
- payload bpw: `3.763246`

## Qwen3-0.6B-Base Public-Calibrated Blend

Selector: `c2_calib_weight_blend_mixed`.

- PMRA NLL: `4.139370`
- `IQ3_XS` NLL: `4.316923`
- `Q3_K_S` NLL: `4.230483`
- same-budget random NLL: `4.491401`
- NLL improvement vs `IQ3_XS`: `0.177553`
- NLL improvement vs `Q3_K_S`: `0.091113`
- NLL improvement vs same-budget random: `0.352031`
- payload bytes: `373,584,896`
- payload bytes vs `IQ3_XS`: `-68,608`
- payload bytes vs `Q3_K_S`: `-10,390,528`
- payload bpw: `3.976251`

## Decision

Gate 8 initial public benchmark mix passes. The release artifact now has
positive public evidence on Wikitext, TinyStories, and LAMBADA. This supports
publishing the method and the experimental GGUF with scoped claims.

Public-calibrated tensor-level PMRA beat `IQ3_XS`, `Q3_K_S`, and same-budget
random at smaller payload size on the public corpora tested for Qwen3 sizes.
