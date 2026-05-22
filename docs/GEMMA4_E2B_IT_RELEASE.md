# Gemma 4 E2B-it PMRA Knapsack Release Notes

## Scope

This is the current PMRA frontier mix in the repo. It is the first file a
Hugging Face reader should inspect if they want the strongest published PMRA
example here.

PMRA here means mixing tensor payloads from existing production GGUF
quantizations. The selector starts from a small GGUF source and promotes
selected tensor groups to stronger source formats when calibration says the
extra bytes are worth it. The output is still one normal GGUF.

## Recommended Mix

```text
gemma4_e2b_it_pmra_calib_knapsack.gguf
```

Hugging Face model repo:

```text
https://huggingface.co/Asystemoffields/gemma-4-E2B-it-PMRA-GGUF
```

- base model: `google/gemma-4-E2B-it`
- source GGUF repo: `mradermacher/gemma-4-E2B-it-GGUF`
- low source: `Q2_K`
- target/control budget: `Q3_K_S`
- stronger sources: `Q3_K_M`, `Q3_K_L`, `IQ4_XS`, `Q4_K_M`
- selector: `c2_calib_knapsack_mixed`
- calibration: Wikitext-2 raw train, 24 prompts
- evaluation: Wikitext-2 raw validation, 256 prompts
- prompt overlap: `0`
- file size: `3,110,215,968` bytes
- tensor payload bytes: `3,094,397,068`
- payload bpw: `5.326615`
- file bpw: `5.353845`
- SHA-256: `a5a80f2628e236a228f2016bcc3ac660a268f2c8757d21d901095c74b60e3d97`
- tensor reload mismatches: `0`

The previous greedy mix remains available as:

```text
gemma4_e2b_it_pmra_calib_greedy.gguf
```

## Selector Result

Lower NLL is better.

| Variant | NLL | Payload bpw | Payload bytes |
|---|---:|---:|---:|
| fp16 reference | 14.381222 | 16.000000 | 9,294,899,782 |
| `Q2_K` | 20.376913 | 5.118105 | 2,973,267,084 |
| `Q3_K_S` target | 17.993582 | 5.326613 | 3,094,396,044 |
| `Q3_K_M` | 15.619944 | 5.483489 | 3,185,529,996 |
| `Q3_K_L` | 15.756687 | 5.622925 | 3,266,532,492 |
| `IQ4_XS` | 16.043206 | 5.670221 | 3,294,008,460 |
| `Q4_K_M` | 13.549753 | 5.873431 | 3,412,059,276 |
| same-budget random | 20.488594 | 5.326613 | 3,094,396,044 |
| PMRA `c2_calib_greedy_mixed` | 13.281400 | 5.326291 | 3,094,208,652 |
| PMRA `c2_calib_knapsack_mixed` | 12.878809 | 5.326613 | 3,094,396,044 |

Key comparisons:

- knapsack PMRA vs `Q3_K_S` target: `+5.114774` NLL improvement.
- knapsack PMRA vs same-budget random: `+7.609785` NLL improvement.
- knapsack PMRA vs greedy PMRA: `+0.402591` NLL improvement.
- selector-reported payload bytes vs `Q3_K_S` target: `0`.
- materialized artifact payload bytes vs `Q3_K_S` target: `+1,024`.
- selected tensor groups: `204`.

## Runtime Check

Local llama.cpp build: `a8fd165`.

- `llama-cli` loaded the GGUF and generated text from a single-turn prompt.
- smoke prompt speed: `30.5` prompt tok/s.
- smoke generation speed: `10.6` tok/s.

## Evidence Files

```text
artifact_report_knapsack.json
artifact_report_knapsack.md
selector_result_knapsack.json
selector_result_knapsack.md
llama_cli_smoke_knapsack.log
```

## Interpretation

The knapsack selector improves the Wikitext validation frontier at the same
`Q3_K_S` selector budget. It spends `187,392` more selector-reported payload
bytes than the greedy selector and improves validation NLL by `0.402591`.

The mix is publishable because it is materialized, reloads with zero tensor
mismatches, carries PMRA metadata, and passes a llama.cpp runtime smoke.

The Qwen3-1.7B mix remains important prior evidence, but Gemma 4 E2B-it
knapsack is the recommended current mix.
